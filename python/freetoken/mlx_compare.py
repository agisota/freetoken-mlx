"""Compare captured FreeToken-MLX runs and enforce benchmark gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

MAX_LOG_BYTES = 64 << 20
REPORT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RunRecord:
    path: str
    label: str
    generated_tokens: int
    elapsed_seconds: float
    wall_tokens_per_second: float
    peak_memory_bytes: int | None
    cache_misses: int | None
    output_sha256: str
    output_bytes: int
    residency: str | None


def _number(value: Any, *, field: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    if positive and number <= 0:
        raise ValueError(f"{field} must be positive")
    if not positive and number < 0:
        raise ValueError(f"{field} cannot be negative")
    return number


def _optional_nonnegative_int(value: Any, *, field: str) -> int | None:
    if value is None:
        return None
    number = _number(value, field=field)
    if not number.is_integer():
        raise ValueError(f"{field} must be an integer")
    return int(number)


def _split_stdout_log(path: Path) -> tuple[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"run log does not exist: {path}")
    size = path.stat().st_size
    if size > MAX_LOG_BYTES:
        raise ValueError(
            f"run log is too large ({size} bytes; limit {MAX_LOG_BYTES}): {path}"
        )
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    report_index = next(
        (index for index in range(len(lines) - 1, -1, -1) if lines[index].strip()),
        None,
    )
    if report_index is None:
        raise ValueError(f"run log is empty: {path}")
    try:
        report = json.loads(lines[report_index].strip())
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"last non-empty stdout line is not a JSON report: {path}: {exc}"
        ) from exc
    if not isinstance(report, dict):
        raise ValueError(f"final JSON report must be an object: {path}")
    if any(line.strip() for line in lines[report_index + 1 :]):
        raise ValueError(f"unexpected content follows the final JSON report: {path}")

    output = "".join(lines[:report_index])
    # `ft generate` prints one separator newline before the final JSON report.
    # Remove exactly that separator while preserving a newline emitted by the model.
    if output.endswith("\r\n"):
        output = output[:-2]
    elif output.endswith("\n"):
        output = output[:-1]
    return output, report


def load_run(path: str | Path, *, label: str) -> RunRecord:
    resolved = Path(path).expanduser().resolve()
    output, report = _split_stdout_log(resolved)
    if report.get("backend") != "mlx":
        raise ValueError(f"report backend must be 'mlx': {resolved}")

    generated = _optional_nonnegative_int(
        report.get("generated_tokens"), field="generated_tokens"
    )
    if generated is None:
        raise ValueError(f"generated_tokens is missing: {resolved}")
    elapsed = _number(
        report.get("elapsed_seconds"), field="elapsed_seconds", positive=True
    )

    memory = report.get("memory")
    if memory is not None and not isinstance(memory, dict):
        raise ValueError(f"memory must be an object or null: {resolved}")
    peak = _optional_nonnegative_int(
        memory.get("peak_bytes") if isinstance(memory, dict) else None,
        field="memory.peak_bytes",
    )

    expert_cache = report.get("expert_cache")
    if expert_cache is not None and not isinstance(expert_cache, dict):
        raise ValueError(f"expert_cache must be an object or null: {resolved}")
    misses = _optional_nonnegative_int(
        expert_cache.get("misses") if isinstance(expert_cache, dict) else None,
        field="expert_cache.misses",
    )

    residency_data = report.get("residency")
    residency = None
    if residency_data is not None:
        if not isinstance(residency_data, dict):
            raise ValueError(f"residency must be an object or null: {resolved}")
        selected = residency_data.get("selected")
        if selected is not None and not isinstance(selected, str):
            raise ValueError(f"residency.selected must be a string: {resolved}")
        residency = selected

    output_bytes = output.encode("utf-8")
    return RunRecord(
        path=str(resolved),
        label=label,
        generated_tokens=generated,
        elapsed_seconds=elapsed,
        wall_tokens_per_second=generated / elapsed,
        peak_memory_bytes=peak,
        cache_misses=misses,
        output_sha256=hashlib.sha256(output_bytes).hexdigest(),
        output_bytes=len(output_bytes),
        residency=residency,
    )


def _median(values: list[float | int | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return statistics.median(present) if present else None


def summarize(records: list[RunRecord]) -> dict[str, Any]:
    if not records:
        raise ValueError("each comparison group needs at least one run")
    return {
        "count": len(records),
        "generated_tokens": sorted({record.generated_tokens for record in records}),
        "median_elapsed_seconds": _median(
            [record.elapsed_seconds for record in records]
        ),
        "median_wall_tokens_per_second": _median(
            [record.wall_tokens_per_second for record in records]
        ),
        "median_peak_memory_bytes": _median(
            [record.peak_memory_bytes for record in records]
        ),
        "median_cache_misses": _median(
            [record.cache_misses for record in records]
        ),
        "output_sha256": sorted({record.output_sha256 for record in records}),
        "residency": sorted(
            {record.residency for record in records if record.residency is not None}
        ),
        "runs": [asdict(record) for record in records],
    }


def compare_runs(
    baseline: list[RunRecord],
    candidate: list[RunRecord],
    *,
    expected_tokens: int,
    min_throughput_ratio: float = 0.97,
    max_peak_memory_ratio: float = 1.05,
    max_cache_miss_ratio: float | None = None,
    require_output_match: bool = False,
    require_same_residency: bool = False,
) -> dict[str, Any]:
    if expected_tokens < 1:
        raise ValueError("expected_tokens must be at least 1")
    if min_throughput_ratio <= 0:
        raise ValueError("min_throughput_ratio must be positive")
    if max_peak_memory_ratio <= 0:
        raise ValueError("max_peak_memory_ratio must be positive")
    if max_cache_miss_ratio is not None and max_cache_miss_ratio <= 0:
        raise ValueError("max_cache_miss_ratio must be positive")

    baseline_summary = summarize(baseline)
    candidate_summary = summarize(candidate)
    violations: list[str] = []

    for record in baseline + candidate:
        if record.generated_tokens != expected_tokens:
            violations.append(
                f"{record.label} run generated {record.generated_tokens} tokens, "
                f"expected exactly {expected_tokens}: {record.path}"
            )

    baseline_tps = baseline_summary["median_wall_tokens_per_second"]
    candidate_tps = candidate_summary["median_wall_tokens_per_second"]
    throughput_ratio = None
    if baseline_tps is None or baseline_tps <= 0:
        violations.append(
            "throughput gate requires a positive baseline token rate; "
            "check generated_tokens and elapsed_seconds"
        )
    elif candidate_tps is None:
        violations.append("throughput gate requires candidate timing metrics")
    else:
        throughput_ratio = candidate_tps / baseline_tps
        if throughput_ratio < min_throughput_ratio:
            violations.append(
                f"throughput ratio {throughput_ratio:.6f} is below "
                f"minimum {min_throughput_ratio:.6f}"
            )

    all_records = baseline + candidate
    missing_peak = [
        record.path for record in all_records if record.peak_memory_bytes is None
    ]
    baseline_peak = baseline_summary["median_peak_memory_bytes"]
    candidate_peak = candidate_summary["median_peak_memory_bytes"]
    peak_memory_ratio = None
    if missing_peak:
        violations.append(
            "peak-memory gate requires memory.peak_bytes in every run; missing: "
            + ", ".join(missing_peak)
        )
    elif baseline_peak == 0:
        peak_memory_ratio = 1.0 if candidate_peak == 0 else None
        if candidate_peak != 0:
            violations.append(
                "candidate reports non-zero peak memory while baseline reports zero"
            )
    else:
        # Both medians are non-null because every run was checked above.
        peak_memory_ratio = candidate_peak / baseline_peak
        if peak_memory_ratio > max_peak_memory_ratio:
            violations.append(
                f"peak-memory ratio {peak_memory_ratio:.6f} exceeds "
                f"maximum {max_peak_memory_ratio:.6f}"
            )

    cache_miss_ratio = None
    if max_cache_miss_ratio is not None:
        missing_misses = [
            record.path for record in all_records if record.cache_misses is None
        ]
        baseline_misses = baseline_summary["median_cache_misses"]
        candidate_misses = candidate_summary["median_cache_misses"]
        if missing_misses:
            violations.append(
                "cache-miss gate requires expert_cache.misses in every run; missing: "
                + ", ".join(missing_misses)
            )
        elif baseline_misses == 0:
            cache_miss_ratio = 1.0 if candidate_misses == 0 else None
            if candidate_misses != 0:
                violations.append(
                    "candidate reports cache misses while baseline reports zero"
                )
        else:
            # Both medians are non-null because every run was checked above.
            cache_miss_ratio = candidate_misses / baseline_misses
            if cache_miss_ratio > max_cache_miss_ratio:
                violations.append(
                    f"cache-miss ratio {cache_miss_ratio:.6f} exceeds "
                    f"maximum {max_cache_miss_ratio:.6f}"
                )

    if require_output_match:
        if any(record.output_bytes == 0 for record in all_records):
            violations.append(
                "output matching was requested, but at least one stdout log has no "
                "generated-text prefix before the JSON report"
            )
        output_hashes = {record.output_sha256 for record in all_records}
        if len(output_hashes) != 1:
            violations.append(
                "generated output differs across baseline/candidate runs: "
                + ", ".join(sorted(output_hashes))
            )

    if require_same_residency:
        missing_residency = [
            record.path for record in all_records if record.residency is None
        ]
        baseline_residency = set(baseline_summary["residency"])
        candidate_residency = set(candidate_summary["residency"])
        if missing_residency:
            violations.append(
                "residency matching requires residency.selected in every run; missing: "
                + ", ".join(missing_residency)
            )
        elif len(baseline_residency) != 1 or len(candidate_residency) != 1:
            violations.append(
                "each group must use one residency path: "
                f"baseline={sorted(baseline_residency)} "
                f"candidate={sorted(candidate_residency)}"
            )
        elif baseline_residency != candidate_residency:
            violations.append(
                "residency sets differ: "
                f"baseline={sorted(baseline_residency)} "
                f"candidate={sorted(candidate_residency)}"
            )

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "pass": not violations,
        "thresholds": {
            "expected_tokens": expected_tokens,
            "min_throughput_ratio": min_throughput_ratio,
            "max_peak_memory_ratio": max_peak_memory_ratio,
            "max_cache_miss_ratio": max_cache_miss_ratio,
            "require_output_match": require_output_match,
            "require_same_residency": require_same_residency,
        },
        "ratios": {
            "median_wall_throughput": throughput_ratio,
            "median_peak_memory": peak_memory_ratio,
            "median_cache_misses": cache_miss_ratio,
        },
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "violations": violations,
    }


def build_parser(prog: str = "ft mlx-compare") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Compare stdout logs captured from `ft generate --backend mlx` and "
            "enforce reproducibility/performance gates."
        ),
    )
    parser.add_argument(
        "--baseline", nargs="+", required=True, metavar="LOG",
        help="baseline stdout log files; shell globs may be used",
    )
    parser.add_argument(
        "--candidate", nargs="+", required=True, metavar="LOG",
        help="candidate stdout log files; shell globs may be used",
    )
    parser.add_argument("--expected-tokens", type=int, required=True)
    parser.add_argument("--min-throughput-ratio", type=float, default=0.97)
    parser.add_argument("--max-peak-memory-ratio", type=float, default=1.05)
    parser.add_argument(
        "--max-cache-miss-ratio", type=float,
        help="optional upper bound for candidate/baseline median cache misses",
    )
    parser.add_argument("--require-output-match", action="store_true")
    parser.add_argument("--require-same-residency", action="store_true")
    parser.add_argument(
        "--write-summary", type=Path,
        help="also write the comparison JSON to this path",
    )
    parser.add_argument(
        "--compact", action="store_true", help="print compact JSON instead of indented JSON"
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    prog: str = "ft mlx-compare",
) -> int:
    args = build_parser(prog).parse_args(argv)
    try:
        baseline = [load_run(path, label="baseline") for path in args.baseline]
        candidate = [load_run(path, label="candidate") for path in args.candidate]
        result = compare_runs(
            baseline,
            candidate,
            expected_tokens=args.expected_tokens,
            min_throughput_ratio=args.min_throughput_ratio,
            max_peak_memory_ratio=args.max_peak_memory_ratio,
            max_cache_miss_ratio=args.max_cache_miss_ratio,
            require_output_match=args.require_output_match,
            require_same_residency=args.require_same_residency,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"{prog}: {exc}", file=sys.stderr)
        return 2

    payload = json.dumps(
        result, sort_keys=True, indent=None if args.compact else 2
    ) + "\n"
    if args.write_summary is not None:
        args.write_summary.parent.mkdir(parents=True, exist_ok=True)
        args.write_summary.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RunRecord",
    "build_parser",
    "compare_runs",
    "load_run",
    "main",
    "summarize",
]
