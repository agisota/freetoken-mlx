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
MAX_REPORT_BYTES = 4 << 20
MAX_COUNTER_VALUE = (1 << 63) - 1
REPORT_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class RunRecord:
    path: str
    label: str
    log_sha256: str
    log_bytes: int
    report_sha256: str
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
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{field} must be finite") from exc
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
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    if isinstance(value, int):
        number = value
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field} must be finite")
        if not value.is_integer():
            raise ValueError(f"{field} must be an integer")
        number = int(value)
    else:
        raise ValueError(f"{field} must be an integer")
    if number < 0:
        raise ValueError(f"{field} cannot be negative")
    if number > MAX_COUNTER_VALUE:
        raise ValueError(
            f"{field} exceeds the supported 64-bit counter range"
        )
    return number


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is not allowed: {key}")
        result[key] = value
    return result


def _parse_report(report_bytes: bytes, path: Path) -> dict[str, Any]:
    if len(report_bytes) > MAX_REPORT_BYTES:
        raise ValueError(
            f"final JSON report is too large ({len(report_bytes)} bytes; "
            f"limit {MAX_REPORT_BYTES}): {path}"
        )
    try:
        text = report_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"final JSON report is not UTF-8: {path}: {exc}") from exc
    try:
        report = json.loads(
            text,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except RecursionError as exc:
        raise ValueError(
            f"final JSON report exceeds the supported nesting depth: {path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"last non-empty stdout line is not a JSON report: {path}: {exc}"
        ) from exc
    except ValueError as exc:
        raise ValueError(f"invalid final JSON report: {path}: {exc}") from exc
    if not isinstance(report, dict):
        raise ValueError(f"final JSON report must be an object: {path}")
    return report


def _split_stdout_log(
    path: Path,
) -> tuple[bytes, dict[str, Any], bytes, bytes]:
    if not path.is_file():
        raise FileNotFoundError(f"run log does not exist: {path}")
    size = path.stat().st_size
    if size > MAX_LOG_BYTES:
        raise ValueError(
            f"run log is too large ({size} bytes; limit {MAX_LOG_BYTES}): {path}"
        )
    data = path.read_bytes()
    # Recheck after the read in case the file grew between stat() and read_bytes().
    if len(data) > MAX_LOG_BYTES:
        raise ValueError(
            f"run log is too large ({len(data)} bytes; limit {MAX_LOG_BYTES}): {path}"
        )
    lines = data.splitlines(keepends=True)
    report_index = next(
        (index for index in range(len(lines) - 1, -1, -1) if lines[index].strip()),
        None,
    )
    if report_index is None:
        raise ValueError(f"run log is empty: {path}")
    report_bytes = lines[report_index].strip()
    report = _parse_report(report_bytes, path)

    output = b"".join(lines[:report_index])
    # `ft generate` prints one separator newline before the final JSON report.
    # Remove exactly that separator while preserving model-emitted line endings.
    if output.endswith(b"\r\n"):
        output = output[:-2]
    elif output.endswith((b"\n", b"\r")):
        output = output[:-1]
    return output, report, data, report_bytes


def load_run(path: str | Path, *, label: str) -> RunRecord:
    resolved = Path(path).expanduser().resolve()
    output, report, raw_log, report_bytes = _split_stdout_log(resolved)
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
        if selected is not None and selected not in {"resident", "offload"}:
            raise ValueError(
                "residency.selected must be 'resident' or 'offload': "
                f"{resolved}"
            )
        residency = selected

    return RunRecord(
        path=str(resolved),
        label=label,
        log_sha256=hashlib.sha256(raw_log).hexdigest(),
        log_bytes=len(raw_log),
        report_sha256=hashlib.sha256(report_bytes).hexdigest(),
        generated_tokens=generated,
        elapsed_seconds=elapsed,
        wall_tokens_per_second=generated / elapsed,
        peak_memory_bytes=peak,
        cache_misses=misses,
        output_sha256=hashlib.sha256(output).hexdigest(),
        output_bytes=len(output),
        residency=residency,
    )


def _median(values: list[float | int | None]) -> float | None:
    if not values or any(value is None for value in values):
        return None
    return float(statistics.median(values))


def _relative_mad(values: list[float]) -> float | None:
    if not values:
        return None
    median = float(statistics.median(values))
    if median <= 0:
        return None
    mad = float(statistics.median(abs(value - median) for value in values))
    return mad / median


def summarize(records: list[RunRecord]) -> dict[str, Any]:
    if not records:
        raise ValueError("each comparison group needs at least one run")
    throughput = [record.wall_tokens_per_second for record in records]
    return {
        "count": len(records),
        "generated_tokens": sorted({record.generated_tokens for record in records}),
        "median_elapsed_seconds": _median(
            [record.elapsed_seconds for record in records]
        ),
        "median_wall_tokens_per_second": _median(throughput),
        "throughput_relative_mad": _relative_mad(throughput),
        "median_peak_memory_bytes": _median(
            [record.peak_memory_bytes for record in records]
        ),
        "median_cache_misses": _median(
            [record.cache_misses for record in records]
        ),
        "complete_metrics": {
            "peak_memory": all(
                record.peak_memory_bytes is not None for record in records
            ),
            "cache_misses": all(
                record.cache_misses is not None for record in records
            ),
            "residency": all(record.residency is not None for record in records),
        },
        "output_sha256": sorted({record.output_sha256 for record in records}),
        "log_sha256": sorted({record.log_sha256 for record in records}),
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
    min_throughput_ratio: float | None = 0.97,
    max_peak_memory_ratio: float = 1.05,
    max_cache_miss_ratio: float | None = None,
    require_output_match: bool = False,
    require_same_residency: bool = False,
    min_runs_per_group: int = 1,
    require_balanced_groups: bool = False,
    max_throughput_relative_mad: float | None = None,
) -> dict[str, Any]:
    if (
        isinstance(expected_tokens, bool)
        or not isinstance(expected_tokens, int)
        or expected_tokens < 1
    ):
        raise ValueError("expected_tokens must be an integer of at least 1")
    if (
        isinstance(min_runs_per_group, bool)
        or not isinstance(min_runs_per_group, int)
        or min_runs_per_group < 1
    ):
        raise ValueError("min_runs_per_group must be an integer of at least 1")
    if min_throughput_ratio is not None:
        min_throughput_ratio = _number(
            min_throughput_ratio,
            field="min_throughput_ratio",
            positive=True,
        )
    max_peak_memory_ratio = _number(
        max_peak_memory_ratio,
        field="max_peak_memory_ratio",
        positive=True,
    )
    if max_cache_miss_ratio is not None:
        max_cache_miss_ratio = _number(
            max_cache_miss_ratio,
            field="max_cache_miss_ratio",
            positive=True,
        )
    if max_throughput_relative_mad is not None:
        max_throughput_relative_mad = _number(
            max_throughput_relative_mad,
            field="max_throughput_relative_mad",
        )

    paths = [record.path for record in baseline + candidate]
    seen: set[str] = set()
    duplicates: set[str] = set()
    for path in paths:
        if path in seen:
            duplicates.add(path)
        seen.add(path)
    if duplicates:
        raise ValueError(
            "each run log may be selected only once; duplicates: "
            + ", ".join(sorted(duplicates))
        )

    baseline_summary = summarize(baseline)
    candidate_summary = summarize(candidate)
    violations: list[str] = []

    if len(baseline) < min_runs_per_group:
        violations.append(
            f"baseline has {len(baseline)} runs; requires at least "
            f"{min_runs_per_group}"
        )
    if len(candidate) < min_runs_per_group:
        violations.append(
            f"candidate has {len(candidate)} runs; requires at least "
            f"{min_runs_per_group}"
        )
    if require_balanced_groups and len(baseline) != len(candidate):
        violations.append(
            "balanced groups were requested, but run counts differ: "
            f"baseline={len(baseline)} candidate={len(candidate)}"
        )

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
        if min_throughput_ratio is not None:
            violations.append(
                "throughput gate requires a positive baseline token rate; "
                "check generated_tokens and elapsed_seconds"
            )
    elif candidate_tps is None:
        if min_throughput_ratio is not None:
            violations.append("throughput gate requires candidate timing metrics")
    else:
        throughput_ratio = candidate_tps / baseline_tps
        if (
            min_throughput_ratio is not None
            and throughput_ratio < min_throughput_ratio
        ):
            violations.append(
                f"throughput ratio {throughput_ratio:.6f} is below "
                f"minimum {min_throughput_ratio:.6f}"
            )

    if max_throughput_relative_mad is not None:
        for label, summary in (
            ("baseline", baseline_summary),
            ("candidate", candidate_summary),
        ):
            relative_mad = summary["throughput_relative_mad"]
            if relative_mad is None:
                violations.append(
                    f"{label} throughput dispersion requires a positive token rate"
                )
            elif relative_mad > max_throughput_relative_mad:
                violations.append(
                    f"{label} throughput relative MAD {relative_mad:.6f} exceeds "
                    f"maximum {max_throughput_relative_mad:.6f}"
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
            "min_runs_per_group": min_runs_per_group,
            "require_balanced_groups": require_balanced_groups,
            "max_throughput_relative_mad": max_throughput_relative_mad,
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
    throughput = parser.add_mutually_exclusive_group()
    throughput.add_argument(
        "--min-throughput-ratio", type=float, default=0.97,
        help="minimum candidate/baseline median token-rate ratio (default: 0.97)",
    )
    throughput.add_argument(
        "--no-throughput-gate", action="store_true",
        help="report throughput but do not fail on its ratio",
    )
    parser.add_argument("--max-peak-memory-ratio", type=float, default=1.05)
    parser.add_argument(
        "--max-cache-miss-ratio", type=float,
        help="optional upper bound for candidate/baseline median cache misses",
    )
    parser.add_argument(
        "--min-runs-per-group", type=int, default=1,
        help="minimum accepted run count in both groups (default: 1)",
    )
    parser.add_argument(
        "--require-balanced-groups", action="store_true",
        help="require baseline and candidate to contain the same number of runs",
    )
    parser.add_argument(
        "--max-throughput-relative-mad", type=float,
        help="optional noise gate for median absolute deviation / median token rate",
    )
    parser.add_argument("--require-output-match", action="store_true")
    parser.add_argument("--require-same-residency", action="store_true")
    parser.add_argument(
        "--write-summary", type=Path,
        help="also write the comparison JSON to this path",
    )
    parser.add_argument(
        "--compact", action="store_true",
        help="print compact JSON instead of indented JSON",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    prog: str = "ft mlx-compare",
) -> int:
    args = build_parser(prog).parse_args(argv)
    summary_path = (
        args.write_summary.expanduser().resolve()
        if args.write_summary is not None
        else None
    )
    try:
        baseline = [load_run(path, label="baseline") for path in args.baseline]
        candidate = [load_run(path, label="candidate") for path in args.candidate]
        if summary_path is not None and str(summary_path) in {
            record.path for record in baseline + candidate
        }:
            raise ValueError("--write-summary must not overwrite an input run log")
        result = compare_runs(
            baseline,
            candidate,
            expected_tokens=args.expected_tokens,
            min_throughput_ratio=(
                None if args.no_throughput_gate else args.min_throughput_ratio
            ),
            max_peak_memory_ratio=args.max_peak_memory_ratio,
            max_cache_miss_ratio=args.max_cache_miss_ratio,
            require_output_match=args.require_output_match,
            require_same_residency=args.require_same_residency,
            min_runs_per_group=args.min_runs_per_group,
            require_balanced_groups=args.require_balanced_groups,
            max_throughput_relative_mad=args.max_throughput_relative_mad,
        )
    except (OSError, UnicodeError, ValueError, OverflowError) as exc:
        print(f"{prog}: {exc}", file=sys.stderr)
        return 2

    try:
        payload = json.dumps(
            result,
            sort_keys=True,
            indent=None if args.compact else 2,
            allow_nan=False,
        ) + "\n"
        if summary_path is not None:
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(payload, encoding="utf-8")
    except (OSError, ValueError, OverflowError) as exc:
        print(f"{prog}: {exc}", file=sys.stderr)
        return 2

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
