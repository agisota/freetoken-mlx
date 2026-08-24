#!/usr/bin/env python3
"""Collect a controlled expert-cache memory/throughput curve on Apple Silicon."""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _free_percent() -> int:
    result = subprocess.run(
        ["memory_pressure", "-Q"], check=True, capture_output=True, text=True,
    )
    match = re.search(r"free percentage:\s*(\d+)%", result.stdout)
    if match is None:
        raise RuntimeError("could not read macOS memory pressure")
    return int(match.group(1))


def _last_json_line(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("backend") == "mlx":
            return value
    raise RuntimeError("generation output did not contain an MLX JSON report")


def _write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--budgets-gb", type=float, nargs="+",
        default=(0.25, 0.50, 0.75, 1.00, 1.25),
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--memory-limit-gb", type=float, default=10.0)
    parser.add_argument("--system-headroom-gb", type=float, default=3.2)
    parser.add_argument(
        "--min-free-percent", type=int, default=35,
        help="abort before a run if macOS reports less reclaimable memory",
    )
    parser.add_argument("--initial-cooldown-seconds", type=float, default=0.0)
    parser.add_argument("--cooldown-seconds", type=float, default=0.0)
    parser.add_argument(
        "--output", type=Path,
        default=Path("paper/data/memory_curve_runs.json"),
    )
    parser.add_argument("--ft", default=".venv/bin/ft")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("this benchmark requires Apple Silicon")
    if args.repeats < 1 or args.max_tokens < 1:
        raise ValueError("repeats and max-tokens must be positive")
    if not args.budgets_gb or any(value <= 0 for value in args.budgets_gb):
        raise ValueError("cache budgets must be positive")
    if args.min_free_percent <= 20:
        raise ValueError("min-free-percent must exceed the reserved 20% floor")
    if args.initial_cooldown_seconds < 0 or args.cooldown_seconds < 0:
        raise ValueError("cooldown durations must be non-negative")

    repo = Path(__file__).resolve().parents[1]
    ft = (repo / args.ft).resolve() if not Path(args.ft).is_absolute() else Path(args.ft)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    payload = {
        "schema_version": 1,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "hardware": {
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "protocol": {
            "model": str(Path(args.model).expanduser().resolve()),
            "prompt": args.prompt,
            "raw_prompt": True,
            "max_tokens": args.max_tokens,
            "repeats": args.repeats,
            "budgets_gb": args.budgets_gb,
            "order": "rotating Latin square; one fresh process per point",
            "profile": "performance",
            "residency": "offload",
            "memory_limit_gb": args.memory_limit_gb,
            "system_headroom_gb": args.system_headroom_gb,
            "allocator_cache_mb": 256,
            "min_free_percent": args.min_free_percent,
            "initial_cooldown_seconds": args.initial_cooldown_seconds,
            "cooldown_seconds": args.cooldown_seconds,
        },
        "runs": [],
    }
    _write_atomic(repo / args.output, payload)

    budgets = list(args.budgets_gb)
    if args.initial_cooldown_seconds:
        print(
            f"initial cooldown={args.initial_cooldown_seconds:.0f}s",
            flush=True,
        )
        time.sleep(args.initial_cooldown_seconds)
    for repeat in range(args.repeats):
        offset = repeat % len(budgets)
        order = budgets[offset:] + budgets[:offset]
        for position, budget in enumerate(order):
            free_percent = _free_percent()
            if free_percent < args.min_free_percent:
                raise RuntimeError(
                    f"aborting safely: macOS free percentage {free_percent}% "
                    f"is below {args.min_free_percent}%"
                )
            command = [
                str(ft), "generate", "--backend", "mlx",
                "--model", args.model,
                "--prompt", args.prompt,
                "--raw-prompt",
                "--max-tokens", str(args.max_tokens),
                "--profile", "performance",
                "--residency", "offload",
                "--quiet-cache",
                "--memory-limit-gb", str(args.memory_limit_gb),
                "--system-headroom-gb", str(args.system_headroom_gb),
                "--allocator-cache-mb", "256",
                "--expert-cache-budget-gb", str(budget),
            ]
            print(
                f"repeat={repeat + 1}/{args.repeats} position={position + 1} "
                f"budget={budget:.2f}GiB free={free_percent}%",
                flush=True,
            )
            completed = subprocess.run(
                command, cwd=repo, capture_output=True, text=True,
            )
            if completed.returncode != 0:
                payload["failure"] = {
                    "command": command,
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                }
                _write_atomic(repo / args.output, payload)
                print(completed.stderr, file=sys.stderr)
                return completed.returncode
            report = _last_json_line(completed.stdout)
            requested_bytes = int(budget * (1 << 30))
            effective_bytes = report["residency"][
                "offload_expert_cache_budget_bytes"
            ]
            if effective_bytes != requested_bytes:
                payload["failure"] = {
                    "reason": "safe runtime budget clamped the treatment",
                    "requested_bytes": requested_bytes,
                    "effective_bytes": effective_bytes,
                    "report": report,
                }
                _write_atomic(repo / args.output, payload)
                return 2
            if report["generated_tokens"] != args.max_tokens:
                payload["failure"] = {
                    "reason": "generation ended before the requested token count",
                    "report": report,
                }
                _write_atomic(repo / args.output, payload)
                return 2
            payload["runs"].append({
                "repeat": repeat,
                "position": position,
                "requested_cache_budget_gb": budget,
                "free_percent_before": free_percent,
                "command": command,
                "report": report,
                "stderr": completed.stderr.strip(),
            })
            _write_atomic(repo / args.output, payload)
            if args.cooldown_seconds and not (
                repeat == args.repeats - 1 and position == len(order) - 1
            ):
                print(f"cooldown={args.cooldown_seconds:.0f}s", flush=True)
                time.sleep(args.cooldown_seconds)

    payload["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_atomic(repo / args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
