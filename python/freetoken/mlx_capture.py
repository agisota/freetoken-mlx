"""Capture a reproducible FreeToken-MLX generation run as an artifact bundle."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from freetoken.mlx_compare import load_run

CAPTURE_SCHEMA_VERSION = 1
_SAFE_ENVIRONMENT_KEYS = (
    "HF_HUB_DISABLE_TELEMETRY",
    "HF_HUB_OFFLINE",
    "MLX_METAL_CACHE_DIR",
    "MLX_METAL_PREWARM",
    "PYTHONHASHSEED",
    "TOKENIZERS_PARALLELISM",
    "TMPDIR",
)
_PACKAGE_NAMES = (
    "freetoken",
    "mlx",
    "mlx-lm",
    "huggingface-hub",
    "transformers",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in _PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _physical_memory_bytes() -> int | None:
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        pages = int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    if page_size <= 0 or pages <= 0:
        return None
    return page_size * pages


def _host_metadata() -> dict[str, object]:
    return {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "physical_memory_bytes": _physical_memory_bytes(),
    }


def _run_git(repo_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _source_metadata() -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[2]
    commit = _run_git(repo_root, "rev-parse", "HEAD")
    dirty_output = _run_git(
        repo_root,
        "status",
        "--porcelain",
        "--untracked-files=no",
    )
    return {
        "repository_root": str(repo_root) if commit is not None else None,
        "git_commit": commit,
        "git_dirty": bool(dirty_output) if dirty_output is not None else None,
    }


def _safe_environment(environment: Mapping[str, str]) -> dict[str, str]:
    return {
        key: environment[key]
        for key in _SAFE_ENVIRONMENT_KEYS
        if key in environment
    }


def _prompt_record(value: str) -> dict[str, object]:
    encoded = value.encode("utf-8")
    return {
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _redact_generate_args(
    generate_args: Sequence[str],
    *,
    include_prompt: bool,
) -> tuple[list[str], dict[str, object] | None]:
    redacted: list[str] = []
    prompt: dict[str, object] | None = None
    index = 0
    while index < len(generate_args):
        arg = generate_args[index]
        if arg == "--prompt" and index + 1 < len(generate_args):
            value = generate_args[index + 1]
            prompt = _prompt_record(value)
            redacted.extend((arg, value if include_prompt else "<redacted>"))
            index += 2
            continue
        if arg.startswith("--prompt="):
            value = arg.split("=", 1)[1]
            prompt = _prompt_record(value)
            redacted.append(arg if include_prompt else "--prompt=<redacted>")
            index += 1
            continue
        redacted.append(arg)
        index += 1
    return redacted, prompt


def _backend_value(generate_args: Sequence[str]) -> str | None:
    value: str | None = None
    for index, arg in enumerate(generate_args):
        if arg == "--backend" and index + 1 < len(generate_args):
            value = generate_args[index + 1]
        elif arg.startswith("--backend="):
            value = arg.split("=", 1)[1]
    return value


def _validate_timeout(timeout_seconds: float | None) -> float | None:
    if timeout_seconds is None:
        return None
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be finite and positive")
    return timeout_seconds


def capture_run(
    output: str | Path,
    generate_args: Sequence[str],
    *,
    label: str | None = None,
    timeout_seconds: float | None = None,
    include_prompt: bool = False,
    command_prefix: Sequence[str] | None = None,
    environment: Mapping[str, str] | None = None,
) -> tuple[int, dict[str, object]]:
    """Run one generation command and atomically publish its evidence bundle."""

    args = list(generate_args)
    if not args:
        raise ValueError("at least one generate argument is required")
    timeout_seconds = _validate_timeout(timeout_seconds)
    if command_prefix is None:
        backend = _backend_value(args)
        if backend != "mlx":
            raise ValueError("captured generation must explicitly use --backend mlx")
        prefix = [sys.executable, "-m", "freetoken.cli", "generate"]
    else:
        prefix = list(command_prefix)
        if not prefix:
            raise ValueError("command_prefix cannot be empty")

    output_path = Path(output).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"capture output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{output_path.name}.", dir=output_path.parent
    ))
    stdout_path = temporary / "stdout.log"
    stderr_path = temporary / "stderr.log"
    manifest_path = temporary / "capture.json"
    command = [*prefix, *args]
    redacted_args, prompt = _redact_generate_args(
        args, include_prompt=include_prompt
    )
    redacted_command = [*prefix, *redacted_args]
    process_environment = dict(os.environ if environment is None else environment)
    started_at = _utc_now()
    started = time.perf_counter()
    process_exit_code: int | None = None
    status = "launch_error"
    error: str | None = None

    try:
        try:
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                completed = subprocess.run(
                    command,
                    stdout=stdout,
                    stderr=stderr,
                    env=process_environment,
                    check=False,
                    timeout=timeout_seconds,
                )
            process_exit_code = completed.returncode
            status = "completed" if completed.returncode == 0 else "failed"
        except subprocess.TimeoutExpired as exc:
            process_exit_code = 124
            status = "timeout"
            error = f"generation exceeded timeout of {exc.timeout} seconds"
        except OSError as exc:
            status = "launch_error"
            error = str(exc)

        finished_at = _utc_now()
        wall_seconds = time.perf_counter() - started
        for path in (stdout_path, stderr_path):
            if not path.exists():
                path.write_bytes(b"")

        parsed_run: dict[str, object] | None = None
        harness_exit_code: int
        if status == "completed":
            try:
                parsed_run = asdict(load_run(stdout_path, label=label or "capture"))
                harness_exit_code = 0
            except (OSError, UnicodeError, ValueError, OverflowError) as exc:
                status = "invalid_report"
                error = str(exc)
                harness_exit_code = 2
        elif status in {"failed", "timeout"}:
            harness_exit_code = 1
        else:
            harness_exit_code = 2

        manifest: dict[str, object] = {
            "schema_version": CAPTURE_SCHEMA_VERSION,
            "status": status,
            "label": label,
            "started_at": started_at,
            "finished_at": finished_at,
            "wall_seconds": wall_seconds,
            "process_exit_code": process_exit_code,
            "error": error,
            "command": {
                "argv": redacted_command,
                "shell": shlex.join(redacted_command),
                "prompt": prompt,
                "prompt_included": include_prompt,
            },
            "host": _host_metadata(),
            "packages": _package_versions(),
            "source": _source_metadata(),
            "environment": _safe_environment(process_environment),
            "files": {
                "stdout": _file_record(stdout_path, temporary),
                "stderr": _file_record(stderr_path, temporary),
            },
            "run": parsed_run,
        }
        manifest_path.write_text(
            json.dumps(
                manifest,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.rename(output_path)
        return harness_exit_code, manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def build_parser(prog: str = "ft mlx-capture") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Run one `ft generate --backend mlx` command and capture stdout, "
            "stderr, exact hashes, environment metadata, and the parsed report."
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--label")
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument(
        "--include-prompt",
        action="store_true",
        help="store prompt text in capture.json; default stores only its hash and size",
    )
    parser.add_argument(
        "generate_args",
        nargs=argparse.REMAINDER,
        help="arguments passed to `ft generate`; place them after `--`",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    prog: str = "ft mlx-capture",
) -> int:
    args = build_parser(prog).parse_args(argv)
    generate_args = list(args.generate_args)
    if generate_args[:1] == ["--"]:
        generate_args = generate_args[1:]
    try:
        exit_code, manifest = capture_run(
            args.output,
            generate_args,
            label=args.label,
            timeout_seconds=args.timeout_seconds,
            include_prompt=args.include_prompt,
        )
    except (OSError, ValueError, OverflowError) as exc:
        print(f"{prog}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "output": str(args.output.expanduser().resolve()),
        "status": manifest["status"],
        "process_exit_code": manifest["process_exit_code"],
    }, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "capture_run", "main"]
