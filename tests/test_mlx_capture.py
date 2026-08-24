from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from freetoken.cli import main as cli_main  # noqa: E402
from freetoken.mlx_capture import capture_run, main  # noqa: E402


def _success_script(output: str = "hello") -> str:
    report = {
        "backend": "mlx",
        "generated_tokens": 1,
        "elapsed_seconds": 0.25,
        "residency": {"selected": "offload"},
        "memory": {"peak_bytes": 1024},
        "expert_cache": {"misses": 4},
    }
    return (
        "import json; "
        f"print({output!r}); "
        f"print(json.dumps({report!r}))"
    )


class MLXCaptureTest(unittest.TestCase):
    def test_completed_capture_is_atomic_hashed_and_prompt_redacted(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            code, manifest = capture_run(
                output,
                ["--prompt", "private prompt"],
                label="candidate",
                command_prefix=[sys.executable, "-c", _success_script()],
                environment={
                    "PYTHONHASHSEED": "0",
                    "UNSAFE_SECRET": "do-not-record",
                },
            )

            self.assertEqual(code, 0)
            self.assertEqual(manifest["status"], "completed")
            self.assertTrue((output / "stdout.log").is_file())
            self.assertTrue((output / "stderr.log").is_file())
            self.assertTrue((output / "capture.json").is_file())
            persisted = json.loads((output / "capture.json").read_text())
            self.assertEqual(persisted["schema_version"], 1)
            self.assertEqual(persisted["command"]["argv"][-1], "<redacted>")
            self.assertNotIn("private prompt", persisted["command"]["shell"])
            prompt = persisted["command"]["prompt"]
            self.assertEqual(prompt["bytes"], len(b"private prompt"))
            self.assertEqual(
                prompt["sha256"], hashlib.sha256(b"private prompt").hexdigest()
            )
            self.assertEqual(persisted["environment"], {"PYTHONHASHSEED": "0"})
            self.assertEqual(persisted["run"]["generated_tokens"], 1)
            self.assertEqual(persisted["run"]["path"], "stdout.log")
            self.assertEqual(persisted["run"]["label"], "candidate")
            stdout = output / "stdout.log"
            self.assertEqual(
                persisted["files"]["stdout"]["sha256"],
                hashlib.sha256(stdout.read_bytes()).hexdigest(),
            )

    def test_include_prompt_is_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            code, manifest = capture_run(
                output,
                ["--prompt=visible"],
                include_prompt=True,
                command_prefix=[sys.executable, "-c", _success_script()],
            )
        self.assertEqual(code, 0)
        self.assertEqual(manifest["command"]["argv"][-1], "--prompt=visible")
        self.assertTrue(manifest["command"]["prompt_included"])

    def test_nonzero_target_exit_still_publishes_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            script = "import sys; print('failed', file=sys.stderr); raise SystemExit(7)"
            code, manifest = capture_run(
                output,
                ["argument"],
                command_prefix=[sys.executable, "-c", script],
            )
            self.assertEqual(code, 1)
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["process_exit_code"], 7)
            self.assertIn("failed", (output / "stderr.log").read_text())
            self.assertIsNone(manifest["run"])

    def test_invalid_success_report_returns_harness_error(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            code, manifest = capture_run(
                output,
                ["argument"],
                command_prefix=[sys.executable, "-c", "print('not a report')"],
            )
            self.assertEqual(code, 2)
            self.assertEqual(manifest["status"], "invalid_report")
            self.assertIn("not a JSON report", manifest["error"])
            self.assertTrue((output / "capture.json").is_file())

    def test_timeout_is_recorded_without_losing_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            code, manifest = capture_run(
                output,
                ["argument"],
                timeout_seconds=0.05,
                command_prefix=[
                    sys.executable,
                    "-c",
                    "import time; time.sleep(1)",
                ],
            )
            self.assertEqual(code, 1)
            self.assertEqual(manifest["status"], "timeout")
            self.assertEqual(manifest["process_exit_code"], 124)
            self.assertTrue((output / "stdout.log").is_file())

    def test_launch_error_is_structured(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            code, manifest = capture_run(
                output,
                ["argument"],
                command_prefix=[str(Path(directory) / "missing-program")],
            )
            self.assertEqual(code, 2)
            self.assertEqual(manifest["status"], "launch_error")
            self.assertIsNone(manifest["process_exit_code"])
            self.assertTrue((output / "capture.json").is_file())

    def test_capture_refuses_to_overwrite_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            output.mkdir()
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                capture_run(
                    output,
                    ["argument"],
                    command_prefix=[sys.executable, "-c", _success_script()],
                )

    def test_default_command_requires_explicit_mlx_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            for args in (["--model", "x"], ["--backend", "cuda"]):
                with self.subTest(args=args), self.assertRaisesRegex(
                    ValueError, "--backend mlx"
                ):
                    capture_run(output, args)

    def test_nonfinite_timeout_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            for timeout in (0, -1, float("nan"), float("inf")):
                with self.subTest(timeout=timeout), self.assertRaisesRegex(
                    ValueError, "finite and positive"
                ):
                    capture_run(
                        output,
                        ["argument"],
                        timeout_seconds=timeout,
                        command_prefix=[sys.executable, "-c", _success_script()],
                    )

    def test_cli_reports_invalid_arguments_with_exit_two(self):
        stderr = StringIO()
        with redirect_stderr(stderr):
            result = main(["--output", "unused", "--", "--model", "x"])
        self.assertEqual(result, 2)
        self.assertIn("--backend mlx", stderr.getvalue())

    def test_cli_help_is_available(self):
        stdout = StringIO()
        with self.assertRaises(SystemExit) as raised, redirect_stdout(stdout):
            main(["--help"])
        self.assertEqual(raised.exception.code, 0)
        self.assertIn("--include-prompt", stdout.getvalue())

    def test_root_cli_exposes_capture_without_importing_accelerators(self):
        modules_before = set(sys.modules)
        stdout = StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(cli_main(["--help"]), 0)
        loaded = set(sys.modules) - modules_before
        self.assertIn("mlx-capture", stdout.getvalue())
        self.assertFalse(
            any(name == "torch" or name.startswith("torch.") for name in loaded)
        )
        self.assertFalse(
            any(name == "mlx" or name.startswith("mlx.") for name in loaded)
        )

    def test_root_cli_dispatches_capture_errors_cleanly(self):
        stderr = StringIO()
        with redirect_stderr(stderr):
            result = cli_main([
                "mlx-capture", "--output", "unused", "--", "--model", "x"
            ])
        self.assertEqual(result, 2)
        self.assertIn("--backend mlx", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
