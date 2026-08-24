from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from freetoken.cli import main as cli_main  # noqa: E402
from freetoken.mlx_compare import (  # noqa: E402
    MAX_LOG_BYTES,
    compare_runs,
    load_run,
    main,
)


def _report(
    *,
    tokens: int | float = 32,
    elapsed: int | float = 8.0,
    peak: int | float | None = 8_000_000_000,
    misses: int | float | None = 100,
    residency: str = "offload",
) -> dict:
    return {
        "backend": "mlx",
        "generated_tokens": tokens,
        "elapsed_seconds": elapsed,
        "residency": {"selected": residency},
        "memory": None if peak is None else {"peak_bytes": peak},
        "expert_cache": None if misses is None else {"misses": misses},
    }


def _write_run(
    path: Path,
    *,
    output: str = "generated text",
    tokens: int | float = 32,
    elapsed: int | float = 8.0,
    peak: int | float | None = 8_000_000_000,
    misses: int | float | None = 100,
    residency: str = "offload",
) -> Path:
    path.write_text(
        output
        + "\n"
        + json.dumps(
            _report(
                tokens=tokens,
                elapsed=elapsed,
                peak=peak,
                misses=misses,
                residency=residency,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_raw_run(
    path: Path,
    *,
    output: bytes,
    report: dict | None = None,
    separator: bytes = b"\n",
    final_newline: bytes = b"\n",
) -> Path:
    payload = json.dumps(report or _report(), separators=(",", ":")).encode("utf-8")
    path.write_bytes(output + separator + payload + final_newline)
    return path


class MLXCompareTest(unittest.TestCase):
    def test_comparison_passes_for_equal_output_and_safe_ratios(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = [
                load_run(_write_run(root / "baseline-1.log", elapsed=8.0), label="baseline"),
                load_run(_write_run(root / "baseline-2.log", elapsed=8.4), label="baseline"),
            ]
            candidate = [
                load_run(
                    _write_run(root / "candidate-1.log", elapsed=7.6, peak=8_100_000_000),
                    label="candidate",
                ),
                load_run(
                    _write_run(root / "candidate-2.log", elapsed=7.8, peak=8_200_000_000),
                    label="candidate",
                ),
            ]
            result = compare_runs(
                baseline,
                candidate,
                expected_tokens=32,
                max_cache_miss_ratio=1.05,
                require_output_match=True,
                require_same_residency=True,
                min_runs_per_group=2,
                require_balanced_groups=True,
            )

        self.assertTrue(result["pass"], result["violations"])
        self.assertEqual(result["schema_version"], 2)
        self.assertGreater(result["ratios"]["median_wall_throughput"], 1.0)
        self.assertLess(result["ratios"]["median_peak_memory"], 1.05)
        self.assertEqual(result["baseline"]["generated_tokens"], [32])
        self.assertEqual(len(baseline[0].log_sha256), 64)
        self.assertEqual(len(baseline[0].report_sha256), 64)

    def test_exact_generated_token_count_is_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = [load_run(_write_run(root / "a.log"), label="baseline")]
            candidate = [
                load_run(_write_run(root / "b.log", tokens=17, elapsed=4.0), label="candidate")
            ]
            result = compare_runs(baseline, candidate, expected_tokens=32)
        self.assertFalse(result["pass"])
        self.assertTrue(any("expected exactly 32" in item for item in result["violations"]))

    def test_output_mismatch_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = [load_run(_write_run(root / "a.log", output="alpha"), label="baseline")]
            candidate = [load_run(_write_run(root / "b.log", output="beta"), label="candidate")]
            result = compare_runs(
                baseline, candidate, expected_tokens=32, require_output_match=True
            )
        self.assertFalse(result["pass"])
        self.assertTrue(any("generated output differs" in item for item in result["violations"]))

    def test_output_hash_preserves_line_endings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = load_run(
                _write_raw_run(root / "crlf.log", output=b"line1\r\nline2"),
                label="baseline",
            )
            candidate = load_run(
                _write_raw_run(root / "lf.log", output=b"line1\nline2"),
                label="candidate",
            )
            self.assertNotEqual(baseline.output_sha256, candidate.output_sha256)
            result = compare_runs(
                [baseline], [candidate], expected_tokens=32, require_output_match=True
            )
        self.assertFalse(result["pass"])

    def test_large_integer_counter_is_rejected_without_overflow(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            huge = 10**400
            baseline = _write_raw_run(
                root / "huge.log",
                output=b"text",
                report=_report(tokens=huge),
            )
            candidate = _write_run(root / "candidate.log")
            stderr = StringIO()
            with redirect_stderr(stderr):
                exit_code = main([
                    "--baseline", str(baseline),
                    "--candidate", str(candidate),
                    "--expected-tokens", "32",
                ])
        self.assertEqual(exit_code, 2)
        self.assertIn("64-bit counter range", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_duplicate_json_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.log"
            path.write_bytes(
                b'text\n{"backend":"mlx","backend":"mlx",'
                b'"generated_tokens":32,"elapsed_seconds":1,'
                b'"memory":{"peak_bytes":1},'
                b'"expert_cache":{"misses":1},'
                b'"residency":{"selected":"offload"}}\n'
            )
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                load_run(path, label="baseline")

    def test_nonstandard_json_constants_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nan.log"
            path.write_bytes(
                b'text\n{"backend":"mlx","generated_tokens":32,'
                b'"elapsed_seconds":NaN,"memory":{"peak_bytes":1},'
                b'"expert_cache":{"misses":1},'
                b'"residency":{"selected":"offload"}}\n'
            )
            with self.assertRaisesRegex(ValueError, "non-finite JSON constant"):
                load_run(path, label="baseline")

    def test_excessive_json_nesting_is_a_controlled_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "deep.log"
            nested = b"[" * 12_000 + b"0" + b"]" * 12_000
            report = (
                b'{"backend":"mlx","generated_tokens":32,'
                b'"elapsed_seconds":1,"memory":{"peak_bytes":1},'
                b'"expert_cache":{"misses":1},'
                b'"residency":{"selected":"offload"},"deep":'
                + nested
                + b"}"
            )
            path.write_bytes(b"text\n" + report + b"\n")
            with self.assertRaisesRegex(ValueError, "nesting depth"):
                load_run(path, label="baseline")

    def test_oversized_log_is_rejected_before_reading_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "oversized.log"
            with path.open("wb") as stream:
                stream.truncate(MAX_LOG_BYTES + 1)
            with patch.object(
                Path,
                "read_bytes",
                side_effect=AssertionError("oversized log payload was read"),
            ), self.assertRaisesRegex(ValueError, "run log is too large"):
                load_run(path, label="baseline")

    def test_missing_peak_memory_cannot_bypass_default_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = [load_run(_write_run(root / "a.log", peak=None), label="baseline")]
            candidate = [load_run(_write_run(root / "b.log", peak=None), label="candidate")]
            result = compare_runs(baseline, candidate, expected_tokens=32)
        self.assertFalse(result["pass"])
        self.assertTrue(any("peak-memory gate requires" in item for item in result["violations"]))

    def test_one_missing_peak_memory_cannot_hide_behind_group_median(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = [
                load_run(_write_run(root / "a-1.log"), label="baseline"),
                load_run(_write_run(root / "a-2.log", peak=None), label="baseline"),
            ]
            candidate = [load_run(_write_run(root / "b.log"), label="candidate")]
            result = compare_runs(baseline, candidate, expected_tokens=32)
        self.assertFalse(result["pass"])
        self.assertTrue(any("in every run" in item for item in result["violations"]))
        self.assertIsNone(result["baseline"]["median_peak_memory_bytes"])
        self.assertFalse(result["baseline"]["complete_metrics"]["peak_memory"])

    def test_zero_token_baseline_fails_without_division_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = [load_run(_write_run(root / "a.log", tokens=0), label="baseline")]
            candidate = [load_run(_write_run(root / "b.log"), label="candidate")]
            result = compare_runs(baseline, candidate, expected_tokens=32)
        self.assertFalse(result["pass"])
        self.assertIsNone(result["ratios"]["median_wall_throughput"])
        self.assertTrue(any("positive baseline token rate" in item for item in result["violations"]))

    def test_cache_miss_gate_requires_metric_in_every_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = [load_run(_write_run(root / "a.log"), label="baseline")]
            candidate = [
                load_run(_write_run(root / "b-1.log"), label="candidate"),
                load_run(_write_run(root / "b-2.log", misses=None), label="candidate"),
            ]
            result = compare_runs(
                baseline,
                candidate,
                expected_tokens=32,
                max_cache_miss_ratio=1.05,
            )
        self.assertFalse(result["pass"])
        self.assertTrue(
            any("cache-miss gate" in item and "in every run" in item for item in result["violations"])
        )

    def test_same_residency_gate_rejects_mixed_group(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = [
                load_run(_write_run(root / "a-1.log", residency="offload"), label="baseline"),
                load_run(_write_run(root / "a-2.log", residency="resident"), label="baseline"),
            ]
            candidate = [load_run(_write_run(root / "b.log", residency="offload"), label="candidate")]
            result = compare_runs(
                baseline, candidate, expected_tokens=32, require_same_residency=True
            )
        self.assertFalse(result["pass"])
        self.assertTrue(any("one residency path" in item for item in result["violations"]))

    def test_evidence_count_and_balance_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = [load_run(_write_run(root / "a.log"), label="baseline")]
            candidate = [
                load_run(_write_run(root / "b-1.log"), label="candidate"),
                load_run(_write_run(root / "b-2.log"), label="candidate"),
            ]
            result = compare_runs(
                baseline,
                candidate,
                expected_tokens=32,
                min_runs_per_group=3,
                require_balanced_groups=True,
            )
        self.assertFalse(result["pass"])
        self.assertTrue(any("baseline has 1 runs" in item for item in result["violations"]))
        self.assertTrue(any("candidate has 2 runs" in item for item in result["violations"]))
        self.assertTrue(any("run counts differ" in item for item in result["violations"]))

    def test_throughput_dispersion_gate_detects_noisy_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = [
                load_run(_write_run(root / f"a-{index}.log", elapsed=elapsed), label="baseline")
                for index, elapsed in enumerate((1.0, 1.0, 10.0, 10.0))
            ]
            candidate = [
                load_run(_write_run(root / f"b-{index}.log", elapsed=1.0), label="candidate")
                for index in range(4)
            ]
            result = compare_runs(
                baseline,
                candidate,
                expected_tokens=32,
                max_throughput_relative_mad=0.1,
            )
        self.assertFalse(result["pass"])
        self.assertGreater(result["baseline"]["throughput_relative_mad"], 0.8)
        self.assertTrue(any("relative MAD" in item for item in result["violations"]))

    def test_duplicate_run_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write_run(Path(directory) / "run.log")
            run = load_run(path, label="baseline")
            with self.assertRaisesRegex(ValueError, "selected only once"):
                compare_runs([run], [run], expected_tokens=32)

    def test_unknown_selected_residency_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write_run(Path(directory) / "run.log")
            lines = path.read_text(encoding="utf-8").splitlines()
            report = json.loads(lines[-1])
            report["residency"]["selected"] = "auto"
            path.write_text(lines[0] + "\n" + json.dumps(report) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "resident.*offload"):
                load_run(path, label="baseline")

    def test_last_nonempty_line_must_be_json_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.log"
            path.write_text("model output\nnot-json\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "last non-empty stdout line"):
                load_run(path, label="baseline")

    def test_root_cli_exposes_comparator_without_importing_accelerators(self):
        modules_before = set(sys.modules)
        stdout = StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(cli_main(["--help"]), 0)
        loaded = set(sys.modules) - modules_before
        self.assertIn("mlx-compare", stdout.getvalue())
        self.assertFalse(any(name == "torch" or name.startswith("torch.") for name in loaded))
        self.assertFalse(any(name == "mlx" or name.startswith("mlx.") for name in loaded))

    def test_nonfinite_thresholds_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = [load_run(_write_run(root / "a.log"), label="baseline")]
            candidate = [load_run(_write_run(root / "b.log"), label="candidate")]
            for kwargs in (
                {"min_throughput_ratio": float("nan")},
                {"max_peak_memory_ratio": float("inf")},
                {"max_cache_miss_ratio": float("nan")},
                {"max_throughput_relative_mad": float("nan")},
            ):
                with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "finite"):
                    compare_runs(baseline, candidate, expected_tokens=32, **kwargs)

    def test_throughput_gate_can_be_disabled_for_correctness_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = [load_run(_write_run(root / "a.log", elapsed=1.0), label="baseline")]
            candidate = [load_run(_write_run(root / "b.log", elapsed=100.0), label="candidate")]
            result = compare_runs(
                baseline,
                candidate,
                expected_tokens=32,
                min_throughput_ratio=None,
                require_output_match=True,
            )
            self.assertTrue(result["pass"], result["violations"])
            self.assertLess(result["ratios"]["median_wall_throughput"], 0.02)
            self.assertIsNone(result["thresholds"]["min_throughput_ratio"])

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main([
                    "--baseline", baseline[0].path,
                    "--candidate", candidate[0].path,
                    "--expected-tokens", "32",
                    "--no-throughput-gate",
                    "--require-output-match",
                    "--compact",
                ])
            self.assertEqual(exit_code, 0)
            self.assertTrue(json.loads(stdout.getvalue())["pass"])

    def test_cli_writes_summary_and_uses_gate_exit_codes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = _write_run(root / "baseline.log", elapsed=8.0)
            candidate = _write_run(root / "candidate.log", elapsed=7.5)
            summary = root / "summary.json"
            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main([
                    "--baseline", str(baseline),
                    "--candidate", str(candidate),
                    "--expected-tokens", "32",
                    "--require-output-match",
                    "--write-summary", str(summary),
                    "--compact",
                ])
            self.assertEqual(exit_code, 0)
            self.assertTrue(summary.is_file())
            self.assertEqual(json.loads(summary.read_text())["schema_version"], 2)
            self.assertTrue(json.loads(stdout.getvalue())["pass"])

            stderr = StringIO()
            with redirect_stderr(stderr):
                invalid_exit = main([
                    "--baseline", str(root / "missing.log"),
                    "--candidate", str(candidate),
                    "--expected-tokens", "32",
                ])
            self.assertEqual(invalid_exit, 2)
            self.assertIn("does not exist", stderr.getvalue())

            stderr = StringIO()
            with redirect_stderr(stderr):
                threshold_exit = main([
                    "--baseline", str(baseline),
                    "--candidate", str(candidate),
                    "--expected-tokens", "32",
                    "--min-throughput-ratio", "nan",
                ])
            self.assertEqual(threshold_exit, 2)
            self.assertIn("must be finite", stderr.getvalue())

            stderr = StringIO()
            with redirect_stderr(stderr):
                collision_exit = main([
                    "--baseline", str(baseline),
                    "--candidate", str(candidate),
                    "--expected-tokens", "32",
                    "--write-summary", str(baseline),
                ])
            self.assertEqual(collision_exit, 2)
            self.assertIn("must not overwrite", stderr.getvalue())

            blocker = root / "not-a-directory"
            blocker.write_text("block", encoding="utf-8")
            stderr = StringIO()
            with redirect_stderr(stderr):
                write_exit = main([
                    "--baseline", str(baseline),
                    "--candidate", str(candidate),
                    "--expected-tokens", "32",
                    "--write-summary", str(blocker / "summary.json"),
                ])
            self.assertEqual(write_exit, 2)
            self.assertTrue(stderr.getvalue().startswith("ft mlx-compare:"))


if __name__ == "__main__":
    unittest.main()
