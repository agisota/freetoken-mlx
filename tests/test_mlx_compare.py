from __future__ import annotations

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
from freetoken.mlx_compare import compare_runs, load_run, main  # noqa: E402


def _write_run(
    path: Path,
    *,
    output: str = "generated text",
    tokens: int = 32,
    elapsed: float = 8.0,
    peak: int | None = 8_000_000_000,
    misses: int | None = 100,
    residency: str = "offload",
) -> Path:
    report = {
        "backend": "mlx",
        "generated_tokens": tokens,
        "elapsed_seconds": elapsed,
        "residency": {"selected": residency},
        "memory": None if peak is None else {"peak_bytes": peak},
        "expert_cache": None if misses is None else {"misses": misses},
    }
    path.write_text(output + "\n" + json.dumps(report) + "\n", encoding="utf-8")
    return path


class MLXCompareTest(unittest.TestCase):
    def test_comparison_passes_for_equal_output_and_safe_ratios(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = [
                load_run(
                    _write_run(root / "baseline-1.log", elapsed=8.0),
                    label="baseline",
                ),
                load_run(
                    _write_run(root / "baseline-2.log", elapsed=8.4),
                    label="baseline",
                ),
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
            )

        self.assertTrue(result["pass"], result["violations"])
        self.assertGreater(result["ratios"]["median_wall_throughput"], 1.0)
        self.assertLess(result["ratios"]["median_peak_memory"], 1.05)
        self.assertEqual(result["baseline"]["generated_tokens"], [32])

    def test_exact_generated_token_count_is_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = [load_run(_write_run(root / "a.log"), label="baseline")]
            candidate = [
                load_run(
                    _write_run(root / "b.log", tokens=17, elapsed=4.0),
                    label="candidate",
                )
            ]
            result = compare_runs(baseline, candidate, expected_tokens=32)

        self.assertFalse(result["pass"])
        self.assertTrue(
            any("expected exactly 32" in violation for violation in result["violations"])
        )

    def test_output_mismatch_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = [
                load_run(_write_run(root / "a.log", output="alpha"), label="baseline")
            ]
            candidate = [
                load_run(_write_run(root / "b.log", output="beta"), label="candidate")
            ]
            result = compare_runs(
                baseline,
                candidate,
                expected_tokens=32,
                require_output_match=True,
            )

        self.assertFalse(result["pass"])
        self.assertTrue(
            any("generated output differs" in violation for violation in result["violations"])
        )

    def test_missing_peak_memory_cannot_bypass_default_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = [
                load_run(_write_run(root / "a.log", peak=None), label="baseline")
            ]
            candidate = [
                load_run(_write_run(root / "b.log", peak=None), label="candidate")
            ]
            result = compare_runs(baseline, candidate, expected_tokens=32)

        self.assertFalse(result["pass"])
        self.assertTrue(
            any("peak-memory gate requires" in item for item in result["violations"])
        )

    def test_one_missing_peak_memory_cannot_hide_behind_group_median(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = [
                load_run(_write_run(root / "a-1.log"), label="baseline"),
                load_run(
                    _write_run(root / "a-2.log", peak=None), label="baseline"
                ),
            ]
            candidate = [
                load_run(_write_run(root / "b.log"), label="candidate")
            ]
            result = compare_runs(baseline, candidate, expected_tokens=32)

        self.assertFalse(result["pass"])
        self.assertTrue(
            any("in every run" in item for item in result["violations"])
        )

    def test_zero_token_baseline_fails_without_division_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = [
                load_run(
                    _write_run(root / "a.log", tokens=0), label="baseline"
                )
            ]
            candidate = [
                load_run(_write_run(root / "b.log"), label="candidate")
            ]
            result = compare_runs(baseline, candidate, expected_tokens=32)

        self.assertFalse(result["pass"])
        self.assertIsNone(result["ratios"]["median_wall_throughput"])
        self.assertTrue(
            any("positive baseline token rate" in item for item in result["violations"])
        )

    def test_cache_miss_gate_requires_metric_in_every_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = [
                load_run(_write_run(root / "a.log"), label="baseline")
            ]
            candidate = [
                load_run(
                    _write_run(root / "b-1.log"), label="candidate"
                ),
                load_run(
                    _write_run(root / "b-2.log", misses=None), label="candidate"
                ),
            ]
            result = compare_runs(
                baseline,
                candidate,
                expected_tokens=32,
                max_cache_miss_ratio=1.05,
            )

        self.assertFalse(result["pass"])
        self.assertTrue(
            any("cache-miss gate" in item and "in every run" in item
                for item in result["violations"])
        )

    def test_same_residency_gate_rejects_mixed_group(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = [
                load_run(
                    _write_run(root / "a-1.log", residency="offload"),
                    label="baseline",
                ),
                load_run(
                    _write_run(root / "a-2.log", residency="resident"),
                    label="baseline",
                ),
            ]
            candidate = [
                load_run(
                    _write_run(root / "b.log", residency="offload"),
                    label="candidate",
                )
            ]
            result = compare_runs(
                baseline,
                candidate,
                expected_tokens=32,
                require_same_residency=True,
            )

        self.assertFalse(result["pass"])
        self.assertTrue(
            any("one residency path" in item for item in result["violations"])
        )

    def test_last_nonempty_line_must_be_json_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.log"
            path.write_text("model output\nnot-json\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "last non-empty stdout line"):
                load_run(path, label="baseline")

    def test_root_cli_exposes_comparator_without_importing_accelerators(self):
        stdout = StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(cli_main(["--help"]), 0)
        self.assertIn("mlx-compare", stdout.getvalue())
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("mlx", sys.modules)

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
            self.assertTrue(json.loads(summary.read_text())["pass"])
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


if __name__ == "__main__":
    unittest.main()
