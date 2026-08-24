from __future__ import annotations

import subprocess
import sys
import unittest
from contextlib import redirect_stderr
from io import StringIO
from unittest.mock import patch

from freetoken.mlx_backend import build_parser, main


class MLXCliTest(unittest.TestCase):
    def test_help_is_torch_free(self):
        proc = subprocess.run(
            [sys.executable, "-m", "freetoken.cli", "generate", "--help"],
            check=False, capture_output=True, text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("--backend {mlx}", proc.stdout)
        self.assertNotIn("torch", sys.modules)

    def test_root_help_lists_expert_quantizer_without_importing_torch(self):
        proc = subprocess.run(
            [sys.executable, "-m", "freetoken.cli", "--help"],
            check=False, capture_output=True, text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("mlx-quantize-experts", proc.stdout)
        self.assertNotIn("torch", sys.modules)

    def test_stable_memory_defaults(self):
        args = build_parser().parse_args(["--backend", "mlx", "--model", "model"])
        self.assertEqual(args.moe_cache_size, 4)
        self.assertIsNone(args.expert_cache_budget_gb)
        self.assertEqual(args.allocator_cache_mb, 256)
        self.assertIsNone(args.memory_limit_gb)
        self.assertEqual(args.profile, "stable")
        self.assertIsNone(args.shard_cache_size)
        self.assertEqual(args.eval_interval, 1)
        self.assertFalse(args.detailed_timing)
        self.assertIsNone(args.grouped_gemv)
        self.assertIsNone(args.async_overlap)
        self.assertEqual(args.cache_policy, "auto")
        self.assertIsNone(args.draft_model)
        self.assertEqual(args.num_draft_tokens, 2)
        self.assertEqual(args.residency, "auto")
        self.assertEqual(args.system_headroom_gb, 2.0)

    def test_performance_profile_is_available(self):
        args = build_parser().parse_args([
            "--backend", "mlx", "--model", "model", "--profile", "performance",
        ])
        self.assertEqual(args.profile, "performance")
        self.assertIsNone(args.grouped_gemv)
        self.assertIsNone(args.async_overlap)
        disabled = build_parser().parse_args([
            "--backend", "mlx", "--model", "model", "--profile", "performance",
            "--no-grouped-gemv",
        ])
        self.assertFalse(disabled.grouped_gemv)
        no_overlap = build_parser().parse_args([
            "--backend", "mlx", "--model", "model", "--profile", "performance",
            "--no-async-overlap",
        ])
        self.assertFalse(no_overlap.async_overlap)

    def test_expert_cache_budget_is_explicit_and_positive(self):
        args = build_parser().parse_args([
            "--backend", "mlx", "--model", "model",
            "--expert-cache-budget-gb", "0.75",
        ])
        self.assertEqual(args.expert_cache_budget_gb, 0.75)
        with self.assertRaisesRegex(ValueError, "must be positive"):
            main([
                "--backend", "mlx", "--model", "model",
                "--expert-cache-budget-gb", "0",
            ])

    def test_metal_oom_is_reported_without_traceback(self):
        stderr = StringIO()
        with patch("freetoken.mlx_backend.run", side_effect=RuntimeError(
            "[METAL] Insufficient Memory"
        )), redirect_stderr(stderr):
            result = main(["--backend", "mlx", "--model", "model"])
        self.assertEqual(result, 1)
        self.assertIn("stopped before exhausting system memory", stderr.getvalue())
        self.assertIn("--moe-cache-size", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
