from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("mlx.core", reason="requires the Apple-Silicon MLX runtime")
pytest.importorskip("mlx_lm", reason="requires the Apple-Silicon mlx-lm runtime")

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.activations import swiglu

from freetoken.mlx_backend import (
    _LayerExpertSource,
    _ShardPool,
    _expert_quantization,
    _expert_cache_fraction,
    _expert_storage_bytes,
    _expected_projection_shapes,
    _linear,
    _make_cached_switch,
    _make_materializer,
    _projection_quantization,
    _resolve_cache_policy,
    _resolve_shard_cache_size,
    _select_residency,
    _validate_draft_tokenizer,
)
from freetoken.expert_banks import ExpertBanks
from freetoken.mlx_cache import MLXOffloadMoeCache


class MLXExpertComputeTest(unittest.TestCase):
    def test_switch_returns_one_output_per_route(self):
        identity = {"weight": mx.eye(2)}
        doubled = {"weight": 2 * mx.eye(2)}
        rows = (((identity, identity, identity), (doubled, doubled, identity)),)
        cache = MLXOffloadMoeCache(
            ExpertBanks("bf16", {"experts": rows}),
            2, 2, lambda source, expert: source[expert]
        )
        switch = _make_cached_switch(mx, nn, swiglu, cache, 0, 1 << 60)
        x = mx.array([[[1.0, 2.0]]])
        indices = mx.array([[[0, 1]]])
        actual = switch(x, indices, overlap=x)
        expected = mx.stack(
            (swiglu(x, x), swiglu(2 * x, 2 * x)), axis=-2
        )
        self.assertEqual(actual.shape, (1, 1, 2, 2))
        np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=1e-5)

    def test_affine_quantized_projection_uses_packed_weight(self):
        mx.random.seed(7)
        weight = mx.random.normal((4, 32)).astype(mx.bfloat16)
        packed, scales, biases = mx.quantize(weight, group_size=32, bits=4)
        x = mx.random.normal((2, 32)).astype(mx.bfloat16)
        projection = {
            "weight": packed,
            "scales": scales,
            "biases": biases,
            "mode": "affine",
            "group_size": 32,
            "bits": 4,
        }
        actual = _linear(mx, x, projection)
        expected = mx.quantized_matmul(
            x, packed, scales, biases, transpose=True,
            group_size=32, bits=4, mode="affine",
        )
        np.testing.assert_array_equal(
            np.array(actual.astype(mx.float32)),
            np.array(expected.astype(mx.float32)),
        )


class ExpertQuantizationTest(unittest.TestCase):
    def test_global_or_expert_specific_config(self):
        global_config = {"quantization": {"group_size": 64, "bits": 4}}
        self.assertEqual(_expert_quantization(global_config), {
            "mode": "affine", "group_size": 64, "bits": 4,
        })
        specific = {
            "quantization": {"group_size": 64, "bits": 4},
            "freetoken_expert_quantization": {
                "mode": "affine", "group_size": 32, "bits": 8,
            },
        }
        self.assertEqual(_expert_quantization(specific)["bits"], 8)
        self.assertIsNone(_expert_quantization({}))

    def test_rejects_unsupported_packing(self):
        with self.assertRaisesRegex(ValueError, "supports affine"):
            _expert_quantization({
                "quantization": {"mode": "mxfp4", "group_size": 64, "bits": 4}
            })

    def test_storage_budget_includes_affine_metadata(self):
        self.assertEqual(_expert_storage_bytes(32, 64, None), 12_288)
        self.assertEqual(_expert_storage_bytes(
            64, 128, {"mode": "affine", "group_size": 64, "bits": 4},
        ), 13_824)
        mixed = _expert_quantization({
            "freetoken_expert_quantization": {
                "mode": "affine", "group_size": 64, "bits": 3,
                "projections": {"down_proj": {"bits": 4}},
            },
        })
        self.assertEqual(_projection_quantization(mixed, "up_proj")["bits"], 3)
        self.assertEqual(_projection_quantization(mixed, "down_proj")["bits"], 4)
        self.assertEqual(_expert_storage_bytes(64, 128, mixed), 11_776)
        self.assertEqual(
            _expected_projection_shapes(64, 128, "up_proj", mixed),
            {"weight": (128, 6), "scales": (128, 1), "biases": (128, 1)},
        )
        self.assertEqual(
            _expected_projection_shapes(64, 128, "down_proj", mixed),
            {"weight": (64, 16), "scales": (64, 2), "biases": (64, 2)},
        )
        with self.assertRaisesRegex(ValueError, "projection quantization"):
            _expert_quantization({
                "freetoken_expert_quantization": {
                    "mode": "affine", "group_size": 64, "bits": 3,
                    "projections": {"down_proj": 4},
                },
            })

    def test_quantized_shard_cache_keeps_mappings_but_bf16_stays_small(self):
        self.assertEqual(_resolve_shard_cache_size(
            None, quantized_experts=True, num_shards=12,
        ), 8)
        self.assertEqual(_resolve_shard_cache_size(
            None, quantized_experts=True, num_shards=2,
        ), 2)
        self.assertEqual(_resolve_shard_cache_size(
            None, quantized_experts=False, num_shards=8,
        ), 1)
        self.assertEqual(_resolve_shard_cache_size(
            3, quantized_experts=False, num_shards=8,
        ), 3)

    def test_adaptive_policy_is_limited_to_quantized_performance_profile(self):
        self.assertEqual(_resolve_cache_policy(
            "auto", profile="performance", quantized_experts=True,
        ), "slru")
        self.assertEqual(_resolve_cache_policy(
            "auto", profile="stable", quantized_experts=True,
        ), "lru")
        self.assertEqual(_resolve_cache_policy(
            "auto", profile="performance", quantized_experts=False,
        ), "lru")
        self.assertEqual(_resolve_cache_policy(
            "slru", profile="stable", quantized_experts=False,
        ), "slru")

    def test_performance_profile_uses_spare_budget_for_bf16_experts(self):
        self.assertEqual(_expert_cache_fraction(
            profile="stable", quantized_experts=False,
        ), 0.20)
        self.assertEqual(_expert_cache_fraction(
            profile="performance", quantized_experts=False,
        ), 0.50)
        self.assertEqual(_expert_cache_fraction(
            profile="performance", quantized_experts=True,
        ), 0.40)


class DraftTokenizerTest(unittest.TestCase):
    class Tokenizer:
        bos_token_id = None
        eos_token_id = 2
        pad_token_id = None

        def __init__(self, vocab):
            self.vocab = vocab

        def get_vocab(self):
            return self.vocab

    def test_matching_token_ids_are_accepted(self):
        _validate_draft_tokenizer(
            self.Tokenizer({"hello": 1}), self.Tokenizer({"hello": 1})
        )

    def test_mismatched_vocabulary_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            _validate_draft_tokenizer(
                self.Tokenizer({"hello": 1}), self.Tokenizer({"hello": 2})
            )


class ResidencySelectionTest(unittest.TestCase):
    class FakeMX:
        @staticmethod
        def device_info():
            return {
                "max_recommended_working_set_size": 12 << 30,
                "memory_size": 16 << 30,
            }

    @staticmethod
    def args(residency="auto"):
        return SimpleNamespace(
            residency=residency,
            memory_limit_gb=None,
            system_headroom_gb=2.0,
        )

    def make_model(self, root: Path, *, mixed=False, size_bytes=None):
        model = root / "model"
        model.mkdir()
        config = {
            "model_type": "qwen2_moe",
            "hidden_size": 2,
            "moe_intermediate_size": 4,
        }
        if mixed:
            config["freetoken_expert_quantization"] = {
                "mode": "affine", "group_size": 64, "bits": 4,
            }
        (model / "config.json").write_text(json.dumps(config))
        payload_size = size_bytes or 7
        tensor_name = (
            "model.layers.0.mlp.experts.0.up_proj.weight"
            if size_bytes is not None else "model.embed_tokens.weight"
        )
        header = json.dumps({
            tensor_name: {
                "dtype": "U8", "shape": [payload_size],
                "data_offsets": [0, payload_size],
            },
        }).encode()
        weights = model / "model-01.safetensors"
        with weights.open("wb") as stream:
            stream.write(len(header).to_bytes(8, "little"))
            stream.write(header)
            stream.truncate(8 + len(header) + payload_size)
        return model

    @patch("freetoken.mlx_backend._memory_pressure_available")
    def test_auto_uses_resident_when_estimate_fits(self, available):
        available.return_value = 14 << 30
        with tempfile.TemporaryDirectory() as directory:
            model = self.make_model(Path(directory), size_bytes=6 << 30)
            decision = _select_residency(
                self.args(), model, None, self.FakeMX,
            )
        self.assertEqual(decision["selected"], "resident")
        self.assertEqual(decision["reason"], "resident_is_safe")

    @patch("freetoken.mlx_backend._memory_pressure_available")
    def test_auto_forces_mixed_checkpoint_through_freetoken(self, available):
        available.return_value = 14 << 30
        with tempfile.TemporaryDirectory() as directory:
            model = self.make_model(Path(directory), mixed=True)
            decision = _select_residency(
                self.args(), model, None, self.FakeMX,
            )
        self.assertEqual(decision["selected"], "offload")
        self.assertIn("requires_freetoken", decision["reason"])

    @patch("freetoken.mlx_backend._memory_pressure_available")
    def test_auto_preserves_headroom_during_memory_pressure(self, available):
        available.return_value = 7 << 30
        with tempfile.TemporaryDirectory() as directory:
            model = self.make_model(Path(directory), size_bytes=6 << 30)
            decision = _select_residency(
                self.args(), model, None, self.FakeMX,
            )
        self.assertEqual(decision["selected"], "offload")
        self.assertIn("current_pressure", decision["reason"])

    @patch("freetoken.mlx_backend._memory_pressure_available")
    def test_offload_reserves_full_draft_from_expert_cache(self, available):
        available.return_value = 14 << 30
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = self.make_model(root)
            draft = root / "draft"
            draft.mkdir()
            (draft / "model.safetensors").write_bytes(b"d" * 1024)
            decision = _select_residency(
                self.args("offload"), model, draft, self.FakeMX,
            )
        self.assertGreater(decision["offload_draft_reserve_bytes"], 1024)
        self.assertEqual(
            decision["offload_expert_cache_budget_bytes"],
            min(
                int(decision["offload_total_budget_bytes"] * 0.20),
                decision["offload_total_budget_bytes"]
                - decision["offload_base_reserve_bytes"]
                - decision["offload_draft_reserve_bytes"],
            ),
        )

    @patch("freetoken.mlx_backend._memory_pressure_available")
    def test_refuses_to_start_when_offload_cannot_preserve_headroom(self, available):
        available.return_value = 2 << 30
        with tempfile.TemporaryDirectory() as directory:
            model = self.make_model(Path(directory))
            with self.assertRaisesRegex(RuntimeError, "insufficient safe memory"):
                _select_residency(
                    self.args("offload"), model, None, self.FakeMX,
                )


class ShardPoolTest(unittest.TestCase):
    def test_selected_array_is_removed_from_retained_mapping(self):
        first = object()
        second = object()

        class FakeMX:
            calls = 0

            @classmethod
            def load(cls, path):
                cls.calls += 1
                return {"first": first, "second": second}

        pool = _ShardPool(FakeMX, capacity=1)
        self.assertIs(pool.take("shard", "first"), first)
        self.assertNotIn("first", pool._mappings["shard"])
        self.assertIs(pool.take("shard", "second"), second)
        self.assertEqual((pool.opens, pool.hits, FakeMX.calls), (1, 1, 1))

        # A consumed key requires a fresh parse instead of being kept alive by
        # the shard pool after the expert cache releases it.
        self.assertIs(pool.take("shard", "first"), first)
        self.assertEqual((pool.opens, FakeMX.calls), (2, 2))

    def test_lazy_materializer_defers_array_evaluation(self):
        class Array:
            pass

        class FakeMX:
            array = Array
            eval_calls = 0

            @staticmethod
            def load(path):
                return {"up": Array(), "gate": Array(), "down": Array()}

            @classmethod
            def eval(cls, *arrays):
                cls.eval_calls += 1

        refs = (
            {"weight": ("shard", "up"), "bits": 4},
            {"weight": ("shard", "gate")},
            {"weight": ("shard", "down")},
        )
        source = _LayerExpertSource((refs,))
        pool = _ShardPool(FakeMX, capacity=0)

        expert = _make_materializer(FakeMX, pool, eager=False)(source, 0)
        self.assertEqual(FakeMX.eval_calls, 0)
        self.assertEqual(expert[0]["bits"], 4)
        _make_materializer(FakeMX, pool, eager=True)(source, 0)
        self.assertEqual(FakeMX.eval_calls, 1)


if __name__ == "__main__":
    unittest.main()
