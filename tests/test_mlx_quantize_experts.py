from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pytest

pytest.importorskip("mlx.core", reason="requires the Apple-Silicon MLX runtime")

import mlx.core as mx

from freetoken.mlx_quantize_experts import quantize_experts


class MLXExpertQuantizerTest(unittest.TestCase):
    def test_keeps_dense_bf16_and_packs_only_routed_experts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "mixed"
            source.mkdir()
            (source / "config.json").write_text(json.dumps({
                "model_type": "qwen2_moe",
                "num_hidden_layers": 1,
                "num_experts": 1,
            }))
            dense = mx.arange(32).astype(mx.bfloat16).reshape(1, 32)
            expert = mx.arange(64).astype(mx.bfloat16).reshape(2, 32)
            mx.save_safetensors(str(source / "model-01.safetensors"), {
                "model.embed_tokens.weight": dense,
                "model.layers.0.mlp.experts.0.up_proj.weight": expert,
                "model.layers.0.mlp.experts.0.gate_proj.weight": expert,
                "model.layers.0.mlp.experts.0.down_proj.weight": expert,
            })

            report = quantize_experts(
                str(source), str(output), bits=4, group_size=32,
            )

            dense_weights = mx.load(str(output / "model-dense-01.safetensors"))
            expert_weights = mx.load(str(output / "model-experts-01.safetensors"))
            config = json.loads((output / "config.json").read_text())
            index = json.loads((output / "model.safetensors.index.json").read_text())
            self.assertEqual(report["dense_tensors"], 1)
            self.assertEqual(report["expert_tensors"], 9)
            self.assertEqual(config["freetoken_expert_quantization"], {
                "mode": "affine", "group_size": 32, "bits": 4,
            })
            self.assertEqual(
                dense_weights["model.embed_tokens.weight"].dtype, mx.bfloat16,
            )
            prefix = "model.layers.0.mlp.experts.0.up_proj"
            self.assertIn(f"{prefix}.weight", expert_weights)
            self.assertIn(f"{prefix}.scales", expert_weights)
            self.assertIn(f"{prefix}.biases", expert_weights)
            self.assertEqual(
                index["weight_map"][f"{prefix}.weight"],
                "model-experts-01.safetensors",
            )

            projection_mixed = root / "projection-mixed"
            mixed_report = quantize_experts(
                str(source), str(projection_mixed), bits=3, down_bits=4,
                group_size=32,
            )
            mixed_config = json.loads(
                (projection_mixed / "config.json").read_text()
            )
            mixed_weights = mx.load(str(
                projection_mixed / "model-experts-01.safetensors"
            ))
            self.assertEqual(mixed_report["down_bits"], 4)
            self.assertEqual(mixed_config["freetoken_expert_quantization"], {
                "mode": "affine", "group_size": 32, "bits": 3,
                "projections": {"down_proj": {"bits": 4}},
            })
            self.assertLess(
                mixed_weights[
                    "model.layers.0.mlp.experts.0.up_proj.weight"
                ].shape[-1],
                mixed_weights[
                    "model.layers.0.mlp.experts.0.down_proj.weight"
                ].shape[-1],
            )

    def test_refuses_to_overwrite_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            output.mkdir()
            with self.assertRaises(FileExistsError):
                quantize_experts(str(source), str(output))

    def test_rejects_incomplete_expert_bank(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "config.json").write_text(json.dumps({
                "model_type": "qwen2_moe",
                "num_hidden_layers": 1,
                "num_experts": 1,
            }))
            expert = mx.zeros((2, 32), dtype=mx.bfloat16)
            mx.save_safetensors(str(source / "model-01.safetensors"), {
                "model.layers.0.mlp.experts.0.up_proj.weight": expert,
            })
            with self.assertRaisesRegex(ValueError, "projections are incomplete"):
                quantize_experts(
                    str(source), str(root / "output"), bits=4, group_size=32,
                )

    def test_rejects_existing_expert_quantization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "config.json").write_text(json.dumps({
                "model_type": "qwen2_moe",
                "num_hidden_layers": 1,
                "num_experts": 1,
                "freetoken_expert_quantization": {
                    "mode": "affine", "group_size": 64, "bits": 4,
                },
            }))
            with self.assertRaisesRegex(ValueError, "already quantized"):
                quantize_experts(str(source), str(root / "output"))


if __name__ == "__main__":
    unittest.main()
