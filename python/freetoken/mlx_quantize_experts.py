"""Create a mixed-precision Qwen MoE checkpoint for the FreeToken MLX bank."""

from __future__ import annotations

import argparse
import glob
import json
import platform
import re
import shutil
import tempfile
from pathlib import Path
from typing import Sequence


SUPPORTED_MODEL_TYPE = "qwen2_moe"
EXPERT_FIELD_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(up_proj|gate_proj|down_proj)\.(weight|scales|biases|bias)$"
)
EXPERT_PREFIX_RE = re.compile(
    r"^model\.layers\.\d+\.mlp\.experts\.\d+\."
)


def _mlx():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("MLX expert quantization requires Apple Silicon")
    try:
        import mlx.core as mx
    except ImportError as exc:
        raise RuntimeError("install freetoken[mlx] before quantizing experts") from exc
    return mx


def _resolve_model(source: str) -> Path:
    path = Path(source).expanduser()
    if path.exists():
        return path.resolve()
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(
        source,
        allow_patterns=[
            "*.json", "*.jinja", "*.model", "*.safetensors",
            "*.tiktoken", "*.txt",
        ],
    ))


def quantize_experts(
    source: str,
    output: str,
    *,
    bits: int = 4,
    down_bits: int | None = None,
    group_size: int = 64,
) -> dict[str, int | str]:
    """Keep dense/shared tensors intact and quantize only routed experts."""
    if bits not in {2, 3, 4, 5, 6, 8}:
        raise ValueError("expert bits must be one of 2, 3, 4, 5, 6, or 8")
    if down_bits is not None and down_bits not in {2, 3, 4, 5, 6, 8}:
        raise ValueError("down projection bits must be one of 2, 3, 4, 5, 6, or 8")
    if group_size not in {32, 64, 128}:
        raise ValueError("expert group size must be 32, 64, or 128")
    model_path = _resolve_model(source)
    output_path = Path(output).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"output already exists: {output_path}")
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"model config is missing: {config_path}")
    config = json.loads(config_path.read_text())
    if config.get("model_type") != SUPPORTED_MODEL_TYPE:
        raise ValueError(
            f"expert quantizer supports exactly model_type={SUPPORTED_MODEL_TYPE!r}"
        )
    if config.get("quantization") is not None or \
            config.get("quantization_config") is not None:
        raise ValueError("source must contain BF16 weights, not global quantization")
    if config.get("freetoken_expert_quantization") is not None:
        raise ValueError("source routed experts are already quantized")
    num_layers = int(config.get("num_hidden_layers", 0))
    num_experts = int(config.get("num_experts", config.get("num_local_experts", 0)))
    if num_layers < 1 or num_experts < 1:
        raise ValueError("Qwen config is missing layer or expert counts")
    weight_files = sorted(glob.glob(str(model_path / "model*.safetensors")))
    if not weight_files:
        raise FileNotFoundError(f"no safetensors weights found in {model_path}")

    mx = _mlx()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{output_path.name}.", dir=output_path.parent,
    ))
    dense_bytes = expert_bytes = dense_tensors = expert_tensors = 0
    expert_projections: dict[tuple[int, int], set[str]] = {}
    weight_map: dict[str, str] = {}
    mx.reset_peak_memory()
    try:
        for shard_id, weight_file in enumerate(weight_files, 1):
            weights = mx.load(weight_file)
            dense = {}
            experts = {}
            for name, value in weights.items():
                match = EXPERT_FIELD_RE.match(name)
                if match is None:
                    if EXPERT_PREFIX_RE.match(name):
                        raise ValueError(f"unsupported routed expert tensor: {name}")
                    dense[name] = value
                    continue
                layer_id, expert_id = int(match.group(1)), int(match.group(2))
                projection, field = match.group(3), match.group(4)
                if field != "weight":
                    raise ValueError(f"source routed experts are already packed: {name}")
                if value.ndim != 2 or value.shape[-1] % group_size:
                    raise ValueError(
                        f"expert weight shape is incompatible with group_size "
                        f"{group_size}: {name} {value.shape}"
                    )
                if value.dtype != mx.bfloat16:
                    raise ValueError(f"expert weight is not BF16: {name} {value.dtype}")
                key = (layer_id, expert_id)
                projections = expert_projections.setdefault(key, set())
                if projection in projections:
                    raise ValueError(f"duplicate routed expert projection: {name}")
                projections.add(projection)
                experts[name] = value
            if dense:
                duplicate = next((name for name in dense if name in weight_map), None)
                if duplicate is not None:
                    raise ValueError(f"duplicate tensor key: {duplicate}")
                destination_name = f"model-dense-{shard_id:02d}.safetensors"
                destination = temporary / destination_name
                mx.save_safetensors(str(destination), dense)
                dense_bytes += sum(value.nbytes for value in dense.values())
                dense_tensors += len(dense)
                weight_map.update({name: destination_name for name in dense})
            del dense
            mx.clear_cache()

            quantized = {}
            for name, value in experts.items():
                projection = EXPERT_FIELD_RE.match(name).group(3)
                projection_bits = (
                    down_bits
                    if projection == "down_proj" and down_bits is not None
                    else bits
                )
                weight, scales, biases = mx.quantize(
                    value, group_size=group_size, bits=projection_bits, mode="affine",
                )
                prefix = name.removesuffix(".weight")
                quantized[name] = weight
                quantized[f"{prefix}.scales"] = scales
                quantized[f"{prefix}.biases"] = biases
            if quantized:
                duplicate = next(
                    (name for name in quantized if name in weight_map), None,
                )
                if duplicate is not None:
                    raise ValueError(f"duplicate tensor key: {duplicate}")
                destination_name = f"model-experts-{shard_id:02d}.safetensors"
                destination = temporary / destination_name
                mx.save_safetensors(str(destination), quantized)
                expert_bytes += sum(value.nbytes for value in quantized.values())
                expert_tensors += len(quantized)
                weight_map.update({name: destination_name for name in quantized})
            del experts, quantized, weights
            mx.clear_cache()

        expected = {
            (layer_id, expert_id)
            for layer_id in range(num_layers)
            for expert_id in range(num_experts)
        }
        if expert_projections.keys() != expected:
            missing = sorted(expected - expert_projections.keys())
            extra = sorted(expert_projections.keys() - expected)
            raise ValueError(
                f"routed expert bank is incomplete: missing={missing[:4]} extra={extra[:4]}"
            )
        required = {"up_proj", "gate_proj", "down_proj"}
        incomplete = [
            key for key, projections in expert_projections.items()
            if projections != required
        ]
        if incomplete:
            raise ValueError(f"routed expert projections are incomplete: {incomplete[:4]}")
        for source_file in model_path.iterdir():
            if not source_file.is_file():
                continue
            if source_file.name == "config.json" or \
                    source_file.name == "model.safetensors.index.json" or \
                    source_file.match("model*.safetensors"):
                continue
            shutil.copy2(source_file, temporary / source_file.name)
        config["freetoken_expert_quantization"] = {
            "mode": "affine", "group_size": group_size, "bits": bits,
        }
        if down_bits is not None and down_bits != bits:
            config["freetoken_expert_quantization"]["projections"] = {
                "down_proj": {"bits": down_bits},
            }
        (temporary / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        index = {
            "metadata": {"total_size": dense_bytes + expert_bytes},
            "weight_map": weight_map,
        }
        (temporary / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2) + "\n"
        )
        if output_path.exists():
            raise FileExistsError(f"output was created during conversion: {output_path}")
        temporary.rename(output_path)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return {
        "source": str(model_path),
        "output": str(output_path),
        "bits": bits,
        "down_bits": down_bits if down_bits is not None else bits,
        "group_size": group_size,
        "dense_tensors": dense_tensors,
        "expert_tensors": expert_tensors,
        "dense_bytes": dense_bytes,
        "expert_bytes": expert_bytes,
        "peak_conversion_bytes": mx.get_peak_memory(),
    }


def build_parser(prog: str = "ft mlx-quantize-experts") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument("--model", required=True, help="BF16 Qwen MoE model ID or path")
    parser.add_argument("--output", required=True, help="new mixed checkpoint directory")
    parser.add_argument("--bits", type=int, choices=(2, 3, 4, 5, 6, 8), default=4)
    parser.add_argument("--down-bits", type=int, choices=(2, 3, 4, 5, 6, 8))
    parser.add_argument("--group-size", type=int, choices=(32, 64, 128), default=64)
    return parser


def main(argv: Sequence[str] | None = None, *, prog: str = "ft mlx-quantize-experts") -> int:
    args = build_parser(prog).parse_args(argv)
    print(json.dumps(quantize_experts(
        args.model, args.output, bits=args.bits, group_size=args.group_size,
        down_bits=args.down_bits,
    ), sort_keys=True))
    return 0


__all__ = ["main", "quantize_experts"]
