"""Minimal, real MLX backend for Qwen1.5-MoE on Apple Silicon."""

from __future__ import annotations

import argparse
import glob
import json
import platform
import re
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Sequence

from freetoken.expert_banks import ExpertBanks
from freetoken.mlx_cache import MLXOffloadMoeCache

SUPPORTED_MODEL_TYPE = "qwen2_moe"
EXPERT_WEIGHT_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(up_proj|gate_proj|down_proj)\.(weight|scales|biases|bias)$"
)


def _mlx_imports():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("the MLX backend requires Apple Silicon (macOS arm64)")
    try:
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm.generate import stream_generate
        from mlx_lm.models.activations import swiglu
        from mlx_lm.tokenizer_utils import load as load_tokenizer
        from mlx_lm.models.qwen2_moe import Model, ModelArgs
        from mlx_lm.utils import load_config
    except ImportError as exc:
        raise RuntimeError("MLX backend dependencies are missing; install freetoken[mlx]") from exc
    return mx, nn, stream_generate, swiglu, load_tokenizer, load_config, Model, ModelArgs


class _LayerExpertSource:
    """FreeToken bank containing paths/keys, never resident expert arrays."""

    def __init__(self, rows: tuple[Any, ...]):
        self.rows = rows


class _ShardPool:
    """Small LRU of parsed safetensors mappings without resident expert rows."""

    def __init__(self, mx, capacity: int, timing: dict[str, int] | None = None):
        self.mx = mx
        self.capacity = capacity
        self._mappings = OrderedDict()
        self.opens = 0
        self.hits = 0
        self.timing = timing

    def take(self, path: str, key: str):
        mapping = self._mappings.pop(path, None)
        if mapping is None or key not in mapping:
            started = time.perf_counter_ns()
            mapping = self.mx.load(path)
            if self.timing is not None:
                self.timing["shard_open_ns"] += time.perf_counter_ns() - started
            self.opens += 1
        else:
            self.hits += 1
        # Removing the selected row is essential: after expert-cache eviction,
        # the parsed shard mapping must not keep its evaluated Metal array alive.
        value = mapping.pop(key)
        self._mappings[path] = mapping
        while len(self._mappings) > self.capacity:
            self._mappings.popitem(last=False)
        return value

    def clear(self):
        self._mappings.clear()


def _make_materializer(mx, shard_pool: _ShardPool, eager: bool = True):
    def materialize(source: _LayerExpertSource, expert_id: int):
        refs = source.rows[expert_id]
        expert = []
        local_loaded = {}
        for projection in refs:
            fields = {}
            for field, reference in projection.items():
                if not isinstance(reference, tuple):
                    fields[field] = reference
                    continue
                path, key = reference
                if shard_pool.capacity == 0:
                    if path not in local_loaded:
                        local_loaded[path] = mx.load(path)
                    fields[field] = local_loaded[path][key]
                else:
                    fields[field] = shard_pool.take(path, key)
            expert.append(fields)
        expert = tuple(expert)
        arrays = [v for projection in expert for v in projection.values() if isinstance(v, mx.array)]
        if eager:
            mx.eval(*arrays)
        return expert

    return materialize


def _expert_quantization(config: dict[str, Any]) -> dict[str, Any] | None:
    quantization = config.get("freetoken_expert_quantization")
    if quantization is None:
        quantization = config.get("quantization")
    if quantization is None:
        return None

    def validate(candidate):
        mode = str(candidate.get("mode", "affine"))
        bits = int(candidate.get("bits", 0))
        group_size = int(candidate.get("group_size", 0))
        if mode != "affine" or bits not in {2, 3, 4, 5, 6, 8} or \
                group_size not in {32, 64, 128}:
            raise ValueError(
                "MLX expert cache supports affine 2/3/4/5/6/8-bit weights "
                "with group_size 32, 64, or 128"
            )
        return {"mode": mode, "bits": bits, "group_size": group_size}

    normalized = validate(quantization)
    overrides = quantization.get("projections")
    if overrides is not None:
        if not isinstance(overrides, dict) or not set(overrides) <= {
            "up_proj", "gate_proj", "down_proj",
        }:
            raise ValueError("invalid expert projection quantization overrides")
        if not all(isinstance(override, dict) for override in overrides.values()):
            raise ValueError("invalid expert projection quantization overrides")
        normalized["projections"] = {
            name: validate(normalized | override)
            for name, override in overrides.items()
        }
    return normalized


def _projection_quantization(
    quantization: dict[str, Any], projection: str,
) -> dict[str, Any]:
    return quantization.get("projections", {}).get(
        projection,
        {name: quantization[name] for name in ("mode", "bits", "group_size")},
    )


def _resolve_shard_cache_size(
    requested: int | None, *, quantized_experts: bool, num_shards: int,
) -> int:
    if requested is not None:
        return requested
    return min(8, num_shards) if quantized_experts else 1


def _resolve_cache_policy(
    requested: str, *, profile: str, quantized_experts: bool,
) -> str:
    if requested not in {"auto", "lru", "slru"}:
        raise ValueError("MLX expert cache policy must be auto, lru, or slru")
    if requested != "auto":
        return requested
    return "slru" if profile == "performance" and quantized_experts else "lru"


def _expert_cache_fraction(*, profile: str, quantized_experts: bool) -> float:
    if quantized_experts:
        return 0.40
    # BF16 experts are much larger, so the former 20% cap left several safe
    # GiB unused in the performance profile and caused avoidable checkpoint
    # page-ins. The residency budget still subtracts dense/runtime/draft bytes
    # and current system headroom before this fraction is applied.
    return 0.50 if profile == "performance" else 0.20


def _expert_storage_bytes(
    hidden_size: int, intermediate_size: int,
    quantization: dict[str, Any] | None,
) -> int:
    if quantization is None:
        return 3 * hidden_size * intermediate_size * 2
    total = 0
    for projection in ("up_proj", "gate_proj", "down_proj"):
        output_size, input_size = (
            (hidden_size, intermediate_size)
            if projection == "down_proj"
            else (intermediate_size, hidden_size)
        )
        projection_quantization = _projection_quantization(
            quantization, projection,
        )
        # MLX packs each input row into uint32 words. Scales and biases use
        # one two-byte value each per affine group. Account for row rounding.
        packed_columns = (
            input_size * projection_quantization["bits"] + 31
        ) // 32
        groups = (
            input_size + projection_quantization["group_size"] - 1
        ) // projection_quantization["group_size"]
        total += output_size * (packed_columns * 4 + groups * 4)
    return total


def _expected_projection_shapes(
    hidden_size: int,
    intermediate_size: int,
    projection: str,
    quantization: dict[str, Any] | None,
) -> dict[str, tuple[int, int]]:
    output_size, input_size = (
        (hidden_size, intermediate_size)
        if projection == "down_proj"
        else (intermediate_size, hidden_size)
    )
    if quantization is None:
        return {"weight": (output_size, input_size)}
    projection_quantization = _projection_quantization(
        quantization, projection,
    )
    packed_columns = (
        input_size * projection_quantization["bits"] + 31
    ) // 32
    groups = (
        input_size + projection_quantization["group_size"] - 1
    ) // projection_quantization["group_size"]
    return {
        "weight": (output_size, packed_columns),
        "scales": (output_size, groups),
        "biases": (output_size, groups),
    }


def _validate_draft_tokenizer(tokenizer, draft_tokenizer) -> None:
    """Reject draft models whose token IDs cannot be compared to the target."""
    target_vocab = tokenizer.get_vocab()
    draft_vocab = draft_tokenizer.get_vocab()
    if target_vocab != draft_vocab:
        raise ValueError("draft model tokenizer vocabulary does not match the target")
    for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if getattr(tokenizer, name, None) != getattr(draft_tokenizer, name, None):
            raise ValueError(f"draft model tokenizer {name} does not match the target")


def _load_draft_model(path: str, tokenizer):
    from mlx_lm.utils import load

    try:
        draft_model, draft_tokenizer = load(path)
    except ValueError as exc:
        # Some older mlx-community Qwen1.5 conversions contain a separate,
        # quantized lm_head even though config.json still says it is tied. MLX
        # correctly rejects the orphan scale/bias arrays. Instantiate the head
        # declared by the checkpoint instead of ignoring those parameters.
        message = str(exc)
        if "lm_head.biases" not in message or "lm_head.scales" not in message:
            raise
        draft_model, draft_tokenizer = load(
            path, model_config={"tie_word_embeddings": False}
        )
    _validate_draft_tokenizer(tokenizer, draft_tokenizer)
    return draft_model


def _linear(mx, x, projection: dict[str, Any]):
    if "scales" in projection:
        y = mx.quantized_matmul(
            x,
            projection["weight"],
            projection["scales"],
            projection.get("biases"),
            transpose=True,
            group_size=projection["group_size"],
            bits=projection["bits"],
            mode=projection.get("mode", "affine"),
        )
    else:
        y = mx.matmul(x, projection["weight"].T)
    if "bias" in projection:
        y = y + projection["bias"]
    return y


def _make_grouped_gemv(mx, swiglu, hidden_size: int, intermediate_size: int):
    """Build a top-4 BF16 GEMV from MLX's installed Metal kernel template."""
    mlx_root = Path(mx.__file__).parent
    gemv_path = mlx_root / "include/mlx/backend/metal/kernels/gemv.h"
    if not gemv_path.is_file():
        raise RuntimeError(f"MLX GEMV template is unavailable: {gemv_path}")
    source = gemv_path.read_text()
    start = source.index("#define MLX_MTL_CONST")
    end = source.index("/// Vector matrix multiplication", start)
    header = source[start:end]
    # GEMVKernel::run is a kernel helper in MLX. Runtime custom kernels pass
    # these dimensions as compile-time constants, so remove entry-point buffer
    # annotations and their constant address-space references.
    header = header.replace("const constant int&", "const int")
    header = header.replace("const constant float&", "const float")
    header = re.sub(r"\s*\[\[buffer\(\d+\)\]\]", "", header)
    header = re.sub(r"\s*\[\[threadgroup\(\d+\)\]\]", "", header)

    up_grid = (intermediate_size + 15) // 16
    down_grid = (hidden_size + 15) // 16
    up_body = f"""
        using G = GEMVKernel<bfloat16_t, 4, 1, 1, 32, 4, 4, false>;
        const device bfloat16_t* matrix = gate0;
        uint expert = threadgroup_position_in_grid.z;
        if (expert == 1) matrix = up0;
        else if (expert == 2) matrix = gate1;
        else if (expert == 3) matrix = up1;
        else if (expert == 4) matrix = gate2;
        else if (expert == 5) matrix = up2;
        else if (expert == 6) matrix = gate3;
        else if (expert == 7) matrix = up3;
        device bfloat16_t* dst = output + expert * {intermediate_size};
        G::run(matrix, x, matrix, dst, {hidden_size}, {intermediate_size},
            {hidden_size}, 1.0f, 0.0f, 0, nullptr,
            threadgroup_position_in_grid, thread_position_in_threadgroup,
            simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
    """
    down_body = f"""
        using G = GEMVKernel<bfloat16_t, 4, 1, 1, 32, 4, 4, false>;
        uint expert = threadgroup_position_in_grid.z;
        const device bfloat16_t* matrix = down0;
        if (expert == 1) matrix = down1;
        else if (expert == 2) matrix = down2;
        else if (expert == 3) matrix = down3;
        const device bfloat16_t* vector = hidden + expert * {intermediate_size};
        device bfloat16_t* dst = output + expert * {hidden_size};
        G::run(matrix, vector, matrix, dst, {intermediate_size}, {hidden_size},
            {intermediate_size}, 1.0f, 0.0f, 0, nullptr,
            threadgroup_position_in_grid, thread_position_in_threadgroup,
            simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
    """
    gate_up = mx.fast.metal_kernel(
        name=f"freetoken_gate_up_4_{hidden_size}_{intermediate_size}",
        input_names=[
            "x", "gate0", "up0", "gate1", "up1",
            "gate2", "up2", "gate3", "up3",
        ],
        output_names=["output"], source=up_body, header=header,
    )
    down = mx.fast.metal_kernel(
        name=f"freetoken_down_4_{hidden_size}_{intermediate_size}",
        input_names=["hidden", "down0", "down1", "down2", "down3"],
        output_names=["output"], source=down_body, header=header,
    )

    def grouped(x, experts):
        if len(experts) != 4 or any(
            any(set(projection) != {"weight"} for projection in expert)
            for expert in experts
        ):
            raise ValueError("grouped GEMV requires four BF16 weight-only experts")
        inputs = [x]
        for up, gate, _ in experts:
            inputs.extend((gate["weight"], up["weight"]))
        projected = gate_up(
            inputs=inputs, grid=(up_grid * 32, 1, 8 * 4),
            threadgroup=(32, 1, 4),
            output_shapes=[(4, 2, intermediate_size)],
            output_dtypes=[mx.bfloat16],
        )[0]
        hidden = swiglu(projected[:, 0, :], projected[:, 1, :])
        output = down(
            inputs=[hidden, *(expert[2]["weight"] for expert in experts)],
            grid=(down_grid * 32, 1, 4 * 4), threadgroup=(32, 1, 4),
            output_shapes=[(4, hidden_size)], output_dtypes=[mx.bfloat16],
        )[0]
        return output.reshape(1, 1, 4, hidden_size)

    return grouped


def _make_cached_switch(
    mx, nn, swiglu, cache: MLXOffloadMoeCache, layer_id: int, memory_limit: int,
    eval_interval: int = 1,
    lazy_materialize: bool = False,
    timing: dict[str, int] | None = None,
    grouped_gemv=None,
):
    class FreeTokenMLXSwitchGLU(nn.Module):
        def __init__(self):
            super().__init__()
            self._layer_id = layer_id

        def __call__(self, x, indices, overlap=None):
            # Routing tensors are tiny (batch=1, top-k); this is the explicit
            # synchronization point before FreeToken admits selected experts.
            started = time.perf_counter_ns()
            routed = [int(v) for v in indices.reshape(-1).tolist()]
            if timing is not None:
                timing["route_sync_ns"] += time.perf_counter_ns() - started
                timing["layers"] += 1
            if overlap is not None:
                # The router dependency is now resolved. Submit the independent
                # shared expert before CPU-side cache admission so Metal compute
                # can overlap safetensors lookup/materialization on a miss.
                mx.async_eval(overlap)
            expert_ids = list(dict.fromkeys(routed))
            if (
                grouped_gemv is not None
                and x.ndim == 3
                and x.shape[:2] == (1, 1)
                and len(routed) == len(expert_ids) == 4
            ):
                started = time.perf_counter_ns()
                experts = [
                    cache.ensure_expert(self._layer_id, expert_id)
                    for expert_id in routed
                ]
                if timing is not None:
                    timing["ensure_ns"] += time.perf_counter_ns() - started
                started = time.perf_counter_ns()
                output = grouped_gemv(x, experts)
                if timing is not None:
                    timing["graph_build_ns"] += time.perf_counter_ns() - started
                expert_ids = []
            else:
                output = None
            for expert_id in expert_ids:
                started = time.perf_counter_ns()
                up, gate, down = cache.ensure_expert(self._layer_id, expert_id)
                if timing is not None:
                    timing["ensure_ns"] += time.perf_counter_ns() - started
                if lazy_materialize and x.shape[-2] > 1:
                    # Long prefill can route many experts per layer. Keep its
                    # weight lifetime eager; delayed materialization is bounded
                    # to batch=1 single-token decode graphs.
                    mx.eval(*(
                        value for projection in (up, gate, down)
                        for value in projection.values()
                        if isinstance(value, mx.array)
                    ))
                started = time.perf_counter_ns()
                expert_y = swiglu(_linear(mx, x, gate), _linear(mx, x, up))
                expert_y = _linear(mx, expert_y, down)
                mask = (indices == expert_id)[..., None]
                routed_y = mx.where(mask, expert_y[..., None, :], 0)
                output = routed_y if output is None else output + routed_y
                if timing is not None:
                    timing["graph_build_ns"] += time.perf_counter_ns() - started
            if output is None:
                raise RuntimeError("Qwen router selected no experts")
            # Bound the lazy graph to one MoE layer. Without this synchronization,
            # evicted expert weights remain referenced by the full 24-layer graph
            # until logits are evaluated and a 16 GiB machine eventually OOMs.
            if (self._layer_id + 1) % eval_interval == 0 or \
                    self._layer_id + 1 == cache.num_layers:
                started = time.perf_counter_ns()
                mx.eval(output)
                if timing is not None:
                    timing["explicit_eval_ns"] += time.perf_counter_ns() - started
                used = mx.get_active_memory() + mx.get_cache_memory()
                if used > int(memory_limit * 0.85):
                    # Preserve at least one decode top-k working set. Shrinking is
                    # monotonic for this run so repeated pressure cannot thrash.
                    cache.resize(max(1, cache.cache_size // 2))
                    mx.clear_cache()
            return output

    return FreeTokenMLXSwitchGLU()


def _make_overlapped_moe_block(mx, nn, original, switch):
    """Qwen MoE block with shared-expert/cache-materialization overlap."""

    class FreeTokenMLXOverlappedMoeBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_experts = original.num_experts
            self.top_k = original.top_k
            self.gate = original.gate
            self.switch_mlp = switch
            self.shared_expert = original.shared_expert
            self.shared_expert_gate = original.shared_expert_gate

        def __call__(self, x):
            gates = mx.softmax(self.gate(x), axis=-1, precise=True)
            indices = mx.stop_gradient(mx.argpartition(
                -gates, kth=self.top_k - 1, axis=-1,
            )[..., :self.top_k])
            scores = mx.take_along_axis(gates, indices, axis=-1)

            shared = self.shared_expert(x)
            shared = mx.sigmoid(self.shared_expert_gate(x)) * shared
            output = self.switch_mlp(x, indices, overlap=shared)
            output = (output * scores[..., None]).sum(axis=-2)
            return output + shared

    return FreeTokenMLXOverlappedMoeBlock()


def load_model_with_freetoken_cache(
    model_path: str,
    cache_size: int,
    *,
    verbose: bool = True,
    memory_limit_gb: float | None = None,
    allocator_cache_mb: int = 256,
    shard_cache_size: int | None = None,
    eval_interval: int = 1,
    profile: str = "stable",
    detailed_timing: bool = False,
    grouped_gemv: bool = False,
    async_overlap: bool = False,
    cache_policy: str = "auto",
    expert_cache_budget_bytes: int | None = None,
):
    (mx, nn, stream_generate, swiglu, load_tokenizer, load_config,
     Model, ModelArgs) = _mlx_imports()
    model_path = Path(model_path)
    device_info = mx.device_info()
    recommended = int(device_info["max_recommended_working_set_size"])
    total_memory = int(device_info["memory_size"])
    memory_limit = (
        int(memory_limit_gb * (1 << 30))
        if memory_limit_gb is not None
        else int(recommended * 0.90)
    )
    if memory_limit < (2 << 30) or memory_limit >= total_memory:
        raise ValueError("MLX memory limit must be at least 2 GiB and below physical memory")
    if allocator_cache_mb < 0:
        raise ValueError("MLX allocator cache limit cannot be negative")
    if shard_cache_size is not None and shard_cache_size < 0:
        raise ValueError("shard cache size cannot be negative")
    if eval_interval < 1:
        raise ValueError("eval interval must be at least 1")
    mx.set_memory_limit(memory_limit)
    mx.set_cache_limit(allocator_cache_mb << 20)
    mx.reset_peak_memory()
    config = load_config(model_path)
    if config.get("model_type") != SUPPORTED_MODEL_TYPE:
        raise ValueError(
            f"MLX backend supports exactly model_type={SUPPORTED_MODEL_TYPE!r}; "
            f"got {config.get('model_type')!r}"
        )
    model = Model(ModelArgs.from_dict(config))
    expert_quantization = _expert_quantization(config)
    cache_policy = _resolve_cache_policy(
        cache_policy, profile=profile,
        quantized_experts=expert_quantization is not None,
    )
    global_quantization = config.get("quantization")
    dense_quantization = (
        _expert_quantization({"quantization": global_quantization})
        if global_quantization is not None else None
    )
    num_experts = int(config.get("num_experts", config.get("num_local_experts", 0)))
    num_layers = int(config["num_hidden_layers"])
    hidden_size = int(config["hidden_size"])
    intermediate_size = int(config["moe_intermediate_size"])

    # Keep raw per-expert tensors independent. mlx-lm's normal sanitize() stacks
    # all 60 experts per layer; evaluating one slice can then materialize that
    # whole stack and defeats offload on a 16 GiB Mac.
    rows: list[list[dict[str, dict[str, Any]]]] = [
        [dict() for _ in range(num_experts)] for _ in range(num_layers)
    ]
    dense_weights = {}
    weight_files = glob.glob(str(model_path / "model*.safetensors"))
    for weight_file in weight_files:
        for name, value in mx.load(weight_file).items():
            match = EXPERT_WEIGHT_RE.match(name)
            if match is None:
                dense_weights[name] = value
                continue
            layer_id, expert_id = int(match.group(1)), int(match.group(2))
            projection, field = match.group(3), match.group(4)
            expected_shapes = _expected_projection_shapes(
                hidden_size, intermediate_size, projection,
                expert_quantization,
            )
            expected_shape = expected_shapes.get(field)
            if expected_shape is None or tuple(value.shape) != expected_shape:
                raise ValueError(
                    "expert tensor shape does not match quantization metadata: "
                    f"{name} expected={expected_shape} actual={tuple(value.shape)}"
                )
            if expert_quantization is None:
                if value.dtype != mx.bfloat16:
                    raise ValueError(f"BF16 expert tensor has wrong dtype: {name}")
            elif field == "weight" and value.dtype != mx.uint32:
                raise ValueError(f"packed expert weight has wrong dtype: {name}")
            elif field != "weight" and value.dtype not in {
                mx.bfloat16, mx.float16, mx.float32,
            }:
                raise ValueError(f"expert quantization metadata has wrong dtype: {name}")
            rows[layer_id][expert_id].setdefault(projection, {})[field] = (
                weight_file, name
            )
    if global_quantization is not None:
        # MLX checkpoints can leave selected modules (notably embeddings)
        # unquantized. Mirror mlx-lm's key-driven predicate instead of applying
        # the global config indiscriminately.
        nn.quantize(
            model,
            group_size=dense_quantization["group_size"],
            bits=dense_quantization["bits"],
            mode=dense_quantization["mode"],
            class_predicate=lambda path, module: f"{path}.scales" in dense_weights,
        )

    sources = []
    for layer_id, layer_rows in enumerate(rows):
        packed_rows = []
        for expert_id, row in enumerate(layer_rows):
            missing = {"up_proj", "gate_proj", "down_proj"} - row.keys()
            if missing:
                raise ValueError(
                    f"expert bank missing layer={layer_id} expert={expert_id}: {sorted(missing)}"
                )
            projections = []
            for name in ("up_proj", "gate_proj", "down_proj"):
                projection = dict(row[name])
                is_quantized = "scales" in projection or "biases" in projection
                if is_quantized != (expert_quantization is not None):
                    raise ValueError(
                        f"inconsistent MLX expert quantization at layer={layer_id} "
                        f"expert={expert_id} projection={name}"
                    )
                if is_quantized:
                    if not {"weight", "scales", "biases"} <= projection.keys():
                        raise ValueError(
                            f"quantized expert projection is incomplete: layer={layer_id} "
                            f"expert={expert_id} projection={name}"
                        )
                    projection.update(_projection_quantization(
                        expert_quantization, name,
                    ))
                projections.append(projection)
            packed_rows.append(tuple(projections))
        sources.append(_LayerExpertSource(tuple(packed_rows)))

    bank_format = "bf16"
    if expert_quantization is not None:
        bank_format = (
            "mlx_affine_mixed"
            if expert_quantization.get("projections")
            else f"mlx_affine{expert_quantization['bits']}"
        )
    banks = ExpertBanks(bank_format, {"experts": tuple(sources)})
    log = (lambda message: print(message, file=sys.stderr, flush=True)) if verbose else None
    expert_bytes = _expert_storage_bytes(
        hidden_size, intermediate_size, expert_quantization,
    )
    cache_fraction = _expert_cache_fraction(
        profile=profile, quantized_experts=expert_quantization is not None,
    )
    cache_budget = (
        expert_cache_budget_bytes
        if expert_cache_budget_bytes is not None
        else int(memory_limit * cache_fraction)
    )
    if cache_budget < 0:
        raise ValueError("expert cache budget cannot be negative")
    safe_slots = min(
        num_layers * num_experts,
        max(1, cache_budget // expert_bytes),
    )
    if profile == "performance":
        cache_size = safe_slots
        eval_interval = 8
    if cache_size > safe_slots:
        if log is not None:
            log(
                f"expert_cache event=clamp requested={cache_size} effective={safe_slots} "
                f"reason=memory_budget"
            )
        cache_size = safe_slots
    timing = ({name: 0 for name in (
        "shard_open_ns", "route_sync_ns", "ensure_ns", "graph_build_ns",
        "explicit_eval_ns", "layers",
    )} if detailed_timing else None)
    # Quantized checkpoints have three tensors per projection and parsing their
    # large shard mappings repeatedly dominates decode. Retain all mappings (up
    # to eight) while consumed expert arrays are still removed individually, so
    # LRU eviction continues to release evaluated weights.
    shard_cache_size = _resolve_shard_cache_size(
        shard_cache_size,
        quantized_experts=expert_quantization is not None,
        num_shards=len(weight_files),
    )
    shard_pool = _ShardPool(mx, shard_cache_size, timing)
    cache = MLXOffloadMoeCache(
        banks, num_experts, cache_size,
        _make_materializer(mx, shard_pool, eager=profile != "performance"),
        eviction_policy=cache_policy,
        log=log,
    )
    if grouped_gemv and expert_quantization is not None and log is not None:
        log("expert_compute event=fallback backend=quantized_matmul reason=packed_weights")
    grouped_expert = (
        _make_grouped_gemv(
            mx, swiglu, int(config["hidden_size"]),
            int(config["moe_intermediate_size"]),
        ) if grouped_gemv and expert_quantization is None else None
    )
    for layer_id, layer in enumerate(model.layers):
        if getattr(getattr(layer, "mlp", None), "switch_mlp", None) is None:
            raise ValueError(f"Qwen2-MoE layer {layer_id} has no switch_mlp expert bank")
        # Detach the stacked expert modules from the model parameter tree. Dense
        # weights are evaluated normally; expert rows are evaluated only on miss.
        switch = _make_cached_switch(
            mx, nn, swiglu, cache, layer_id, memory_limit, eval_interval,
            lazy_materialize=profile == "performance",
            timing=timing,
            grouped_gemv=grouped_expert,
        )
        if async_overlap:
            layer.mlp = _make_overlapped_moe_block(mx, nn, layer.mlp, switch)
        else:
            layer.mlp.switch_mlp = switch
    model.load_weights(list(dense_weights.items()), strict=False)
    model.eval()
    mx.eval(model.parameters())
    tokenizer = load_tokenizer(model_path, eos_token_ids=config.get("eos_token_id"))
    backend_info = {
        "grouped_gemv": grouped_expert is not None,
        "cache_policy": cache_policy,
        "expert_quantization": expert_quantization,
        "expert_bytes": expert_bytes,
    }
    return (
        mx, stream_generate, model, tokenizer, cache, memory_limit, shard_pool,
        backend_info,
    )


def _resolve_model_path(model: str) -> Path:
    path = Path(model).expanduser()
    if path.exists():
        return path.resolve()
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(
        model,
        allow_patterns=[
            "*.json", "*.jinja", "*.model", "*.py", "*.safetensors",
            "*.tiktoken", "*.txt",
        ],
    ))


def _memory_pressure_available(total_memory: int) -> int | None:
    """Return macOS's reclaimable-memory estimate without adding a dependency."""
    try:
        result = subprocess.run(
            ["memory_pressure", "-Q"], check=True, capture_output=True,
            text=True, timeout=2,
        )
        match = re.search(r"free percentage:\s*(\d+)%", result.stdout)
        if match is not None:
            return total_memory * int(match.group(1)) // 100
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _checkpoint_storage(model_path: Path) -> tuple[int, int]:
    """Return checkpoint and non-expert payload bytes from safetensors headers."""
    files = list(model_path.glob("model*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no model safetensors found in {model_path}")
    checkpoint_bytes = sum(path.stat().st_size for path in files)
    dense_bytes = 0
    for path in files:
        with path.open("rb") as stream:
            raw_length = stream.read(8)
            if len(raw_length) != 8:
                raise ValueError(f"invalid safetensors header in {path}")
            header_length = int.from_bytes(raw_length, "little")
            if header_length <= 0 or header_length > path.stat().st_size - 8:
                raise ValueError(f"invalid safetensors header length in {path}")
            header = json.loads(stream.read(header_length))
        for name, metadata in header.items():
            if name == "__metadata__" or EXPERT_WEIGHT_RE.match(name):
                continue
            start, end = metadata["data_offsets"]
            dense_bytes += int(end) - int(start)
    return checkpoint_bytes, dense_bytes


def _select_residency(args, model_path: Path, draft_path: Path | None, mx):
    config = json.loads((model_path / "config.json").read_text())
    if config.get("model_type") != SUPPORTED_MODEL_TYPE:
        raise ValueError(
            f"MLX backend supports exactly model_type={SUPPORTED_MODEL_TYPE!r}; "
            f"got {config.get('model_type')!r}"
        )
    device_info = mx.device_info()
    recommended = int(device_info["max_recommended_working_set_size"])
    total_memory = int(device_info["memory_size"])
    memory_limit = (
        int(args.memory_limit_gb * (1 << 30))
        if args.memory_limit_gb is not None else int(recommended * 0.90)
    )
    if memory_limit < (2 << 30) or memory_limit >= total_memory:
        raise ValueError("MLX memory limit must be at least 2 GiB and below physical memory")
    headroom = int(args.system_headroom_gb * (1 << 30))
    if headroom < 0 or headroom >= total_memory:
        raise ValueError("system headroom must be non-negative and below physical memory")
    weight_bytes, dense_bytes = _checkpoint_storage(model_path)
    draft_bytes = (
        sum(path.stat().st_size for path in draft_path.glob("model*.safetensors"))
        if draft_path is not None else 0
    )
    if draft_path is not None and draft_bytes == 0:
        raise FileNotFoundError(f"no draft-model safetensors found in {draft_path}")
    # Measured resident MLX peaks track checkpoint bytes closely. Add 5% for
    # packing/alignment plus 512 MiB for prompt/KV/runtime allocations.
    estimated_peak = int((weight_bytes + draft_bytes) * 1.05) + (512 << 20)
    pressure_available = _memory_pressure_available(total_memory)
    budgets = {
        "mlx_working_set": memory_limit,
        "physical_headroom": total_memory - headroom,
    }
    if pressure_available is not None:
        budgets["current_pressure"] = max(0, pressure_available - headroom)

    mixed_experts = config.get("freetoken_expert_quantization") is not None
    if args.residency == "resident" and mixed_experts:
        raise ValueError(
            "mixed expert checkpoints require --residency offload or auto"
        )
    if args.residency != "auto":
        selected = args.residency
        reason = "forced_by_user"
    elif mixed_experts:
        selected = "offload"
        reason = "mixed_expert_checkpoint_requires_freetoken"
    else:
        failed = next(
            (name for name, budget in budgets.items() if estimated_peak > budget),
            None,
        )
        selected = "offload" if failed is not None else "resident"
        reason = f"estimated_peak_exceeds_{failed}" if failed else "resident_is_safe"

    # A speculative draft remains fully resident even while the target experts
    # are offloaded. Reserve its checkpoint plus packing/runtime overhead from
    # both the process budget and the expert-cache allowance. This prevents the
    # target cache from filling memory that the draft needs later in generation.
    offload_total_budget = min(budgets.values())
    draft_reserve = (
        int(draft_bytes * 1.10) + (256 << 20) if draft_bytes else 0
    )
    # Dense/shared tensors remain resident on the FreeToken path. Safetensors
    # headers let us account for them precisely without mapping model payloads.
    offload_base_reserve = int(dense_bytes * 1.10) + (512 << 20)
    offload_model_budget = max(0, offload_total_budget - draft_reserve)
    quantized_experts = (
        mixed_experts or config.get("quantization") is not None
    )
    cache_fraction = _expert_cache_fraction(
        profile=getattr(args, "profile", "stable"),
        quantized_experts=quantized_experts,
    )
    expert_cache_budget = max(
        0,
        min(
            int(offload_total_budget * cache_fraction),
            offload_total_budget - offload_base_reserve - draft_reserve,
        ),
    )
    one_expert_bytes = _expert_storage_bytes(
        int(config["hidden_size"]), int(config["moe_intermediate_size"]),
        _expert_quantization(config),
    )
    if selected == "offload" and expert_cache_budget < one_expert_bytes:
        raise RuntimeError(
            "insufficient safe memory for FreeToken offload after reserving "
            "dense weights, runtime, system headroom, and the speculative draft; "
            "close memory-heavy applications, remove --draft-model, or lower "
            "--system-headroom-gb"
        )
    return {
        "requested": args.residency,
        "selected": selected,
        "reason": reason,
        "checkpoint_bytes": weight_bytes,
        "dense_checkpoint_bytes": dense_bytes,
        "draft_checkpoint_bytes": draft_bytes,
        "estimated_resident_peak_bytes": estimated_peak,
        "memory_limit_bytes": memory_limit,
        "system_headroom_bytes": headroom,
        "pressure_available_bytes": pressure_available,
        "resident_budgets": budgets,
        "offload_total_budget_bytes": offload_total_budget,
        "offload_draft_reserve_bytes": draft_reserve,
        "offload_base_reserve_bytes": offload_base_reserve,
        "offload_model_budget_bytes": offload_model_budget,
        "offload_expert_cache_budget_bytes": expert_cache_budget,
    }


def _generate(mx, stream_generate, model, tokenizer, draft_model, args):
    prompt = args.prompt
    if not args.raw_prompt and getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
    started = time.perf_counter()
    generated = 0
    accepted_draft_tokens = 0
    generation_options = (
        {"draft_model": draft_model, "num_draft_tokens": args.num_draft_tokens}
        if draft_model is not None else {}
    )
    for response in stream_generate(
        model, tokenizer, prompt, max_tokens=args.max_tokens,
        **generation_options,
    ):
        print(response.text, end="", flush=True)
        generated = response.generation_tokens
        accepted_draft_tokens += int(response.from_draft)
    mx.synchronize()
    print()
    return {
        "generated_tokens": generated,
        "elapsed_seconds": time.perf_counter() - started,
        "speculative": {
            "enabled": draft_model is not None,
            "draft_model": args.draft_model,
            "num_draft_tokens": args.num_draft_tokens if draft_model is not None else 0,
            "accepted_tokens": accepted_draft_tokens,
            "draft_output_fraction": (
                accepted_draft_tokens / generated if generated else 0.0
            ),
        },
    }


def _run_resident(args, model_path: Path, draft_path: Path | None, decision, mx,
                  stream_generate) -> int:
    from mlx_lm.utils import load

    mx.set_memory_limit(decision["memory_limit_bytes"])
    mx.set_cache_limit(args.allocator_cache_mb << 20)
    mx.reset_peak_memory()
    model, tokenizer = load(str(model_path))
    draft_model = (
        _load_draft_model(str(draft_path), tokenizer)
        if draft_path is not None else None
    )
    generation = _generate(
        mx, stream_generate, model, tokenizer, draft_model, args,
    )
    config = json.loads((model_path / "config.json").read_text())
    report = {
        "backend": "mlx",
        "model_type": SUPPORTED_MODEL_TYPE,
        "profile": args.profile,
        "residency": decision,
        "expert_quantization": config.get("quantization"),
        "expert_cache": None,
        "shard_cache": None,
        "grouped_gemv": "mlx_native",
        "async_overlap": False,
        "memory": {
            "limit_bytes": decision["memory_limit_bytes"],
            "active_bytes": mx.get_active_memory(),
            "allocator_cache_bytes": mx.get_cache_memory(),
            "peak_bytes": mx.get_peak_memory(),
        },
    } | generation
    print(json.dumps(report, sort_keys=True))
    return 0


def run(args: argparse.Namespace) -> int:
    if args.batch_size != 1:
        raise ValueError("the first MLX backend supports only --batch-size 1")
    (mx, _, stream_generate, _, _, _, _, _) = _mlx_imports()
    model_path = _resolve_model_path(args.model)
    draft_path = (
        _resolve_model_path(args.draft_model)
        if args.draft_model is not None else None
    )
    decision = _select_residency(args, model_path, draft_path, mx)
    print(
        f"mlx_residency selected={decision['selected']} reason={decision['reason']} "
        f"estimated_peak={decision['estimated_resident_peak_bytes']}",
        file=sys.stderr, flush=True,
    )
    if decision["selected"] == "resident":
        return _run_resident(
            args, model_path, draft_path, decision, mx, stream_generate,
        )
    grouped_gemv = (
        args.grouped_gemv
        if args.grouped_gemv is not None
        else args.profile == "performance"
    )
    async_overlap = (
        args.async_overlap
        if args.async_overlap is not None
        else args.profile == "performance"
    )
    expert_cache_budget = decision["offload_expert_cache_budget_bytes"]
    if args.expert_cache_budget_gb is not None:
        requested_budget = int(args.expert_cache_budget_gb * (1 << 30))
        decision["offload_expert_cache_policy_budget_bytes"] = expert_cache_budget
        decision["offload_expert_cache_requested_bytes"] = requested_budget
        expert_cache_budget = min(expert_cache_budget, requested_budget)
        decision["offload_expert_cache_budget_bytes"] = expert_cache_budget
    (mx, stream_generate, model, tokenizer, cache,
     memory_limit, shard_pool, backend_info) = load_model_with_freetoken_cache(
        str(model_path),
        args.moe_cache_size,
        verbose=not args.quiet_cache,
        memory_limit_gb=args.memory_limit_gb,
        allocator_cache_mb=args.allocator_cache_mb,
        shard_cache_size=args.shard_cache_size,
        eval_interval=args.eval_interval,
        profile=args.profile,
        detailed_timing=args.detailed_timing,
        grouped_gemv=grouped_gemv,
        async_overlap=async_overlap,
        cache_policy=args.cache_policy,
        expert_cache_budget_bytes=expert_cache_budget,
    )
    draft_model = None
    if draft_path is not None:
        draft_model = _load_draft_model(str(draft_path), tokenizer)
    generation = _generate(
        mx, stream_generate, model, tokenizer, draft_model, args,
    )
    report = {
        "backend": "mlx",
        "model_type": SUPPORTED_MODEL_TYPE,
        "profile": args.profile,
        "residency": decision,
        "grouped_gemv": backend_info["grouped_gemv"],
        "async_overlap": async_overlap,
        "cache_policy": backend_info["cache_policy"],
        "expert_quantization": backend_info["expert_quantization"],
        "expert_cache": cache.stats(),
        "memory": {
            "limit_bytes": memory_limit,
            "active_bytes": mx.get_active_memory(),
            "allocator_cache_bytes": mx.get_cache_memory(),
            "peak_bytes": mx.get_peak_memory(),
        },
        "shard_cache": {"capacity": shard_pool.capacity, "hits": shard_pool.hits,
                        "opens": shard_pool.opens},
    } | generation
    if shard_pool.timing is not None:
        report["timing"] = {
            key.removesuffix("_ns") + "_ms": round(value / 1_000_000, 3)
            for key, value in shard_pool.timing.items() if key.endswith("_ns")
        } | {"layers": shard_pool.timing["layers"]}
    print(json.dumps(report, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ft generate")
    parser.add_argument("--backend", choices=("mlx",), required=True)
    parser.add_argument("--model", required=True, help="HF model id or local snapshot path")
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--residency", choices=("auto", "resident", "offload"), default="auto",
        help="auto uses native MLX only when full residency is safe (default: auto)",
    )
    parser.add_argument(
        "--system-headroom-gb", type=float, default=2.0,
        help="memory reserved for macOS and other processes in auto mode (default: 2)",
    )
    parser.add_argument(
        "--profile", choices=("stable", "performance"), default="stable",
        help="stable minimizes memory; performance uses the measured safe cache/eval preset",
    )
    parser.add_argument("--moe-cache-size", type=int, default=4)
    parser.add_argument(
        "--expert-cache-budget-gb", type=float,
        help=(
            "cap expert-cache bytes for controlled Pareto measurements; "
            "the runtime still clamps this to its safe memory budget"
        ),
    )
    parser.add_argument(
        "--memory-limit-gb", type=float,
        help="MLX working-set limit; default is 90%% of the device recommendation",
    )
    parser.add_argument(
        "--allocator-cache-mb", type=int, default=256,
        help="maximum MLX free-buffer cache (default: 256 MiB)",
    )
    parser.add_argument(
        "--shard-cache-size", type=int,
        help="parsed shard mappings; auto keeps quantized shards, one for BF16",
    )
    parser.add_argument(
        "--eval-interval", type=int, default=1,
        help="experimental number of MoE layers per MLX graph evaluation",
    )
    parser.add_argument("--quiet-cache", action="store_true")
    parser.add_argument(
        "--detailed-timing", action="store_true",
        help="report lightweight MLX expert-path timings without adding synchronization",
    )
    parser.add_argument(
        "--grouped-gemv", action=argparse.BooleanOptionalAction, default=None,
        help="MLX-template top-4 BF16 grouped GEMV (default: performance profile)",
    )
    parser.add_argument(
        "--async-overlap", action=argparse.BooleanOptionalAction, default=None,
        help=(
            "overlap shared expert compute and cache misses "
            "(default: performance profile)"
        ),
    )
    parser.add_argument(
        "--cache-policy", choices=("auto", "lru", "slru"), default="auto",
        help="expert eviction policy (default: auto)",
    )
    parser.add_argument(
        "--raw-prompt", action="store_true",
        help="do not apply the tokenizer chat template (lowest-memory smoke test)",
    )
    parser.add_argument(
        "--draft-model",
        help="compatible small MLX model used for speculative decoding",
    )
    parser.add_argument(
        "--num-draft-tokens", type=int, default=2,
        help="speculative tokens proposed per target verification step (default: 2)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be at least 1")
    if args.moe_cache_size < 1:
        raise ValueError("--moe-cache-size must be at least 1")
    if args.expert_cache_budget_gb is not None and args.expert_cache_budget_gb <= 0:
        raise ValueError("--expert-cache-budget-gb must be positive")
    if args.num_draft_tokens < 1:
        raise ValueError("--num-draft-tokens must be at least 1")
    try:
        return run(args)
    except RuntimeError as exc:
        message = str(exc)
        if "memory" not in message.lower() and "metal" not in message.lower():
            raise
        print(
            "FreeToken MLX stopped before exhausting system memory. "
            "Retry with --residency offload, a shorter/raw prompt, a smaller "
            "--moe-cache-size, or a lower --memory-limit-gb; if enabled, omit "
            "--draft-model to release its memory.\n"
            f"MLX error: {message}",
            file=sys.stderr,
        )
        return 1


__all__ = ["load_model_with_freetoken_cache", "main"]
