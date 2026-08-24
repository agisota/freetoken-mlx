# FreeToken-MLX

FreeToken expert offload for Apple Silicon.

This repository is an independently maintained fork of [FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken). The upstream CUDA/Torch runtime remains in the tree. The Apple Silicon implementation is a separate MLX backend under `python/freetoken/mlx_backend.py`.

## What it does

The MLX backend targets one concrete problem: running `Qwen/Qwen1.5-MoE-A2.7B` on Macs where the full checkpoint does not fit safely in unified memory.

It keeps dense/shared weights resident and loads routed experts on demand through a bounded cross-layer cache. If the whole checkpoint is safe to keep resident, it uses native `mlx-lm` instead of forcing offload.

Current MLX scope:

- macOS on Apple Silicon;
- `model_type=qwen2_moe`;
- tested with `Qwen/Qwen1.5-MoE-A2.7B`;
- batch size 1;
- local text generation;
- BF16 experts, fully quantized MLX checkpoints, and mixed BF16-dense / quantized-expert checkpoints;
- optional speculative decoding;
- LRU or segmented-LRU expert eviction;
- bounded MLX working-set and allocator-cache limits;
- JSON metrics for residency, memory, cache activity, timing, and speculative acceptance.

This is **not** a complete MLX port of the upstream FreeToken engine. CUDA attention, distributed serving, scheduler/server code, and most upstream model support remain CUDA/Torch code.

## Why offload exists

Native `mlx-lm` is preferable when the model fits. Offload adds synchronization, cache lookup, safetensors reads, and expert materialization.

The useful case is capacity: a model that would otherwise hit Metal OOM can remain runnable by keeping only routed experts resident.

The runtime therefore has three decisions:

1. **resident** — use native `mlx-lm` when estimated peak memory fits the MLX working set, system headroom, and current macOS memory pressure;
2. **offload** — use the FreeToken expert cache when full residency is unsafe or when the checkpoint stores experts separately from dense weights;
3. **fail early** — refuse to start when the requested safety headroom cannot accommodate even the offload path.

## Measured results

These numbers are from Apple M4 / 16 GB unified memory, batch size 1. They are measurements for the listed setup, not hardware-independent claims.

| Path | Throughput | Peak MLX memory |
| --- | ---: | ---: |
| native `mlx-lm`, BF16 | Metal OOM before token 1 | reported 28.63 GB |
| FreeToken offload, BF16 | 1.88 tok/s | 5.83 GB |
| FreeToken offload, BF16, larger safe cache | 2.05 tok/s | 9.03 GB |
| BF16 dense + Q4 experts | 4.49 tok/s | 7.93 GB |
| native resident full-Q4 | selected automatically when safe | ~8.8 GB |

A controlled Q4 cache sweep from 55 to 275 experts reduced misses by 31.4% while median throughput moved from 3.02 to 3.16 tok/s. The runs overlapped, so the defensible conclusion is that larger cache reliably trades memory for fewer expert loads; the throughput gain was modest.

The strongest offload improvement came from reducing expert bytes rather than from making one GEMV kernel faster. See [`docs/MLX_PERFORMANCE_TUNING_GUIDE.md`](docs/MLX_PERFORMANCE_TUNING_GUIDE.md).

## Install

```bash
git clone https://github.com/agisota/freetoken-mlx.git
cd freetoken-mlx
uv venv --python 3.12 --seed
uv pip install -e '.[mlx]'
```

The root `install.sh` is the inherited Linux/NVIDIA installer. It is not the installer for the MLX backend.

## Run

Minimal smoke test:

```bash
.venv/bin/ft generate \
  --backend mlx \
  --model Qwen/Qwen1.5-MoE-A2.7B \
  --prompt Hello \
  --raw-prompt \
  --max-tokens 1 \
  --batch-size 1 \
  --moe-cache-size 4
```

Longer decode with the measured performance preset:

```bash
.venv/bin/ft generate \
  --backend mlx \
  --model Qwen/Qwen1.5-MoE-A2.7B \
  --prompt Hello \
  --raw-prompt \
  --max-tokens 32 \
  --profile performance \
  --quiet-cache
```

For reproducible comparisons, pin the memory envelope as well:

```bash
.venv/bin/ft generate \
  --backend mlx \
  --model Qwen/Qwen1.5-MoE-A2.7B \
  --prompt Hello \
  --raw-prompt \
  --max-tokens 32 \
  --profile performance \
  --residency offload \
  --memory-limit-gb 10 \
  --system-headroom-gb 3.2 \
  --expert-cache-budget-gb 4 \
  --quiet-cache \
  --detailed-timing
```

## How the MLX path works

### 1. Residency selection

Before loading weights, the backend reads checkpoint metadata, estimates resident peak memory, reserves system headroom and optional draft-model memory, and checks current macOS memory pressure.

`--residency auto` is the default. Use `resident` or `offload` only for debugging and controlled benchmarks.

### 2. Expert bank

The loader keeps expert tensors as `(shard path, tensor key)` references instead of letting `mlx-lm` stack every expert into one resident tensor. Dense weights are loaded normally.

On a cache miss, only the selected expert's `up_proj`, `gate_proj`, and `down_proj` arrays are materialized.

### 3. Expert cache

`MLXOffloadMoeCache` is shared across layers. It records hits, misses, loads, evictions, resident entries, and capacity.

- `lru` is the conservative policy;
- `slru` protects reused experts from scan-like eviction after warm-up;
- `auto` uses SLRU only for quantized experts in the performance profile.

### 4. Memory control

The backend configures both MLX working-set and allocator-cache limits. During decode it monitors active + allocator-cache memory. If usage crosses 85% of the configured limit, the expert cache is shrunk and the MLX cache is cleared.

Important flags:

| Flag | Purpose |
| --- | --- |
| `--memory-limit-gb` | MLX process working-set ceiling |
| `--system-headroom-gb` | memory reserved for macOS and other processes |
| `--expert-cache-budget-gb` | explicit cap for controlled experiments |
| `--allocator-cache-mb` | MLX free-buffer cache ceiling |
| `--moe-cache-size` | requested expert entries before byte-budget clamping |
| `--cache-policy` | `auto`, `lru`, or `slru` |
| `--residency` | `auto`, `resident`, or `offload` |

### 5. Performance profile

`--profile performance` currently enables the measured fast-path defaults:

- larger safe expert-cache budget;
- graph evaluation every eight MoE layers;
- lazy expert materialization during single-token decode;
- BF16 top-4 grouped GEMV using an MLX Metal kernel template;
- shared-expert / cache-miss overlap;
- SLRU for quantized experts.

Each optimization has a disable switch where useful. Do not treat the preset as universally faster: re-benchmark when model, MLX version, prompt shape, or hardware changes.

## Quantized experts

The mixed checkpoint format keeps dense/attention/router/shared-expert weights at their original precision while quantizing only routed experts.

Supported expert weight format:

- affine 2/3/4/5/6/8-bit;
- group size 32, 64, or 128;
- optional per-projection overrides for `up_proj`, `gate_proj`, and `down_proj`.

Example: Q3 for up/gate and Q4 for down reduced routed-expert storage by about 16.7% in the measured checkpoint. It improved throughput in that experiment because more experts fit in the same cache budget. It is not the default quality setting.

## Speculative decoding

Use a tokenizer-compatible MLX draft model:

```bash
.venv/bin/ft generate \
  --backend mlx \
  --model Qwen/Qwen1.5-MoE-A2.7B \
  --draft-model <compatible-mlx-model> \
  --num-draft-tokens 2 \
  --max-tokens 64
```

The draft remains resident. Its memory is reserved before the target expert-cache budget is calculated. The runtime rejects draft/tokenizer pairs with different vocabularies or special-token IDs.

## Output and observability

The final stdout line is JSON. Depending on the selected path it includes:

- residency decision and reason;
- checkpoint and safety budgets;
- generated tokens and elapsed time;
- peak, active, and allocator-cache memory;
- expert-cache hits/misses/loads/evictions;
- shard-cache opens/hits;
- grouped-GEMV and async-overlap state;
- speculative acceptance;
- optional detailed timing.

Cache event logs go to stderr and can be disabled with `--quiet-cache`.

## Tests

MLX-specific tests cover:

- expert routing and output shape;
- affine quantized matmul;
- quantization metadata and storage accounting;
- LRU/SLRU behavior;
- shard mapping lifetime;
- residency selection and system headroom;
- draft-tokenizer compatibility;
- CLI defaults and memory-error handling.

Run on Apple Silicon with the MLX extra installed:

```bash
.venv/bin/python -m pytest tests/test_mlx_backend.py tests/test_mlx_cache.py tests/test_mlx_cli.py tests/test_mlx_quantize_experts.py -q
```

Tests that use real checkpoints are documented in [`tests/README.md`](tests/README.md).

## Known limitations

- one MLX model family is supported;
- batch size is fixed at 1;
- routing admission crosses a Metal→CPU synchronization boundary because selected expert IDs are converted to a Python list;
- expert misses still depend on safetensors/page-cache behavior;
- the MLX backend does not expose the upstream OpenAI/Anthropic server path;
- no MLX tensor-parallel implementation is wired into this backend;
- no persistent prompt-cache or quantized-KV controls are exposed yet;
- the custom BF16 grouped GEMV depends on MLX's installed Metal kernel template and is therefore version-sensitive.

## Next performance work

The highest-value experiments are now above the individual GEMV-kernel level:

1. remove or amortize the per-layer Metal→CPU routing synchronization;
2. expose MLX prompt-cache, rotating/quantized KV cache, and prefill controls;
3. test `mx.compile()` on stable dense/router/shared-expert subgraphs while avoiding shape-driven recompilation;
4. redesign expert storage for fewer random shard reads and lower page-in amplification;
5. add trace-driven cache policies using real expert-route sequences;
6. evaluate multi-Mac sharding with MLX distributed/JACCL only after single-node I/O is no longer dominant;
7. turn performance claims into machine-readable benchmark gates in CI.

## Project layout

```text
python/freetoken/mlx_backend.py           MLX model loading, routing, offload, generation
python/freetoken/mlx_cache.py             Apple-Silicon expert cache
python/freetoken/mlx_quantize_experts.py  expert-only checkpoint conversion
tests/test_mlx_*.py                       MLX unit/behavior tests
docs/MLX_PERFORMANCE_TUNING_GUIDE.md      benchmark evidence and tuning notes
benchmarks/                               benchmark assets
paper/                                    research write-up and measurements
```

## License

Apache-2.0. Upstream attribution and fork status are preserved in the repository history and license files.
