# FreeToken-MLX

Bounded MoE expert offload for Apple Silicon.

FreeToken-MLX is an independently maintained fork of [FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken). The upstream CUDA/Torch engine remains in the repository. The Apple-Silicon implementation is a separate MLX backend in `python/freetoken/mlx_backend.py`.

## Scope

The backend solves one concrete problem: run `Qwen/Qwen1.5-MoE-A2.7B` on Macs where full residency is unsafe.

It keeps dense and shared weights resident, materializes only routed experts, and bounds their cross-layer cache. When the full checkpoint fits, `--residency auto` selects native `mlx-lm` instead of paying the offload overhead.

Current MLX support:

- macOS on Apple Silicon;
- `model_type=qwen2_moe`, tested with `Qwen/Qwen1.5-MoE-A2.7B`;
- batch size 1 and local text generation;
- BF16, full MLX quantization, and BF16-dense/quantized-expert checkpoints;
- LRU and warm-up SLRU expert eviction;
- optional speculative decoding;
- bounded MLX working-set, allocator-cache, and system-headroom controls;
- machine-readable runtime and A/B comparison reports.

This is not a complete MLX port of FreeToken. CUDA attention, serving, scheduling, distributed execution, and most upstream model support remain CUDA/Torch code.

## Why offload exists

Native `mlx-lm` is the preferred path when the checkpoint fits. Offload adds route synchronization, cache admission, safetensors access, and expert materialization.

Its value is capacity:

1. **resident** — use native MLX when the estimated peak fits all safety budgets;
2. **offload** — keep only routed experts resident when full residency is unsafe or the checkpoint uses FreeToken's expert-only format;
3. **fail early** — refuse to start when dense weights, runtime state, an optional draft, and requested macOS headroom leave too little memory for one expert.

## Measured reference results

Apple M4, 16 GB unified memory, batch size 1:

| Path | Throughput | Peak MLX memory |
| --- | ---: | ---: |
| native `mlx-lm`, BF16 | Metal OOM before token 1 | reported 28.63 GB |
| FreeToken offload, BF16 | 1.88 tok/s | 5.83 GB |
| FreeToken offload, BF16, larger safe cache | 2.05 tok/s | 9.03 GB |
| BF16 dense + Q4 experts | 4.49 tok/s | 7.93 GB |
| native resident full-Q4 | selected automatically when safe | ~8.8 GB |

A controlled Q4 cache sweep from 55 to 275 experts reduced misses by 31.4%; median throughput moved from 3.02 to 3.16 tok/s with overlapping run ranges. The defensible result is fewer loads for more memory, not a large universal speedup.

The strongest offload gain came from reducing expert bytes, not from another isolated GEMV tweak. See [`docs/MLX_PERFORMANCE_TUNING_GUIDE.md`](docs/MLX_PERFORMANCE_TUNING_GUIDE.md).

## Install

```bash
git clone https://github.com/agisota/freetoken-mlx.git
cd freetoken-mlx
uv venv --python 3.12 --seed
uv pip install -e '.[mlx]'
```

The root `install.sh` is the inherited Linux/NVIDIA installer. It does not install the MLX backend.

## Quick start

Minimal offload smoke test:

```bash
.venv/bin/ft generate \
  --backend mlx \
  --model Qwen/Qwen1.5-MoE-A2.7B \
  --prompt Hello \
  --raw-prompt \
  --max-tokens 1 \
  --batch-size 1 \
  --residency offload \
  --moe-cache-size 4
```

Measured performance preset:

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

For controlled measurements, pin the path and memory envelope:

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

`--max-tokens` is an upper bound: generation can stop earlier on EOS. Do not label a run “32-token” unless the final JSON reports `generated_tokens: 32`.

## Expert-only quantization

The converter preserves dense, attention, router, and shared-expert tensors and quantizes only routed experts. It processes one shard at a time instead of loading the complete BF16 checkpoint as one resident model.

Conservative Q4 experts:

```bash
.venv/bin/ft mlx-quantize-experts \
  --model Qwen/Qwen1.5-MoE-A2.7B \
  --output ./Qwen1.5-MoE-A2.7B-BF16-Q4Experts \
  --bits 4 \
  --group-size 64

.venv/bin/ft generate \
  --backend mlx \
  --model ./Qwen1.5-MoE-A2.7B-BF16-Q4Experts \
  --prompt Hello --raw-prompt --max-tokens 32 \
  --profile performance --quiet-cache
```

Experimental Q3 up/gate with Q4 down:

```bash
.venv/bin/ft mlx-quantize-experts \
  --model Qwen/Qwen1.5-MoE-A2.7B \
  --output ./Qwen1.5-MoE-A2.7B-BF16-Q3UpGate-Q4Down \
  --bits 3 \
  --down-bits 4 \
  --group-size 64
```

Quantizer contract:

- affine 2/3/4/5/6/8-bit weights;
- group size 32, 64, or 128;
- `--bits` applies to `up_proj` and `gate_proj`;
- `--down-bits` optionally overrides only `down_proj`.

Q3 up/gate + Q4 down reduced routed-expert storage by about 16.7% in the measured checkpoint. It is an opt-in speed/capacity tradeoff, not the default quality setting. Uniform Q4 remains the conservative default until the deployment workload is evaluated.

## Speculative decoding

Speculation is also opt-in. The draft remains resident and reduces the target expert-cache budget.

The measured beneficial pair used a full-Q4 target and compatible Q4 draft:

```bash
.venv/bin/ft generate \
  --backend mlx \
  --model mlx-community/Qwen1.5-MoE-A2.7B-4bit \
  --draft-model mlx-community/Qwen1.5-0.5B-4bit \
  --num-draft-tokens 2 \
  --prompt Hello \
  --raw-prompt \
  --max-tokens 64 \
  --profile performance \
  --residency offload \
  --quiet-cache
```

In the recorded 64-token A/B, this reduced mean time from 12.77 s to 11.45 s, with 57.8% of output tokens accepted from the draft. An 8-token request was slower. A BF16-target speculative run also regressed: 54.84 s for 64 tokens, cache shrink, and slower completion than ordinary BF16 decode. Re-benchmark the exact target, draft, prompt length, and memory envelope before enabling speculation.

The runtime rejects draft/tokenizer pairs with different vocabularies or special-token IDs.

## Runtime controls

| Flag | Purpose |
| --- | --- |
| `--residency` | `auto`, `resident`, or `offload` |
| `--memory-limit-gb` | MLX working-set ceiling |
| `--system-headroom-gb` | memory reserved for macOS and other processes |
| `--expert-cache-budget-gb` | explicit expert-cache byte cap for experiments |
| `--allocator-cache-mb` | MLX free-buffer cache ceiling |
| `--moe-cache-size` | requested expert entries before byte-budget clamping |
| `--cache-policy` | `auto`, `lru`, or `slru` |
| `--profile` | `stable` or measured `performance` preset |

`--profile performance` currently forces the safe cache size and graph evaluation every eight MoE layers. It therefore overrides an explicit `--eval-interval`. To isolate evaluation cadence, use `--profile stable`, set `--eval-interval` explicitly, and opt into any other fast paths you need with their individual flags.

The performance preset also enables lazy single-token expert materialization, BF16 top-4 grouped GEMV, shared-expert/cache-miss overlap, and SLRU for quantized experts. These are hardware- and version-sensitive measurements, not universal defaults.

## Benchmark gates

Capture stdout and stderr separately. The final stdout line is the runtime JSON; the preceding bytes are generated output.

```bash
.venv/bin/ft mlx-compare \
  --baseline runs/A/*.stdout \
  --candidate runs/B/*.stdout \
  --expected-tokens 128 \
  --min-throughput-ratio 0.97 \
  --max-peak-memory-ratio 1.05 \
  --max-cache-miss-ratio 1.05 \
  --require-output-match \
  --require-same-residency \
  --write-summary runs/comparison.json
```

The comparator rejects EOS-shortened runs, hashes generated output, compares median wall throughput and peak memory, optionally gates cache misses, and returns stable process exit codes. Full Apple-Silicon procedures are in [`tests/MLX.md`](tests/MLX.md).

## Architecture

1. The residency selector reads checkpoint metadata and evaluates the MLX working set, physical headroom, and current macOS memory pressure.
2. The loader stores expert tensors as `(shard path, tensor key)` references; dense weights load normally.
3. On a miss, only one routed expert's `up_proj`, `gate_proj`, and `down_proj` are materialized.
4. `MLXOffloadMoeCache` shares capacity across layers and records hits, misses, loads, and evictions.
5. Decode monitors active plus allocator-cache memory. Above 85% of the configured limit, it shrinks the expert cache and clears the MLX allocator cache.

The final report includes residency decisions, safety budgets, elapsed time, memory, expert/shard cache counters, fast-path state, speculative acceptance, and optional timing. Cache events go to stderr and can be disabled with `--quiet-cache`.

## Tests

Generic CI now covers the torch-free MLX control plane on Python 3.10 and 3.13: source compilation, CLI dispatch, LRU/SLRU semantics, and the benchmark comparator.

On Apple Silicon:

```bash
.venv/bin/python -m pytest \
  tests/test_mlx_backend.py \
  tests/test_mlx_cache.py \
  tests/test_mlx_cli.py \
  tests/test_mlx_compare.py \
  tests/test_mlx_quantize_experts.py \
  -q
```

Real-checkpoint MLX generation is not yet automated in repository CI. Follow [`tests/MLX.md`](tests/MLX.md) for the smoke, resident/offload correctness, A/B, and memory-pressure procedures.

## Known limitations

- one MLX model family and batch size 1;
- local generation only; the upstream OpenAI/Anthropic server path is not wired to MLX;
- route admission crosses a Metal→CPU synchronization boundary for every MoE layer;
- cache misses still depend on safetensors and filesystem page locality;
- prompt-cache, rotating/quantized KV, and explicit prefill controls are not exposed yet;
- no Apple-Silicon CI runner or automated real-checkpoint performance gate;
- no MLX tensor-parallel path;
- the grouped BF16 GEMV reads an installed private MLX kernel template and is version-sensitive;
- package/release metadata still mixes this backend with the inherited CUDA distribution.

## Next work

1. separate MLX package/release identity from inherited CUDA publication;
2. add a pinned Apple-Silicon CI and benchmark runner;
3. expose prompt/KV cache and prefill controls with split TTFT/prefill/decode metrics;
4. record route traces and miss latency, then simulate policies offline;
5. repack experts contiguously around `(layer, expert)` miss service;
6. test predictors that improve on the failed previous-token prefetch baseline;
7. evaluate `mx.compile()` only on stable subgraphs;
8. prototype device-side resident/miss lookup;
9. consider multi-Mac/JACCL only after single-node I/O and synchronization stop dominating.

## Layout

```text
python/freetoken/mlx_backend.py           model loading, routing, offload, generation
python/freetoken/mlx_cache.py             Apple-Silicon expert cache
python/freetoken/mlx_quantize_experts.py  expert-only checkpoint conversion
python/freetoken/mlx_compare.py           captured-run parser and benchmark gates
tests/test_mlx_*.py                       MLX unit and behavior tests
tests/MLX.md                              real-checkpoint validation procedure
docs/MLX_PERFORMANCE_TUNING_GUIDE.md      evidence, failures, and experiment queue
paper/                                    research write-up and raw measurements
```

## License

Apache-2.0. Upstream attribution and fork status are preserved in the repository history and license files.
