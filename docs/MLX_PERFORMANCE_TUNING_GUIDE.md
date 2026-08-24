# MLX performance tuning

This document records what has actually improved FreeToken-MLX on Apple Silicon, what did not, and what should be tested next.

The current reference setup is `Qwen/Qwen1.5-MoE-A2.7B`, batch size 1, Apple M4 with 16 GB unified memory. Treat every number below as configuration-specific.

## Bottom line

If the whole model fits safely, use native `mlx-lm`.

FreeToken-MLX is useful when full residency does not fit. In that regime the dominant cost is usually expert bytes moving through storage/page cache/unified memory, not Python arithmetic and not one isolated GEMV kernel.

That changes the optimization order:

1. reduce expert bytes;
2. reduce expert misses;
3. reduce shard/page-in overhead;
4. overlap independent work;
5. remove Metal→CPU synchronization;
6. only then optimize individual kernels.

## Benchmark contract

Do not compare runs unless these are fixed:

- checkpoint and quantization layout;
- prompt and chat-template mode;
- generated token count;
- batch size;
- residency mode;
- MLX memory limit;
- system headroom;
- expert-cache byte budget;
- expert-cache policy;
- shard-cache size;
- performance-profile switches;
- draft model and draft-token count;
- MLX / mlx-lm versions.

Use fresh processes. Alternate A/B order (`A,B,B,A`). Record medians and all raw runs, not the best run. Keep generated output when exactness matters.

Recommended controlled envelope for a 16 GB Mac:

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

Run at least short decode (32 tokens), medium decode (128 tokens), and a prompt-heavy case. Cache policies that help at 128 tokens can be neutral or worse at 32.

## Metrics to keep

Every benchmark should retain:

- wall-clock generation time;
- generation tok/s;
- generated output or output hash;
- MLX active/cache/peak memory;
- expert cache hits, misses, loads, evictions, resident entries and capacity;
- shard opens and hits;
- selected residency path and all memory budgets;
- speculative accepted-token count and acceptance fraction;
- detailed route/ensure/eval timing when enabled;
- machine, OS, MLX and mlx-lm version;
- thermal/cooldown metadata when doing close A/B comparisons.

`route_sync_ms` is not pure Python overhead. `indices.tolist()` is a Metal→CPU synchronization point, so that timer can include upstream GPU work that had not completed yet.

## What worked

### Native residency when safe

Offload is a capacity mechanism, not a universal fast path. The selector should continue to choose native `mlx-lm` whenever the whole checkpoint fits inside the process working set, physical headroom, and current memory-pressure budget.

### Bounded lazy evaluation

Evaluating every MoE layer leaves performance on the table. Letting the graph span the whole model keeps evicted expert arrays alive too long and can OOM.

The current performance preset evaluates every eight MoE layers. In the reference BF16 run this moved decode from roughly 1.43 to 1.88 tok/s while keeping memory bounded.

### Shared-expert overlap

Once router indices are known, the shared expert is independent of CPU-side expert-cache admission. Scheduling it before safetensors lookup produced a small repeatable gain.

Keep this switchable. Scheduler behavior can change across MLX releases.

### Keep parsed quantized shard mappings

Quantized experts have packed weights plus scale/bias metadata. Reopening and reparsing shard mappings on every miss was expensive.

The useful policy was:

- retain up to eight shard mappings for quantized checkpoints;
- remove each consumed expert array from the retained mapping so cache eviction can release it;
- keep the BF16 default conservative.

At a fixed 600-expert cache this moved full-Q4 decode from about 3.6 to 5.0 tok/s and mixed BF16-dense/Q4-expert decode from about 3.20 to 4.42 tok/s in the recorded runs. Shard-parse time fell from about 2.4 s to 0.24 s.

### Quantize routed experts

Reducing bytes had the largest effect on the offload path.

Mixed BF16-dense/Q4-expert reached 4.49 tok/s at 7.93 GB peak in the reference run, versus 1.88 tok/s for BF16 offload at 5.83 GB.

This is the expected direction: fewer bytes per expert means cheaper misses and more experts resident under the same memory budget.

### Per-projection expert precision

The checkpoint format permits different affine precision for `up_proj`, `gate_proj`, and `down_proj`.

Q3 up/gate + Q4 down reduced routed-expert storage by about 16.7% compared with uniform Q4. Under the same safe byte budget it increased cache capacity from 864 to 1,015 experts.

Recorded 128-token A/B:

| Layout | Mean time | Misses | Peak |
| --- | ---: | ---: | ---: |
| uniform Q4 | 17.52 s | 3,489 | ~8.02 GB |
| Q3 up/gate + Q4 down | 15.77 s | 2,822 | ~8.03 GB |

A fixed-864-slot comparison still showed about 3.4% speed improvement and ~0.62 GB lower peak, separating lower packed-weight cost from the benefit of extra cache slots.

Quality proxy on 135 teacher-forced tokens:

| Format | NLL |
| --- | ---: |
| BF16 | 2.618 |
| uniform Q4 | 2.707 |
| Q3 up/gate + Q4 down | 2.725 |
| uniform Q3 | 2.865 |

This supports mixed precision as an opt-in speed/capacity tradeoff. It is not enough evidence to replace Q4 as the conservative default.

### SLRU after warm-up

Pure LRU is vulnerable to scan-like expert traffic. The current SLRU keeps a small protected segment initially, then grows it after several cache-sized request windows.

At 864 entries:

- 32-token misses were effectively unchanged;
- 128-token misses fell from 3,801 to 3,489;
- throughput improved by about 4%;
- peak memory was unchanged.

Auto-selection enables SLRU only for quantized experts in the performance profile.

### Use safe spare memory for BF16 cache

The old BF16 performance budget left safe memory unused. Raising the performance-profile fraction from 20% to 50%, while still clamping against all safety budgets, increased the reference cache from 121 to 304 experts.

Recorded effect:

- 32-token misses: 2,353 → 1,949;
- 128-token misses: 9,672 → 8,183;
- 128-token time: 64.59 s → 62.33 s;
- peak: 5.86 GB → 9.03 GB under a 10 GB MLX limit.

This is a capacity trade: spend memory only when the requested system headroom still survives.

## What did not work

### Faster isolated GEMV did not materially move end-to-end decode

One thread-layout change improved isolated BF16 expert GEMV from 1.534 ms to 1.243 ms, roughly 19%.

End-to-end generation moved by about 1%. Some layouts also changed numerical reduction order and therefore generated output.

This is an Amdahl's-law result: the kernel was not the dominant wall-clock component.

### Batching packed Q4 matmuls regressed

Stacking routed Q4 weights to call larger native quantized matmuls looked attractive but added packing/copy/shape overhead on the decode path. The total path regressed in the tested configuration.

Do not revive this without measuring total bytes copied and graph-build cost.

### Unlimited lazy graphs OOM

Deferring evaluation across all MoE layers prevents evicted expert arrays from dying because the graph still references them. This defeats the cache's memory model.

Any attempt to increase `eval_interval` must record peak memory and must include an OOM guard.

## Current bottlenecks

### 1. Metal→CPU route synchronization

The current cached switch does this per MoE layer:

```python
routed = [int(v) for v in indices.reshape(-1).tolist()]
```

That is the admission boundary: MLX must make routing indices visible to Python before the CPU can decide which experts to load.

This is now one of the most important architectural bottlenecks because it prevents a fully asynchronous decode pipeline.

Candidate experiments:

- batch route decisions for several layers where graph structure permits it;
- prefetch from previous-token route history before the current route is synchronized;
- maintain a device-side hot-expert table and synchronize only misses;
- predict likely experts per layer from a route trace and issue speculative page-ins;
- move more admission state into an MLX/Metal primitive if the cache policy can be expressed compactly.

Success criterion: lower wall time with the same output and no increase in miss-induced memory growth. Do not use `route_sync_ms` alone as proof.

### 2. Random expert storage reads

On a miss, the runtime resolves `(shard path, key)` and materializes three projections. Even when shard metadata is cached, payload access still depends on filesystem/page-cache locality.

The next storage experiment should build an expert-oriented layout:

```text
layer/expert -> contiguous gate + up + down payload
```

Compare it against the current Hugging Face shard layout with the same quantization and cache trace.

Measure:

- bytes read per miss;
- page faults / filesystem reads;
- miss service latency;
- first-token latency;
- 32/128-token throughput;
- conversion-time and disk-size overhead.

A repacked checkpoint is justified only if end-to-end miss service improves enough to compensate for another artifact format.

### 3. Prompt/KV work is largely delegated to default mlx-lm behavior

The backend currently calls `stream_generate` without exposing the useful cache controls already present in mlx-lm.

High-value additions:

- persistent prompt cache for repeated prefixes;
- `max_kv_size` / rotating KV for bounded long-context memory;
- KV quantization (`kv_bits`, group size, quantization start);
- explicit prefill step size;
- separate prefill and decode metrics.

These matter especially for agent harnesses, where the system prompt and tool schema are often reused across many requests.

### 4. Compile stable subgraphs

MLX `mx.compile()` can fuse graphs and reduce graph overhead, but recompilation can occur when shapes or dtypes change.

Do **not** compile the dynamic cache-miss/materialization path first.

Test compilation on stable components:

- router softmax/top-k block;
- shared expert;
- dense post-routing reduction;
- resident path decode step;
- fixed-shape BF16 expert compute once expert arrays are already resident.

Benchmark cold compile separately from steady-state decode. A harness should report compile count and warm-up cost.

### 5. Prefetch rather than smarter eviction alone

Once cache policy is reasonably good, preventing a miss before it blocks decode can be worth more than a few points of hit rate.

Build a trace recorder keyed by `(layer, token_index)` and replay route sequences offline. Evaluate predictors such as:

- previous-token same-layer expert set;
- exponentially decayed expert frequency per layer;
- transition table from previous expert set;
- small Markov predictor;
- top-N hot experts pinned per layer;
- sequential prefetch for experts expected within the next 1–2 layers.

Metrics must include wasted-prefetch bytes, not just hit rate.

### 6. Multi-Mac scaling is not the first fix

MLX currently supports distributed collectives including ring and JACCL. That makes tensor/model sharding across Apple Silicon machines possible in principle.

It does not automatically solve this backend's current bottleneck. If local decode is dominated by route synchronization and storage page-ins, adding network collectives can make it worse.

Only evaluate multi-Mac after producing a time breakdown showing enough compute or memory-capacity pressure to justify communication.

## Suggested experiment queue

| Priority | Experiment | Expected upside | Main risk |
| --- | --- | --- | --- |
| P0 | expose prompt cache + KV controls | large for repeated/long contexts | quality loss with too-small/quantized KV |
| P0 | route-trace instrumentation + miss latency | enables correct optimization | instrumentation perturbation |
| P0 | expert-oriented contiguous storage | large if page-in dominates | new artifact format |
| P1 | route-based expert prefetch | hides miss latency | wasted bandwidth/memory |
| P1 | `mx.compile()` stable subgraphs | lower graph overhead | recompilation/cold-start |
| P1 | adaptive byte-budget controller | better pressure handling | oscillation/thrash |
| P2 | device-side admission experiment | removes sync boundary | implementation complexity |
| P2 | multi-Mac/JACCL sharding | more capacity | communication overhead |

## Reproducibility gate

A performance change should not become a default unless the repository contains:

1. the exact benchmark command;
2. machine and software metadata;
3. raw result JSON for every run;
4. an A/B summary generated from those files;
5. correctness comparison;
6. memory-safety comparison;
7. a disable switch for risky fast paths;
8. a regression threshold suitable for CI or a documented reason CI cannot run it.

Prefer machine-readable benchmark artifacts over prose claims.

## Safety rules

Performance tuning on unified memory can make the rest of macOS unusable before Python receives a clean OOM.

Keep all of these controls:

- MLX working-set limit;
- allocator-cache limit;
- explicit system headroom;
- current memory-pressure check;
- draft-model reserve;
- dense/runtime reserve;
- expert-cache byte clamp;
- runtime cache shrink when pressure rises;
- clean error reporting for Metal memory failures.

A benchmark that is 5% faster because it borrows memory from the operating system is not a valid improvement.

## Interpretation

The project has already crossed the point where another clever micro-kernel is the obvious answer.

The current optimization frontier is the **movement and scheduling of expert state**: how many bytes each expert costs, how often it must be loaded, how those loads map to files/pages, and how much of the miss latency can be hidden before the next token blocks.

That is where the next large end-to-end gains are most likely to come from.
