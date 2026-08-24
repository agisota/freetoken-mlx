# MLX performance tuning

This document records what improved FreeToken-MLX on Apple Silicon, what failed, and how to test the next change without fooling yourself.

Reference setup: `Qwen/Qwen1.5-MoE-A2.7B`, batch size 1, Apple M4 with 16 GB unified memory. Every number is configuration-specific.

## Bottom line

Use native `mlx-lm` when the whole model fits safely.

FreeToken-MLX matters when it does not. In that regime, expert-state movement usually dominates: checkpoint bytes, page locality, cache misses, and the synchronization required before Python can admit an expert. One faster GEMV does not fix those costs.

Optimization order:

1. reduce expert bytes;
2. reduce misses;
3. reduce miss service and page-in amplification;
4. overlap independent work;
5. remove or amortize Metal→CPU synchronization;
6. optimize a kernel only after profiling shows it dominates.

## Benchmark contract

A comparison is invalid unless these are fixed:

- checkpoint revision and quantization layout;
- prompt and chat-template mode;
- exact generated-token count;
- batch size and sampling behavior;
- residency path;
- MLX memory limit and macOS headroom;
- expert-cache byte budget, capacity, policy, and shard-cache size;
- profile and individual fast-path switches;
- draft model and proposed draft-token count;
- Python, macOS, MLX, and mlx-lm versions.

Use fresh processes and alternate order (`A,B,B,A`). Keep every raw run. Report medians or paired summaries, never the best sample.

`--max-tokens 32` is not proof of a 32-token run: mlx-lm stops at EOS. The final report must contain `generated_tokens: 32`, or the run must be discarded. `ft mlx-compare --expected-tokens 32` enforces that rule.

Recommended 16 GB safety envelope:

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
  --detailed-timing \
  > runs/A/run-1.stdout \
  2> runs/A/run-1.stderr
```

Run at least:

- exact 32-token decode;
- exact 128-token decode;
- a prompt-heavy case;
- a pressure case near the intended memory ceiling.

A policy that helps at 128 tokens can be neutral or harmful at 32.

## Machine-readable gate

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

The comparator:

- parses only the last non-empty stdout line as the runtime JSON;
- treats preceding stdout bytes as generated output and hashes them;
- rejects missing or malformed metrics;
- rejects EOS-shortened runs;
- compares median wall throughput, peak memory, and optionally cache misses;
- emits explicit violations and exits `0`/`1`/`2` for pass/gate failure/input failure.

See [`../tests/MLX.md`](../tests/MLX.md) for the full capture and validation procedure.

## Metrics to retain

Every run should preserve:

- wall time and generated tokens;
- generated text or output hash;
- MLX active, allocator-cache, and peak memory;
- expert-cache hits, misses, loads, evictions, resident entries, and capacity;
- shard-cache opens and hits;
- residency decision and every memory budget;
- speculative accepted-token count and fraction;
- route/ensure/eval timing when enabled;
- machine, OS, Python, MLX, and mlx-lm versions;
- checkpoint/config fingerprint;
- order, cooldown, and thermal notes for close comparisons.

`route_sync_ms` is not a pure Python timer. `indices.tolist()` is a Metal→CPU synchronization point, so it can absorb unfinished upstream Metal work.

## Changes that earned their place

### Native residency when safe

Offload is a capacity mechanism. The selector should keep native `mlx-lm` for checkpoints that fit the MLX working set, physical headroom, and current pressure budget.

### Bounded lazy evaluation

Evaluating every MoE layer leaves graph performance unused. Deferring all 24 layers keeps evicted arrays alive through graph references and can OOM.

The measured compromise is evaluation every eight MoE layers. In the reference BF16 run, the performance preset moved decode from about 1.43 to 1.88 tok/s while keeping memory bounded.

Current implementation caveat: `--profile performance` forces `eval_interval=8` and ignores an explicit `--eval-interval`. For a cadence A/B, use `--profile stable`, set the interval explicitly, and enable any other fast paths independently.

### Shared-expert overlap

After routing indices are available, the shared expert is independent of CPU-side cache admission. Scheduling it before safetensors materialization produced a small repeatable gain.

Keep the switch. MLX scheduling can change between releases.

### Retained quantized shard mappings

Quantized experts contain packed weights plus scale and bias metadata. Reopening and reparsing mappings on each miss was expensive.

The useful policy:

- retain up to eight mappings for quantized checkpoints;
- remove each consumed expert array from the mapping so expert-cache eviction can release it;
- retain only one mapping by default for BF16.

At a fixed 600-expert cache, recorded full-Q4 decode moved from about 3.6 to 5.0 tok/s; mixed BF16-dense/Q4-expert moved from about 3.20 to 4.42 tok/s. Shard parse time fell from about 2.4 s to 0.24 s.

### Expert-only quantization

Reducing bytes produced the largest offload gain.

Mixed BF16-dense/Q4-expert reached 4.49 tok/s at 7.93 GB peak in the reference run, versus 1.88 tok/s for BF16 offload at 5.83 GB.

This is the expected mechanism: cheaper misses and more resident experts under the same byte budget.

### Projection-aware precision

The current converter exposes one base precision for `up_proj` and `gate_proj`, plus an optional `down_proj` override.

Q3 up/gate + Q4 down reduced routed-expert storage by about 16.7% relative to uniform Q4 and increased capacity from 864 to 1,015 experts under the same safe byte budget.

Recorded 128-token A/B:

| Layout | Mean time | Misses | Peak |
| --- | ---: | ---: | ---: |
| uniform Q4 | 17.52 s | 3,489 | ~8.02 GB |
| Q3 up/gate + Q4 down | 15.77 s | 2,822 | ~8.03 GB |

At a fixed 864 slots, the mixed layout remained about 3.4% faster and used about 0.62 GB less peak memory.

Teacher-forced proxy on 135 tokens:

| Format | NLL |
| --- | ---: |
| BF16 | 2.618 |
| uniform Q4 | 2.707 |
| Q3 up/gate + Q4 down | 2.725 |
| uniform Q3 | 2.865 |

This supports an opt-in speed/capacity tradeoff. Uniform Q4 remains the conservative default.

### SLRU after warm-up

Pure LRU is vulnerable to scan-like traffic. The current SLRU starts with a small protected segment, then expands it after several cache-sized request windows.

At 864 entries:

- 32-token misses were effectively unchanged;
- 128-token misses fell from 3,801 to 3,489;
- throughput improved by about 4%;
- peak memory was unchanged.

Auto-selection uses SLRU only for quantized experts in the performance profile.

### Safe spare memory for BF16

The old BF16 performance budget left safe memory unused. Raising its cache fraction from 20% to 50%, while preserving every clamp, increased the reference cache from 121 to 304 experts.

Recorded effect:

- 32-token misses: 2,353 → 1,949;
- 128-token misses: 9,672 → 8,183;
- 128-token time: 64.59 s → 62.33 s;
- peak: 5.86 GB → 9.03 GB under a 10 GB MLX limit.

This is a deliberate memory-for-loads trade, not free speed.

### Full-Q4 speculative decoding on a compatible workload

A compatible Q4 0.5B draft helped the full-Q4 MoE target in the recorded 64-token workload:

- mean time: 12.77 s → 11.45 s;
- 57.8% of output tokens accepted from the draft;
- peak memory: 5.71 GB → 6.31 GB.

An 8-token request was slower. Keep speculation opt-in and measure target calls, acceptance, cache capacity after draft reservation, and total wall time.

## Experiments that failed

### Faster isolated GEMV barely moved end-to-end decode

A thread-layout change improved isolated BF16 expert GEMV from 1.534 ms to 1.243 ms, about 19%.

End-to-end generation moved by about 1%. Some layouts changed reduction order and generated output. The kernel was not the dominant component.

### Packed-Q4 batching regressed

Stacking routed Q4 weights reduced dispatch count but added per-token packing, copying, and shape work. The tested path regressed and used more allocator cache.

A future fused path needs physically contiguous cache slots; dynamic stacking removes the intended benefit.

### Unlimited lazy graphs OOM

A full-model lazy graph keeps references to evicted experts until logits are evaluated. This defeats bounded residency.

Any larger evaluation interval must retain an OOM guard and record peak memory.

### More retained BF16 mappings did not help

BF16 has less quantization metadata per projection. Retaining additional mappings increased retention without a measured throughput gain. The default remains one.

### Alternative BF16 eviction policies were worse

At 300 slots, SLRU increased 32-token misses from 1,953 to 2,086 and slowed inference by about 4%. Offline simulation also favored global LRU over per-layer LRU and LFU for that trace.

Do not transfer a Q4 policy to BF16 without replaying the actual route geometry.

### Previous-token same-layer prefetch was a negative baseline

Only 26.5% of selected experts overlapped between adjacent tokens in the same layer. Prefetching all four prior experts caused roughly three wasted admissions for every useful one and polluted the bounded BF16 cache.

Do not repeat this as the default predictor. Any new prefetch method must beat this measured baseline in useful-prefetch ratio, wasted bytes, miss latency, and end-to-end time.

### BF16-target speculation regressed

A resident Q4 0.5B draft reduced the target expert-cache budget while short BF16 verification chunks routed many unique experts. The recorded 64-token run took 54.84 s, shrank the cache, and finished slower than ordinary BF16 decode.

Speculation must be evaluated per target representation and output length.

### Q2 experts were fast but damaged output

Q2 reached about 6.18 tok/s on a smoke prompt and visibly degraded the sample. Throughput without a quality evaluation is not a valid improvement.

## Current bottlenecks

### 1. Metal→CPU route synchronization

Every MoE layer currently crosses this boundary:

```python
routed = [int(v) for v in indices.reshape(-1).tolist()]
```

Python owns expert-cache admission and file materialization, so the route must become host-visible before the miss can be serviced.

Useful experiments:

- maintain a device-side resident bitset and synchronize only miss IDs;
- batch decisions where graph structure permits;
- prefetch using a predictor trained on route traces, not naive previous-token reuse;
- move compact admission state into an MLX/Metal primitive;
- separate route wait from miss materialization in instrumentation.

Success means lower wall time with identical output and bounded memory. A lower `route_sync_ms` alone is insufficient.

### 2. Checkpoint-oriented storage

A miss resolves `(shard path, tensor key)` for three projections. Cached mappings remove metadata parse cost, but payload access still follows Hugging Face shard locality.

Next experiment:

```text
(layer, expert) -> contiguous gate + up + down payload
```

Compare both layouts on the same route trace and quantization. Record bytes read, faults, p50/p95 miss service, TTFT, 32/128-token throughput, disk size, and conversion cost.

### 3. Prompt and KV controls are not exposed

The backend calls `mlx_lm.stream_generate` without exposing:

- persistent prompt-cache load/save for repeated prefixes;
- `max_kv_size` / rotating KV for bounded non-speculative generation;
- KV quantization bits, group size, and start step;
- explicit prefill step size;
- separate TTFT, prefill, and decode metrics.

Important compatibility limit: mlx-lm 0.31 drops `max_kv_size` when `draft_model` is used, and its speculative path does not implement rotating KV. Do not document or test that flag as active under speculation until a compatible path exists.

These controls matter most for agent workloads with repeated system prompts and tool schemas.

### 4. Stable subgraph compilation

`mx.compile()` can reduce graph overhead, but dynamic shapes or dtypes can trigger recompilation.

Do not compile cache miss/materialization first. Test stable pieces:

- router softmax/top-k;
- shared expert;
- weighted reduction;
- resident decode step;
- fixed-shape resident expert compute.

Report cold compile time, compile count, warm throughput, memory, and output.

### 5. Trace-driven prefetch and policy

Create a trace keyed by `(token, layer)` and replay it offline. Evaluate:

- decayed expert frequency per layer;
- transition tables over complete expert sets;
- a small Markov predictor;
- pinned hot sets;
- one- or two-layer look-ahead where computation permits;
- Belady's offline upper bound.

Keep previous-token same-layer reuse only as the measured negative baseline: 26.5% overlap and about three wasted admissions per useful one.

Metrics must include wasted bytes and cache pollution, not only hit rate.

### 6. Multi-Mac is not the first fix

MLX distributed collectives make multi-Mac sharding possible. They do not remove local route synchronization or page-ins.

Add network collectives only after a time breakdown shows enough compute or capacity pressure to justify them.

## Experiment queue

| Priority | Experiment | Expected upside | Main risk |
| --- | --- | --- | --- |
| P0 | Apple-Silicon CI + pinned benchmark lane | continuous validity | runner cost/noise |
| P0 | runtime report schema + prefill/decode split | trustworthy gates | compatibility work |
| P0 | prompt cache and supported KV controls | large on repeated/long contexts | cache correctness/quality |
| P0 | route trace + miss latency distribution | identifies the real target | instrumentation perturbation |
| P0 | contiguous expert artifact | lower page-in cost | another format |
| P1 | trace-based prefetch | hide miss latency | wasted I/O and pollution |
| P1 | `mx.compile()` stable subgraphs | less graph overhead | recompilation/cold start |
| P1 | adaptive byte-budget controller | better pressure response | oscillation |
| P2 | device-side admission prototype | remove sync boundary | implementation complexity |
| P2 | multi-Mac/JACCL sharding | more capacity | communication overhead |

Package/release identity is a separate P0 repository requirement and should be resolved before publishing MLX artifacts, independent of this performance queue.

## Default-change gate

A performance change should not become default without:

1. exact commands;
2. machine and software metadata;
3. raw stdout/stderr for every run;
4. exact generated-token enforcement;
5. A/B summary generated from the raw logs;
6. output or quality comparison;
7. peak-memory and headroom comparison;
8. a disable switch for risky fast paths;
9. an automated threshold, or a precise explanation of why the required Mac lane is still manual.

## Safety

Unified-memory pressure can make macOS unusable before Python receives a clean OOM. Preserve:

- MLX working-set limit;
- allocator-cache limit;
- explicit system headroom;
- live pressure check;
- dense/runtime and draft reserves;
- expert-cache byte clamp;
- monotonic cache shrink under pressure;
- clean Metal-memory errors.

A 5% gain obtained by taking memory from the operating system is a regression.

## Interpretation

The optimization frontier is expert-state movement and scheduling: bytes per expert, miss frequency, file/page locality, and how much miss latency can be hidden before the next token blocks.

That is where the next material end-to-end gain is most likely.
