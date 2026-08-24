# FreeToken-MLX audit

Audit date: 2026-08-24.

Scope: repository structure, MLX runtime, offload and memory behavior, tests, packaging/release, observability, security boundaries, documentation, and performance priorities.

## Executive assessment

FreeToken-MLX contains a real Apple-Silicon MoE offload implementation. Routed expert IDs drive cache admission; misses materialize projections from safetensors; MLX executes the expert path; eviction changes resident state; memory limits and pressure checks alter runtime behavior.

The largest repository risk is separation, not a fake backend. The tree still combines the upstream CUDA engine, Linux installer, CUDA release machinery, PyPI name `freetoken`, and an MLX backend with a much narrower support contract.

The next material MLX gains are also above an isolated GEMV: route synchronization, cache misses, storage locality, bytes per expert, and reused prompt/KV state.

This PR improves the control plane by adding:

- a pure-stdlib `ft mlx-compare` gate for captured runs;
- exact generated-token, output, throughput, peak-memory, cache-miss, and residency checks;
- generic CI for torch-free MLX CLI/cache/comparator behavior on Python 3.10 and 3.13;
- an Apple-Silicon real-checkpoint validation procedure in `tests/MLX.md`;
- documentation corrections for quantization, speculation, evaluation cadence, and failed prefetch experiments.

It does not add an Apple-Silicon runner or change the Metal runtime.

## Findings

### P0 — Package and release identity remain ambiguous

`pyproject.toml` still names the distribution `freetoken`, describes a broader OpenAI/Anthropic-compatible runtime, and advertises Linux/CUDA. MLX is an optional extra. The tagged release workflow builds manylinux wheels and targets the same PyPI project.

Risk:

- fork artifacts can be published under an upstream-compatible identity;
- an MLX user receives metadata and installation paths centered on CUDA/server behavior;
- release provenance for the Apple-Silicon backend is unclear.

Required decision:

- create a distinct MLX distribution, or explicitly document ownership of the `freetoken` package identity;
- separate macOS/arm64 and CUDA publication lanes;
- assert package names and platforms before upload;
- keep credentials behind separate protected environments.

This release-safety work must precede multi-Mac experiments or public MLX artifact publication.

### P0 — No Apple-Silicon CI executes the real backend

The new generic CI compiles sources and tests the torch-free control plane. It intentionally cannot validate Metal execution, unified-memory pressure, private MLX kernel-template compatibility, or real-checkpoint output.

A complete Mac lane needs:

1. torch-free CLI/import tests on macOS arm64;
2. MLX unit tests for every supported MLX/mlx-lm pair;
3. a pinned small real-checkpoint smoke test;
4. resident/offload output comparison;
5. raw benchmark artifacts and regression thresholds;
6. pressure/OOM behavior under a fixed safety envelope.

Until then, `tests/MLX.md` is a manual procedure, not continuous proof.

### P0 — Per-layer Metal→CPU route synchronization is architectural

The offload switch calls `indices.reshape(-1).tolist()` before Python can admit or materialize experts. This makes the route host-visible at every MoE layer.

The important cost is the synchronization boundary, not Python list construction.

Experiments:

- device-side resident bitset and host transfer only for misses;
- trace-trained prefetch that improves on the failed previous-token baseline;
- batched admission decisions where graph structure permits;
- compact admission in an MLX/Metal primitive;
- separate route wait, file access, materialization, and evaluation timing.

### P0 — Expert storage is optimized for checkpoints, not misses

The loader retains `(safetensors path, tensor key)` references. This avoids stacked resident experts, but a miss still follows Hugging Face shard layout.

Build and compare an expert-oriented artifact with contiguous gate/up/down payloads per `(layer, expert)`. Use the same route trace and quantization, and record bytes, page faults, p50/p95 miss latency, TTFT, throughput, disk size, and conversion cost.

### P0 — Prompt and KV controls are missing

Generation delegates to `mlx_lm.stream_generate` without exposing prompt-cache, supported rotating KV, KV quantization, or prefill-step controls.

Add:

- persistent prompt-cache load/save;
- `max_kv_size` for non-speculative generation;
- KV bits/group size/start step;
- prefill step size;
- TTFT, prefill tok/s, and decode tok/s.

Compatibility must be explicit: mlx-lm 0.31 ignores `max_kv_size` under speculative decoding.

### P1 — Runtime reporting is useful but not yet a stable schema

The runtime JSON already contains memory, residency, cache, timing, and speculative fields. The new comparator gives comparison artifacts `schema_version: 1`, but the source runtime report itself is still unversioned.

Add to the runtime report:

- schema version and git commit;
- Python/macOS/MLX/mlx-lm versions;
- hardware identifier and memory;
- model revision and config/checkpoint fingerprint;
- prompt hash;
- TTFT/prefill/decode split;
- miss latency quantiles;
- grouped-kernel template hash and fallback reason.

### P1 — Memory prediction remains heuristic

Resident peak is estimated from checkpoint bytes plus fixed overhead. Offload reserves dense bytes and fixed runtime/draft allowances, then assigns a cache fraction.

Improve it by persisting observed peaks keyed by hardware, OS, MLX version, and checkpoint layout; report prediction error; use pressure trend plus hysteresis rather than a single shrink threshold.

### P1 — Custom Metal GEMV depends on private MLX headers

`_make_grouped_gemv` reads and rewrites `include/mlx/backend/metal/kernels/gemv.h` from the installed MLX package.

Required hardening:

- probe expected signatures;
- record MLX version and source hash;
- fall back to native matmul on incompatibility;
- test the fallback;
- vendor only after license/provenance and maintenance review.

### P1 — `mx.compile()` has not been evaluated on stable subgraphs

Do not compile cache miss and materialization first. Test stable routing, shared-expert, weighted reduction, resident decode, and fixed-shape resident expert compute. Separate cold compile from warm throughput and record compile count.

### P1 — Cache work should become trace-driven

LRU and warm-up SLRU are understandable but not model-aware. Add route tracing and an offline simulator for:

- current LRU and SLRU;
- decayed frequency and pinned hot sets;
- transition/Markov prefetch;
- Belady's offline upper bound.

Report loads, loaded bytes, wasted-prefetch bytes, pollution, and required resident capacity.

Do not use naive previous-token same-layer prefetch as a recommendation: measured overlap was 26.5%, with roughly three wasted admissions per useful one.

### P1 — Test coverage is behavior-heavy and integration-light

Existing tests cover cache semantics, quantized math, residency decisions, CLI handling, shard lifetime, conversion, and tokenizer compatibility. This PR adds comparison-gate tests and a manual real-checkpoint procedure.

Still missing from automated Mac CI:

- native resident versus offload output on the same checkpoint;
- deterministic output across cache sizes/policies for exact paths;
- pressure recovery under a tight envelope;
- private-kernel fallback;
- speculative correctness and acceptance;
- corrupted mixed-projection checkpoints;
- repeated-prefix prompt-cache correctness after implementation.

### P1 — Root installer is unrelated to MLX

`install.sh` is a Linux/NVIDIA wheel installer. At repository root it implies the wrong Apple-Silicon path.

Rename or move it to a CUDA-specific location. Add a macOS installer only when standalone installation is genuinely supported; until then, keep the editable MLX install in README authoritative.

### P2 — Repository boundaries remain mixed

A clearer shape:

```text
python/freetoken/          shared/upstream core
python/freetoken/mlx/      MLX backend, cache, conversion, comparison
scripts/mlx/               MLX benchmark/install helpers
scripts/cuda/              inherited CUDA packaging
tests/mlx/                 MLX tests and fixtures
benchmarks/mlx/            raw and summarized MLX runs
docs/mlx/                  MLX architecture and performance
```

This is not a performance prerequisite, but it clarifies ownership and audit scope.

### P2 — Multi-Mac is possible but premature

MLX distributed collectives and JACCL make sharding possible. They do not remove local route sync or page-ins. Network collectives should be added only after a profile proves that compute or capacity, rather than local I/O/synchronization, justifies them.

## Security review

No critical MLX-specific remote-code execution path was identified in the reviewed local generation path.

Trust boundaries:

- Hugging Face snapshots, tokenizer/config files, and safetensors metadata are untrusted input;
- auxiliary files are accepted through a download allowlist;
- the grouped kernel consumes source from the installed MLX package;
- the inherited Linux installer bootstraps `uv` through `curl | sh` and is not an MLX install path;
- a self-hosted CUDA build runner must remain unreachable from fork PR code;
- publication credentials must remain separated and reviewer-gated.

Hardening:

- support and report an exact model revision;
- fingerprint config and checkpoint metadata;
- keep remote tokenizer/model code disabled by default;
- validate shard count, header size, dimensions, and tensor count before derived allocation;
- keep GitHub Actions pinned by commit;
- separate MLX and CUDA release credentials and package assertions.

## Validation status

Validated from source and pure-Python execution:

- route-driven cache admission and materialization design;
- BF16 and quantized expert compute paths;
- memory-budget and pressure decisions;
- LRU/SLRU semantics;
- quantization metadata and storage accounting;
- tokenizer compatibility checks;
- runtime report parsing;
- exact-token/output/memory/cache/residency comparison gates;
- CLI dispatch without importing CUDA;
- Linux/CUDA package and installer mismatch with the repository's MLX name.

Not executed in this environment:

- real Apple-Silicon generation;
- Metal kernel compatibility/performance;
- unified-memory pressure/OOM behavior;
- real-checkpoint quality comparisons.

## Recommended implementation order

1. add a pinned Apple-Silicon CI/benchmark lane and version the runtime report;
2. split MLX package/release identity from inherited CUDA publication;
3. expose prompt/KV/prefill controls and split TTFT/prefill/decode metrics;
4. add route traces, miss latency, and an offline cache simulator;
5. prototype contiguous expert storage;
6. implement trace-based prefetch that beats the 26.5% overlap baseline;
7. evaluate `mx.compile()` on stable subgraphs;
8. prototype device-side resident/miss lookup;
9. only then evaluate multi-Mac/JACCL scaling.
