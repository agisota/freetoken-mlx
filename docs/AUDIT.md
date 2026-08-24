# FreeToken-MLX audit

Audit date: 2026-08-24.

Scope: repository structure, MLX runtime path, memory/offload behavior, tests, packaging/release surface, observability, documentation, and next performance work.

## Executive assessment

FreeToken-MLX contains a real Apple-Silicon MoE offload implementation, not a mock adapter. Routed expert IDs drive cache admission; misses materialize expert projections from safetensors; MLX performs the expert computation; eviction changes resident state; memory limits and pressure checks affect runtime behavior.

The main weakness is product/repository separation. The repository still carries the upstream CUDA engine, Linux installer, CUDA release workflow, package name `freetoken`, and metadata that describe a broader server runtime than the MLX backend actually provides. The code is more credible than the packaging and docs make it look.

The next large MLX speedups are unlikely to come from another isolated GEMV tweak. The critical path is expert-state movement and scheduling: route synchronization, cache misses, shard/page locality, bytes per expert, and reusable prompt/KV state.

## Findings

### P0 — Release/package identity is ambiguous

`pyproject.toml` still publishes the distribution as `freetoken`. Its description says it is a local MoE runtime with OpenAI/Anthropic API compatibility, while the classifiers advertise Linux/CUDA. MLX is only an optional extra.

At the same time `.github/workflows/release.yml` builds Linux manylinux wheels and publishes them to the PyPI project `freetoken`.

Risk:

- accidental publication of fork builds under an upstream-compatible package identity;
- users installing the package for MLX and receiving metadata centered on CUDA/server behavior;
- inability to reason cleanly about which release contains the Apple-Silicon backend.

Recommended fix:

- create a distinct MLX distribution identity or explicitly declare that this repository is the canonical publisher of `freetoken`;
- create a macOS/arm64 MLX release lane independent of CUDA wheels;
- make package metadata platform-conditional or split runtime distributions;
- protect release workflows with environments and explicit package-name assertions.

### P0 — No CI lane validates the real MLX runtime on Apple Silicon

The repository has MLX unit tests, but the visible workflows are wheel/release oriented and CUDA/Linux centered.

Required CI layers:

1. torch-free import/CLI tests on macOS arm64;
2. MLX unit tests on every supported MLX/mlx-lm pair;
3. a small real-checkpoint smoke test;
4. a benchmark job on a pinned Apple-Silicon runner that uploads raw JSON;
5. threshold checks for output equivalence, peak memory, cache behavior, and throughput regression.

Without an Apple-Silicon runner, performance claims remain manually reproducible rather than continuously validated.

### P0 — Per-layer Metal→CPU route synchronization is architectural

The offload switch converts routing indices with `indices.reshape(-1).tolist()` before cache admission. This forces routing state to cross from the MLX graph to Python for every MoE layer.

It is necessary in the current design because Python owns the cache and file materialization. It also blocks a fully asynchronous decode pipeline.

Recommended experiments:

- trace previous-token routes and prefetch likely next experts before synchronization;
- maintain per-layer hot sets and synchronize only when the route is outside the resident set;
- batch admission decisions where model structure allows it;
- prototype device-side resident-bitset lookup and return only miss IDs;
- record miss-service latency separately from upstream GPU wait time.

Do not optimize `tolist()` as a Python operation; the important cost is the synchronization boundary.

### P0 — Expert storage is checkpoint-oriented, not miss-oriented

The backend keeps `(safetensors path, key)` references. This avoids resident expert stacks, but cache misses still follow the Hugging Face shard layout.

The best next storage experiment is an expert-oriented artifact where gate/up/down payloads for one `(layer, expert)` are contiguous. This can reduce page-in amplification and random payload access.

The benchmark must compare the same route trace and quantization under both layouts and record bytes/page faults/miss latency, not only tok/s.

### P0 — Prompt/KV cache controls are missing from the FreeToken CLI

Current generation delegates to `mlx_lm.stream_generate` without exposing prompt-cache, rotating KV, KV quantization, or prefill-step controls available in mlx-lm.

This is especially important for agent workloads with a large repeated system/tool prefix.

Add:

- persistent prompt-cache load/save;
- maximum KV size;
- KV bits/group size/quantization start;
- prefill step size;
- separate prefill and decode timing/tok/s.

### P1 — Memory prediction is useful but still heuristic

Resident peak is estimated from checkpoint bytes plus fixed packing/runtime allowances. Offload reserves dense checkpoint bytes plus fixed overhead and then allocates a fraction to expert cache.

This is reasonable for a first selector but it should learn from observed runs.

Recommended improvement:

- persist observed `(checkpoint layout, OS, MLX version, hardware) -> resident peak` records;
- use measured peak as the first predictor on subsequent runs;
- report prediction error;
- clamp automatic cache growth based on live pressure trend rather than a single 85% threshold;
- add hysteresis before shrinking/growing to prevent oscillation.

### P1 — Custom Metal GEMV depends on MLX private installed headers

`_make_grouped_gemv` reads `include/mlx/backend/metal/kernels/gemv.h`, rewrites fragments, and compiles a runtime Metal kernel.

This is effective but version-sensitive. A semver-compatible MLX update can change internal headers without treating this code as public API.

Recommended fix:

- feature-probe the expected header signatures;
- emit the exact MLX version and kernel-source hash in benchmark output;
- fall back cleanly to native matmul when the template is incompatible;
- add a test that intentionally exercises the fallback;
- consider vendoring the minimal kernel implementation only if license/provenance and maintenance are acceptable.

### P1 — `mx.compile()` is not evaluated on stable subgraphs

MLX compilation can fuse work and reduce graph overhead, but compiling dynamic cache/materialization functions can cause recompilation and erase the gain.

Test it only on stable-shape pieces first:

- routing softmax/top-k;
- shared expert;
- score-weighted reduction;
- resident decode step;
- fixed-shape expert compute after weights are resident.

Record cold compile time, steady-state speed, memory, and compile count.

### P1 — Cache policy should become trace-driven

LRU and the current warm-up SLRU are simple and understandable. They are not model-aware.

Add a route-trace mode and offline simulator so policies can be compared without running the model repeatedly. Include:

- LRU;
- current SLRU;
- per-layer LFU/decayed frequency;
- pinned hot set;
- Markov/transition prefetch;
- Belady offline upper bound.

The simulator should report hit rate, loads, bytes loaded, wasted prefetch, and required resident bytes.

### P1 — Observability needs schema/versioning

The final JSON report is already useful, but it should become a stable benchmark artifact.

Add:

- `report_schema_version`;
- git commit;
- Python, macOS, MLX, mlx-lm versions;
- hardware identifier and memory size;
- checkpoint fingerprint/config hash;
- prompt hash rather than prompt text by default;
- prefill/decode split;
- cache miss latency histogram or quantiles;
- kernel-source/version identifiers;
- optional route trace stored separately.

### P1 — Test coverage is behavior-heavy but integration-light

The MLX tests verify cache semantics, quantization math, memory decisions, CLI behavior, and some expert computation. That is a good base.

Missing high-value tests:

- full real model generation compared with native mlx-lm for a resident-safe checkpoint;
- offload generation compared with a resident reference on the same weights;
- deterministic output across cache sizes/policies for exact paths;
- OOM/pressure recovery under a deliberately tight memory envelope;
- custom-kernel fallback;
- speculative decode acceptance/correctness;
- mixed-projection checkpoint corruption cases;
- repeated-prefix prompt-cache correctness once added.

### P1 — Root installer is unrelated to MLX

`install.sh` is a Linux/NVIDIA wheel installer. Keeping it at repository root makes `freetoken-mlx` look installable through the wrong path.

Recommended fix:

- rename it to `install-linux-cuda.sh` or move it under `scripts/`;
- add `install-macos.sh` only if a standalone installer is actually supported;
- keep README instructions authoritative until that exists.

### P2 — Multi-Mac distributed MLX is possible but premature

Current MLX supports distributed collectives, including ring communication and JACCL. Multi-Mac sharding can increase available memory and compute.

Do not prioritize it until a profiler shows that local storage/synchronization is no longer dominant. Otherwise network collectives will stack on top of the existing stalls.

### P2 — Repository should separate inherited upstream surfaces

The root tree mixes:

- CUDA engine/server/distributed runtime;
- Apple-Silicon MLX backend;
- CUDA wheel cache;
- paper and benchmark artifacts;
- desktop/Linux release machinery.

A clearer shape would be:

```text
python/freetoken/          shared/upstream core
python/freetoken/mlx/      Apple-Silicon backend and cache
scripts/mlx/               conversion/benchmark/install helpers
scripts/cuda/              inherited CUDA packaging helpers
tests/mlx/                 all MLX tests
benchmarks/mlx/            raw and summarized MLX runs
docs/mlx/                  MLX architecture/performance docs
```

This is not required for performance, but it makes maintenance and audit boundaries explicit.

## Security review

No critical remote-code execution path was identified in the MLX-specific code reviewed here. The MLX CLI is local-only and the backend does not expose the upstream server surface.

Important trust boundaries remain:

- Hugging Face model snapshots and tokenizer/config files are untrusted input;
- `snapshot_download` can download auxiliary files matching the allowlist;
- safetensors headers are parsed before loading payloads and should continue to receive strict bounds/shape validation;
- the custom Metal kernel consumes source from the locally installed MLX package;
- root Linux installer bootstraps `uv` through `curl | sh`, which is unrelated to MLX and should not be presented as the Apple-Silicon install path;
- release workflows use a self-hosted build runner and should remain inaccessible to fork PR code.

Recommended hardening:

- support optional model revision/commit pinning and report the resolved revision;
- add checkpoint/config fingerprints to reports;
- keep `trust_remote_code` off unless a future model explicitly requires it;
- validate safetensors dimensions/counts before allocating derived structures;
- cap total shard count/header size processed by local tools where practical;
- pin GitHub Actions by commit SHA (already done in the reviewed release workflow) and keep publication behind protected environments;
- separate MLX and CUDA release credentials/lane.

## Validation status

Validated from repository source:

- real MLX cache admission and eviction path;
- route-driven expert materialization;
- quantized and BF16 expert compute paths;
- memory-budget and pressure checks;
- LRU/SLRU unit tests;
- mixed quantization metadata validation;
- speculative tokenizer compatibility checks;
- JSON runtime reporting;
- Linux/CUDA release and installer mismatch with the repository's MLX name.

Not executed in this audit environment:

- real Apple-Silicon model generation;
- Metal kernel benchmarks;
- memory-pressure/OOM experiments;
- real-checkpoint quality comparisons.

Those require an Apple-Silicon runner with model weights. The repository should automate them rather than relying on prose evidence.

## Recommended implementation order

1. add an Apple-Silicon CI/benchmark lane and report schema;
2. expose prompt/KV cache controls and prefill/decode metrics;
3. add route traces + offline cache simulator;
4. prototype expert-oriented contiguous storage;
5. implement route-based prefetch;
6. evaluate `mx.compile()` on stable subgraphs;
7. prototype device-side resident/miss lookup;
8. only then evaluate multi-Mac/JACCL scaling;
9. split MLX packaging/release identity from inherited CUDA release machinery.
