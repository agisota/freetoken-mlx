# FreeToken-MLX audit

Audit date: 2026-08-24.

Scope: repository structure, MLX runtime, offload and memory behavior, tests, packaging/release, observability, security boundaries, documentation, and performance priorities.

## Executive assessment

FreeToken-MLX contains a real Apple-Silicon MoE offload implementation. Routed expert IDs drive cache admission; misses materialize projections from safetensors; MLX executes the expert path; eviction changes resident state; memory limits and pressure checks alter runtime behavior.

The largest repository risk is separation, not a fake backend. The tree still combines the upstream CUDA engine, Linux installer, CUDA release machinery, PyPI name `freetoken`, and an MLX backend with a much narrower support contract.

The next material MLX gains are also above an isolated GEMV: route synchronization, cache misses, storage locality, bytes per expert, and reused prompt/KV state.

This PR now improves the control plane with:

- `ft mlx-capture`, an atomic per-run bundle with stdout/stderr, hashes, host/package/git metadata, prompt redaction, timeouts, and structured failure states;
- `ft mlx-compare`, a strict schema-versioned parser and A/B gate for exact tokens, byte-exact output, sample completeness, balance, noise, throughput, memory, cache misses, and residency;
- a versioned runtime generation layer with rendered-prompt fingerprints, first-output latency, prompt/decode rates, finish reason, speculative accounting, and requested KV/prefill controls;
- fail-fast support for rotating KV, ordinary Q4/Q8 KV quantization, and explicit prefill chunking within the actual `mlx-lm 0.31` compatibility envelope;
- generic CI for torch-free MLX CLI/cache/generation/capture/comparison behavior on Python 3.10 and 3.13;
- an Apple-Silicon real-checkpoint validation procedure in `tests/MLX.md`;
- documentation corrections for quantization, speculation, evaluation cadence, failed prefetch experiments, clean OOM behavior, and long-context constraints.

It does not add an Apple-Silicon runner or change route computation, expert math, or Metal kernels.

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

The generic CI compiles sources and tests the torch-free control plane. It intentionally cannot validate Metal execution, unified-memory pressure, private MLX kernel-template compatibility, real-checkpoint output, or long-context quality.

A complete Mac lane needs:

1. torch-free CLI/import tests on macOS arm64;
2. MLX unit tests for every supported MLX/mlx-lm pair;
3. a pinned small real-checkpoint smoke test;
4. resident/offload output comparison;
5. ordinary, rotating, Q4-KV, and Q8-KV long-context cases;
6. raw capture bundles and regression thresholds;
7. pressure/OOM behavior under a fixed safety envelope.

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

Build and compare an expert-oriented artifact with contiguous gate/up/down payloads per `(layer, expert)`. Use the same route trace and quantization, and record bytes, page faults, p50/p95 miss latency, first-output latency, throughput, disk size, and conversion cost.

### P0 — Persistent prompt cache and real KV validation remain

The CLI now exposes the supported upstream generation controls:

- `max_kv_size` for non-speculative rotating KV;
- Q4/Q8 ordinary KV quantization with group size and start offset;
- explicit prefill chunk size;
- rendered-prompt hash and token/rate metrics;
- first-output latency, decode rate, derived durations, and finish reason.

It also rejects unsupported combinations before model loading:

- rotating KV with a draft model;
- rotating KV with KV quantization, because `RotatingKVCache.to_quantized` is not implemented in `mlx-lm 0.31`;
- a rotating limit that does not exceed the four retained prefix tokens.

Remaining work:

- persistent prompt-cache load/save for repeated prefixes;
- Apple-Silicon A/B evidence for each KV mode and prefill size;
- quality evaluation for quantized KV;
- long-context pressure tests that account for both KV and expert-cache residency.

### P1 — Runtime and evidence schemas are versioned but still incomplete

The generation report now has `runtime_report_schema_version: 1` and includes rendered-prompt fingerprints, prompt/decode split, first-output latency, requested KV settings, finish reason, and corrected speculative acceptance. The comparison schema is version 2. Capture manifests are version 1 and add source-log/report hashes, host/package/git metadata, safe environment fields, command provenance, and structured status.

Still missing or incomplete:

- resolved model revision and config/checkpoint fingerprint in every runtime report;
- exact MLX/private-kernel template hash and fallback reason;
- miss-service latency quantiles;
- a process-level cold-start split for download, model load, prefill, and first token;
- a compatibility/migration policy for future schema versions.

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

Pure-Python coverage now includes cache semantics, quantized math, residency decisions, CLI handling, generation-option forwarding, compatibility guards, speculative-accounting deduplication, shard lifetime, conversion, strict runtime-log parsing, atomic capture bundles, prompt redaction, timeout/failure states, and A/B gates.

Still missing from automated Mac CI:

- native resident versus offload output on the same checkpoint;
- deterministic output across cache sizes/policies for exact paths;
- ordinary versus rotating versus quantized KV on real long prompts;
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
python/freetoken/mlx/      MLX backend, generation, cache, conversion, capture, comparison
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

Hardening already added to benchmark evidence:

- strict JSON rejects duplicate keys and non-finite constants;
- numeric counters are bounded to a signed 64-bit range;
- log and final-report sizes are capped;
- output is hashed as raw bytes without newline normalization;
- capture output cannot overwrite an existing path or broken symlink;
- prompt text is excluded from capture metadata unless explicitly requested;
- only an allowlist of non-secret environment fields is recorded;
- failed and timed-out runs preserve evidence without being presented as successful runtime reports.

Remaining hardening:

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
- extended generation-option forwarding and compatibility guards;
- rendered-prompt, prompt/decode, KV-setting, finish-reason, and first-output report fields;
- speculative acceptance is not double-counted on the final repeated response;
- atomic capture bundles, prompt redaction, hashes, timeouts, and structured failures;
- strict token/output/memory/cache/residency/sample/noise comparison gates;
- CLI dispatch without importing CUDA or MLX for control-plane help/tests;
- Linux/CUDA package and installer mismatch with the repository's MLX name.

Local control-plane validation for the new generation, capture, and comparison layers completed 48 tests. This does not substitute for the missing Mac lane.

Not executed in this environment:

- real Apple-Silicon generation;
- real rotating or quantized KV behavior on Qwen MoE;
- Metal kernel compatibility/performance;
- unified-memory pressure/OOM behavior;
- real-checkpoint quality comparisons.

## Recommended implementation order

1. add a pinned Apple-Silicon CI/benchmark lane;
2. split MLX package/release identity from inherited CUDA publication;
3. add persistent prompt-cache load/save and validate ordinary/rotating/Q4/Q8 KV plus prefill settings on real long contexts;
4. add resolved model/checkpoint/kernel fingerprints and miss-latency quantiles;
5. add route traces and an offline cache simulator;
6. prototype contiguous expert storage;
7. implement trace-based prefetch that beats the 26.5% overlap baseline;
8. evaluate `mx.compile()` on stable subgraphs;
9. prototype device-side resident/miss lookup;
10. only then evaluate multi-Mac/JACCL scaling.
