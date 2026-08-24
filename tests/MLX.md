# Validating the MLX backend

The MLX compute path requires an Apple-Silicon Mac. Generic CI can validate the torch-free CLI, cache policy, capture harness, and benchmark comparator, but it cannot execute Metal kernels or reproduce unified-memory behavior.

## 1. Install the development environment

```bash
uv venv --python 3.12 --seed
uv pip install -e '.[mlx,dev]'
```

## 2. Run the MLX unit suite

```bash
.venv/bin/python -m pytest \
  tests/test_mlx_backend.py \
  tests/test_mlx_cache.py \
  tests/test_mlx_cli.py \
  tests/test_mlx_capture.py \
  tests/test_mlx_compare.py \
  tests/test_mlx_quantize_experts.py \
  -q
```

These tests cover routing math, quantized projections, cache semantics, residency selection, shard lifetime, CLI behavior, conversion, evidence capture, and comparison gates. They do not prove that a real checkpoint completes on Metal.

## 3. Capture a real-checkpoint smoke run

`mlx-capture` launches a fresh `ft generate` subprocess and atomically publishes `stdout.log`, `stderr.log`, and `capture.json`.

```bash
mkdir -p runs/smoke
.venv/bin/ft mlx-capture \
  --output runs/smoke/run-1 \
  --label smoke \
  --timeout-seconds 1800 \
  -- \
  --backend mlx \
  --model Qwen/Qwen1.5-MoE-A2.7B \
  --prompt Hello \
  --raw-prompt \
  --max-tokens 1 \
  --batch-size 1 \
  --residency offload \
  --moe-cache-size 4
```

A successful bundle must have `capture.json.status: "completed"`. The last non-empty line of `stdout.log` must be a JSON object with:

- `backend: "mlx"`;
- `generated_tokens: 1`;
- `residency.selected: "offload"`;
- a non-null `expert_cache`;
- positive `memory.peak_bytes`.

The bytes before that JSON line are generated output. Cache events are in `stderr.log`. The capture manifest records exact hashes, host/package/git metadata, and the parsed runtime record. Prompt text is redacted from the manifest by default; only its UTF-8 size and SHA-256 are stored.

## 4. Compare resident and offload correctness

Use a checkpoint that safely supports both paths, such as the full-Q4 conversion. Keep sampling deterministic and compare the same prompt.

```bash
MODEL=mlx-community/Qwen1.5-MoE-A2.7B-4bit
mkdir -p runs/resident runs/offload

.venv/bin/ft mlx-capture \
  --output runs/resident/run-1 \
  --label resident \
  --timeout-seconds 1800 \
  -- \
  --backend mlx --model "$MODEL" \
  --prompt Hello --raw-prompt --max-tokens 32 \
  --profile stable --residency resident --quiet-cache

.venv/bin/ft mlx-capture \
  --output runs/offload/run-1 \
  --label offload \
  --timeout-seconds 1800 \
  -- \
  --backend mlx --model "$MODEL" \
  --prompt Hello --raw-prompt --max-tokens 32 \
  --profile stable --residency offload \
  --expert-cache-budget-gb 4 --moe-cache-size 600 --quiet-cache

read_generated_tokens() {
  .venv/bin/python - "$1" <<'PYTHON'
import json
import pathlib
import sys

lines = pathlib.Path(sys.argv[1]).read_bytes().splitlines()
report = json.loads(next(line for line in reversed(lines) if line.strip()))
print(report["generated_tokens"])
PYTHON
}

BASELINE=runs/resident/run-1/stdout.log
CANDIDATE=runs/offload/run-1/stdout.log
BASELINE_TOKENS=$(read_generated_tokens "$BASELINE")
CANDIDATE_TOKENS=$(read_generated_tokens "$CANDIDATE")
test "$BASELINE_TOKENS" = "$CANDIDATE_TOKENS" || {
  echo "generated-token mismatch: resident=$BASELINE_TOKENS offload=$CANDIDATE_TOKENS" >&2
  exit 1
}
test "$BASELINE_TOKENS" -gt 0
EXPECTED_TOKENS=$BASELINE_TOKENS

.venv/bin/ft mlx-compare \
  --baseline "$BASELINE" \
  --candidate "$CANDIDATE" \
  --expected-tokens "$EXPECTED_TOKENS" \
  --no-throughput-gate \
  --require-output-match \
  --max-peak-memory-ratio 1.05 \
  --write-summary runs/resident-vs-offload.json
```

`--max-tokens 32` is only an upper bound because mlx-lm stops at EOS. Both runs must report the same positive `generated_tokens`; set `EXPECTED_TOKENS` to that exact value.

This recipe disables the throughput gate because it checks output correctness across two intentionally different runtime paths. Do not require identical residency here. Output matching is byte-exact: CRLF, LF, and bare-CR differences produce different hashes.

## 5. Controlled performance A/B

Run fresh processes in alternating order, for example `A,B,B,A`. Use at least three valid bundles per group for a claim intended to become a default. Keep the exact model, prompt, token count, memory envelope, and residency path fixed.

```bash
.venv/bin/ft mlx-compare \
  --baseline runs/A/*/stdout.log \
  --candidate runs/B/*/stdout.log \
  --expected-tokens 128 \
  --min-runs-per-group 3 \
  --require-balanced-groups \
  --min-throughput-ratio 0.97 \
  --max-peak-memory-ratio 1.05 \
  --max-cache-miss-ratio 1.05 \
  --require-output-match \
  --require-same-residency \
  --write-summary runs/comparison.json
```

For close results, add `--max-throughput-relative-mad <ratio>` to reject groups whose median absolute deviation divided by median throughput exceeds the runner's established noise budget.

Exit codes are stable for harnesses:

- `0`: all gates passed;
- `1`: logs were valid, but one or more gates failed;
- `2`: invalid arguments, malformed logs, or harness I/O failure.

The comparison JSON schema is versioned. It includes raw run records, medians, relative throughput dispersion, ratios, thresholds, byte-exact output hashes, source-log/report hashes, and explicit violations. The parser rejects duplicate JSON keys, non-finite constants, counters outside the signed 64-bit range, duplicate inputs, and oversized logs/reports.

## 6. Memory-pressure validation

Unified-memory failures can degrade the whole desktop before Python receives a clean exception. Close unrelated workloads and keep an explicit safety envelope:

```bash
--memory-limit-gb 10 --system-headroom-gb 3.2
```

Run pressure experiments through `mlx-capture`. It preserves a bundle for completed, failed, timed-out, and invalid-report attempts.

A valid pressure test must show that:

1. the target process either completes or exits through the clean MLX memory error path;
2. the requested macOS headroom remains available;
3. cache shrink does not change deterministic output on completed runs;
4. completed runs have `capture.json.status: "completed"`, a final stdout runtime report, and retained stderr;
5. clean memory-error exits have `capture.json.status: "failed"`, a non-zero `process_exit_code`, and the MLX memory diagnostic in `stderr.log`. They are not required to emit a success-shaped final runtime JSON.

A timeout is not equivalent to a clean memory failure. Treat `capture.json.status: "timeout"` as a separate failure mode and investigate the retained logs.

## Current automation gap

There is still no repository-managed Apple-Silicon runner. Real-checkpoint generation, Metal kernel compatibility, and performance thresholds remain manual until that lane exists. The generic CI workflow intentionally tests only the torch-free control plane.
