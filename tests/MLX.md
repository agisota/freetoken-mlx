# Validating the MLX backend

The MLX compute path requires an Apple-Silicon Mac. Generic CI can validate the torch-free CLI, cache policy, and benchmark comparator, but it cannot execute Metal kernels or reproduce unified-memory behavior.

## 1. Install the development environment

```bash
uv venv --python 3.12 --seed
uv pip install -e '.[mlx,dev]'
```

Record the environment before publishing a result:

```bash
mkdir -p runs/meta
sw_vers > runs/meta/macos.txt
system_profiler SPHardwareDataType > runs/meta/hardware.txt
.venv/bin/python --version > runs/meta/python.txt
.venv/bin/python -m pip show mlx mlx-lm > runs/meta/mlx-packages.txt
```

## 2. Run the MLX unit suite

```bash
.venv/bin/python -m pytest \
  tests/test_mlx_backend.py \
  tests/test_mlx_cache.py \
  tests/test_mlx_cli.py \
  tests/test_mlx_compare.py \
  tests/test_mlx_quantize_experts.py \
  -q
```

These tests cover routing math, quantized projections, cache semantics, residency selection, shard lifetime, CLI behavior, conversion, and comparison gates. They do not prove that a real checkpoint completes on Metal.

## 3. Real-checkpoint smoke test

```bash
mkdir -p runs/smoke
.venv/bin/ft generate \
  --backend mlx \
  --model Qwen/Qwen1.5-MoE-A2.7B \
  --prompt Hello \
  --raw-prompt \
  --max-tokens 1 \
  --batch-size 1 \
  --residency offload \
  --moe-cache-size 4 \
  > runs/smoke/stdout.log \
  2> runs/smoke/stderr.log
```

The last non-empty stdout line must be a JSON object with:

- `backend: "mlx"`;
- `generated_tokens: 1`;
- `residency.selected: "offload"`;
- a non-null `expert_cache`;
- positive `memory.peak_bytes`.

The text before that JSON line is the generated output. Cache events are written to stderr unless `--quiet-cache` is used.

## 4. Compare resident and offload correctness

Use a checkpoint that safely supports both paths, such as the full-Q4 conversion. Keep sampling deterministic and compare the same prompt.

```bash
MODEL=mlx-community/Qwen1.5-MoE-A2.7B-4bit
mkdir -p runs/resident runs/offload

.venv/bin/ft generate \
  --backend mlx --model "$MODEL" \
  --prompt Hello --raw-prompt --max-tokens 32 \
  --profile stable --residency resident --quiet-cache \
  > runs/resident/run-1.stdout 2> runs/resident/run-1.stderr

.venv/bin/ft generate \
  --backend mlx --model "$MODEL" \
  --prompt Hello --raw-prompt --max-tokens 32 \
  --profile stable --residency offload \
  --expert-cache-budget-gb 4 --moe-cache-size 600 --quiet-cache \
  > runs/offload/run-1.stdout 2> runs/offload/run-1.stderr

read_generated_tokens() {
  .venv/bin/python - "$1" <<'PYTHON'
import json
import pathlib
import sys

lines = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
report = json.loads(next(line for line in reversed(lines) if line.strip()))
print(report["generated_tokens"])
PYTHON
}

BASELINE_TOKENS=$(read_generated_tokens runs/resident/run-1.stdout)
CANDIDATE_TOKENS=$(read_generated_tokens runs/offload/run-1.stdout)
test "$BASELINE_TOKENS" = "$CANDIDATE_TOKENS" || {
  echo "generated-token mismatch: resident=$BASELINE_TOKENS offload=$CANDIDATE_TOKENS" >&2
  exit 1
}
test "$BASELINE_TOKENS" -gt 0
EXPECTED_TOKENS=$BASELINE_TOKENS

.venv/bin/ft mlx-compare \
  --baseline runs/resident/run-1.stdout \
  --candidate runs/offload/run-1.stdout \
  --expected-tokens "$EXPECTED_TOKENS" \
  --no-throughput-gate \
  --require-output-match \
  --max-peak-memory-ratio 1.05 \
  --write-summary runs/resident-vs-offload.json
```

`--max-tokens 32` is only an upper bound because mlx-lm stops at EOS. Both runs must report the same positive `generated_tokens`; set `EXPECTED_TOKENS` to that exact value. The comparator then rejects either a shortened/mismatched run instead of silently comparing different decode lengths.

This recipe disables the throughput gate because it checks output correctness across two intentionally different runtime paths. Do not require identical residency here. For ordinary performance A/B tests, keep the throughput gate and use `--require-same-residency` so both groups exercise one identical path.

## 5. Controlled performance A/B

Run fresh processes in alternating order, for example `A,B,B,A`, and capture stdout and stderr separately. Use at least three valid runs per group for a claim intended to become a default.

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

Exit codes are stable for harnesses:

- `0`: all gates passed;
- `1`: logs were valid, but one or more gates failed;
- `2`: invalid arguments or malformed/missing logs.

The comparison JSON includes the raw run records, medians, ratios, thresholds, output hashes, and explicit violations.

## 6. Memory-pressure validation

Unified-memory failures can degrade the whole desktop before Python receives a clean exception. Close unrelated workloads and keep an explicit safety envelope:

```bash
--memory-limit-gb 10 --system-headroom-gb 3.2
```

A valid pressure test must show that:

1. the process either completes or exits through the clean MLX memory error path;
2. the requested macOS headroom remains available;
3. cache shrink does not change deterministic output;
4. the final report and stderr log are retained.

## Current automation gap

There is still no repository-managed Apple-Silicon runner. Real-checkpoint generation, Metal kernel compatibility, and performance thresholds remain manual until that lane exists. The generic CI workflow intentionally tests only the torch-free control plane.
