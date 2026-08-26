# Тесты

Запускайте команды из корня репозитория. Backend-dependent suites должны либо получить backend, либо явно пропустить тест; collection error из-за отсутствующего backend — не успешный результат.

## Матрица

| Среда | Команда | Что проверяет |
| --- | --- | --- |
| macOS arm64 + MLX | `uv run --python 3.12 --extra mlx --extra dev python -m pytest tests/test_mlx_backend.py tests/test_mlx_cache.py tests/test_mlx_cli.py tests/test_mlx_quantize_experts.py -q` | MLX loader, cache, CLI, expert quantization |
| Linux + CUDA | `uv run --extra dev python -m pytest tests/ -m 'not slow'` | CUDA/Torch runtime, server, scheduler and kernels |
| Hosted CPU CI | `PYTHONPATH=python uv run --no-project --with pytest --with fastapi --with uvicorn --with pydantic --with httpx python -m pytest tests/test_docs_contract.py tests/daemon/test_daemon_import_safety.py tests/daemon/test_daemon_serve_manager.py tests/test_mlx_cache.py tests/test_mlx_cli.py -q` | Backend-neutral CLI, daemon, cache и documentation contracts |
| Any source checkout | `PYTHONPATH=python python -m freetoken.cli --help` | Lazy CLI dispatcher |

## Markers

- `slow`: longer deterministic work; exclude with `-m 'not slow'`.
- `needs_weights`: requires a local model fixture; exclude with `-m 'not needs_weights'`.

## Local-weight inputs

| Variable | Used by |
| --- | --- |
| `FREETOKEN_TEST_MODEL` | AIME evaluation and fallback real-model tests |
| `FREETOKEN_AIME_SERIES` | AIME dataset: `aime24`, `aime25` or `aime26` |
| `FREETOKEN_AIME{24,25,26}_JSONL` | JSONL path for the selected AIME series |
| `FREETOKEN_AIME_REQ` | Request id, comma-separated ids or `all` |
| `FREETOKEN_AIME_MAX_TOKENS` | Token budget per sample |
| `FREETOKEN_AIME_SAMPLES` | Number of samples |
| `FREETOKEN_AIME_MIN_FREE_GIB` | Required free GPU memory |
| `FREETOKEN_TEST_MOE_CACHE_SIZE` | Enables offload MoE for the fixture |
| `FREETOKEN_TEST_MEM_RATIO` | Offload memory ratio |
| `FREETOKEN_REBUILD_TEST_MODEL` | Small local model for cache-rebuild server test |
| `FREETOKEN_GEMMA4_GGUF_GLOB` | Local Gemma-4 GGUF glob |

Run the containing test module after a production fix. Full real-weight tests are opt-in; never make CI download an unpinned checkpoint implicitly.
