# Установка

## Требования

- Linux x86_64, GPU NVIDIA, драйвер r580+ (CUDA 13)
- Python >= 3.10; рекомендуется [uv](https://docs.astral.sh/uv/) (подойдёт и
  обычный `pip` + `venv`)

## Способ 1: установка из PyPI

```bash
uv venv && source .venv/bin/activate
uv pip install "freetoken[accel]"
```

CUDA-ядра JIT-компилируются при первом использовании; нужен CUDA 13 toolkit с
`nvcc` на PATH.

## Способ 2: установка из исходников

```bash
git clone https://github.com/FlashML-org/FreeToken.git && cd FreeToken
uv venv && source .venv/bin/activate
uv pip install -e ".[accel]"
```

## Проверка

```bash
source .venv/bin/activate
ft --version
ft serve --model ~/path/to/Qwen3.6-35B-A3B
curl http://127.0.0.1:1919/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.6-35B-A3B","messages":[{"role":"user","content":"hi"}]}'
```

Далее — [quickstart.md](quickstart.md).
