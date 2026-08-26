# Установка

FreeToken-MLX содержит два разных runtime-пути. Выберите один; Linux/CUDA и macOS/MLX не являются взаимозаменяемыми.

## Linux + CUDA

Нужны Linux, совместимый GPU NVIDIA и драйвер CUDA. В корне репозитория:

```bash
git clone https://github.com/agisota/freetoken-mlx.git
cd freetoken-mlx
uv venv --python 3.11 --seed
uv pip install -e '.[accel,dev]'
.venv/bin/ft --help
```

Последняя команда печатает список подкоманд `ft`. Для запуска сервера требуются локальные веса модели и GPU; см. [quickstart](quickstart.md).

Корневой `install.sh` — наследованный Linux/NVIDIA-install helper. Не запускайте его на macOS.

## macOS + Apple Silicon

Нужны macOS arm64 и Python 3.12:

```bash
git clone https://github.com/agisota/freetoken-mlx.git
cd freetoken-mlx
uv venv --python 3.12 --seed
uv pip install -e '.[mlx,dev]'
.venv/bin/ft generate --help
```

Ожидаемый результат последней команды — справка команды `generate`. Она не скачивает веса.

MLX backend поддерживает только `model_type=qwen2_moe`, протестирован с `Qwen/Qwen1.5-MoE-A2.7B`, и рассчитан на batch size 1. См. [README](../README.md) и [models](models.md).

## Проверка исходников

На установленной платформенной поверхности:

```bash
.venv/bin/python -m compileall -q python
```

Команда завершается без вывода при успешной компиляции. Для тестовой матрицы см. [tests/README.md](../tests/README.md).
