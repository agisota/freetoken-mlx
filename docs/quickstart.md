# Быстрый старт

Сначала выполните установку для своей платформы: [Linux/CUDA или macOS/MLX](install.md).

## Узнать доступные команды

```bash
.venv/bin/ft --help
```

Ожидаемый результат — девять top-level команд: `generate`, `mlx-quantize-experts`, `serve`, `shell`, `ctl`, `daemon`, `launch`, `checkpoint` и `bench`.

## Локальная MLX-генерация

На Apple Silicon команда ниже скачивает модель при первом запуске и печатает текст, затем JSON-отчёт:

```bash
.venv/bin/ft generate \
  --backend mlx \
  --model Qwen/Qwen1.5-MoE-A2.7B \
  --prompt Hello \
  --raw-prompt \
  --max-tokens 16 \
  --batch-size 1
```

Используйте `--residency auto` по умолчанию. Для воспроизводимого offload-измерения задайте `--residency offload`, `--memory-limit-gb` и `--system-headroom-gb`.

## Преобразовать routed experts

Команда создаёт новый mixed checkpoint; исходный checkpoint не изменяется:

```bash
.venv/bin/ft mlx-quantize-experts \
  --model /path/to/qwen-moe \
  --output /path/to/qwen-moe-q4-experts \
  --bits 4 \
  --group-size 64
```

Поддерживаются affine bits `2, 3, 4, 5, 6, 8` и group size `32, 64, 128`.

## CUDA server

На Linux/CUDA запуск требует локальной модели и совместимого GPU:

```bash
.venv/bin/ft serve --model /path/to/model
curl http://127.0.0.1:1919/v1/models
```

Сервер по умолчанию слушает loopback. Последняя команда возвращает JSON с доступной моделью. Не открывайте `--host 0.0.0.0` без отдельного контроля доступа.

## Следующий шаг

- Полная справка флагов: [CLI reference](cli.md).
- Поддержка моделей: [models](models.md).
- MLX memory/cache tuning: [MLX performance tuning](MLX_PERFORMANCE_TUNING_GUIDE.md).
- Управление долгоживущим CUDA server: [`ft daemon`](../python/freetoken/daemon/README.md).
