# Справочник CLI

Запускайте справку из установленного окружения: она является источником точных флагов для установленной версии.

```bash
.venv/bin/ft <command> --help
```

## Команды верхнего уровня

| Команда | Назначение | Проверка справки |
| --- | --- | --- |
| `generate` | Локальная генерация через Apple-Silicon MLX backend. | `ft generate --help` |
| `mlx-quantize-experts` | Квантизирует только routed experts и создаёт новый checkpoint. | `ft mlx-quantize-experts --help` |
| `serve` | Запускает CUDA/Torch server с OpenAI и Anthropic API. | `ft serve --help` |
| `shell` | Открывает terminal client для server. | `ft shell --help` |
| `ctl` | Читает состояние и управляет запущенным server. | `ft ctl --help` |
| `daemon` | Управляет supervisor без импорта torch для `ft serve`. | `ft daemon --help` |
| `launch` | Готовит и запускает поддерживаемый agent client через server. | `ft launch --help` |
| `checkpoint` | Конвертирует HF safetensors checkpoint в FTW. | `ft checkpoint --help` |
| `bench` | Запускает micro-benchmark; текущая subcommand — `bw`. | `ft bench --help` |

## Проверяемые поверхности команд

### `ft generate`

Обязательны `--backend mlx` и `--model`. Defaults: `--prompt Hello`,
`--max-tokens 1`, `--batch-size 1`, `--moe-cache-size 4`. Остальные флаги —
только MLX-поверхность (residency и memory budgets, cache policy и размеры,
profiling, speculative decoding через `--draft-model`); исчерпывающий список с
defaults печатает `ft generate --help`.

### `ft mlx-quantize-experts`

Обязательны `--model` и `--output`. Defaults: `--bits 4`, `--group-size 64`; `--down-bits` optional. Допустимые bits: `2,3,4,5,6,8`; group sizes: `32,64,128`.

### `ft ctl`

Default server URL — `http://127.0.0.1:1919`, default `--timeout` — `10` seconds. `--json` печатает raw JSON. Список subcommands и их параметры печатает `ft ctl --help`.

### `ft serve`

CUDA/Torch server по умолчанию использует `--host 127.0.0.1` и `--port 1919`.
`--cors-origins` по умолчанию разрешает только local Tauri/Vite origins; пустая
строка отключает CORS, `*` разрешает любой origin. Model и accelerator options
зависят от CUDA runtime, поэтому их полный список и defaults выводит
`ft serve --help` на Linux/CUDA environment.

### `ft shell`

`--server` и `--base-url` — aliases. Если flag не указан, используется
`FREETOKEN_HOST`, иначе `http://127.0.0.1:1919`. Команда подключается к уже
работающему server и не запускает model.

### `ft daemon`

Server mode: `--host 127.0.0.1`, `--port 1900`, `--state-dir`, `--token`,
`--default-serve-port 1919`, `--serve-python`, `--grace 10.0`,
`--poll-interval 1.0`, `--oom-child-score 500`, `--no-oom`, `--auto-restart`,
`--stop-serve-on-exit`, `--log-capacity 4000`, `--setsid`, `--log-level info`.
Client verbs используют `--url` (default `FREETOKEN_DAEMON_URL` или
`http://127.0.0.1:1900`), `--token` и `--timeout`; `start`/`switch` принимают
model, optional `--port` и opaque `ft serve` args после `--`.

### `ft launch`

Первый positional argument — поддерживаемый agent. `--server` принимает origin
или `/v1` URL и без него использует `FREETOKEN_HOST` либо
`http://127.0.0.1:1919`. `--dry-run` не меняет configuration, `-y/--yes`
автоматически подтверждает prompts, `--force-reinstall` повторяет installer,
`--config` настраивает без launch, `--install-only` только устанавливает agent.

### `ft checkpoint`

Обязательны `--model` и `--out`. Defaults: `--dtype bfloat16`, `--moe-backend offload`, `--shard-gib 8.0`; без `--device` используется CUDA device 0.

### `ft bench bw`

`ft bench bw --help` документирует CPU/PCIe measurement flags. Defaults: `--threshold 2.0`, `--device 0`, `--cpu-iters 8`, `--pcie-mib 256`, `--pcie-iters 30`.

## Переменные окружения

Command help остаётся источником runtime flags. Tests используют документированные `FREETOKEN_*` inputs в [tests/README.md](../tests/README.md). Endpoint/token daemon описаны в [runtime guide](../python/freetoken/daemon/README.md).

Не переносите команду из другого backend: MLX не имеет server path, а `serve`, `checkpoint` и `bench bw` требуют CUDA/Torch surface.
