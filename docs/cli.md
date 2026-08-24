# CLI reference

```
ft <command> [args]
```

| Команда | Назначение |
|---|---|
| `ft serve` | Запустить API-сервер (OpenAI `/v1/*`, Anthropic `/v1/messages`, Responses) |
| `ft shell` | Общаться с сервером в терминале |
| `ft ctl` | Опрашивать и управлять запущенным сервером по HTTP |
| `ft launch` | Настроить и запустить кодинг-агента против сервера |
| `ft checkpoint` | Конвертировать HF-чекпоинт в формат быстрой загрузки FTW |
| `ft bench bw` | Замерить пропускную способность CPU vs PCIe для калибровки MoE-бэкенда |

`ft --version` печатает установленную версию (без torch; nightly-колёса несут
штамп сборки `+g<sha>`, тегированные релизы — голую версию). Каждая команда
поддерживает `--help`.

## ft serve

```bash
ft serve --model <path-or-hf-id> [options]
```

`--model` — единственный обязательный флаг: dtype, бэкенд attention, MoE-бэкенд,
размер MoE-кэша, ёмкость KV, размеры CUDA-графов и парсеры tool-call/reasoning
определяются автоматически из чекпоинта и GPU.

### Модель

| Флаг | По умолчанию | Значение |
|---|---|---|
| `--model-path`, `--model` | required | Локальный каталог, id HF-репозитория или FTW-каталог (определяется автоматически) |
| `--served-model-name` | basename of `--model` | Id модели, который отдаёт `/v1/models` |

### Сервер и рантайм

| Флаг | По умолчанию | Значение |
|---|---|---|
| `--host` | 127.0.0.1 | Адрес привязки |
| `--port` | 1919 | Порт привязки |
| `--max-running-requests` | 4 | Максимум одновременно выполняемых запросов |
| `--max-output-tokens` | 32768 | Бюджет вывода по умолчанию для запросов без своего |
| `--max-seq-len-override` | из чекпоинта | Максимальная длина последовательности |
| `--max-prefill-length` | 8192 | Размер чанка chunked-prefill в токенах |
| `--cuda-graph-max-bs`, `--graph` | = max running requests | Максимальный batch size, захваченный как CUDA-графы |
| `--decode-log-interval` | 40 | Строка статуса планировщика каждые N шагов декодирования |

### KV-кэш и память

| Флаг | По умолчанию | Значение |
|---|---|---|
| `--memory-ratio` | 0.9 | Доля свободной VRAM, которую может использовать движок (веса + MoE-кэш + KV) |
| `--num-pages` / `--num-tokens` | auto | Переопределение ёмкости KV в страницах / токенах (взаимоисключающие; auto подбирает размер из VRAM, оставшейся после весов и MoE-кэша) |
| `--page-size` | 1 | Размер KV-страницы; DSV4 форсирует 128, бэкенду TRTLLM нужны 16/32/64, SWA-модели требуют 1 |
| `--cache-type` | radix | `radix` (переиспользование префиксов; SWA/GDN-aware варианты выбираются автоматически) или `naive` |
| `--attention-backend`, `--attn` | auto | `trtllm`/`fi`/`fa`/`triton`/`dsv4_sparse`/`dsa`; допустима пара `prefill,decode`; auto подбирает по модели + GPU |

### MoE offload

Что делает каждый бэкенд — см. [models.md](models.md#moe-backends).

| Флаг | По умолчанию | Значение |
|---|---|---|
| `--moe-backend` | auto | `fused`/`offload`/`cpu`/`hybrid`; auto → offload, либо hybrid при наличии профиля `ft bench bw` |
| `--moe-cache-size` / `--moe-cache-rate` / `--moe-cache-auto` | auto | Размер GPU-кэша экспертов: слоты / доля всех экспертов / подбор из свободной VRAM (взаимоисключающие; auto включён по умолчанию для семейства offload) |
| `--kv-reserve-tokens` | 8192 | Минимум KV-токенов, резервируемый до заполнения экспертами через `--moe-cache-auto` |
| `--moe-cpu-threads` | физические ядра | Потоки CPU-воркеров для исполнителя cpu/hybrid |
| `--moe-cpu-layers` | все на GPU | С `offload`: какие MoE-слои декодируются на CPU (`3,7,11`, количество или доля) |
| `--moe-hybrid-max-fetch` | auto | С `hybrid`: максимум экспертов, забираемых по PCIe на слой за шаг; остальные считаются на CPU |
| `--moe-prefill-hit-d2d` | off | Prefill: копировать попавших в кэш экспертов на устройстве, стримить только промахи (CUDA >= 13) |
| `--disable-moe-prefill-overlap` | overlap on | Отключить двухбуферное перекрытие копирования prefill |

### Поведение API

| Флаг | По умолчанию | Значение |
|---|---|---|
| `--sampling-defaults` | model | Заполнять незаданные параметры сэмплинга из `generation_config.json` чекпоинта (`none` = дефолты фреймворка) |
| `--tool-call-parser` | auto | Формат tool-call; определяется автоматически по семейству модели |
| `--reasoning-parser` | auto | Разделяет chain-of-thought на `reasoning_content`; определяется автоматически; `off` отключает |
| `--enable-cache-report` | off | Отчитываться о попаданиях prefix-кэша в блоке usage каждого ответа |

## ft shell

```bash
ft shell                                    # attach to a running server
ft shell --model ~/models/Qwen3.6-35B-A3B   # serve + chat in one process
```

- Режим подключения общается с `--server URL` (по умолчанию `http://127.0.0.1:1919`)
- `/help` внутри шелла перечисляет команды (`/think`, `/cache`, `/reset`).

## ft ctl

```bash
ft ctl [--base-url http://127.0.0.1:1919] [--timeout 10] [--json] <subcommand>
```

| Подкоманда | Endpoint | Назначение |
|---|---|---|
| `health` | `GET /health` | Статус сервера, модель, прогресс загрузки |
| `stats` | `GET /v1/stats` | Пропускная способность, латентность, VRAM, занятость пулов |
| `generate [prompt] [--max-tokens N] [--ignore-eos]` | `POST /generate` | Сырой smoke-тест completion (без chat-шаблона) |
| `cache` | `GET /v1/cache/status` | Таблица пулов кэша |
| `cache --moe N \| --kv N \| --mamba N \| --swa N [--wait 300]` | `POST /v1/cache/rebuild` | Изменение размера пулов на живом сервере без рестарта (суффиксы `k`/`m`; `--kv`/`--swa` в токенах) |
| `requests [--since N] [--limit N]` | `GET /v1/requests` | Кольцо недавних запросов |

## ft launch

```bash
ft launch {claude,codex,dsh,hermes,openclaw,opencode} [options] [-- <agent args>]
```

Определяет обслуживаемую модель через `/v1/models`, записывает конфиг
провайдера агента, устанавливает CLI агента при отсутствии и запускает его.
Облачные API-ключи (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, …) удаляются из
окружения дочернего процесса, чтобы агент не мог молча переключиться на
платный endpoint.

| Флаг | Значение |
|---|---|
| `--server URL` | Сервер, на который направить агента (по умолчанию `http://127.0.0.1:1919`) |
| `--dry-run` | Напечатать планируемые изменения конфига и команду, ничего не трогая |
| `-y`, `--yes` | Одобрять запросы установки/конфигурации |
| `--config` | Только настроить, без запуска |
| `--install-only` | Только установить CLI агента (сервер не нужен) |
| `--force-reinstall` | Перезапустить установщик агента |
| `-- <args>` | Передаются агенту дословно |

## ft checkpoint

```bash
ft checkpoint --model <hf_dir> --out <ftw_dir> [--dtype bfloat16] [--moe-backend offload] [--shard-gib 8] [--device cuda:0]
```

Конвертирует HF-чекпоинт safetensors в FTW — самодостаточный формат быстрой
загрузки FreeToken; укажите выходной каталог в `ft serve --model`. `--moe-backend
offload` (по умолчанию) упаковывает экспертов в offload-банки; `--moe-backend triton`
оставляет их плотными для resident-обслуживания. Оговорки FTW см. в
[models.md](models.md#notes).

## ft bench bw

```bash
ft bench bw                       # once per machine
ft bench bw --dtype nvfp4,bf16    # only the formats you serve
```

Замеряет пропускную способность host-RAM vs PCIe реальными ядрами cpu/offload
MoE и пишет профиль (`~/.cache/freetoken/benchbw.json`), который читают `ft serve
--moe-backend auto` и `--moe-hybrid-max-fetch -1`. Профили ключуются по формату
экспертов + имени GPU, поэтому профиль с другого железа игнорируется, а не
применяется ошибочно. Флаги выбора: `--dtype`, `--model`, `--formats`,
`--isa`; правило решения: `--threshold` (по умолчанию 2.0 — рекомендовать hybrid,
когда пропускная способность CPU > 2× PCIe).
