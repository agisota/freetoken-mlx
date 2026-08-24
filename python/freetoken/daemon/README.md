# `ft daemon` — простой демон-режим для FreeToken

Небольшая, надёжная управляющая плоскость **без torch**, которая владеет жизненным циклом дочернего
процесса `ft serve` и открывает управление / логи / метрики по HTTP. Движок становится постоянным
сервисом; всё, что умеет HTTP, — тонкий клиент. Этот файл — дизайн-референс.

```
client (ft ctl / curl / any HTTP client)             chat traffic → serve DIRECTLY
        │ HTTP control plane (loopback :1900)                     │
        ▼                                                         ▼
   ft daemon  ──spawn / signal / tail──▶  ft serve  (model · inference · MAY crash)
   (no torch)                              └─ /health /v1/stats  (per-serve control API)
        ▲
   systemd  Restart=always · RestartSec=1 · KillMode=process
```

## Почему без torch (единственное непреложное требование)

Демон импортирует **только** stdlib + `fastapi` + `uvicorn` (+ мелкие чтения `/proc` + опциональный
`pynvml`). Он никогда не импортирует `torch` / CUDA / `flashinfer` / `sgl_kernel`, а также что-либо из
`freetoken.server.*` (его `__init__` тянет torch) или `freetoken.utils.*` (его `__init__` тянет
transformers). Именно это делает его неубиваемым: CUDA-сбой или сегфолт нативного расширения убивает
процесс, который их загрузил, — а демон не загружает ни того, ни другого. Вся рискованная работа живёт
в изолированном дочернем процессе `ft serve`. Это требование enforced тестом-стражем импорта
`tests/daemon/test_daemon_import_safety.py`.

## Запуск сервера

```bash
ft daemon --host 127.0.0.1 --port 1900         # bare/flags = run the daemon server
# or as a service (survives logout, auto-restarts): see ft-daemon.service
```

Состояние (single-instance lock, pidfile serve для повторного усыновления, логи на каждый serve)
живёт в `--state-dir` (по умолчанию `~/.freetoken/daemon`, переопределяется `$FREETOKEN_DAEMON_DIR`).

## Управление им (`ft daemon <verb>` — свой вход, отличный от `ft ctl`)

`ft daemon` с **глаголом** — это клиент (управляет запущенным демоном по HTTP); голый `ft daemon`
запускает сервер. `ft ctl` не тронут — он адресует запущенный *serve*, а не демон.

```bash
ft daemon self                                 # daemon self-health
ft daemon start MODEL --port 1919 -- --moe-cache-auto   # args after -- go to ft serve
ft daemon status
ft daemon logs                                 # stream engine logs (SSE)
ft daemon health                               # proxied serve /health (camelCased)
ft daemon metrics                              # engine-only RAM(PSS)+VRAM footprint
ft daemon switch OTHER_MODEL                    # stop old + start new
ft daemon stop
# Recovery only: permit a degraded receipt if the failed engine cannot seal final totals.
ft daemon stop --force
```

Адресуйте нестандартный демон через `--url http://host:1900` (или `$FREETOKEN_DAEMON_URL`) и
`--token`/`$FREETOKEN_DAEMON_TOKEN`.

## HTTP API (camelCase JSON, по умолчанию loopback)

| Метод / путь | Примечания |
| --- | --- |
| `GET /health` | Самодиагностика демона; всегда отвечает, никогда не закрыт `--token`. |
| `POST /engine/start` `{model,port,args[]}` | Идемпотентен по полному `(model,port,args)`; другая конфигурация на том же порту → `409`. |
| `POST /engine/stop` `{force?:false}` | Закрыть приём запросов, drain/abort, надёжно поставить в очередь квитанцию финального учёта, затем `SIGTERM`→grace→`SIGKILL`. Сбой prepare/outbox сохраняет движок. |
| `POST /engine/switch` `{model,port,args[],force?:false}` | Одна сериализованная транзакция stop-accounting-start. |
| `GET /engine/status` | `{running,pid,model,port,uptimeS,lastExitCode,…}`; переживает любой отдельный serve. |
| `GET /engine/logs?since=` | SSE, без ANSI, tqdm-`\r` схлопнуты, replay кольца, `id:<seq>`, возобновление по `Last-Event-ID`. |
| `GET /engine/metrics` | `{ramBytes,vramBytes}` — только собственный след дерева serve. |
| `GET /engine/health` | Проксированный serve `/health` + достижимость демона. |
| `GET /engine/stats` | Проксированный serve `/v1/stats`. |
| `GET /accounting/pending` | Неподтверждённые надёжные квитанции финального учёта, воспроизводимые после падения Desktop/клиента. |
| `POST /accounting/ack` `{receiptId}` | Идемпотентно удаляет квитанцию только после того, как клиент надёжно её применил. |
| `POST /checkpoint/start\|cancel` | Контролируемый `ft checkpoint` (эксклюзив GPU: сначала останавливает serve). |

Задайте `--token` (или `$FREETOKEN_DAEMON_TOKEN`), чтобы требовать заголовок `X-FT-Token` на всём,
кроме `/health`.

К деструктивному эндпоинту serve `POST /v1/admin/prepare-stop` демон обращается только через
loopback; serve отвергает не-loopback вызывающих, даже когда его inference-API слушает на
`0.0.0.0`. `force` никогда не подразумевается: это явное решение о восстановлении, которое может
записать null-итоги для ненаблюдаемого хвоста, когда сломанный движок нельзя ни осушить, ни опросить.

## Самосохранение (в этом весь смысл)

- **Единственный владелец:** flock pidfile → не более одного демона; при старте он **повторно
  усыновляет** ещё работающий serve, записанный в pidfile (безопасно к переиспользованию PID за счёт
  start-time + идентичности argv), так что сбой демона никогда не оставляет живой движок сиротой.
- **Движок переживает демон:** по `SIGTERM` демон по умолчанию *отцепляется* (оставляет serve
  работать); движок убивает только `POST /engine/stop`. Пара к этому — `KillMode=process` в systemd.
- **OOM-политика:** демон периодически повышает `oom_score_adj` дерева serve, делая 22-ГБ serve —
  а не крошечного демона — предпочтительной жертвой ядра.
- **Никогда не блокирует / никогда не бросает наружу:** spawn/kill/`/proc`/NVML/прокси выполняются вне
  event loop; ошибки обработчиков становятся 5xx; деградировавший старт работает даже без serve /
  со устаревшим pidfile.
- **Политика падений:** крах serve фиксируется (`lastExitCode`) и сообщается, но не перезапускается
  вслепую (слепые циклы рестарта при OOM от слишком большой модели). Опционально — `--auto-restart`.
