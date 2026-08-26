# ft daemon

`ft daemon` — supervisor без импорта torch для CUDA/Torch процесса `ft serve`. Он владеет lifecycle, logs и control endpoints; inference traffic продолжает идти прямо в `ft serve`.

```text
client ──control HTTP──> ft daemon ──spawn / signal / logs──> ft serve
client ──inference HTTP──────────────────────────────────────> ft serve
```

Daemon намеренно не импортирует `torch`, CUDA и server modules. Поэтому crash native engine может завершить `ft serve`, не уронив control plane.

## Запуск

```bash
ft daemon --host 127.0.0.1 --port 1900
```

State по умолчанию находится в `~/.freetoken/daemon`; override — `--state-dir` или `FREETOKEN_DAEMON_DIR`.

## Команды управления

```bash
ft daemon self
ft daemon start /path/to/model --port 1919 -- --moe-cache-auto
ft daemon status
ft daemon logs
ft daemon health
ft daemon metrics
ft daemon switch /path/to/other-model
ft daemon stop
```

Точные arguments печатает `ft daemon --help`. Голый `ft daemon` запускает supervisor; `ft daemon <verb>` управляет уже запущенным. `ft ctl` адресует `ft serve`, а не daemon.

## HTTP граница

Default daemon endpoint — loopback `http://127.0.0.1:1900`. Для другого daemon используйте `--url` или `FREETOKEN_DAEMON_URL`. `--token` или `FREETOKEN_DAEMON_TOKEN` требует `X-FT-Token` на всех endpoints, кроме `/health`.

| Endpoint | Назначение |
| --- | --- |
| `GET /health` | Self-health daemon. |
| `POST /engine/start` | Запускает одну `ft serve` configuration. |
| `POST /engine/stop` | Останавливает engine; `force` всегда explicit. |
| `POST /engine/switch` | Serialized stop/start transition. |
| `GET /engine/status`, `/logs`, `/metrics`, `/health`, `/stats` | State engine, logs, resource use и proxied endpoints. |
| `GET /accounting/pending`, `POST /accounting/ack` | Persistent final-accounting receipts. |
| `POST /checkpoint/start\|cancel` | Controlled checkpoint conversion. |

Daemon использует single-instance lock, хранит PID/identity child process и может re-adopt matching live `ft serve` после собственного restart. Он не перезапускает crashed engine без явной настройки.

## Безопасность

`ft serve` по умолчанию bind к loopback. Daemon обращается к `POST /v1/admin/prepare-stop` только через loopback; не публикуйте ни одну control plane без authenticated boundary.
