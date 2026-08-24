# Быстрый старт

Предполагается, что FreeToken установлен — см. [install.md](install.md).

## Запуск сервера

```bash
ft serve --model ~/models/Qwen3.6-35B-A3B
```

`--model` принимает и id репозитория Hugging Face. Всё остальное — dtype,
бэкенды attention и MoE, размеры кэшей, парсеры tool-call и reasoning —
определяется автоматически из чекпоинта и GPU; флаги см. в [cli.md](cli.md).
Сервер готов, когда в логе появляется `API server is ready to serve on 127.0.0.1:1919`.

## Отправка запроса

Проверьте, какая модель обслуживается:

```bash
curl http://127.0.0.1:1919/v1/models
```

Затем используйте этот id в поле `model`:

```bash
curl http://127.0.0.1:1919/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.6-35B-A3B",
    "messages": [{"role": "user", "content": "What is a Mixture-of-Experts model?"}],
    "max_tokens": 256,
    "stream": true
  }'
```

FreeToken обслуживает OpenAI API (`/v1/chat/completions`, `/v1/responses`,
`/v1/models`) и Anthropic API (`/v1/messages`,
`/v1/messages/count_tokens`), поэтому клиентская библиотека любого из них
работает, если указать её base URL на сервер.

## Чат в терминале

Простой TUI для взаимодействия с сервером:

```bash
ft shell                                    # attach to the server above
ft shell --model ~/models/Qwen3.6-35B-A3B   # start an engine and chat, one process
```

`/help` перечисляет команды внутри шелла. Режим подключения не требует GPU,
поэтому он также управляет сервером на другой машине (`--server URL`).

## Использование кодинг-агента

```bash
ft launch claude   # claude / codex / dsh / hermes / openclaw / opencode
```

Записывает конфиг провайдера этого агента, устанавливает его CLI при
отсутствии и запускает его против вашего сервера. `--dry-run` показывает
изменения без применения.
