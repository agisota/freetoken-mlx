# Аудит FreeToken-MLX

Дата: 2026-08-24; ревизия дефектов: 2026-08-26.

## Область и метод

Проверены структура репозитория, MLX path, memory/offload behavior, tests,
packaging/release surface, documentation и workflow. Source inspection не
заменяет запуск с model weights: ниже отдельно указано, что подтверждено
локально, а что требует Apple-Silicon или CUDA evidence.

## Краткий вердикт

MLX path — реальная реализация expert offload, а не adapter: routing IDs управляют
cache admission, misses материализуют projections из safetensors, MLX вычисляет
expert, eviction меняет resident state, а memory budget и pressure влияют на
решение runtime.

Главный риск был в границах продукта: package и release surfaces всё ещё несут
наследие CUDA server, тогда как MLX backend имеет узкий contract. Главные
изменения этой ветки — явная матрица платформ, functional repository URLs,
hosted validation вместо неподтверждённого nightly publishing и tests для
documentation/network boundary.

Главный результат аудита — пять подтверждённых source-анализом дефектов
жизненного цикла и валидации входа; они исправляются в этой ветке, сводка в
разделе «Исправленные дефекты».

## Проверенные факты

- ServerArgs.server_host по умолчанию равен 127.0.0.1; default CORS allowlist
  содержит только local Tauri/Vite origins и не содержит wildcard.
- Daemon строит subprocess argv списком и не использует shell.
- _ShardPool.take() удаляет selected arrays из retained mapping; MLX tests
  проверяют это memory-lifetime правило.
- MLX backend принимает только model_type=qwen2_moe; текущая проверенная model —
  Qwen/Qwen1.5-MoE-A2.7B, batch size 1.
- Release workflow с тегами ссылается на существующие scripts. Ночной workflow
  больше не требует недоказанных self-hosted runner и publishing secret.
- Documentation contract проверяет visible CLI command table против фактического
  ft --help, clone URLs, relative Markdown links, fences и workflow script paths.

## Исправленные расхождения

| Приоритет | Наблюдение | Исправление |
| --- | --- | --- |
| P0 | Source install clone указывал на upstream вместо этого форка. | Все functional clone URLs используют agisota/freetoken-mlx. |
| P0 | Nightly workflow зависел от self-hosted runner и secret, которые не подтверждены доступным GitHub inventory. | Workflow стал hosted validation; tag publishing остаётся в release.yml. |
| P0 | MLX tests падали при collection без MLX. | MLX-only modules используют explicit importorskip с причиной. |
| P1 | README и docs смешивали Linux/CUDA и Apple-Silicon claims. | Добавлены support matrix, platform-specific install/test paths и limitations. |
| P1 | Документация могла тихо расходиться с dispatcher и workflow. | Добавлен tests/test_docs_contract.py. |

## Исправленные дефекты

Каждый дефект подтверждён чтением исходников с конкретным триггером
вход→отказ; измерений с весами модели не проводилось.

| # | Дефект | Триггер | Исправление | Регрессионный тест |
| --- | --- | --- | --- | --- |
| 1 | `POST /bench/run`: дочерний процесс бенчмарка переживал отмену SSE-стрима. | Клиент отключается во время stdout-стрима; финализация генератора не имела `finally`, ребёнок продолжал держать GPU и писать общий profile. | Ребёнок в отдельной процессной группе; `try/finally` вокруг чтения, ожидания и выдачи результата; SIGTERM → 5 s grace → SIGKILL → гарантированный reap. | `tests/daemon/test_daemon_app.py` |
| 2 | `stream_with_cancellation`: отмена без гарантии доставки `AbortMsg`. | `asyncio.create_task(self.abort_user(uid))` без владельца — teardown запроса мог завершиться раньше доставки, исключения задачи становились предупреждениями. | `await asyncio.shield(...)` в обработчике отмены; идемпотентный `abort_user` (ровно один `AbortMsg`). | `tests/server/test_stream_cancellation.py` |
| 3 | Гонка при конкурентном `POST /v1/cache/rebuild`. | Оба запроса видели `maintenance_state == "serving"` до установки `"rebuilding"`; единственный слот `_pending_rebuild` перезаписывался. | Резервирование под `asyncio.Lock` с повторной проверкой состояния; `CacheRebuildRequest.timeout` ограничен `0 < t <= 3600`. | `tests/server/test_rebuild_maintenance.py` |
| 4 | Невалидные sampling-параметры молча доходили до движка. | `resolve_sampling` пересылал произвольные float; глубокая нормализация клампила вместо ошибки клиента. | Валидация `temperature` (конечное, >= 0), `top_p` (0, 1], `top_k` (`-1` или >= 1) до `SamplingParams`; адаптеры возвращают клиентскую ошибку 400/422. | Параметризованные тесты в `tests/server/test_openai_api.py`, `tests/server/test_anthropic_api.py`, `tests/server/test_responses_api.py` |
| 5 | Daemon-модели запросов принимали некорректный ввод. | `StartBody.port` без границ сокета доходил до мутации lifecycle; `args`-поля использовали изменяемые list-literal defaults. | `port` — optional int `1..65535`; все три `args` — `Field(default_factory=list)`. | `tests/daemon` (модельные и route-тесты) |

Состояние валидации на 2026-08-26: пункты 4 и 5 реализованы в рабочем дереве,
их модули запускаются; пункты 1–3 реализуются в этой ветке, указанные
регрессионные модули должны проходить перед merge. Прогонов на CUDA с весами
или на real-model MLX для этих исправлений не выполнялось (см. «Ограничения
evidence»).

## Ограничения evidence

Не выполнены в этой среде:

- real MLX model generation, memory-pressure/OOM runs и Metal kernel benchmarks;
- CUDA 13 runtime, model-weight e2e и needs_weights tests;
- performance regression gate на Apple-Silicon runner.

Наличие source/test contracts не является заменой этим измерениям.

## Документация исследований

paper/*.md проверены на executable commands, local links, identifiers, URLs,
metrics и cited data. Их исследовательские claims и числа не переписывались:
план запрещает менять такую evidence без свежей валидации. Operator-facing
README и guides очищены от diary-like prose отдельно.

## Рекомендуемый порядок дальнейшей работы

1. Добавить macOS arm64 MLX CI с маленьким checkpoint smoke и raw JSON artifact.
2. Добавить persistent prompt/KV controls и separate prefill/decode metrics.
3. Записать route traces, miss latency и offline cache simulator.
4. Сравнить Hugging Face shard layout с expert-oriented contiguous storage.
5. Проверить route prefetch и mx.compile() только на stable subgraphs.
6. Рассматривать multi-Mac/JACCL после source-backed time breakdown.
