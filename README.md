# FreeToken-MLX

FreeToken expert offload для Apple Silicon.

> [!IMPORTANT]
> FreeToken-MLX — независимо поддерживаемый форк
> [FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken), а не
> официальный релиз FlashML. В репозитории сохранены исходный CUDA/Torch runtime
> и архитектура expert bank; MLX-реализация находится в
> `python/freetoken/mlx_backend.py`.

## Назначение

MLX backend решает узкую задачу: запустить `Qwen/Qwen1.5-MoE-A2.7B` на Mac, где
полный checkpoint небезопасно держать в unified memory. Dense/shared weights
остаются резидентными, а выбранные routed experts материализуются по требованию
через ограниченный общий cache. Если checkpoint безопасно помещается полностью,
runtime выбирает native `mlx-lm`, а не offload.

## Матрица поддержки

| Поверхность | Поддержка |
| --- | --- |
| Linux/CUDA engine | Наследованный FreeToken runtime: server, scheduler, distributed и CUDA paths. |
| macOS/Apple Silicon MLX | `model_type=qwen2_moe`, протестированный `Qwen/Qwen1.5-MoE-A2.7B`, batch size 1, локальная генерация. |
| MLX checkpoint formats | BF16, full quantized и mixed BF16-dense/quantized-expert. |
| Не входит в MLX backend | OpenAI/Anthropic server, tensor parallel, произвольные model families, persistent prompt cache. |

Не переносите утверждения CUDA registry на MLX: loader проверяет `model_type` до
выделения весов.

## Установка

### macOS + MLX

```bash
git clone https://github.com/agisota/freetoken-mlx.git
cd freetoken-mlx
uv venv --python 3.12 --seed
uv pip install -e '.[mlx,dev]'
```

### Linux + CUDA

```bash
git clone https://github.com/agisota/freetoken-mlx.git
cd freetoken-mlx
uv venv --python 3.11 --seed
uv pip install -e '.[accel,dev]'
```

Корневой `install.sh` относится к наследованному Linux/NVIDIA пути и не является
установщиком MLX.

## Быстрый старт MLX

На Apple Silicon команда скачивает model при первом запуске, печатает текст и
завершающий JSON-отчёт:

```bash
.venv/bin/ft generate \
  --backend mlx \
  --model Qwen/Qwen1.5-MoE-A2.7B \
  --prompt Hello \
  --raw-prompt \
  --max-tokens 16 \
  --batch-size 1
```

`--residency auto` выбирает native resident MLX, если оценка пика, system
headroom и текущее pressure это допускают; иначе выбирает expert offload. Для
контролируемого сравнения укажите `--residency offload`, `--memory-limit-gb`
и `--system-headroom-gb`.

## API и CLI

`ft serve` поднимает локальный HTTP-сервер с эндпоинтами в форматах OpenAI и
Anthropic; `ft ctl` управляет запущенным сервером, `ft shell` даёт интерактивный
терминал. Полный набор подкоманд: `ft --help`; справочник флагов и переменных
окружения: [docs/cli.md](docs/cli.md).

## Измерения

Результаты ниже получены на Apple M4 с 16 GB unified memory, batch size 1. Это
данные конкретной конфигурации, а не гарантия на другом железе. Первоисточник
цифр — `paper/data/measurements.json`, `artifact_commit`
`7b3feed2cc26d277254d29641d819ee133293deb`; календарная дата измерений в
артефакте не зафиксирована.

| Путь | Пропускная способность | Peak MLX memory |
| --- | ---: | ---: |
| native `mlx-lm`, BF16 | Metal OOM до token 1 | заявлено 28.63 GB |
| FreeToken offload, BF16 | 1.88 tok/s | 5.83 GB |
| FreeToken offload, BF16, больше safe cache | 2.05 tok/s | 9.03 GB |
| BF16 dense + Q4 experts | 4.49 tok/s | 7.93 GB |
| native resident full-Q4 | выбирается автоматически, если безопасно | около 8.8 GB |

Cache sweep от 55 до 275 experts сократил misses на 31.4%, а медианная
пропускная способность изменилась с 3.02 до 3.16 tok/s. Надёжный вывод — больше
cache обменивает память на меньшее число загрузок; выигрыш tok/s в этой серии
умеренный.

## Как работает MLX путь

1. **Выбор резидентности.** Перед загрузкой runtime читает metadata, оценивает
   peak memory, резервирует память для macOS и optional draft model, затем
   учитывает pressure.
2. **Expert bank.** Вместо stack всех experts loader хранит ссылки
   `(shard path, tensor key)`. На miss материализуются только `up_proj`,
   `gate_proj` и `down_proj` выбранного expert.
3. **Кэш.** `MLXOffloadMoeCache` общий для layers и сообщает hits, misses,
   loads, evictions и capacity. Политики: `lru`, `slru`, `auto`.
4. **Память.** Runtime ограничивает MLX working set и allocator cache; при
   превышении 85% configured limit уменьшает expert cache и очищает MLX cache.

Подробная методика и ограничения: [MLX performance tuning](docs/MLX_PERFORMANCE_TUNING_GUIDE.md).

## Квантизация и speculative decoding

`ft mlx-quantize-experts` сохраняет dense/attention/router/shared-expert
weights и квантует только routed experts. Поддерживаются affine bits
`2,3,4,5,6,8`, group size `32,64,128` и optional overrides для
`up_proj`, `gate_proj`, `down_proj`.

Speculative decoding требует tokenizer-compatible MLX draft model. Draft остаётся
резидентным и вычитается из expert-cache budget до запуска.

## Проверка

На macOS arm64 с MLX extra:

```bash
.venv/bin/python -m pytest \
  tests/test_mlx_backend.py tests/test_mlx_cache.py \
  tests/test_mlx_cli.py tests/test_mlx_quantize_experts.py -q
```

Проверка завершает 32 MLX unit/behavior tests на референсной среде. Полная
матрица приведена в [tests/README.md](tests/README.md).

## Ограничения и дальнейшая работа

Главные текущие задержки — Metal→CPU sync routing IDs, miss service latency
safetensors/page cache и отсутствие controls для prompt/KV cache. Следующие
эксперименты: trace маршрутов, contiguous expert storage, prefetch, `mx.compile()`
на stable subgraphs и только затем multi-Mac scaling.

## Лицензия

Apache-2.0. Статус форка и attribution к upstream сохранены в этом README и
license-артефактах.
