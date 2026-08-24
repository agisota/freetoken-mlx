# benchmarks

Запуск из корня репозитория с `PYTHONPATH=python:.`, привязка к одному GPU
(`CUDA_VISIBLE_DEVICES=0`). Подробности — в `--help` / docstring каждого скрипта.

**`bench_decode_moe.py`** — tok/s декода при bs=1 для обслуживаемой MoE-модели. Запускает `ft serve`
для каждого бэкенда и замеряет приход токенов по стриминговому `/v1/chat/completions`, так что
в числа входит весь путь обслуживания. Промпт AIME-25, сэмплирование по рекомендациям чекпоинта.

```bash
python benchmarks/bench_decode_moe.py --model /path/to/model --backend offload,cpu,hybrid
```

**`bench_load_weight_generic.py`** — время загрузки банка экспертов: последовательная загрузка vs параллельный O_DIRECT
vs предупакованный FTW, каждый режим в своём subprocess. Только Linux; FTW размещается в
`/var/tmp` (`--ftw-dir` переопределяет; примерно размером с чекпоинт).

```bash
python benchmarks/bench_load_weight_generic.py --model /path/to/model
```

**`bench_offload_cache_copy.py`** — синтетический бенчмарк (без чекпоинта): стоимость копирования экспертов
за слой при декоде (`ensure_experts` + `copy_missing`), перебор по раскладке банка x слотам кэша x
размеру батча x доле промахов.

```bash
python benchmarks/bench_offload_cache_copy.py
```

Для пропускной способности RAM хоста vs PCIe и выбора бэкенда offload/hybrid используйте вместо этого
`ft bench bw` — он записывает JSON-профиль, который читает движок.
