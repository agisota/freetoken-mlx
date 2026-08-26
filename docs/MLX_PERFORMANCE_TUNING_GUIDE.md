# Тюнинг производительности MLX

Документ фиксирует измеренные эффекты FreeToken-MLX на Apple Silicon и условия,
без которых сравнение недействительно. Референсная конфигурация:
`Qwen/Qwen1.5-MoE-A2.7B`, batch size 1, Apple M4 с 16 GB unified memory.

## Главный вывод

Если модель безопасно помещается в память, выбирайте native `mlx-lm`.
FreeToken-MLX нужен для capacity: он удерживает только routed experts и делает
иначе OOM checkpoint запускаемым. В offload path главный расход — bytes экспертов
в storage/page cache/unified memory, а не одна операция GEMV.

Приоритет оптимизации:

1. уменьшить bytes экспертов;
2. уменьшить cache misses;
3. уменьшить shard/page-in overhead;
4. перекрыть независимую работу;
5. уменьшить Metal→CPU synchronization;
6. затем оптимизировать isolated kernels.

## Контракт бенчмарка

Фиксируйте checkpoint и quantization layout, prompt/template, token count, batch
size, residency, MLX memory limit, system headroom, cache budget/policy, shard
cache, profile switches, draft model и версии MLX/mlx-lm. Используйте fresh
processes и порядок A,B,B,A; храните все raw runs и median, а не лучший прогон.

Рекомендуемый envelope для Mac 16 GB:

```bash
.venv/bin/ft generate \
  --backend mlx \
  --model Qwen/Qwen1.5-MoE-A2.7B \
  --prompt Hello \
  --raw-prompt \
  --max-tokens 32 \
  --profile performance \
  --residency offload \
  --memory-limit-gb 10 \
  --system-headroom-gb 3.2 \
  --expert-cache-budget-gb 4 \
  --quiet-cache \
  --detailed-timing
```

Отдельно измеряйте 32 tokens, 128 tokens и prompt-heavy case. Policy, полезная
на длинном decode, может не помочь на коротком.

## Метрики

Сохраняйте wall time, tok/s, output или его hash, active/cache/peak MLX memory,
cache hits/misses/loads/evictions, shard opens/hits, residency decision/budgets,
speculative acceptance, detailed timing, hardware/OS/MLX versions и cooldown
metadata. `route_sync_ms` — не только Python overhead: `indices.tolist()`
синхронизирует Metal с CPU.

## Подтверждённые решения

### Native residency при достаточной памяти

Offload — механизм capacity, не универсальный fast path. Selector должен
оставлять native `mlx-lm`, когда peak укладывается в working set, physical
headroom и pressure budget.

### Bounded lazy evaluation

Граф на все MoE layers удерживает evicted arrays живыми и может привести к OOM.
Текущий performance profile вычисляет граф каждые восемь MoE layers. В референсном
BF16 run это изменило decode примерно с 1.43 до 1.88 tok/s при bounded memory.

### Shared-expert overlap

После определения router IDs shared expert независим от CPU cache admission.
Его раннее выполнение давало небольшой повторяемый выигрыш. Опция должна
оставаться отключаемой при изменениях MLX scheduler.

### Parsed quantized shard mappings

Quantized shards содержат packed weights и scale/bias metadata. Полезная policy:
удерживать до восьми mappings для quantized checkpoints, но удалять consumed array
из mapping при materialization. Так cache eviction действительно освобождает expert.

В записанных run при fixed 600-expert cache full-Q4 вырос примерно с 3.6 до 5.0
tok/s, BF16-dense/Q4-expert — с 3.20 до 4.42 tok/s. BF16 default остаётся
консервативным.

### Квантизация routed experts

Mixed BF16-dense/Q4-expert показал 4.49 tok/s при 7.93 GB peak против 1.88 tok/s
и 5.83 GB у BF16 offload. Меньше bytes на expert означает меньше стоимость miss и
больше resident entries при том же budget.

## Неподтверждённые или отклонённые подходы

- Изолированный GEMV выиграл около 19%, но end-to-end decode изменился примерно
  на 1%; kernel не был главным bottleneck.
- Batching packed Q4 matmul добавил packing/copy/shape overhead и регрессировал.
- Unlimited lazy graph удерживал evicted arrays и приводил к OOM.

Не возвращайте эти подходы без полного end-to-end measurement, identical output и
peak-memory guard.

## Следующая очередь работы

1. recorder route traces и miss-latency;
2. persistent prompt cache, rotating/quantized KV и prefill controls;
3. expert-oriented contiguous storage;
4. route-based prefetch с учётом wasted bytes;
5. `mx.compile()` только для stable subgraphs;
6. device-side resident/miss lookup;
7. multi-Mac/JACCL после доказательства, что local I/O и sync больше не доминируют.

## Ограничения

Evidence охватывает одну модель, один M4, batch size 1 и ограниченный набор
prompts. Page cache, thermal state и system pressure меняют абсолютное время.
Новые performance claims требуют machine-readable raw artefacts и CI gate на
Apple Silicon; этот Linux hosted workflow не заменяет такую проверку.
