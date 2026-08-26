# Бенчмарки

Запускайте из корня репозитория. Бенчмарки относятся к Linux/CUDA, если script не говорит обратного; закрепите GPU через `CUDA_VISIBLE_DEVICES=0` и сохраните model revision, backend, prompt, token count и raw JSON.

| Script | Назначение | Команда |
| --- | --- | --- |
| `bench_decode_moe.py` | End-to-end streaming decode throughput для выбранных MoE backends. | `python benchmarks/bench_decode_moe.py --model /path/to/model --backend offload,cpu,hybrid` |
| `bench_load_weight_generic.py` | Загрузка expert bank для sequential, O_DIRECT и FTW layouts. Только Linux; FTW по умолчанию в `/var/tmp`. | `python benchmarks/bench_load_weight_generic.py --model /path/to/model` |
| `bench_offload_cache_copy.py` | Synthetic cost копирования cache без checkpoint. | `python benchmarks/bench_offload_cache_copy.py` |

Для host-RAM versus PCIe bandwidth и выбора backend используйте:

```bash
ft bench bw --help
```

Бенчмарк пишет JSON profile, который читает engine. Сравнивайте fresh processes с fixed inputs и median, а не один самый быстрый run.
