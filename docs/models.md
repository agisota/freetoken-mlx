# Поддерживаемые модели

FreeToken загружает HF-чекпоинты safetensors напрямую (плюс нативный GGUF для
Gemma-4). Перечисленные ниже чекпоинты проверены — под них настроены готовые
ядра; работают и другие чекпоинты тех же архитектур.

| Модель | HF-чекпоинты |
|---|---|
| DeepSeek-V4 | [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) |
| GLM-5.2 | [nvidia/GLM-5.2-NVFP4](https://huggingface.co/nvidia/GLM-5.2-NVFP4) |
| GLM-4.7 | [nvidia/GLM-4.7-NVFP4](https://huggingface.co/nvidia/GLM-4.7-NVFP4) |
| Qwen3.6 / Qwen3.5 MoE | [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) ([-FP8](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8)), [nvidia/Qwen3.6-35B-A3B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4), [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) ([-FP8](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-FP8)) |
| Qwen3.6 dense | [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) ([-FP8](https://huggingface.co/Qwen/Qwen3.6-27B-FP8)), [nvidia/Qwen3.6-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-27B-NVFP4) |
| Qwen3-MoE | [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) |
| gpt-oss | [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b), [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) |
| Gemma-4 | [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it), [nvidia/Gemma-4-26B-A4B-NVFP4](https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4), [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it), [nvidia/Gemma-4-31B-IT-NVFP4](https://huggingface.co/nvidia/Gemma-4-31B-IT-NVFP4) .. |
| MiniMax-M2.5 | [nvidia/MiniMax-M2.5-NVFP4](https://huggingface.co/nvidia/MiniMax-M2.5-NVFP4) |
| Muse-Glimmer | [meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B), [RedHatAI/Muse-Glimmer-30B-NVFP4](https://huggingface.co/RedHatAI/Muse-Glimmer-30B-NVFP4) |

## MoE-бэкенды

`ft serve --moe-backend {auto,fused,offload,cpu,hybrid}`:

- **fused** — эксперты резидентно на GPU (нужна VRAM); никогда не выбирается автоматически.
- **offload** — эксперты живут в host RAM, на GPU — LRU-кэш слотов экспертов;
  промахи стримятся по PCIe.
- **cpu** — промахи считаются на CPU вместо загрузки.
- **hybrid** — за шаг часть промахов забирается по PCIe, остальные считаются
  на CPU, с перекрытием. Запустите `ft bench bw` один раз на машину для
  калибровки разделения.
- **auto** — dense-модели всегда разрешаются в `fused`; MoE-модели — в
  `offload`, с повышением до `hybrid`, когда кэшированный профиль `ft bench bw`
  это рекомендует.

## Примечания

- Конвертация `ft checkpoint` необязательна — она заранее преобразует чекпоинт
  в формат быстрой загрузки FreeToken, а `ft serve --model` определяет результат автоматически.
- Чекпоинты DeepSeek-V4 должны сохранять подкаталог `inference/config.json` —
  авторитетные аргументы модели читаются оттуда.
- Мультимодальные чекпоинты обслуживаются только текстом.
