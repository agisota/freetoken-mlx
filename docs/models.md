# Модели и совместимость backend

## Apple Silicon MLX

MLX backend принимает только `model_type=qwen2_moe`. Проверенный target:

| Backend | Model | Constraints |
| --- | --- | --- |
| MLX | `Qwen/Qwen1.5-MoE-A2.7B` | macOS arm64, batch size 1, local generation |

Не выводите более широкую MLX support из CUDA registry. MLX loader отвергает другой `model_type` до allocation weights.

## Linux/CUDA runtime registry

Server registry использует Hugging Face `architectures`. Сейчас зарегистрированы:

| Family | Registered architecture examples |
| --- | --- |
| Llama / Mistral | `LlamaForCausalLM`, `MistralForCausalLM`, `Mistral3ForConditionalGeneration` |
| Qwen | `Qwen2ForCausalLM`, `Qwen3ForCausalLM`, `Qwen3MoeForCausalLM`, `Qwen3_5ForConditionalGeneration`, `Qwen3_5MoeForConditionalGeneration` |
| DeepSeek / GLM | `DeepseekV4ForCausalLM`, `Glm4MoeForCausalLM`, `GlmMoeDsaForCausalLM` |
| MiniMax | `MiniMaxM2ForCausalLM`, `MiniMaxM3SparseForConditionalGeneration`, `MiniMaxM3SparseForCausalLM` |
| Gemma | `Gemma4ForCausalLM`, `Gemma4ForConditionalGeneration`, `Gemma4UnifiedForConditionalGeneration`, `Gemma4GGUFForCausalLM` |
| Other | `MuseGlimmerForConditionalGeneration`, `GptOssForCausalLM` |

Registered architecture не гарантирует запуск. Checkpoint format, quantization, CUDA compute capability, optional native packages и доступная память определяют, загрузится ли конкретная model.

## Проверка checkpoint

Прочитайте `config.json` до download или conversion больших weights:

```bash
jq '.architectures, .model_type' /path/to/model/config.json
```

Для MLX `model_type` должен быть `qwen2_moe`. Для CUDA entry в `architectures` должен совпадать с registry выше. Финальная проверка — `ft serve --help` и соответствующие tests.
