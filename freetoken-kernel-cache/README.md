# freetoken-kernel-cache

Этот companion Linux wheel содержит prebuilt TVM FFI shared libraries для CUDA runtime FreeToken. macOS MLX backend его не использует.

## Сборка

Запускайте из корня репозитория в release build environment:

```bash
scripts/build-release-wheels.sh
```

Artifacts появляются в `dist/`:

```text
dist/freetoken-<version>-cp312-cp312-linux_x86_64.whl
dist/freetoken_kernel_cache-<version>+cu130-py3-none-linux_x86_64.whl
```

Optional inputs:

```bash
FREETOKEN_BUILD_OUT_DIR=/tmp/freetoken-dist \
FREETOKEN_BUILD_PYTHON=.venv/bin/python \
scripts/build-release-wheels.sh
```

## Установка и валидация

```bash
FREETOKEN_WHEEL=dist/freetoken-<version>-cp312-cp312-linux_x86_64.whl \
FREETOKEN_KERNEL_CACHE_WHEEL=dist/freetoken_kernel_cache-<version>+cu130-py3-none-linux_x86_64.whl \
bash install.sh
FREETOKEN_DISABLE_JIT=1 ft serve --model /path/to/model
```

С `FREETOKEN_DISABLE_JIT=1` отсутствующий cache entry завершает запуск ошибкой вместо compile во время runtime. Это release validation mode.
