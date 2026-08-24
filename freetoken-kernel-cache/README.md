# freetoken-kernel-cache

`freetoken-kernel-cache` — сопутствующее wheel-расширение для `freetoken`. Оно поставляет
предсобранные разделяемые библиотеки TVM FFI-ядер, чтобы типовые пути рантайма могли загружать
`.so`-файлы напрямую вместо компиляции через nvcc при первом использовании.

## Сборка обоих wheel-пакетов

Используйте верхнеуровневый хелпер, чтобы собрать runtime-wheel и соответствующий kernel-cache
wheel одной командой:

```bash
scripts/build-release-wheels.sh
```

По умолчанию артефакты пишутся в `dist/`:

```text
dist/freetoken-<version>-cp312-cp312-linux_x86_64.whl
dist/freetoken_kernel_cache-<version>+cu130-py3-none-linux_x86_64.whl
```

Полезные настройки:

```bash
FREETOKEN_BUILD_OUT_DIR=/tmp/freetoken-dist \
FREETOKEN_BUILD_PYTHON=.venv/bin/python \
scripts/build-release-wheels.sh
```

Чтобы собрать только подмножество кэша для быстрой проверки:

```bash
FREETOKEN_KERNEL_CACHE_SPECS=freetoken__store_1024_128_1_false \
scripts/build-release-wheels.sh
```

## Содержимое wheel-пакета

В cache-wheel лежит по каталогу на каждое ядро:

```text
freetoken_kernel_cache/
  jit_cache/
    freetoken__store_1024_128_1_false/
      freetoken__store_1024_128_1_false.so
```

Во время выполнения `freetoken.kernel.utils.load_jit()` и `load_aot()` ищут
`freetoken_kernel_cache.get_jit_cache_dir()` и загружают
`<jit_cache>/<kernel_name>/<kernel_name>.so`, прежде чем откатиться к JIT.

## Установка

`install.sh` устанавливает оба wheel-пакета. Передайте оба явно:

```bash
FREETOKEN_WHEEL=dist/freetoken-0.1.1-cp312-cp312-linux_x86_64.whl \
FREETOKEN_KERNEL_CACHE_WHEEL=dist/freetoken_kernel_cache-0.1.1+cu130-py3-none-linux_x86_64.whl \
bash install.sh
```

Если cache-wheel лежит рядом с runtime-wheel, `install.sh` может автоматически найти соседний
`freetoken_kernel_cache-*.whl`, когда задан только `FREETOKEN_WHEEL`.

Для валидации релиза отключите runtime-JIT:

```bash
FREETOKEN_DISABLE_JIT=1 ft serve --model <path>
```

С этим флагом любой промах кэша немедленно приводит к ошибке вместо компиляции на лету.
