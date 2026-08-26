"""Regression tests: crafted index files must not escape the checkpoint folder."""

import json
import sys
import types

import pytest

torch = pytest.importorskip("torch", reason="checkpoint loaders require the torch runtime")

# flashlib is a Linux-only native dep (see pyproject); ShardReader's path
# resolution never touches it, so a minimal stand-in lets this module import
# on macOS while real flashlib keeps winning on Linux (setdefault).
_flashlib = types.ModuleType("flashlib")
_flashlib_kernels = types.ModuleType("flashlib.kernels")
_flashlib_slot = types.ModuleType("flashlib.kernels.slot_cache")
_flashlib_slot.N_STATS = 0

class _Stat:  # noqa: D101 - shape-only placeholder
    pass

_flashlib_slot.Stat = _Stat
sys.modules.setdefault("flashlib", _flashlib)
sys.modules.setdefault("flashlib.kernels", _flashlib_kernels)
sys.modules.setdefault("flashlib.kernels.slot_cache", _flashlib_slot)

from freetoken.checkpoint.ftw import FTWReader
from freetoken.models.loader import ShardReader


def test_ftw_index_shard_escape_refused(tmp_path):
    (tmp_path / "freetoken_weight.json").write_text(
        json.dumps(
            {
                "format": "freetoken_weight",
                "shards": [{"file": "../evil.ftw", "global_off": 0}],
                "tensors": [],
            }
        )
    )
    with pytest.raises(ValueError, match="escapes"):
        FTWReader(str(tmp_path))


def test_weight_map_shard_escape_refused(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer.w": "../evil.safetensors"}})
    )
    with pytest.raises(ValueError, match="escapes"):
        ShardReader(str(tmp_path), torch.device("cpu"))
