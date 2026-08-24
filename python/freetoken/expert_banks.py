"""Backend-neutral FreeToken expert-bank contract.

CUDA loaders and the Apple-Silicon MLX backend both produce this exact bundle;
movement and compute backends decide how its bank rows are materialized.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ExpertBanks:
    quant_format: str
    sources: dict[str, Any]
    gate_up_alpha: Any | None = field(default=None)
    down_alpha: Any | None = field(default=None)
    layer_residency: list[str] | None = field(default=None)
    streamed: bool = False


__all__ = ["ExpertBanks"]
