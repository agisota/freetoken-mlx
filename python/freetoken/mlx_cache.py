"""Torch-free MLX implementation of FreeToken's expert-bank slot cache."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable

from freetoken.expert_banks import ExpertBanks


class MLXOffloadMoeCache:
    """Unified cross-layer LRU cache for evaluated MLX expert weights.

    This is the Apple-Silicon counterpart of ``moe.offload_cache.OffloadMoeCache``:
    the cache is shared across layers, admits only routed expert rows from registered
    banks, and exposes hit/miss/eviction counters. Unified memory replaces the CUDA
    pinned-host/PCIe copy operation; ``materialize`` is the backend-specific transfer.
    """

    def __init__(self, banks: ExpertBanks, num_experts: int, cache_size: int,
                 materialize: Callable[[Any, int], Any], *,
                 eviction_policy: str = "lru",
                 log: Callable[[str], None] | None = None) -> None:
        if cache_size < 1:
            raise ValueError("MLX expert cache size must be at least 1")
        sources = banks.sources.get("experts", ())
        if not sources or num_experts < 1:
            raise ValueError("MLX expert banks must contain layers and experts")
        if eviction_policy not in {"lru", "slru"}:
            raise ValueError("MLX expert cache policy must be lru or slru")
        self.banks = banks
        self.num_layers = len(sources)
        self.num_experts = num_experts
        self.cache_size = cache_size
        self._materialize = materialize
        self._log = log
        self.eviction_policy = eviction_policy
        self._slots: OrderedDict[tuple[int, int], Any] = OrderedDict()
        self._probation: OrderedDict[tuple[int, int], None] = OrderedDict()
        self._protected: OrderedDict[tuple[int, int], None] = OrderedDict()
        self._requests = 0
        self._protected_capacity = self._slru_protected_capacity()
        self.hits = self.misses = self.loads = self.evictions = 0

    def ensure_expert(self, layer_id: int, expert_id: int) -> Any:
        if not 0 <= layer_id < self.num_layers:
            raise IndexError(f"layer id {layer_id} outside expert bank")
        if not 0 <= expert_id < self.num_experts:
            raise IndexError(f"expert id {expert_id} outside expert bank")
        self._requests += 1
        if self.eviction_policy == "slru":
            # A small protected segment avoids preserving one-off prefill and
            # early decode routes. After four full-cache request windows, grow
            # it to retain experts proven hot over a longer conversation.
            self._protected_capacity = self._slru_protected_capacity()
        key = (layer_id, expert_id)
        if key in self._slots:
            self.hits += 1
            value = self._slots[key]
            if self.eviction_policy == "lru":
                self._slots.move_to_end(key)
            elif key in self._protected:
                self._protected.move_to_end(key)
            else:
                self._probation.pop(key)
                self._protected[key] = None
                if len(self._protected) > self._protected_capacity:
                    demoted, _ = self._protected.popitem(last=False)
                    self._probation[demoted] = None
            self._emit("hit", key)
            return value

        # Materialize before mutating the LRU so a failed bank load is atomic.
        value = self._materialize(self.banks.sources["experts"][layer_id], expert_id)
        self.misses += 1
        self.loads += 1
        if len(self._slots) == self.cache_size:
            evicted = self._evict_one()
            self.evictions += 1
            self._emit("evict", evicted)
        self._slots[key] = value
        if self.eviction_policy == "slru":
            self._probation[key] = None
        self._emit("miss", key)
        return value

    def _evict_one(self) -> tuple[int, int]:
        if self.eviction_policy == "lru":
            key, _ = self._slots.popitem(last=False)
            return key
        segment = self._probation if self._probation else self._protected
        key, _ = segment.popitem(last=False)
        self._slots.pop(key)
        return key

    def _slru_protected_capacity(self) -> int:
        fraction = 0.80 if self._requests > 4 * self.cache_size else 0.10
        return max(1, int(self.cache_size * fraction))

    def _emit(self, event: str, key: tuple[int, int]) -> None:
        if self._log is not None:
            self._log(f"expert_cache event={event} layer={key[0]} expert={key[1]}")

    def resize(self, cache_size: int) -> None:
        """Shrink the resident set immediately; growth only changes admission capacity."""
        if cache_size < 1:
            raise ValueError("MLX expert cache size must be at least 1")
        self.cache_size = cache_size
        self._protected_capacity = self._slru_protected_capacity()
        while len(self._protected) > self._protected_capacity:
            demoted, _ = self._protected.popitem(last=False)
            self._probation[demoted] = None
        while len(self._slots) > cache_size:
            evicted = self._evict_one()
            self.evictions += 1
            self._emit("evict", evicted)

    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "loads": self.loads,
                "evictions": self.evictions, "resident": len(self._slots),
                "capacity": self.cache_size}


__all__ = ["MLXOffloadMoeCache"]
