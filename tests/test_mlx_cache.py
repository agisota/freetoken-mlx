from __future__ import annotations

import unittest

from freetoken.expert_banks import ExpertBanks
from freetoken.mlx_cache import MLXOffloadMoeCache


class MLXOffloadMoeCacheTest(unittest.TestCase):
    def test_hit_miss_and_lru_eviction(self):
        loads = []
        cache = MLXOffloadMoeCache(
            ExpertBanks("bf16", {"experts": ("layer0",)}),
            3, 2,
            lambda source, expert: loads.append((source, expert)) or f"expert-{expert}",
        )
        self.assertEqual(cache.ensure_expert(0, 0), "expert-0")
        cache.ensure_expert(0, 1)
        cache.ensure_expert(0, 0)  # makes expert 1 the LRU
        cache.ensure_expert(0, 2)
        self.assertEqual(loads, [("layer0", 0), ("layer0", 1), ("layer0", 2)])
        self.assertEqual(
            cache.stats(),
            {"hits": 1, "misses": 3, "loads": 3, "evictions": 1,
             "resident": 2, "capacity": 2},
        )
        cache.ensure_expert(0, 1)
        self.assertEqual(cache.stats()["misses"], 4)

    def test_failed_load_does_not_mutate_cache(self):
        def fail(_source, _expert):
            raise RuntimeError("bank read failed")

        cache = MLXOffloadMoeCache(
            ExpertBanks("bf16", {"experts": (object(),)}), 1, 1, fail
        )
        with self.assertRaisesRegex(RuntimeError, "bank read failed"):
            cache.ensure_expert(0, 0)
        self.assertEqual(cache.stats()["resident"], 0)
        self.assertEqual(cache.stats()["misses"], 0)

    def test_bounds(self):
        cache = MLXOffloadMoeCache(
            ExpertBanks("bf16", {"experts": (object(),)}), 2, 1, lambda s, e: e
        )
        with self.assertRaises(IndexError):
            cache.ensure_expert(1, 0)
        with self.assertRaises(IndexError):
            cache.ensure_expert(0, 2)

        with self.assertRaisesRegex(ValueError, "policy"):
            MLXOffloadMoeCache(
                ExpertBanks("bf16", {"experts": (object(),)}),
                1, 1, lambda s, e: e, eviction_policy="unknown",
            )

    def test_resize_evicts_lru(self):
        cache = MLXOffloadMoeCache(
            ExpertBanks("bf16", {"experts": (object(),)}), 3, 3, lambda s, e: e
        )
        for expert in range(3):
            cache.ensure_expert(0, expert)
        cache.resize(1)
        self.assertEqual(cache.stats()["resident"], 1)
        self.assertEqual(cache.stats()["capacity"], 1)
        self.assertEqual(cache.stats()["evictions"], 2)

    def test_slru_protects_reused_expert_from_scan(self):
        cache = MLXOffloadMoeCache(
            ExpertBanks("bf16", {"experts": (object(),)}),
            6, 3, lambda _source, expert: expert, eviction_policy="slru",
        )
        cache.ensure_expert(0, 0)
        cache.ensure_expert(0, 1)
        cache.ensure_expert(0, 0)  # promote expert 0
        for expert in (2, 3, 4, 5):
            cache.ensure_expert(0, expert)
        misses = cache.stats()["misses"]
        self.assertEqual(cache.ensure_expert(0, 0), 0)
        self.assertEqual(cache.stats()["misses"], misses)


if __name__ == "__main__":
    unittest.main()
