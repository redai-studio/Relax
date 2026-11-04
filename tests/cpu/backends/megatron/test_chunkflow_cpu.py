# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Dependency-free chunk and pipeline-plan checks."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).parents[4]


def _load_chunkflow() -> SimpleNamespace:
    package = "relax.backends.megatron.chunkflow"
    folder = ROOT / "relax/backends/megatron/chunkflow"
    modules = {}
    with patch.dict(sys.modules):
        for name in ("relax", "relax.backends", "relax.backends.megatron", package):
            module = ModuleType(name)
            module.__path__ = []
            sys.modules[name] = module
        for leaf in ("scheduler", "units"):
            name = f"{package}.{leaf}"
            spec = importlib.util.spec_from_file_location(name, folder / f"{leaf}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            modules[leaf] = module
    return SimpleNamespace(**modules)


class ChunkflowPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cf = _load_chunkflow()

    def test_parent_plan_merge_keeps_dependency_and_parent_order(self):
        plans = []
        for lengths in ([13, 2], [3]):
            chunks = self.cf.scheduler.build_chunks(lengths, 4)
            units = self.cf.units.build_micro_batches(chunks, "staggered")
            plans.append(self.cf.units.make_model_plan(units, chunks, 1, 4))
        merged = self.cf.units.merge_model_plans(plans, 4)
        self.assertEqual(merged.unit_ids, (0, 1, 2, 3, 4, 5))
        self.assertEqual(merged.forward, (4, 0, 1, 2, 3, 5))
        self.assertEqual(merged.backward, (4, 3, 2, 1, 0, 5))
        self.assertEqual(merged.recompute, frozenset({0, 1, 2}))
        self.assertEqual(merged.warmups, (6, 5, 4, 3))

    def test_chunk_split_and_packing_preserve_tokens_and_prefixes(self):
        lengths = [2050, 300, 200]
        chunks = self.cf.scheduler.build_chunks(lengths, 1024)
        dependent = [chunk for chunk in chunks if chunk.is_dependent]
        self.assertEqual([c.segments[0].prefix for c in dependent], [0, 1024, 2048])
        self.assertEqual([c.num_tokens for c in dependent], [1024, 1024, 2])
        self.assertEqual(sum(c.num_tokens for c in chunks), sum(lengths))
        standalone = [chunk for chunk in chunks if not chunk.is_dependent]
        self.assertEqual([c.num_tokens for c in standalone], [500])
        units = self.cf.units.build_micro_batches(chunks, "staggered")
        self.assertEqual([u.chunk_ids for u in units], [[c.chunk_id] for c in chunks])


if __name__ == "__main__":
    unittest.main()
