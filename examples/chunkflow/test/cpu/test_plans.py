# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Dependency-free chunk and pipeline-plan checks."""

from __future__ import annotations

import unittest

from examples.chunkflow import scheduler, units


class PlanTests(unittest.TestCase):
    def test_parent_plan_merge_keeps_dependency_and_parent_order(self):
        plans = []
        for lengths in ([13, 2], [3]):
            chunks = scheduler.build_chunks(lengths, 4)
            batches = units.build_micro_batches(chunks, "staggered")
            plans.append(units.make_model_plan(batches, chunks, 1, 4))
        merged = units.merge_model_plans(plans, 4)
        self.assertEqual(merged.unit_ids, (0, 1, 2, 3, 4, 5))
        self.assertEqual(merged.forward, (4, 0, 1, 2, 3, 5))
        self.assertEqual(merged.backward, (4, 3, 2, 1, 0, 5))
        self.assertEqual(merged.recompute, frozenset({0, 1, 2}))
        self.assertEqual(merged.warmups, (6, 5, 4, 3))

    def test_chunk_split_and_packing_preserve_tokens_and_prefixes(self):
        lengths = [2050, 300, 200]
        chunks = scheduler.build_chunks(lengths, 1024)
        dependent = [chunk for chunk in chunks if chunk.is_dependent]
        self.assertEqual([c.segments[0].prefix for c in dependent], [0, 1024, 2048])
        self.assertEqual([c.num_tokens for c in dependent], [1024, 1024, 2])
        self.assertEqual(sum(c.num_tokens for c in chunks), sum(lengths))
        standalone = [chunk for chunk in chunks if not chunk.is_dependent]
        self.assertEqual([c.num_tokens for c in standalone], [500])
        batches = units.build_micro_batches(chunks, "staggered")
        self.assertEqual([u.chunk_ids for u in batches], [[c.chunk_id] for c in chunks])


if __name__ == "__main__":
    unittest.main()
