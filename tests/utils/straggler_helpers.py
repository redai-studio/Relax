# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Shared fakes for the CPU-only straggler profiler tests."""

from relax.utils.straggler.detector import RankMeta
from relax.utils.straggler.stats import FIELD_INDEX, NUM_FIELDS


class FakeEvent:
    """Stands in for ``torch.cuda.Event``: ``record`` stamps a global counter
    so ``elapsed_time`` is the number of records between the two events."""

    clock = 0.0

    def __init__(self):
        self.stamp = None
        self.done = True

    def record(self):
        FakeEvent.clock += 1.0
        self.stamp = FakeEvent.clock

    def query(self):
        return self.done

    def elapsed_time(self, other):
        return other.stamp - self.stamp


def row(steps=1, **fields):
    """One per-rank window vector; ``fields`` are per-step values summed over
    ``steps``."""
    values = [0.0] * NUM_FIELDS
    values[FIELD_INDEX["num_steps"]] = steps
    for name, value in fields.items():
        values[FIELD_INDEX[name]] = value * steps
    return values


def meta(world, pp_size=1, tp_size=1):
    """``world`` ranks laid out as PP stages of equal size, TP fastest."""
    ranks_per_stage = world // pp_size
    return [
        RankMeta(
            rank=r,
            dp=(r % ranks_per_stage) // tp_size,
            tp=r % tp_size,
            pp=r // ranks_per_stage,
            host=f"h{r // 8}",
            device=r % 8,
        )
        for r in range(world)
    ]
