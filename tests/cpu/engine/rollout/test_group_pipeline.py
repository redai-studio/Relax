# Copyright (c) 2026 Relax Authors. All Rights Reserved.
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from threading import enumerate as enumerate_threads

import pytest

from relax.engine.rollout.group_pipeline import GroupPipeline


def test_group_pipeline_overlaps_and_bounds_both_writer_and_reward():
    writer_started, release_writer, score_started, release_score = (Event() for _ in range(4))
    seen = []

    def prepare(value):
        if value == 0:
            writer_started.set()
            assert release_writer.wait(3)
        return value

    def score(value):
        if value == 0:
            score_started.set()
            assert release_score.wait(3)
        seen.append(value)

    pipeline = GroupPipeline(prepare, score, 2)
    try:
        pipeline.reserve()
        pipeline.submit(0)
        assert writer_started.wait(3)
        # A second generation can start despite a blocked sidecar writer.
        pipeline.reserve()
        pipeline.submit(1)
        release_writer.set()
        assert score_started.wait(3)
        with ThreadPoolExecutor(max_workers=1) as caller:
            credit = caller.submit(pipeline.reserve)
            try:
                assert not credit.done()  # Completed writer != free reward credit.
            finally:
                release_score.set()
            credit.result(timeout=3)
        pipeline.submit(2)
        pipeline.finish()
        assert seen == [0, 1, 2]
    finally:
        release_writer.set()
        release_score.set()
        pipeline.close()
    assert not any(t.name.startswith(("rollout-prepare", "rollout-reward")) for t in enumerate_threads())


@pytest.mark.parametrize("limit", [0, True])
def test_group_pipeline_rejects_invalid_credit_limit(limit):
    with pytest.raises(ValueError, match="overlap_pending_groups"):
        GroupPipeline(lambda x: x, lambda x: x, limit)
