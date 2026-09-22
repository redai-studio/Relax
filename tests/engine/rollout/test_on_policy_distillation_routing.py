# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Unit tests for ``relax/engine/rollout/on_policy_distillation.py`` teacher
routing.

Two independent layers must both hold:

1. MOPD routing by ``data_source`` picks *which teacher model* — a text sample
   must never reach the VL teacher.
2. Replica selection spreads a teacher's load across *copies of that same model*
   at GRPO-group granularity, so the ``n_samples_per_prompt`` samples sharing a
   prompt prefix are prefilled once instead of once per replica.

The regression these lock down: selecting the replica with
``group_index % len(replicas)`` satisfies (2) but breaks load balance, because
``group_index`` is a global counter that does not restart per ``data_source``.
On an interleaved multi-source dataset one teacher then sees only every k-th
index and every one of its groups maps to the same replica.

Run with: pytest tests/engine/rollout/test_on_policy_distillation_routing.py -v
"""

from argparse import Namespace
from collections import Counter

import pytest

from relax.engine.rollout import on_policy_distillation as opd
from relax.utils.types import Sample


TEXT_REPLICAS = ["http://text-r0/generate", "http://text-r1/generate"]
VL_REPLICAS = ["http://vl-r0/generate", "http://vl-r1/generate"]
TEXT_SOURCE = "dapo-math-17k"
VL_SOURCE = "multimodal-open-r1"

N_SAMPLES_PER_PROMPT = 8
NUM_GROUPS = 32


@pytest.fixture(autouse=True)
def _clear_router_state():
    """Router state is module-level; isolate every test."""
    opd._TEACHER_URL_RR.clear()
    opd._TEACHER_GROUP_REPLICA.clear()
    yield
    opd._TEACHER_URL_RR.clear()
    opd._TEACHER_GROUP_REPLICA.clear()


def _args() -> Namespace:
    return Namespace(
        opd_teacher_routes_map={TEXT_SOURCE: TEXT_REPLICAS, VL_SOURCE: VL_REPLICAS},
        opd_teacher_key="data_source",
    )


def _sample(data_source: str, group_index: int | None) -> Sample:
    return Sample(group_index=group_index, metadata={"data_source": data_source})


def _interleaved_picks() -> dict[tuple[str, int], list[str]]:
    """Two data_sources alternating by ``group_index``, mirroring a merged
    dataset.

    Returns ``(data_source, group_index) -> [url per sample]``.
    """
    args = _args()
    picks: dict[tuple[str, int], list[str]] = {}
    for group_index in range(NUM_GROUPS):
        data_source = TEXT_SOURCE if group_index % 2 == 0 else VL_SOURCE
        picks[(data_source, group_index)] = [
            opd._pick_teacher_url(args, _sample(data_source, group_index)) for _ in range(N_SAMPLES_PER_PROMPT)
        ]
    return picks


def test_on_policy_distillation_routing_keeps_data_source_isolated():
    for (data_source, _), urls in _interleaved_picks().items():
        expected = TEXT_REPLICAS if data_source == TEXT_SOURCE else VL_REPLICAS
        assert set(urls) <= set(expected), f"{data_source} leaked to another teacher: {set(urls)}"


def test_on_policy_distillation_routing_keeps_group_on_one_replica():
    for key, urls in _interleaved_picks().items():
        assert len(set(urls)) == 1, f"group {key} was split across replicas {sorted(set(urls))}"


def test_on_policy_distillation_routing_balances_replicas_when_sources_interleave():
    picks = _interleaved_picks()
    for data_source, replicas in ((TEXT_SOURCE, TEXT_REPLICAS), (VL_SOURCE, VL_REPLICAS)):
        per_replica = Counter(urls[0] for (src, _), urls in picks.items() if src == data_source)
        counts = [per_replica.get(url, 0) for url in replicas]
        assert max(counts) - min(counts) <= 1, f"{data_source} load collapsed onto a subset of replicas: {counts}"


def test_on_policy_distillation_routing_group_index_modulo_would_collapse():
    """Guard the rejected alternative so the trap stays documented.

    Not a test of production code: it asserts that the naive ``group_index %
    len(replicas)`` really does collapse on interleaved sources, which is why
    ``_pick_replica`` uses a per-teacher group cursor instead.
    """
    text_groups = [g for g in range(NUM_GROUPS) if g % 2 == 0]
    counts = [sum(1 for g in text_groups if g % len(TEXT_REPLICAS) == i) for i in range(len(TEXT_REPLICAS))]
    assert min(counts) == 0 and max(counts) == len(text_groups)


def test_on_policy_distillation_routing_falls_back_without_group_index():
    """Samples with no ``group_index`` (eval, replay) keep per-sample round-
    robin."""
    args = _args()
    urls = [opd._pick_teacher_url(args, _sample(TEXT_SOURCE, None)) for _ in range(4)]
    assert urls == [TEXT_REPLICAS[0], TEXT_REPLICAS[1], TEXT_REPLICAS[0], TEXT_REPLICAS[1]]


def test_on_policy_distillation_routing_single_replica_is_unconditional():
    args = Namespace(opd_teacher_routes_map={TEXT_SOURCE: [TEXT_REPLICAS[0]]}, opd_teacher_key="data_source")
    urls = {opd._pick_teacher_url(args, _sample(TEXT_SOURCE, g)) for g in range(5)}
    assert urls == {TEXT_REPLICAS[0]}


def test_on_policy_distillation_routing_survives_group_map_eviction():
    """The group->replica map is bounded; eviction must not break affinity."""
    args = _args()
    opd._TEACHER_GROUP_REPLICA.update({("pad", i): 0 for i in range(opd._MAX_TEACHER_GROUP_REPLICA + 1)})
    urls = [opd._pick_teacher_url(args, _sample(TEXT_SOURCE, 999)) for _ in range(N_SAMPLES_PER_PROMPT)]
    assert len(opd._TEACHER_GROUP_REPLICA) < opd._MAX_TEACHER_GROUP_REPLICA
    assert len(set(urls)) == 1


def test_on_policy_distillation_routing_missing_routing_key_raises():
    args = _args()
    with pytest.raises(ValueError, match="missing key 'data_source'"):
        opd._pick_teacher_url(args, Sample(group_index=0, metadata={}))


def test_on_policy_distillation_routing_unknown_data_source_raises():
    args = _args()
    with pytest.raises(KeyError, match="no teacher route"):
        opd._pick_teacher_url(args, _sample("not-a-source", 0))
