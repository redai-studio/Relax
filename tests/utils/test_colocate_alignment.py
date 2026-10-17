# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Colocated weight sync must pair each engine worker with the training rank on
the same physical GPU.

The verdict is a pure function of the gathered rank devices and the engine
reports, so it is tested here on CPU with synthetic identities; the wiring
tests check that the Megatron updater refuses before building IPC groups and
that an engine reports the devices SGLang was launched on.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from relax.utils.colocate_alignment import check_colocate_alignment


def _ranks(*nodes):
    """Rank devices for ``(node_id, num_gpus)`` nodes, one rank per GPU."""
    return [(node, f"uuid:{node}-{gpu}") for node, count in nodes for gpu in range(count)]


def _engine(node, gpus):
    return {"node_id": node, "devices": [f"uuid:{node}-{gpu}" for gpu in gpus]}


def test_colocate_alignment_accepts_engines_on_their_ranks_gpus():
    ranks = _ranks(("a", 4), ("b", 4))
    engines = [_engine("a", (0, 1)), _engine("a", (2, 3)), _engine("b", (0, 1)), _engine("b", (2, 3))]

    assert check_colocate_alignment(ranks, engines, [0, 2, 4, 6], [2, 2, 2, 2]) == ([], [])


def test_colocate_alignment_rejects_swapped_workers():
    ranks = _ranks(("a", 2))
    # Same GPUs, wrong order: worker 0 would open rank 0's handle on GPU 1.
    problems, _ = check_colocate_alignment(ranks, [_engine("a", (1, 0))], [0], [2])

    assert len(problems) == 1 and "rank 0" in problems[0] and "worker 0" in problems[0]


def test_colocate_alignment_rejects_an_engine_on_another_node():
    ranks = _ranks(("a", 2), ("b", 2))
    problems, _ = check_colocate_alignment(ranks, [_engine("b", (0, 1)), _engine("a", (0, 1))], [0, 2], [2, 2])

    assert len(problems) == 2


def test_colocate_alignment_rejects_a_scaled_out_offset_landing_on_rank_zero():
    # A scaled-out engine's offset is relative to its own placement group.
    ranks = _ranks(("a", 4))
    engines = [_engine("a", (0, 1)), _engine("a", (2, 3)), _engine("a", (2, 3))]
    problems, _ = check_colocate_alignment(ranks, engines, [0, 2, 0], [2, 2, 2])

    assert any("both claim training rank 0" in problem for problem in problems)
    assert any("engine 2 workers" in problem for problem in problems)


def test_colocate_alignment_rejects_two_ranks_on_one_gpu():
    ranks = [("a", "uuid:a-0"), ("a", "uuid:a-0")]
    problems, _ = check_colocate_alignment(ranks, [], [], [])

    assert problems == ["training ranks 0 and 1 share device ('a', 'uuid:a-0')"]


def test_colocate_alignment_rejects_ranges_outside_the_world_and_bad_reports():
    ranks = _ranks(("a", 2))

    problems, _ = check_colocate_alignment(ranks, [_engine("a", (0, 1))], [1], [2])
    assert "outside world size 2" in problems[0]
    problems, _ = check_colocate_alignment(ranks, [_engine("a", (0, 1, 2))], [0], [2])
    assert "3 device(s) for 2 rank(s)" in problems[0]


def test_colocate_alignment_checks_a_multi_node_engine_on_its_own_node():
    ranks = _ranks(("a", 2), ("b", 2))

    assert check_colocate_alignment(ranks, [_engine("a", (0, 1))], [0], [4]) == ([], [])


def test_colocate_alignment_reports_engines_without_devices_as_unverified():
    ranks = _ranks(("a", 2))

    assert check_colocate_alignment(ranks, [{"node_id": "a", "devices": None}], [0], [2]) == ([], [0])


def test_colocate_alignment_falls_back_to_visible_index(monkeypatch):
    from relax.utils import colocate_alignment

    module = SimpleNamespace(get_device_properties=lambda index: SimpleNamespace())
    monkeypatch.setattr(colocate_alignment.device_utils, "get_torch_device_module", lambda: module)
    monkeypatch.setattr(colocate_alignment.device_utils, "get_visible_devices_env_var", lambda: "TEST_VISIBLE")
    monkeypatch.setenv("TEST_VISIBLE", "4,5")

    assert colocate_alignment.local_device_identity(1) == "index:5"
    monkeypatch.delenv("TEST_VISIBLE")
    assert colocate_alignment.local_device_identity(3) == "index:3"


# ---------------------------------------------------------------------------
# Wiring.
# ---------------------------------------------------------------------------
def test_sglang_engine_reports_the_devices_it_was_launched_on(monkeypatch):
    pytest.importorskip("sglang")
    from relax.backends.sglang import sglang_engine
    from relax.utils import colocate_alignment

    monkeypatch.setattr(colocate_alignment, "local_device_identity", lambda index: f"uuid:{index}")
    monkeypatch.setattr(sglang_engine.ray, "get_runtime_context", lambda: SimpleNamespace(get_node_id=lambda: "a"))
    engine = object.__new__(sglang_engine.SGLangEngine)
    engine.args = SimpleNamespace(rollout_num_gpus_per_engine=4, num_gpus_per_node=2)
    engine.num_gpus_per_engine = None
    engine._launched_base_gpu_id = 6

    # A 4-GPU engine over 2-GPU nodes drives 2 GPUs on this node.
    assert engine.get_device_identity() == {"node_id": "a", "devices": ["uuid:6", "uuid:7"]}
    engine._launched_base_gpu_id = None
    assert engine.get_device_identity() == {"node_id": "a", "devices": None}


def test_megatron_updater_refuses_misaligned_engines(monkeypatch):
    pytest.importorskip("megatron.core")
    from relax.backends.megatron.weight_update import update_weight_from_tensor as module

    engine = MagicMock()
    engine.get_device_identity.remote.return_value = {"node_id": "a", "devices": ["uuid:a-1", "uuid:a-0"]}
    monkeypatch.setattr(module.ray, "get", lambda refs: list(refs))

    with pytest.raises(module.ColocateAlignmentError, match="other GPUs"):
        module.UpdateWeightFromTensor._check_colocate_alignment(_ranks(("a", 2)), [engine], [0], [2])

    engine.get_device_identity.remote.return_value = {"node_id": "a", "devices": ["uuid:a-0", "uuid:a-1"]}
    module.UpdateWeightFromTensor._check_colocate_alignment(_ranks(("a", 2)), [engine], [0], [2])


def test_megatron_updater_checks_alignment_only_when_ranks_share_engine_gpus():
    pytest.importorskip("megatron.core")
    from relax.backends.megatron.weight_update import update_weight_from_tensor as module

    # Hybrid hands weights to rollout's own GPUs over IPC; plain colocation must pair GPUs.
    assert module._shares_training_gpus(SimpleNamespace(hybrid=False, colocate=True))
    assert not module._shares_training_gpus(SimpleNamespace(hybrid=True, colocate=True))
