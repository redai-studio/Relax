# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU checks of image-built HybridEP opt-in and collective preflight."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


ROOT = Path(__file__).resolve().parents[4]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def preflight(monkeypatch, tmp_path):
    env = load("hybrid_env", ROOT / "relax/utils/env.py")
    monkeypatch.setitem(sys.modules, "relax.utils.env", env)
    import importlib.metadata

    monkeypatch.setenv("RELAX_HYBRID_EP_COMMIT", "abc123def")
    monkeypatch.setattr(
        importlib.metadata,
        "distribution",
        lambda name: SimpleNamespace(version="1.0+abc123", locate_file=lambda path: tmp_path / "site-packages"),
    )
    monkeypatch.delenv("NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN", raising=False)
    expert, world = object(), object()
    dist = ModuleType("torch.distributed")
    dist.group = SimpleNamespace(WORLD=world)
    dist.get_world_size = lambda group: 32 if group is world else 8
    dist.get_rank = lambda group: 0
    dist.get_global_rank = lambda group, rank: rank
    dist.get_process_group_ranks = lambda group: list(range(dist.get_world_size(group)))

    def gather(output, value, group):
        assert group is expert
        output[:] = [dict(value, rank=i) if isinstance(value, dict) else value for i in range(8)]

    dist.all_gather_object = Mock(side_effect=gather)
    torch = ModuleType("torch")
    torch.distributed = dist
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)
    mpu = SimpleNamespace(get_expert_tensor_and_model_parallel_group=lambda: expert)
    for name, value in [
        ("tensor_model", 8),
        ("pipeline_model", 4),
        ("context", 1),
        ("expert_model", 8),
        ("expert_tensor", 1),
    ]:
        setattr(mpu, "get_" + name + "_parallel_world_size", lambda value=value: value)
    core = ModuleType("megatron.core")
    core.mpu = mpu
    megatron = ModuleType("megatron")
    megatron.core = core
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    detect = Mock(return_value=8)
    for name in ("_relax_hybrid_ep", "_relax_hybrid_ep.deep_ep_cpp", "_relax_hybrid_ep.hybrid_ep_cpp"):
        module = ModuleType(name)
        module.__file__ = str(tmp_path / "site-packages" / name / "module.so")
        module.ExtendedMemoryAllocator = lambda: SimpleNamespace(detect_accessible_ranks=detect)
        monkeypatch.setitem(sys.modules, name, module)
    logger = Mock()
    monkeypatch.setitem(sys.modules, "relax.utils.logging_utils", SimpleNamespace(get_logger=lambda name: logger))
    module = load("hybrid_preflight", ROOT / "relax/backends/megatron/hybridep_preflight.py")
    return module, mpu, dist, detect, expert, logger


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
@pytest.mark.parametrize("pp", [2, 4])
@pytest.mark.parametrize("world", [32, 64])
def test_valid_group_uses_real_peer_detection(preflight, tp, pp, world):
    module, mpu, dist, detect, expert, logger = preflight
    mpu.get_tensor_model_parallel_world_size = lambda: tp
    mpu.get_pipeline_model_parallel_world_size = lambda: pp
    dist.get_world_size = lambda group: world if group is dist.group.WORLD else 8
    module.validate_hybridep_worker_group()
    detect.assert_called_once_with(expert)
    assert dist.all_gather_object.call_count == 2
    logger.info.assert_called_once()


@pytest.mark.parametrize("tp", [2, 4, 8])
def test_cross_node_group_is_rejected_before_peer_detection(preflight, tp):
    module, mpu, dist, detect, _, _ = preflight
    mpu.get_tensor_model_parallel_world_size = lambda: tp

    def gather(output, value, group):
        output[:] = [dict(value, rank=i, hostname=f"node-{i // 4}") for i in range(8)]

    dist.all_gather_object.side_effect = gather
    with pytest.raises(RuntimeError, match="intranode preflight rejected"):
        module.validate_hybridep_worker_group()
    detect.assert_not_called()


@pytest.mark.parametrize("failure", ["topology", "forced_peers", "import_path", "detected_peers"])
def test_bad_group_fails_collectively(preflight, monkeypatch, failure):
    module, mpu, dist, detect, _, _ = preflight
    if failure == "topology":
        mpu.get_context_parallel_world_size = lambda: 2
    elif failure == "forced_peers":
        monkeypatch.setenv("NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN", "8")
    elif failure == "import_path":
        monkeypatch.setattr(sys.modules["_relax_hybrid_ep"], "__file__", "/unrelated/deep_ep.py")
    else:
        detect.return_value = 7
    with pytest.raises(RuntimeError):
        module.validate_hybridep_worker_group()
    assert dist.all_gather_object.called
    if failure != "detected_peers":
        detect.assert_not_called()


@pytest.mark.parametrize("failure", [None, "cross_node", "detected_peers", "world_size"])
def test_35b_ep4_requires_exact_layout_and_four_accessible_peers(preflight, failure):
    module, mpu, dist, detect, expert, logger = preflight
    for name, value in [("tensor_model", 1), ("pipeline_model", 2), ("context", 4), ("expert_model", 4)]:
        setattr(mpu, "get_" + name + "_parallel_world_size", lambda value=value: value)
    dist.get_world_size = lambda group: (32 if failure == "world_size" else 16) if group is dist.group.WORLD else 4

    def gather(output, value, group):
        assert group is expert and len(output) == 4
        output[:] = [
            dict(value, rank=i, hostname=f"node-{i // 2}" if failure == "cross_node" else "node-0")
            if isinstance(value, dict)
            else value
            for i in range(4)
        ]

    dist.all_gather_object.side_effect = gather
    detect.return_value = 3 if failure == "detected_peers" else 4
    if failure:
        with pytest.raises(RuntimeError):
            module.validate_hybridep_worker_group()
        if failure != "detected_peers":
            detect.assert_not_called()
    else:
        module.validate_hybridep_worker_group()
        detect.assert_called_once_with(expert)
        assert dist.all_gather_object.call_count == 2
        logger.info.assert_called_once()


@pytest.mark.parametrize("value", ["", "0", "8"])
def test_forced_peer_override_presence_is_rejected(preflight, monkeypatch, value):
    module, _, _, detect, _, _ = preflight
    monkeypatch.setenv("NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN", value)
    with pytest.raises(RuntimeError, match="Forced HybridEP"):
        module.validate_hybridep_worker_group()
    detect.assert_not_called()


@pytest.mark.parametrize("failure", ["duplicate", "wrong_member", "size", "remote_import", "inconsistent_detection"])
def test_actual_collective_members_are_validated(preflight, failure):
    module, _, dist, detect, _, _ = preflight
    if failure == "size":
        dist.get_world_size = lambda group: 32 if group is dist.group.WORLD else 7

    def gather(output, value, group):
        if isinstance(value, dict):
            output[:] = [dict(value, rank=i) for i in range(len(output))]
            if failure == "duplicate":
                output[-1]["rank"] = 0
            elif failure == "wrong_member":
                output[-1]["rank"] = 15
            elif failure == "remote_import":
                output[-1]["errors"] = ["Installed module validation failed"]
        else:
            output[:] = [value] * len(output)
            if failure == "inconsistent_detection":
                output[-1] = 4

    dist.all_gather_object.side_effect = gather
    with pytest.raises(RuntimeError):
        module.validate_hybridep_worker_group()
    if failure != "inconsistent_detection":
        detect.assert_not_called()


@pytest.mark.parametrize(
    "name,value", [("expert_tensor", 2), ("expert_model", 16), ("tensor_model", 3), ("pipeline_model", 0)]
)
def test_unsupported_or_nondivisible_geometry_fails(preflight, name, value):
    module, mpu, _, detect, _, _ = preflight
    setattr(mpu, "get_" + name + "_parallel_world_size", lambda: value)
    with pytest.raises(RuntimeError, match="Unsupported topology"):
        module.validate_hybridep_worker_group()
    detect.assert_not_called()
