# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU checks of image-built HybridEP opt-in and collective preflight."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "examples/models/qwen3.8-flash-next/scripts/run-qwen38-qsa-sft-32xgpu-blackwell.sh"


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


@pytest.mark.parametrize("tp", [2, 8])
def test_valid_group_uses_real_peer_detection(preflight, tp):
    module, mpu, dist, detect, expert, logger = preflight
    mpu.get_tensor_model_parallel_world_size = lambda: tp
    module.validate_hybridep_worker_group()
    detect.assert_called_once_with(expert)
    assert dist.all_gather_object.call_count == 2
    logger.info.assert_called_once()


@pytest.mark.parametrize("tp", [2, 8])
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


def test_alignment_script_dry_run_preserves_baseline_and_worker_environment():
    env = dict(
        os.environ,
        DRY_RUN="1",
        MODEL_DIR="/model",
        DATA_DIR="/data",
        SAVE_DIR="/save",
        MEGATRON="/image/megatron",
        RELAX_HYBRID_EP_COMMIT="abc123def",
        LD_LIBRARY_PATH="/image/nvshmem/lib",
    )
    result = subprocess.run(
        ["bash", str(SCRIPT), "--num-rollout", "50"],
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr
    config = json.loads(result.stdout)
    argv = config["argv"]

    def value(flag):
        return argv[len(argv) - 1 - argv[::-1].index(flag) + 1]

    assert value("--num-rollout") == "50"
    assert value("--tensor-model-parallel-size") == "2"
    assert value("--pipeline-model-parallel-layout") == "Et*12|t*14|t*14|t*8mL"
    assert value("--data-pad-size-multiplier") == "16384"
    assert value("--lr-decay-iters") == "2023"
    assert value("--eval-interval") == "100"
    assert json.loads(value("--multimodal-keys")) == {"image": "images"}
    assert value("--image-max-token-num") == "128"
    assert "--sft-image-preprocess-on-rank" in argv
    assert "--sft-packing-strategy" not in argv
    assert value("--save") == value("--load")
    assert value("--save-interval") == "50"
    assert value("--max-actor-ckpt-to-keep") == "2"
    actor = json.loads(value("--train-env-vars"))
    driver = config["runtime_env"]["env_vars"]
    for name, expected in {
        "RELAX_HYBRID_EP_COMMIT": "abc123def",
        "LD_LIBRARY_PATH": "/image/nvshmem/lib",
        "CUDA_DEVICE_MAX_CONNECTIONS": "32",
        "RELAX_QWEN38_QSA_COMMUNITY_BACKEND": "mixed",
        "RELAX_QWEN38_ROUTER_SORTED_REPLAY": "1",
        "RELAX_QWEN38_GDN_OUTPUT_SAFE_CONFIG": "1",
        "RELAX_SFT_FUSED_LINEAR_CE": "1",
        "RELAX_SFT_MTP_ACCEPTANCE_METRICS": "0",
    }.items():
        assert actor[name] == driver[name] == expected
