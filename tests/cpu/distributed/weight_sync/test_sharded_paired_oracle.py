# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Two-rank control-chain regression for the frozen sharded raw-byte oracle."""

from __future__ import annotations

import json
import os
import sys
import types
from contextlib import ExitStack
from datetime import timedelta
from pathlib import Path
from time import monotonic
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from task_delta_weight_sync.sharded_gpu_audit import AuditState


class _Ref:
    def __init__(self, callback):
        self.callback = callback


class _Remote:
    def __init__(self, owner, method):
        self.owner, self.method = owner, method

    def remote(self, *args, **kwargs):
        return _Ref(lambda: getattr(self.owner, self.method)(*args, **kwargs))


class _ActorId:
    def __init__(self, value: str):
        self.value = value

    def hex(self) -> str:
        return self.value


class _AuditModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bf16 = nn.Parameter(torch.tensor([0.5, -0.0], dtype=torch.bfloat16), requires_grad=False)
        self.fp32 = nn.Parameter(torch.tensor([1.25, -3.0], dtype=torch.float32), requires_grad=False)


class _FakeEngine:
    def __init__(self, index: int):
        self._actor_id = _ActorId(f"paired-engine-{index}")
        self.models = [_AuditModel(), _AuditModel()]
        self.paused = False
        self.probe_calls = 0
        self.capture_snapshot_ids: dict[str, list[tuple[int, ...]]] = {}
        methods = {
            "pause_generation": "_pause_generation",
            "continue_generation": "_continue_generation",
            "run_memory_audit": "_run_memory_audit",
            "disk_weight_generation_probe": "_disk_weight_generation_probe",
            "update_weights_from_distributed": "_update_weights_from_distributed",
        }
        for method, implementation in methods.items():
            setattr(self, method, _Remote(self, implementation))

    def _pause_generation(self, **_kwargs):
        self.paused = True
        return True

    def _continue_generation(self, **_kwargs):
        for model in self.models:
            snapshot = getattr(model, "_relax_cpu_audit_snapshot", None)
            if snapshot is not None and snapshot.state in {AuditState.REFERENCE_PENDING, AuditState.POISONED}:
                raise RuntimeError("Cannot continue a poisoned audit model")
        self.paused = False
        return True

    def _run_memory_audit(self, control, _timeout):
        from task_delta_weight_sync.sharded_gpu_audit import load_audit

        if not self.paused:
            raise RuntimeError("Audit controls require paused engine generation")
        for tp_rank, model in enumerate(self.models):
            member = f"{self._actor_id.hex()}:tp:{tp_rank}"
            local_control = dict(control, member=member, tp_rank=tp_rank)
            encoded = json.dumps(local_control, sort_keys=True)
            load_audit(model, [(encoded, torch.empty(0, dtype=torch.uint8))])
            if control["command"] == "capture":
                snapshot = model._relax_cpu_audit_snapshot
                ids = tuple(id(entry.snapshot) for entry in snapshot._entries)
                self.capture_snapshot_ids.setdefault(control["update_id"], []).append(ids)
        return {"success": True}

    def _disk_weight_generation_probe(self, _timeout):
        for model in self.models:
            snapshot = getattr(model, "_relax_cpu_audit_snapshot", None)
            if snapshot is not None and snapshot.state in {AuditState.REFERENCE_PENDING, AuditState.POISONED}:
                raise RuntimeError("Post-poison generation probe is forbidden")
        self.probe_calls += 1
        return True

    def _update_weights_from_distributed(self, **_kwargs):
        return True


class _FakeLock:
    def __init__(self):
        self.acquire = _Remote(self, "_acquire")
        self.release = _Remote(self, "_release")

    def _acquire(self):
        return True

    def _release(self):
        return True


class _FakeManager:
    def __init__(self, engines):
        self.engines = tuple(engines)
        self.lock = _FakeLock()
        self.get_rollout_engines_and_lock = _Remote(self, "_get")

    def _get(self):
        return self.engines, self.lock, None, [2, 2, 2, 2], [0, 2, 4, 6]


class _FakeRay(types.ModuleType):
    def __init__(self):
        super().__init__("ray")

    @staticmethod
    def get(value, **_kwargs):
        if isinstance(value, (list, tuple)):
            return [_FakeRay.get(item) for item in value]
        if isinstance(value, _Ref):
            return value.callback()
        return value

    @staticmethod
    def kill(_engine, **_kwargs):
        return None


class _FakeEvent:
    def __init__(self, enable_timing=True):
        assert enable_timing

    def record(self):
        return None

    def synchronize(self):
        return None

    def elapsed_time(self, _other):
        return 1.0


class _FakeNativeModule(types.ModuleType):
    def __init__(self):
        super().__init__("fake_native_weight_update")
        self.dist = dist


class _FakeNative:
    module: _FakeNativeModule | None = None
    ray: _FakeRay | None = None
    fail_independent_reference = False

    def __init__(self, _args, model, weights_getter, **_kwargs):
        self.model = model
        self.weights_getter = weights_getter
        self.rollout_engines = ()
        self.calls = 0
        self.weight_version = 0

    def connect_rollout_engines(self, engines, _lock, **_kwargs):
        self.rollout_engines = tuple(engines)

    def update_weights(self, *, resume_generation=False):
        if resume_generation:
            raise AssertionError("Oracle reference must remain paused")
        self.calls += 1
        named = list(self.weights_getter().items())
        refs = self.module.update_weights_from_distributed(
            "fake-native",
            dist.group.WORLD,
            self.calls,
            self.rollout_engines,
            named,
        )
        self.ray.get(refs)
        self.weight_version += 1
        # The production wrapper deliberately leaves generation paused for
        # both timed FULL controls and the post-complement reference load.
        for engine in self.rollout_engines:
            engine.paused = True
        omit_one = self.fail_independent_reference and self.calls >= 2
        if dist.get_rank() == 0:
            with torch.no_grad():
                for engine_index, engine in enumerate(self.rollout_engines):
                    for tp_rank, model in enumerate(engine.models):
                        for name, source in named:
                            target = getattr(model, name)
                            if omit_one and engine_index == 0 and tp_rank == 0 and name == "bf16":
                                target.view(-1)[:-1].copy_(source.view(-1)[:-1])
                            else:
                                target.copy_(source)


def _fake_update_weights(self, _manager, *, full_only=None):
    rank = dist.get_rank()
    self.test_sharded_calls.append("full_tcp" if full_only else "delta_tcp")
    if not full_only:
        with torch.no_grad():
            self.weights_getter()["bf16"].add_(1)
            self.weights_getter()["fp32"].add_(1)
    if rank == 0:
        with torch.no_grad():
            for engine in self._ab_engines:
                for model in engine.models:
                    for name, source in self.weights_getter().items():
                        getattr(model, name).copy_(source)
    return {
        "source_ranks": [
            dict(expert_payload_bytes=4, nonexpert_payload_bytes=8, tcp_control_bytes=1) for _ in range(2)
        ],
        "receivers": [dict(control_json_bytes=1) for _ in range(4)],
    }


def _fake_gate(error):
    errors = [None, None]
    dist.all_gather_object(errors, error, group=dist.group.WORLD)
    if any(errors):
        raise RuntimeError(f"test updater gate: {errors}")


def _run_scenario(rank: int, directory: Path, *, fail_reference: bool):
    from relax.backends.megatron.weight_update.update_weight_from_memory import UpdateWeightFromMemory
    from relax.backends.megatron.weight_update.update_weight_from_sharded_memory import UpdateWeightFromShardedMemory
    from relax.distributed.weight_sync.config import MemorySyncConfig

    directory.mkdir(parents=True, exist_ok=True)
    run_root = directory / "run"
    (run_root / "logs").mkdir(parents=True, exist_ok=True)
    os.environ["TENSORBOARD_DIR"] = str(run_root / "logs" / "tensorboard")
    os.environ["TMPDIR"] = str(run_root / "tmp")
    Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    config = types.SimpleNamespace(layout="qwen35_owner_local", full_only=False, sync_timeout_s=12)
    engines = [_FakeEngine(index) for index in range(4)]
    manager = _FakeManager(engines)
    source = _AuditModel()
    weights_getter = lambda: {"bf16": source.bf16, "fp32": source.fp32}
    updater = object.__new__(UpdateWeightFromShardedMemory)
    updater.config = config
    updater.iterator = types.SimpleNamespace(model=source, model_name="cpu-audit-test")
    updater.weights_getter = weights_getter
    updater._gate = _fake_gate
    updater.sources = {}
    updater.engines = ()
    updater.failed = False
    updater.test_sharded_calls = []

    native_module = _FakeNativeModule()
    native_module.UpdateWeightFromDistributed = _FakeNative
    _FakeNative.module = native_module
    _FakeNative.ray = _FakeRay()
    _FakeNative.fail_independent_reference = fail_reference

    import relax.distributed.weight_sync.provenance as provenance
    import task_delta_weight_sync.audit_cpu_budget as budget_module
    import task_delta_weight_sync.measure_sharded as hook_module

    original_import = hook_module.importlib.import_module

    def import_module(name, *args, **kwargs):
        if name == "relax.backends.megatron.weight_update.update_weight_from_distributed":
            return native_module
        return original_import(name, *args, **kwargs)

    original_budget_init = budget_module.AuditCpuBudget.__init__

    def zero_margin_budget(self, directory, member, *, limit_bytes, margin_bytes=64 << 30, available=None):
        original_budget_init(
            self,
            directory,
            member,
            limit_bytes=limit_bytes,
            margin_bytes=0,
            available=lambda: 1 << 60,
        )

    class _ConfigLoader:
        @classmethod
        def load(cls, _path):
            return config

    # The hook resolves MemorySyncConfig at import time, so patch the public
    # class method and the CPU lease constructor only for this process.
    patches = [
        patch.dict(sys.modules, {"ray": _FakeRay()}),
        patch.object(MemorySyncConfig, "load", _ConfigLoader.load),
        patch.object(hook_module.importlib, "import_module", side_effect=import_module),
        patch.object(UpdateWeightFromShardedMemory, "update_weights", _fake_update_weights),
        patch.object(
            UpdateWeightFromMemory,
            "_close_generation",
            lambda self, _manager: _quarantine_fake_engines(self.engines),
        ),
        patch.object(provenance, "runtime_identity", return_value={"test": "cpu"}),
        patch.object(budget_module.AuditCpuBudget, "__init__", zero_margin_budget),
        patch.object(torch.cuda, "Event", _FakeEvent),
    ]
    with ExitStack() as stack:
        for item in patches:
            stack.enter_context(item)
        UpdateWeightFromShardedMemory._dws_paired = False
        hook_module.install(types.SimpleNamespace(memory_weight_sync_config="cpu-test.json"))
        try:
            if fail_reference:
                with pytest.raises(RuntimeError, match="Raw GPU audit mismatch"):
                    updater.update_weights(manager)
            else:
                first = updater.update_weights(manager)
                second = updater.update_weights(manager)
                assert first["arm"] == second["arm"] == "delta_tcp"
        except Exception:
            raise

    return updater, engines, run_root


def _quarantine_fake_engines(engines):
    for engine in engines:
        engine.paused = True
    return []


def _worker(rank: int, rendezvous: str, output_root: str) -> None:
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=20))
    try:
        from relax.utils import distributed_utils

        with patch.object(distributed_utils, "get_gloo_group", return_value=dist.group.WORLD):
            successful_root = Path(output_root) / "success"
            success = _run_scenario(rank, successful_root, fail_reference=False)
            failed_root = Path(output_root) / "failure"
            failure = _run_scenario(rank, failed_root, fail_reference=True)

            outcomes = [None, None]
            outcomes[rank] = "ok"
            dist.all_gather_object(outcomes, outcomes[rank], group=dist.group.WORLD)
            assert outcomes == ["ok", "ok"]

            if rank == 0:
                _assert_success(success[1], success[2])
                _assert_reference_failure(failure[0], failure[1], failure[2])
            dist.barrier(group=dist.group.WORLD)
    finally:
        dist.destroy_process_group()


def _assert_success(engines, run_root: Path) -> None:
    reports = [json.loads(path.read_text()) for path in sorted((run_root / "paired-measurements").glob("*.json"))]
    reports = [item for item in reports if item.get("status") == "VERIFIED"]
    assert [item["optimizer_update"] for item in reports] == [0, 1]
    assert reports[0]["full_control_order"] == ["full_tcp", "nccl"]
    assert reports[1]["full_control_order"] == ["nccl", "full_tcp"]
    assert all(item["comparison"].startswith("all actual native parameter bytes") for item in reports)

    audit_root = run_root / "audit"
    members = [f"paired-engine-{engine}:tp:{tp}" for engine in range(4) for tp in range(2)]
    for update_index in range(2):
        update_id = f"optimizer-{update_index:06d}"
        for member in members:
            inventory = json.loads((audit_root / member / "inventory.json").read_text())
            phases = [
                "delta-before-full",
                "full_tcp-compare",
                "nccl-compare",
                "complement-before-reference",
                "independent-full-reference",
            ]
            receipts = [
                json.loads((audit_root / member / f"{update_id}-{phase}.json").read_text()) for phase in phases
            ]
            assert all(item["success"] for item in receipts)
            assert receipts[-1]["compared_bytes"] == inventory["snapshot_bytes"]
            assert receipts[-1]["mismatched_bytes"] == 0
            assert receipts[0]["time"] < min(item["time"] for item in receipts[1:])

    for engine in engines:
        captures = [engine.capture_snapshot_ids[f"optimizer-{index:06d}"] for index in range(2)]
        assert captures[0] == captures[1]
        assert engine.probe_calls == 2  # One measured native arm per paired optimizer update.
        assert not engine.paused


def _assert_reference_failure(updater, engines, run_root: Path) -> None:
    audit_root = run_root / "audit"
    member = "paired-engine-0:tp:0"
    assert json.loads((audit_root / member / "optimizer-000000-nccl-compare.json").read_text())["success"]
    failed_receipt = json.loads((audit_root / member / "optimizer-000000-independent-full-reference.json").read_text())
    assert failed_receipt["success"] is False
    assert "Raw GPU audit mismatch" in failed_receipt["error"]
    assert not (run_root / "paired-measurements" / "000000.json").exists()
    assert updater.failed
    assert all(engine.paused for engine in engines)
    assert all(engine.probe_calls == 1 for engine in engines)
    states = [model._relax_cpu_audit_snapshot.state for engine in engines for model in engine.models]
    assert all(state in {AuditState.POISONED, AuditState.REFERENCE_PENDING} for state in states)
    assert AuditState.POISONED in states


def test_sharded_paired_oracle_requires_independent_full_after_complement(tmp_path) -> None:
    context = mp.start_processes(
        _worker,
        args=(f"file://{tmp_path / 'paired-gloo-rendezvous'}", str(tmp_path / "evidence")),
        nprocs=2,
        join=False,
        start_method="fork",
    )
    deadline = monotonic() + 55
    try:
        while not context.join(timeout=1):
            if monotonic() > deadline:
                pytest.fail("Paired oracle control chain exceeded 55 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=3)
