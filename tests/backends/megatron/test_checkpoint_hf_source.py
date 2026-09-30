# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from concurrent.futures import ThreadPoolExecutor
from threading import Event, current_thread
from types import SimpleNamespace

import pytest
import torch


pytest.importorskip("megatron.training.checkpointing", exc_type=ImportError)

from relax.backends.megatron import checkpoint, compat
from relax.utils.model_source import ModelSource


def test_select_hf_load_source_remaps_replaced_hf_path():
    args = SimpleNamespace(
        model_source=ModelSource("s3://bucket/model/"),
        _model_source_original_hf_checkpoint="/models/original",
        hf_checkpoint="/dev/shm/model",
        ref_load="/dev/shm/model",
    )

    assert checkpoint._select_hf_load_source(args, "/models/original/") == "/dev/shm/model"


def test_select_hf_load_source_preserves_distinct_int4_reference():
    args = SimpleNamespace(
        model_source=ModelSource("s3://bucket/packed-model/"),
        _model_source_original_hf_checkpoint="/models/packed-int4",
        hf_checkpoint="/dev/shm/packed-model",
        ref_load="/models/bf16-reference",
    )

    assert checkpoint._select_hf_load_source(args, args.ref_load) == args.ref_load


def test_load_checkpoint_preserves_valid_megatron_resume(monkeypatch, tmp_path):
    resume_path = tmp_path / "model"
    resume_path.mkdir()
    (resume_path / "latest_checkpointed_iteration.txt").write_text("1")
    (resume_path / "iter_0000001").mkdir()
    args = SimpleNamespace(load=str(resume_path))
    expected = (1, 2)

    monkeypatch.setattr(checkpoint, "get_args", lambda: args)
    monkeypatch.setattr(checkpoint, "_alias_renamed_transfer_queue_enum", lambda: None)
    monkeypatch.setattr(checkpoint, "_load_checkpoint_metadata", lambda *a, **kw: {})
    monkeypatch.setattr(checkpoint, "_load_checkpoint_megatron", lambda **_kwargs: expected)

    def fail_hf_load(**_kwargs):
        raise AssertionError("valid Megatron resume must not reach the HF loader")

    monkeypatch.setattr(checkpoint, "_load_checkpoint_hf", fail_hf_load)

    model = [SimpleNamespace(role="actor")]
    assert checkpoint.load_checkpoint(model, None, None, {}, False) == expected


@pytest.mark.parametrize(
    "case",
    ["full", "template", "other_sharding", "partial_offload", "non_precision_aware", "non_hdo", "no_optim", "error"],
)
def test_full_checkpoint_preserves_hdo_step_only_when_applicable(monkeypatch, tmp_path, case):
    from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    parameter = torch.nn.Parameter(torch.zeros(1))
    inner = torch.nn.Parameter(torch.zeros(1))
    state = {"step": torch.tensor(1.0), "exp_avg": torch.ones(1), "exp_avg_sq": torch.ones(1)}
    cpu_optimizer = torch.optim.AdamW([inner])
    cpu_optimizer.state[inner] = state
    hdo = SimpleNamespace() if case == "non_hdo" else object.__new__(HybridDeviceOptimizer)
    hdo.param_update_in_fp32 = True
    hdo.offload_fraction = 0.5 if case == "partial_offload" else 1.0
    hdo.gpu_optimizer = None
    hdo.cpu_optimizers = [cpu_optimizer]
    hdo.state = {parameter: state}
    hdo.inner_param_to_orig_param = {inner: parameter}
    optimizer = object.__new__(DistributedOptimizer)
    optimizer.optimizer = hdo
    optimizer.config = SimpleNamespace(use_precision_aware_optimizer_no_fp8_or_ds_fp8=case != "non_precision_aware")
    optimizer.ddp_config = SimpleNamespace(use_megatron_fsdp=False)
    saved = {
        "optimizer": {"param_groups": [{"step": 7}]},
        "param_state": {},
        "param_state_sharding_type": "other" if case == "other_sharding" else "dp_reshardable",
    }
    if case == "template":
        saved.pop("param_state")
    calls = []
    moment = state["exp_avg"]

    def original_load(current, incoming):
        assert current is optimizer and incoming is saved
        calls.append("load")
        state["step"] = torch.tensor(1.0)
        if case == "error":
            raise RuntimeError("simulated DCP failure")
        return "loaded"

    def load_megatron(**_kwargs):
        assert (DistributedOptimizer.load_state_dict is original_load) == (case == "no_optim")
        if case != "no_optim":
            assert optimizer.load_state_dict(saved) == "loaded"
        return 7, 0

    resume_path = tmp_path / "checkpoint"
    resume_path.mkdir()
    (resume_path / "latest_checkpointed_iteration.txt").write_text("7")
    (resume_path / "iter_0000007").mkdir()
    args = SimpleNamespace(load=str(resume_path), optimizer_cpu_offload=True, no_load_optim=case == "no_optim")
    monkeypatch.setattr(DistributedOptimizer, "load_state_dict", original_load)
    monkeypatch.setattr(checkpoint, "get_args", lambda: args)
    monkeypatch.setattr(checkpoint, "_alias_renamed_transfer_queue_enum", lambda: None)
    monkeypatch.setattr(checkpoint, "_read_lora_checkpoint_metadata", lambda _path, ckpt_step=None: None)
    monkeypatch.setattr(checkpoint, "patch_hybrid_optimizer_native_fp32_checkpoint_load", lambda: False)
    monkeypatch.setattr(checkpoint, "_load_checkpoint_megatron", load_megatron)

    if case == "error":
        with pytest.raises(RuntimeError, match="simulated DCP failure"):
            checkpoint.load_checkpoint(None, optimizer, None, {}, False)
    else:
        assert checkpoint.load_checkpoint(None, optimizer, None, {}, False) == (7, 0)
    assert DistributedOptimizer.load_state_dict is original_load
    assert calls == ([] if case == "no_optim" else ["load"])
    assert state["step"].item() == (7 if case == "full" else 1)
    assert cpu_optimizer.state[inner] is state
    assert state["exp_avg"] is moment and torch.equal(moment, torch.ones(1))
    assert torch.equal(parameter, torch.zeros(1))


def test_hf_initialization_does_not_install_optimizer_step_fix(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text("{}")
    args = SimpleNamespace(load=str(tmp_path), optimizer_cpu_offload=True)

    def unexpected_fix():
        raise AssertionError("HF initialization must not install optimizer resume fixes")

    monkeypatch.setattr(checkpoint, "get_args", lambda: args)
    monkeypatch.setattr(checkpoint, "preserve_hdo_dp_reshardable_steps_on_load", unexpected_fix)
    monkeypatch.setattr(checkpoint, "_load_checkpoint_hf", lambda **_kwargs: (0, 0))

    assert checkpoint.load_checkpoint(None, object(), None, {}, False) == (0, 0)


def test_hdo_step_fix_nested_exceptions_restore_loader(monkeypatch):
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    optimizer = object.__new__(DistributedOptimizer)
    optimizer.optimizer = object()  # Exercise the unchanged non-HDO pass-through.
    saved = {}

    def original_load(current, incoming):
        assert current is optimizer and incoming is saved
        return "loaded"

    monkeypatch.setattr(DistributedOptimizer, "load_state_dict", original_load)
    with pytest.raises(RuntimeError, match="outer failure"):
        with compat.preserve_hdo_dp_reshardable_steps_on_load():
            outer_load = DistributedOptimizer.load_state_dict
            assert outer_load.__wrapped__ is original_load
            with pytest.raises(RuntimeError, match="inner failure"):
                with compat.preserve_hdo_dp_reshardable_steps_on_load():
                    assert DistributedOptimizer.load_state_dict.__wrapped__ is outer_load
                    assert optimizer.load_state_dict(saved) == "loaded"
                    raise RuntimeError("inner failure")
            assert DistributedOptimizer.load_state_dict is outer_load
            assert optimizer.load_state_dict(saved) == "loaded"
            raise RuntimeError("outer failure")
    assert DistributedOptimizer.load_state_dict is original_load


@pytest.mark.parametrize("second_raises", [False, True])
def test_hdo_step_fix_concurrent_contexts_restore_loader(monkeypatch, second_raises):
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    first_entered = Event()
    second_attempted = Event()
    second_entered = Event()
    release_first = Event()
    release_second = Event()
    original_lock = compat._HDO_STEP_LOAD_LOCK
    second_thread = None

    class ObservedLock:
        """Observe the second real RLock acquisition without timing-based
        sleeps."""

        def __enter__(self):
            if current_thread() is second_thread:
                second_attempted.set()
            return original_lock.__enter__()

        def __exit__(self, *args):
            return original_lock.__exit__(*args)

    def original_load(*_args):
        return "loaded"

    monkeypatch.setattr(compat, "_HDO_STEP_LOAD_LOCK", ObservedLock())
    monkeypatch.setattr(DistributedOptimizer, "load_state_dict", original_load)

    def first_load():
        with compat.preserve_hdo_dp_reshardable_steps_on_load():
            first_entered.set()
            assert release_first.wait(10)

    def second_load():
        nonlocal second_thread
        second_thread = current_thread()
        with compat.preserve_hdo_dp_reshardable_steps_on_load():
            second_entered.set()
            # Capturing the first context's wrapper would leave it installed
            # when this second context exits after the first one.
            assert DistributedOptimizer.load_state_dict.__wrapped__ is original_load
            assert release_second.wait(10)
            if second_raises:
                raise RuntimeError("second failure")

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(first_load)
        second = None
        try:
            assert first_entered.wait(10)
            second = pool.submit(second_load)
            assert second_attempted.wait(10)
            assert not second_entered.is_set()
            release_first.set()
            first.result(timeout=10)
            assert second_entered.wait(10)
        finally:
            release_first.set()
            release_second.set()
        assert second is not None
        if second_raises:
            with pytest.raises(RuntimeError, match="second failure"):
                second.result(timeout=10)
        else:
            second.result(timeout=10)
    assert DistributedOptimizer.load_state_dict is original_load
