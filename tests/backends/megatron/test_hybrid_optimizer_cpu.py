# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise real AdamW and HDO checkpoint methods on CPU tensors.

Run with Megatron installed; use the same legacy-image compatibility entry
point as Relax's checkpoint loader. CPU copies stand in for device shards;
these tests do not exercise CUDA DMA, streams, distributed collectives, or the
complete distributed checkpoint loader.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
hybrid_module = pytest.importorskip("megatron.core.optimizer.cpu_offloading.hybrid_optimizer")

from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer  # noqa: E402

from relax.backends.megatron.compat import (  # noqa: E402
    _patch_hdo_group_steps,
    patch_hybrid_optimizer_native_fp32_checkpoint_load,
    preserve_hdo_dp_reshardable_steps_on_load,
)


HybridDeviceOptimizer = hybrid_module.HybridDeviceOptimizer


@pytest.fixture(autouse=True)
def isolate_hybrid_load_compatibility(monkeypatch: pytest.MonkeyPatch) -> None:
    # Restore the installed implementation after each test, including old images.
    monkeypatch.setattr(
        HybridDeviceOptimizer,
        "_update_fp32_params_by_new_state",
        HybridDeviceOptimizer._update_fp32_params_by_new_state,
    )


def _legacy_indexed_load(self) -> None:
    if self.param_update_in_fp32:
        for param, state in self.state.items():
            self.param_to_fp32_param[param].data.copy_(state["master_param"])


def _legacy_guarded_load(self) -> None:
    if self.param_update_in_fp32:
        for param, state in self.state.items():
            fp32_param = self.param_to_fp32_param.get(param)
            if fp32_param is not None:
                fp32_param.data.copy_(state["master_param"])


@pytest.fixture(params=[None, _legacy_indexed_load, _legacy_guarded_load], ids=["installed", "indexed", "guarded"])
def hybrid_load_implementation(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    # Exercise both historical image implementations even when CI upgrades its image.
    if request.param is not None:
        monkeypatch.setattr(HybridDeviceOptimizer, "_update_fp32_params_by_new_state", request.param)


def _cpu_hybrid(*, fused: bool) -> HybridDeviceOptimizer:
    """Construct the CPU state portion without creating CUDA streams."""
    originals = [
        torch.tensor([0.375, -0.625], dtype=torch.bfloat16),
        torch.tensor([1.001, -2.003], dtype=torch.float32),
    ]
    groups = [
        {
            "params": [parameter],
            "lr": lr,
            "betas": (0.8, 0.95),
            "eps": 1e-7,
            "weight_decay": 0.1,
            "fused": fused,
        }
        for parameter, lr in zip(originals, (0.025, 0.005), strict=True)
    ]
    optimizer = object.__new__(HybridDeviceOptimizer)
    torch.optim.Optimizer.__init__(optimizer, groups, {"cpu_optimizer_cls": torch.optim.AdamW})
    optimizer.param_update_in_fp32 = True
    optimizer.param_to_inner_param = {parameter: parameter.float().clone() for parameter in originals}
    optimizer.inner_param_to_orig_param = {
        inner: parameter for parameter, inner in optimizer.param_to_inner_param.items()
    }
    optimizer.param_to_fp32_param = {
        parameter: inner
        for parameter, inner in optimizer.param_to_inner_param.items()
        if parameter.dtype != torch.float32
    }
    optimizer.gpu_params_map_cpu_copy = dict(optimizer.param_to_inner_param)
    optimizer.cpu_copys_map_gpu_param = dict(optimizer.inner_param_to_orig_param)
    cpu_groups = [
        {**group, "params": [optimizer.param_to_inner_param[parameter] for parameter in group["params"]]}
        for group in optimizer.param_groups
    ]
    optimizer.cpu_optimizers = HybridDeviceOptimizer.build_cpu_optimizer_list(torch.optim.AdamW, cpu_groups)
    optimizer.gpu_optimizer = None
    return optimizer


def _cpu_step(optimizer: HybridDeviceOptimizer, step: int) -> None:
    for index, inner in enumerate(optimizer.param_to_inner_param.values()):
        inner.grad = torch.tensor([0.125 * (step + 1), -0.25 * (index + 1)], dtype=torch.float32)
    for cpu_optimizer in optimizer.cpu_optimizers:
        cpu_optimizer.step()
    optimizer._sync_sub_optimizers_state_to_hdo()


def _distributed_state_loader(optimizer: HybridDeviceOptimizer) -> DistributedOptimizer:
    """Supply only metadata consumed by the real model-space state loader."""
    patch_hybrid_optimizer_native_fp32_checkpoint_load()
    distributed = object.__new__(DistributedOptimizer)
    distributed.optimizer = optimizer
    distributed.config = SimpleNamespace(use_precision_aware_optimizer_no_fp8_or_ds_fp8=True)
    originals = list(optimizer.param_to_inner_param)
    distributed.model_param_group_index_map = {parameter: (index, 0) for index, parameter in enumerate(originals)}
    distributed.gbuf_ranges = [{parameter.dtype: [{"param_map": {parameter: {}}}]} for parameter in originals]
    return distributed


@pytest.mark.parametrize("fused", [False, True])
def test_hybrid_cpu_model_space_resume_preserves_bf16_and_native_fp32(fused: bool, hybrid_load_implementation) -> None:
    reference = _cpu_hybrid(fused=fused)
    for step in range(3):
        _cpu_step(reference, step)

    # Use actual AdamW state_dicts. The common checkpoint loads step separately
    # from the model-space payload, just as DistributedOptimizer does.
    saved_states = [copy.deepcopy(optimizer.state_dict()) for optimizer in reference.cpu_optimizers]
    restored = _cpu_hybrid(fused=fused)
    payload = {}
    for index, (parameter, saved) in enumerate(zip(restored.param_to_inner_param, saved_states, strict=True)):
        state = saved["state"][saved["param_groups"][0]["params"][0]]
        restored.state[parameter]["step"] = state["step"].clone()
        payload[index] = {
            "fp32_param": state["master_param"].clone(),
            "exp_avg": state["exp_avg"].clone(),
            "exp_avg_sq": state["exp_avg_sq"].clone(),
        }
    _distributed_state_loader(restored).load_parameter_state_from_fs_model_space(payload)

    for expected, actual in zip(
        reference.param_to_inner_param.values(), restored.param_to_inner_param.values(), strict=True
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual.dtype == torch.float32

    for step in range(3, 6):
        _cpu_step(reference, step)
        _cpu_step(restored, step)
    for reference_optimizer, restored_optimizer in zip(reference.cpu_optimizers, restored.cpu_optimizers, strict=True):
        reference_param = reference_optimizer.param_groups[0]["params"][0]
        restored_param = restored_optimizer.param_groups[0]["params"][0]
        torch.testing.assert_close(restored_param, reference_param, rtol=0, atol=0)
        for name in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                restored_optimizer.state[restored_param][name],
                reference_optimizer.state[reference_param][name],
                rtol=0,
                atol=0,
            )


def test_hybrid_cpu_hf_reload_refreshes_bf16_and_native_fp32_copies() -> None:
    optimizer = _cpu_hybrid(fused=True)
    for parameter in optimizer.param_to_inner_param:
        parameter.fill_(3.007)
    optimizer.update_fp32_param_by_new_param()
    for parameter, inner in optimizer.param_to_inner_param.items():
        assert inner.dtype == torch.float32
        torch.testing.assert_close(inner, parameter.float(), rtol=0, atol=0)


def test_hybrid_cpu_load_compatibility_is_idempotent(hybrid_load_implementation) -> None:
    patch_hybrid_optimizer_native_fp32_checkpoint_load()
    patched = HybridDeviceOptimizer._update_fp32_params_by_new_state
    assert not patch_hybrid_optimizer_native_fp32_checkpoint_load()
    assert HybridDeviceOptimizer._update_fp32_params_by_new_state is patched
    optimizer = _cpu_hybrid(fused=False)
    optimizer.param_update_in_fp32 = False
    # An inactive FP32 master path must not try to read missing master state.
    for parameter in optimizer.param_to_inner_param:
        optimizer.state[parameter] = {}
    optimizer._update_fp32_params_by_new_state()


@pytest.mark.parametrize("offload_fraction", [0.8, 1.0])
@pytest.mark.parametrize("fused", [False, True])
def test_dp_reshardable_cpu_adam_updates_match_after_template_step_replacement(monkeypatch, offload_fraction, fused):
    reference = _cpu_hybrid(fused=fused)
    restored = _cpu_hybrid(fused=fused)
    for step in range(3):
        _cpu_step(reference, step)
    _cpu_step(restored, 0)
    restored.offload_fraction = offload_fraction
    # The GPU child is a group-counter sentinel, not a CUDA execution test.
    gpu_parameter = torch.tensor([9.0])
    gpu_state = {"exp_avg": torch.tensor([2.0])}
    gpu_child = SimpleNamespace(
        state={gpu_parameter: gpu_state}, param_groups=[{"params": [gpu_parameter], "step": 3}]
    )
    if offload_fraction < 1:
        restored.gpu_optimizer = gpu_child
        restored.state[gpu_parameter] = gpu_state
        restored.inner_param_to_orig_param[gpu_parameter] = gpu_parameter
    distributed = _distributed_state_loader(restored)
    distributed.ddp_config = SimpleNamespace(use_megatron_fsdp=False)
    payload = {}
    for index, (expected, actual) in enumerate(
        zip(reference.param_to_inner_param, restored.param_to_inner_param, strict=True)
    ):
        inner = restored.param_to_inner_param[actual]
        inner.copy_(reference.param_to_inner_param[expected])
        state = reference.state[expected]
        payload[index] = {
            actual.dtype: [
                [
                    {
                        "padding": False,
                        "param": inner,
                        "exp_avg": state["exp_avg"].clone(),
                        "exp_avg_sq": state["exp_avg_sq"].clone(),
                        # DCP unwraps the local nonpersistent template step, not saved step3.
                        "step": torch.tensor(1.0),
                    }
                ]
            ]
        }
    saved = {
        "optimizer": {"param_groups": [{"step": 3}, {"step": 3}]},
        "param_state": payload,
        "param_state_sharding_type": "dp_reshardable",
    }

    def load_parameter_payload(self, state_dict):
        self.load_parameter_state_from_dp_reshardable(state_dict["param_state"])

    monkeypatch.setattr(DistributedOptimizer, "load_state_dict", load_parameter_payload)
    with preserve_hdo_dp_reshardable_steps_on_load():
        distributed.load_state_dict(saved)
    for child in restored.cpu_optimizers:
        assert all(state["step"].item() == 3 for state in child.state.values())
    assert "step" not in gpu_state
    assert gpu_child.param_groups[0]["step"] == 3
    for step in range(3, 6):
        for optimizer in (reference, restored):
            for index, inner in enumerate(optimizer.param_to_inner_param.values()):
                inner.grad = torch.tensor([0.125 * (step + 1), -0.25 * (index + 1)])
            for child in optimizer.cpu_optimizers:
                child.step()
    for expected, actual in zip(reference.cpu_optimizers, restored.cpu_optimizers, strict=True):
        expected_param = expected.param_groups[0]["params"][0]
        actual_param = actual.param_groups[0]["params"][0]
        torch.testing.assert_close(actual_param, expected_param, rtol=0, atol=0)
        for key in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                actual.state[actual_param][key], expected.state[expected_param][key], rtol=0, atol=0
            )


@pytest.mark.parametrize("unknown_owner", [False, True])
def test_mixed_hdo_resume_drops_only_cpu_owned_group_counter_before_chained_save(monkeypatch, unknown_owner):
    from megatron.core.optimizer import distrib_optimizer
    from megatron.core.optimizer.optimizer import ChainedOptimizer

    class HDO(HybridDeviceOptimizer):
        pass

    outer = [torch.nn.Parameter(torch.ones(1)) for _ in range(4)]
    inner = [torch.nn.Parameter(torch.ones(1)) for _ in range(4)]
    hdo = object.__new__(HDO)
    # CPU-only, GPU-only, and mixed-owner outer groups; child groups are reordered.
    groups = [
        {"params": [outer[0]], "step": 20},
        {"params": [outer[2]], "step": 20},
        {"params": [outer[1], outer[3]], "step": 20},
    ]
    if unknown_owner:
        groups.append({"params": [torch.nn.Parameter(torch.ones(1))], "step": 20})
    torch.optim.Optimizer.__init__(hdo, groups, {})
    hdo.offload_fraction = 0.8
    hdo.param_update_in_fp32 = True
    hdo.inner_param_to_orig_param = dict(zip(inner, outer, strict=True))
    hdo.cpu_optimizers = [torch.optim.AdamW([inner[i]], fused=True) for i in (1, 0)]
    for child in hdo.cpu_optimizers:
        parameter = child.param_groups[0]["params"][0]
        for _ in range(20):
            parameter.grad = torch.ones_like(parameter)
            child.step()
    # Group-owned GPU counters are CPU sentinels; this does not execute CUDA Adam.
    hdo.gpu_optimizer = SimpleNamespace(
        param_groups=[{"params": [inner[i]], "step": 20} for i in (3, 2)],
        state={inner[i]: {"exp_avg": torch.ones(1), "exp_avg_sq": torch.ones(1)} for i in (2, 3)},
    )
    _patch_hdo_group_steps(HDO)
    hdo._sync_sub_optimizers_state_to_hdo()
    moments = [child.state[next(iter(child.state))]["exp_avg"] for child in hdo.cpu_optimizers]
    distributed = object.__new__(DistributedOptimizer)
    distributed.optimizer = hdo
    distributed.config = SimpleNamespace(use_precision_aware_optimizer_no_fp8_or_ds_fp8=True)
    distributed.ddp_config = SimpleNamespace(use_megatron_fsdp=False)
    distributed.grad_scaler = None
    saved = {
        "optimizer": {"param_groups": [{"step": 20} for _ in groups]},
        "param_state": {},
        "param_state_sharding_type": "dp_reshardable",
    }

    def load_template(self, state_dict):
        for child in self.optimizer.cpu_optimizers:
            for state in child.state.values():
                state["step"] = torch.tensor(1.0)

    monkeypatch.setattr(DistributedOptimizer, "load_state_dict", load_template)
    with preserve_hdo_dp_reshardable_steps_on_load():
        distributed.load_state_dict(saved)
    assert "step" not in hdo.param_groups[0]
    assert [hdo.param_groups[i]["step"] for i in (1, 2)] == [20, 20]
    if unknown_owner:
        assert hdo.param_groups[3]["step"] == 20
    for _ in range(20):
        for child in hdo.cpu_optimizers:
            child.step()
        for group in hdo.gpu_optimizer.param_groups:
            group["step"] += 1
        hdo._sync_sub_optimizers_state_to_hdo()
    chained = object.__new__(ChainedOptimizer)
    chained.chained_optimizers = [distributed]
    if unknown_owner:
        with pytest.raises(AssertionError, match="steps"):
            chained._synchronize_steps()
        return
    assert chained._synchronize_steps() == 40
    # Exercise the installed serializer's HDO branch, as on the training image.
    monkeypatch.setattr(distrib_optimizer, "HAVE_APEX_OR_TE", True)
    checkpoint = distributed.state_dict()
    assert all(group["step"] == 40 for group in checkpoint["optimizer"]["param_groups"])
    for child, moment in zip(hdo.cpu_optimizers, moments, strict=True):
        parameter = child.param_groups[0]["params"][0]
        state = child.state[parameter]
        assert state is hdo.state[hdo.inner_param_to_orig_param[parameter]]
        assert state["step"].item() == 40
        assert state["exp_avg"] is moment
        assert state["master_param"] is parameter
