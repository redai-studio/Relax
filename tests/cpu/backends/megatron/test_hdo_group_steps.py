# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest
import torch

from relax.backends.megatron.compat import _patch_hdo_group_steps


def make_hdo():
    class HDO:
        def _sync_sub_optimizers_state_to_hdo(self):
            self.synced = True

        def update(self):
            for child in self.sub_optimizers:
                if isinstance(child, torch.optim.AdamW):
                    continue
                for group in child.param_groups:
                    group.update({k: v for k, v in self.param_groups[0].items() if k != "params"})
                    group["step"] = group.get("step", 0) + 1
            self._sync_sub_optimizers_state_to_hdo()

    outer, inner = object(), object()
    hdo = HDO()
    hdo.param_groups = [{"params": [outer]}]
    hdo.inner_param_to_orig_param = {inner: outer}
    hdo.sub_optimizers = [SimpleNamespace(param_groups=[{"params": [inner]}])]
    return hdo


def test_group_step_survives_save_restore_and_keeps_advancing():
    original = make_hdo()
    assert _patch_hdo_group_steps(type(original))
    assert not _patch_hdo_group_steps(type(original))
    for _ in range(200):
        original.update()
    saved = {k: v for k, v in original.param_groups[0].items() if k != "params"}
    assert saved["step"] == 200
    resumed = make_hdo()
    _patch_hdo_group_steps(type(resumed))
    resumed.param_groups[0].update(saved)
    for expected in (201, 202, 203):
        resumed.update()
        original.update()
        assert resumed.param_groups[0]["step"] == original.param_groups[0]["step"] == expected
        assert resumed.sub_optimizers[0].param_groups[0]["step"] == expected


def test_cpu_per_parameter_adam_counter_is_not_group_counter():
    hdo = make_hdo()
    cpu = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))])
    cpu.param_groups[0]["step"] = 5
    hdo.sub_optimizers.append(cpu)
    _patch_hdo_group_steps(type(hdo))
    hdo.update()
    assert hdo.param_groups[0]["step"] == 1
    assert cpu.param_groups[0]["step"] == 5


def test_conflicting_inner_group_steps_fail_instead_of_silent_overwrite():
    hdo = make_hdo()
    inner = next(iter(hdo.inner_param_to_orig_param))
    hdo.sub_optimizers[0].param_groups[0]["step"] = 3
    hdo.sub_optimizers.append(SimpleNamespace(param_groups=[{"params": [inner], "step": 4}]))
    _patch_hdo_group_steps(type(hdo))
    with pytest.raises(RuntimeError, match="disagree"):
        hdo._sync_sub_optimizers_state_to_hdo()


def test_reordered_child_groups_keep_distinct_outer_steps():
    hdo = make_hdo()
    outer_a, outer_b, inner_a, inner_b = (object() for _ in range(4))
    hdo.param_groups = [{"params": [outer_a]}, {"params": [outer_b]}]
    hdo.inner_param_to_orig_param = {inner_a: outer_a, inner_b: outer_b}
    groups = [{"params": [inner_b], "step": 17}, {"params": [inner_a], "step": 3}]
    hdo.sub_optimizers = [SimpleNamespace(param_groups=groups)]
    _patch_hdo_group_steps(type(hdo))
    hdo._sync_sub_optimizers_state_to_hdo()
    assert [group["step"] for group in hdo.param_groups] == [3, 17]
    for group in groups:
        group["step"] += 1
    hdo._sync_sub_optimizers_state_to_hdo()
    assert [group["step"] for group in hdo.param_groups] == [4, 18]


def test_installed_hdo_state_dict_keeps_group_counter_and_moment_aliases():
    module = pytest.importorskip("megatron.core.optimizer.cpu_offloading.hybrid_optimizer")

    class HDO(module.HybridDeviceOptimizer):
        pass

    hdo = object.__new__(HDO)
    outer = torch.nn.Parameter(torch.ones(2))
    inner = torch.nn.Parameter(torch.ones(2))
    torch.optim.Optimizer.__init__(hdo, [outer], {})
    moment = torch.zeros(2)
    hdo.cpu_optimizers = []
    hdo.gpu_optimizer = SimpleNamespace(
        param_groups=[{"params": [inner], "step": 200}],
        state={inner: {"exp_avg": moment, "exp_avg_sq": moment.clone()}},
    )
    hdo.inner_param_to_orig_param = {inner: outer}
    hdo.param_update_in_fp32 = True
    _patch_hdo_group_steps(HDO)
    hdo._sync_sub_optimizers_state_to_hdo()
    saved = hdo.state_dict()
    assert saved["param_groups"][0]["step"] == 200
    assert hdo.state[outer]["exp_avg"] is moment
    assert hdo.state[outer]["master_param"] is inner
