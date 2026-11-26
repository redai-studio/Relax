# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import sys
from types import SimpleNamespace

import torch

from relax.backends.sglang.native_expert_layout import describe_native_fp8_experts


def test_native_fp8_expert_probe_checks_local_mapping(monkeypatch):
    local_by_global = {2: 0, 3: 1}

    class NativeExperts(torch.nn.Module):
        layer_id = 0
        moe_ep_rank = _expert_storage_rank = 1
        moe_ep_size = _num_local_routed = num_local_experts = 2
        moe_tp_size = 1
        _num_global_routed = 4
        num_fused_shared_experts = 0

        def __init__(self):
            super().__init__()
            for name in ("w13_weight", "w2_weight"):
                parameter = torch.nn.Parameter(torch.empty(2, 4, dtype=torch.float8_e4m3fn), requires_grad=False)
                parameter.weight_loader = self.weight_loader
                self.register_parameter(name, parameter)

        def weight_loader(self):
            raise AssertionError("An expert ownership probe must not load weights")

        def _map_global_expert_id_to_local_expert_id(self, expert):
            return local_by_global.get(expert, -1)

    moe = SimpleNamespace(
        enable_eplb=False,
        ep_num_redundant_experts=0,
        init_expert_location="trivial",
        elastic_ep_backend=None,
        ep_join_mode=None,
        enable_waterfill=False,
    )
    monkeypatch.setitem(
        sys.modules, "sglang.srt.layers.moe.fused_moe_triton.layer", SimpleNamespace(FusedMoE=NativeExperts)
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.eplb.expert_location",
        SimpleNamespace(get_global_expert_location_metadata=lambda: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.runtime_context",
        SimpleNamespace(
            get_exec=lambda: SimpleNamespace(moe=moe),
            get_parallel=lambda: SimpleNamespace(elastic_ep_initial_size=None, ep_join_rank_offset=0),
        ),
    )
    model = torch.nn.ModuleDict({"experts": NativeExperts()})
    (layout,) = describe_native_fp8_experts(model)
    assert (layout["module"], layout["expert_begin"], layout["expert_end"]) == ("experts", 2, 4)

    local_by_global.update({2: 1, 3: 0})
    assert describe_native_fp8_experts(model) is None
