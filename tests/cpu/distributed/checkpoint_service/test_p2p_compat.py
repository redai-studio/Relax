# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from relax.distributed.checkpoint_service.backends.shard_p2p.remote.compat import ParallelismContext


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
@pytest.mark.parametrize("optional_fields", [False, True])
def test_replica_context_uses_target_coordinates_and_restores_them(monkeypatch, tp, optional_fields):
    original = dict(tp_rank=19, tp_size=32, world_group=object())
    current = dict(original)
    fields = {
        "world_rank",
        "world_size",
        "tp_rank",
        "tp_size",
        "pp_rank",
        "pp_size",
        "attn_tp_rank",
        "attn_tp_size",
        "attn_dp_rank",
        "attn_dp_size",
        "attn_cp_rank",
        "attn_cp_size",
        "dcp_rank",
        "dcp_size",
        "dcp_enabled",
        "attn_dcp_rank",
        "attn_dcp_size",
        "moe_tp_rank",
        "moe_tp_size",
        "moe_ep_rank",
        "moe_ep_size",
        "moe_dp_rank",
        "moe_dp_size",
    } | {name + "_group" for name in ("world", "tp", "pp", "attn_tp", "attn_cp", "dcp", "moe_tp", "moe_ep", "moe_dp")}
    if optional_fields:
        fields.update(("dp_rank", "dp_size", "launch_world_rank", "launch_world_size", "shared_experts_tp_group"))

    @contextmanager
    def override(**values):
        assert set(values) <= fields
        previous = dict(current)
        current.clear()
        current.update(values)
        try:
            yield
        finally:
            current.clear()
            current.update(previous)

    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.runtime_context",
        SimpleNamespace(_PARALLEL_FIELDS=frozenset(fields), get_parallel=lambda: SimpleNamespace(override=override)),
    )
    legacy = SimpleNamespace(
        **dict.fromkeys(
            ("_WORLD", "_TP", "_PP", "_ATTN_TP", "_ATTN_CP", "_DCP", "_MOE_TP", "_MOE_EP", "_MOE_DP"), object()
        )
    )
    attention = SimpleNamespace(_ATTN_DP_RANK=17, _ATTN_DP_SIZE=32)
    saved = dict(vars(legacy))
    monkeypatch.setitem(sys.modules, "sglang.srt.distributed.parallel_state", legacy)
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.dp_attention", attention)
    for rank in range(tp):
        with pytest.raises(RuntimeError, match="intentional exit"):
            with ParallelismContext((rank, tp)):
                assert current["tp_rank"] == current["moe_tp_rank"] == rank
                assert current["tp_size"] == current["moe_tp_size"] == tp
                assert (current["world_rank"], current["world_size"]) == (rank, tp)
                for name in ("world", "tp", "attn_tp", "moe_tp") + (("shared_experts_tp",) if optional_fields else ()):
                    group = current[name + "_group"]
                    assert (group.rank_in_group, group.world_size, group.ranks) == (rank, tp, list(range(tp)))
                    with pytest.raises(RuntimeError, match="cannot use group operation"):
                        group.all_reduce
                assert current["pp_size"] == current["moe_ep_size"] == 1
                outer = legacy._TP
                assert outer is current["tp_group"] and legacy._PP.world_size == 1
                assert (attention._ATTN_DP_RANK, attention._ATTN_DP_SIZE) == (0, 1)
                with ParallelismContext((0, 1)):
                    assert legacy._TP.world_size == 1
                assert legacy._TP is outer
                raise RuntimeError("intentional exit")
        assert current == original
        assert vars(legacy) == saved
        assert (attention._ATTN_DP_RANK, attention._ATTN_DP_SIZE) == (17, 32)


@pytest.mark.parametrize("layout", [(-1, 4), (4, 4), (0, 0), (False, 2), (0, 4.0)])
def test_replica_context_rejects_invalid_coordinates(monkeypatch, layout):
    monkeypatch.setitem(sys.modules, "sglang.srt.runtime_context", SimpleNamespace(get_parallel=lambda: None))
    with pytest.raises(NotImplementedError):
        with ParallelismContext(layout):
            pytest.fail("Invalid replica layout reached model construction")


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_actual_sglang_context_accepts_layout_only_target_groups(monkeypatch, tp):
    pytest.importorskip("sglang")
    from sglang.srt.distributed import parallel_state
    from sglang.srt.layers import dp_attention
    from sglang.srt.runtime_context import get_parallel

    getters = {
        "get_world_group": "world",
        "get_tp_group": "tp",
        "get_pp_group": "pp",
        "get_attn_tp_group": "attn_tp",
        "get_attn_cp_group": "attn_cp",
        "get_dcp_group": "dcp",
        "get_moe_tp_group": "moe_tp",
        "get_moe_ep_group": "moe_ep",
        "get_moe_dp_group": "moe_dp",
    }
    saved = {name: value for name, value in vars(parallel_state).items() if name.startswith("_") and name.isupper()}
    saved_attention = (dp_attention._ATTN_DP_RANK, dp_attention._ATTN_DP_SIZE)
    for rank in range(tp):
        with ParallelismContext((rank, tp)):
            parallel = get_parallel()
            assert (parallel.tp_rank, parallel.tp_size) == (rank, tp)
            assert (parallel.moe_tp_group.rank_in_group, parallel.moe_tp_group.world_size) == (rank, tp)
            assert parallel_state.get_tp_group() is parallel.tp_group
            assert parallel_state.get_pp_group() is parallel.pp_group
            assert parallel_state.get_moe_ep_group() is parallel.moe_ep_group
            for getter, field in getters.items():
                assert getattr(parallel_state, getter)() is getattr(parallel, field + "_group")
            with pytest.raises(RuntimeError, match="cannot use group operation"):
                parallel.tp_group.all_reduce
        assert all(getattr(parallel_state, name) is value for name, value in saved.items())
        assert (dp_attention._ATTN_DP_RANK, dp_attention._ATTN_DP_SIZE) == saved_attention


def test_layout_context_refuses_pd_multiplexed_groups(monkeypatch):
    pytest.importorskip("sglang")
    from sglang.srt.distributed import parallel_state

    monkeypatch.setattr(parallel_state, "_ENABLE_PDMUX_P_TP", True)
    saved = parallel_state._TP
    with pytest.raises(NotImplementedError, match="PD-multiplexed"):
        with ParallelismContext((0, 2)):
            pytest.fail("Unsupported group route reached model construction")
    assert parallel_state._TP is saved


def test_layout_process_groups_cannot_fall_back_to_the_default_group(tmp_path):
    pytest.importorskip("sglang")
    import torch
    import torch.distributed as dist
    from sglang.srt.distributed import parallel_state

    assert not dist.is_initialized()
    dist.init_process_group("gloo", init_method=(tmp_path / "layout-gloo").as_uri(), rank=0, world_size=1)
    try:
        default = dist.distributed_c10d._get_default_group()
        with ParallelismContext((0, 1)):
            group = parallel_state.get_tp_group()
            for forbidden in (group.cpu_group, group.device_group):
                assert forbidden is not None and forbidden is not default
                with pytest.raises((RuntimeError, ValueError, TypeError)):
                    dist.all_reduce(torch.ones(1), group=forbidden)
                with pytest.raises((RuntimeError, ValueError, TypeError)):
                    dist.barrier(group=forbidden)
        value = torch.ones(1)
        dist.all_reduce(value, group=default)
        assert value.item() == 1
    finally:
        dist.destroy_process_group()


def test_actual_native_loader_postprocess_runs_model_hook():
    pytest.importorskip("sglang")
    from relax.distributed.checkpoint_service.backends.shard_p2p.remote.compat import post_load_weights

    calls = []
    model = SimpleNamespace(post_load_weights=lambda: calls.append("native-model-fixup"))
    post_load_weights(model)
    assert calls == ["native-model-fixup"]


@pytest.mark.parametrize("requires_args", [False, True])
def test_replica_publishes_server_args_with_both_native_initializer_signatures(monkeypatch, requires_args):
    from relax.distributed.checkpoint_service.backends.shard_p2p.remote.compat import publish_server_args

    server_args, calls = object(), []

    def initialize_with_args(args):
        assert args is server_args
        calls.append("initialize")

    def initialize_without_args():
        calls.append("initialize")

    initialize = initialize_with_args if requires_args else initialize_without_args
    for name, values in (
        ("sglang.srt.layers.moe", dict(initialize_moe_config=initialize)),
        ("sglang.srt.layers.quantization.fp8_utils", dict(initialize_fp8_gemm_config=initialize)),
        ("sglang.srt.layers.quantization.fp4_utils", dict(initialize_fp4_gemm_config=initialize)),
        ("sglang.srt.server_args", dict(set_global_server_args_for_scheduler=lambda args: calls.append(args))),
    ):
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(**values))
    publish_server_args(server_args)
    assert calls == [server_args, "initialize", "initialize", "initialize"]


def test_replica_does_not_retry_a_native_initializer_internal_type_error(monkeypatch):
    from relax.distributed.checkpoint_service.backends.shard_p2p.remote.compat import publish_server_args

    calls = []

    def initialize(args):
        calls.append(args)
        raise TypeError("native initializer failed internally")

    for name, values in (
        ("sglang.srt.layers.moe", dict(initialize_moe_config=initialize)),
        ("sglang.srt.layers.quantization.fp8_utils", dict(initialize_fp8_gemm_config=initialize)),
        ("sglang.srt.layers.quantization.fp4_utils", dict(initialize_fp4_gemm_config=initialize)),
        ("sglang.srt.server_args", dict(set_global_server_args_for_scheduler=lambda args: None)),
    ):
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(**values))
    server_args = object()
    with pytest.raises(TypeError, match="native initializer failed internally"):
        publish_server_args(server_args)
    assert calls == [server_args]


@pytest.mark.parametrize("restore_hook", [False, True])
def test_replica_uses_native_restore_capability_and_preserves_its_layout(restore_hook):
    pytest.importorskip("sglang")
    import torch
    from sglang.srt.layers.quantization.base_config import QuantizeMethodBase

    from relax.distributed.checkpoint_service.backends.shard_p2p.remote.model_replica import (
        _run_module_postprocess_and_restore,
    )

    calls = []

    class Method(QuantizeMethodBase):
        def apply(self, layer, *args, **kwargs):
            raise AssertionError("Layout discovery must not execute a model forward")

        def process_weights_after_loading(self, layer):
            calls.append("process")
            layer.weight.data = layer.weight.data.t()

    module = torch.nn.Linear(3, 2, bias=False)
    module.quant_method = Method()
    if restore_hook:

        def restore(layer):
            calls.append("restore")
            layer.weight.data = layer.weight.data.t()

        module.quant_method.restore_weights_before_loading = restore
    layouts = _run_module_postprocess_and_restore(
        "linear", module, {id(module.weight): module.weight.shape}, (0, 1), torch.device("cpu")
    )
    assert calls == (["process", "restore"] if restore_hook else ["process"])
    assert tuple(layouts[id(module.weight)].shape) == ((2, 3) if restore_hook else (3, 2))
    assert module.weight.numel() == 0 and module.weight.device.type == "cpu"
