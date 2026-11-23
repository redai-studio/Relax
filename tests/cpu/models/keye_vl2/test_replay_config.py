# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Execute Keye runtime configuration on CPU.

Only unused Bridge/MCore/TE imports are isolated; no model forward or CUDA
kernel is substituted.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from torch import nn


@pytest.fixture
def modules(monkeypatch):
    root = Path(__file__).resolve().parents[4] / "relax/models/keye_vl2"
    for name, path in (("relax.models", root.parent), ("relax.models.keye_vl2", root)):
        package = ModuleType(name)
        package.__path__ = [str(path)]
        monkeypatch.setitem(sys.modules, name, package)
    externals = {
        "megatron.bridge.models.gpt_provider": dict(GPTModelProvider=object, transformer_engine_layer_spec=None),
        "megatron.core.extensions.transformer_engine": dict(
            TEColumnParallelLinear=nn.Module, TERowParallelLinear=nn.Module, TENorm=nn.Module
        ),
        "megatron.core.transformer.attention": dict(SelfAttention=nn.Module),
        "megatron.core.transformer.experimental_attention_variant.dsa": dict(DSAIndexerLossAutoScaler=None),
        "megatron.core.transformer.moe.moe_utils": dict(router_gating_linear=None),
        "megatron.core.transformer.moe.router": dict(TopKRouter=nn.Module),
        "megatron.core.transformer.spec_utils": dict(ModuleSpec=object),
        "megatron.core.tensor_parallel.mappings": dict(reduce_scatter_to_sequence_parallel_region=None),
    }
    for name, attributes in externals.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    loaded = {}
    for name in ("indexer", "replay", "sparse_attention", "topk", "attention", "linear", "provider"):
        spec = importlib.util.spec_from_file_location(f"relax.models.keye_vl2.{name}", root / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        loaded[name] = module
    return loaded


def test_keye_without_replay_preserves_indexer_training_and_freeze_config(modules) -> None:
    provider = modules["provider"].KeyeModelProvider()
    provider.context_parallel_size = 2
    provider.position_embedding_type = "rope"
    args = SimpleNamespace(
        multimodal_keys=None, cp_partition_mode="zigzag", dsa_indexer_loss_coeff=0.0, freeze_params_name_list=None
    )
    provider.configure_runtime(args)
    assert provider.keye_indexer_loss_coeff == 0.01
    assert args.freeze_params_name_list is None


@pytest.mark.parametrize("coefficient,frozen", [(None, [r"\.sa_indexer\."]), (0.0, ["vision"])])
def test_keye_replay_uses_existing_freeze_path(modules, coefficient, frozen) -> None:
    provider = modules["provider"].KeyeModelProvider()
    provider.context_parallel_size = 2
    provider.position_embedding_type = "rope"
    args = SimpleNamespace(
        multimodal_keys=None,
        cp_partition_mode="zigzag",
        use_rollout_indexer_replay=True,
        dsa_indexer_loss_coeff=0.01,
        freeze_params_name_list=frozen,
    )
    with pytest.raises(ValueError, match="zero indexer loss"):
        provider.configure_runtime(args)
    args.dsa_indexer_loss_coeff = coefficient
    provider.configure_runtime(args)
    assert provider.keye_indexer_loss_coeff == 0
    assert args.freeze_params_name_list == list(dict.fromkeys([*frozen, r"\.sa_indexer\."]))


def test_keye_capture_is_installed_by_model_adapter(monkeypatch, modules):
    from relax.distributed.replay_manager.config import ReplayConfig
    from relax.distributed.replay_manager.core import replay_data_manager

    def stub(name, **attributes):
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    keye_sglang = stub("relax.models.keye_vl2.sglang")
    path = Path(modules["replay"].__file__).with_name("sglang") / "replay.py"
    spec = importlib.util.spec_from_file_location(f"{keye_sglang.__name__}.replay", path)
    adapter = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, adapter)
    spec.loader.exec_module(adapter)
    keye_sglang.replay = adapter

    features = SimpleNamespace(enable_return_indexer_topk=False)
    parallel = SimpleNamespace(pp_size=1, enable_dp_attention=False, attn_tp_rank=0)
    stub(
        "sglang.srt.runtime_context",
        get_exec=lambda: SimpleNamespace(features=features),
        get_parallel=lambda: parallel,
        get_schedule=lambda: SimpleNamespace(chunked_prefill_size=8),
    )
    captures = []

    class Buffer:
        def __init__(self, *args):
            self.config = args

        def capture_compacted(self, layer_id, indices, counts, *, row_offset):
            captures.append((layer_id, indices, counts, row_offset))

    monkeypatch.setattr(adapter, "KeyeBitmapCapturer", Buffer)
    state = SimpleNamespace(capturer=None)
    indexer = stub(
        "sglang.srt.state_capturer.indexer_topk",
        create_indexer_capturer=lambda **_kwargs: "native",
        get_global_indexer_capturer=lambda: state.capturer,
    )
    stub("sglang.srt.state_capturer", indexer_topk=indexer)
    runner = stub("sglang.srt.model_executor.model_runner", create_indexer_capturer=indexer.create_indexer_capturer)
    stub("sglang.srt.model_executor", model_runner=runner)
    monkeypatch.setattr(replay_data_manager, "config", ReplayConfig(delivery="reference"))
    adapter.install_indexer_capturer()

    text = SimpleNamespace(model_type="KeyeVL2", sa_config={"topk": 2048}, num_hidden_layers=4)
    options = dict(
        model_config=SimpleNamespace(hf_text_config=text), num_tokens=16, max_running_requests=2, device="cuda"
    )
    assert runner.create_indexer_capturer(**options) is None
    assert adapter.active_capturer() is None
    features.enable_return_indexer_topk = True
    parallel.attn_tp_rank = 1
    assert runner.create_indexer_capturer(**options) is None
    parallel.attn_tp_rank = 0
    state.capturer = runner.create_indexer_capturer(**options)
    assert state.capturer.config == (16, 8, 4, "cuda")
    indices, counts = object(), object()
    adapter.active_capturer().capture_compacted(1, indices, counts, row_offset=2)
    assert captures == [(1, indices, counts, 2)]
