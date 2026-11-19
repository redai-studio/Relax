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
