# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Check stage-local DSA installation without importing CUDA-only Core
modules."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from tests.cpu.models import test_kimi_k3


@pytest.fixture
def installer(monkeypatch):
    class Block(torch.nn.Module):
        def __init__(self, attention=None):
            super().__init__()
            self.config = SimpleNamespace(
                recompute_granularity="full", recompute_method="uniform", recompute_num_layers=1
            )
            self.layers = torch.nn.ModuleList([attention] if attention is not None else [torch.nn.Linear(2, 2)])

        def _checkpointed_forward(self, hidden, packed_seq_params=None):
            return hidden

    class Attention(torch.nn.Module):
        def __init__(self, dtype=torch.bfloat16):
            super().__init__()
            self.core_attention = torch.nn.Linear(2, 2)
            core = self.core_attention
            core.config = SimpleNamespace(params_dtype=dtype, dsa_kernel_backend="tilelang", dsa_indexer_loss_coeff=0)
            core.index_share = False
            core.skip_topk = False
            core.indexer = torch.nn.Linear(2, 2).requires_grad_(False)

    block_module = ModuleType("megatron.core.transformer.transformer_block")
    block_module.TransformerBlock = Block
    monkeypatch.setitem(sys.modules, block_module.__name__, block_module)
    dsa = ModuleType("relax.models.kimi_k3.dsa")
    dsa.KimiK31DSAAttention = Attention
    monkeypatch.setitem(sys.modules, dsa.__name__, dsa)
    path = Path(__file__).resolve().parents[3] / "relax/models/kimi_k3/dsa_recompute.py"
    spec = importlib.util.spec_from_file_location("_k31_stage_installer", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return SimpleNamespace(module=module, Block=Block, Attention=Attention)


def test_kda_only_pipeline_stage_is_unchanged(installer):
    block = installer.Block()
    model = torch.nn.Sequential(block)
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    installer.module.configure_model_indexer_reuse(model)
    assert not hasattr(block, "_relax_dsa_checkpoint_scope")
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
    hidden = torch.randn(2, 2)
    assert block._checkpointed_forward(hidden) is hidden


def test_dsa_stage_keeps_installation_and_dtype_validation(installer):
    attention = installer.Attention()
    block = installer.Block(attention)
    installer.module.configure_model_indexer_reuse(torch.nn.Sequential(block), 2048)
    assert block._relax_dsa_checkpoint_scope
    assert attention.core_attention._relax_dsa_selection_reuse
    assert attention.core_attention.fused_indexer_block_size == 2048
    with pytest.raises(ValueError, match="BF16"):
        installer.module.configure_model_indexer_reuse(installer.Block(installer.Attention(torch.float32)))
    with pytest.raises(ValueError, match="chunk size"):
        installer.module.configure_model_indexer_reuse(installer.Block(), 512)


@pytest.mark.parametrize("is_k31", [False, True])
def test_provider_validates_global_model_before_stage_installation(monkeypatch, installer, is_k31):
    modules = test_kimi_k3.k3_modules.__wrapped__(monkeypatch)
    package = modules.provider.__package__
    ops = ModuleType(package + ".training_ops")
    ops.configure_training_ops = lambda: None
    monkeypatch.setitem(sys.modules, ops.__name__, ops)
    monkeypatch.setitem(sys.modules, package + ".dsa_recompute", installer.module)
    monkeypatch.setenv("RELAX_KIMI_K31_DSA_SELECTION_REUSE", "1")
    monkeypatch.setenv("RELAX_KIMI_K31_DSA_FUSED_INDEXER", "0")
    provider = modules.provider.KimiK3VLModelProvider(kimi_dsa_config={"dsa_layers": [4]} if is_k31 else None)
    parent = type(provider).__mro__[1]
    original = parent.provide
    constructed = []

    def provide(self, **kwargs):
        constructed.append(True)
        return original(self, **kwargs)

    monkeypatch.setattr(parent, "provide", provide)
    if is_k31:
        assert isinstance(provider.provide_language_model(), torch.nn.Module)
        assert constructed == [True]
    else:
        with pytest.raises(ValueError, match="requires a K3.1 DSA model"):
            provider.provide_language_model()
        assert not constructed
