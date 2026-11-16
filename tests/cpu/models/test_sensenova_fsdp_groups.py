# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU regression tests for SenseNova FSDP branch ownership and hooks."""

import pytest
import torch


def _branch_group_model(training_mode="lora"):
    pytest.importorskip("torchvision")
    from relax.models.sensenova.adapter import SenseNovaSFTModel
    from relax.models.sensenova.model import SenseNovaU1ForCausalLM

    pipe = torch.nn.Module()
    pipe.dit = torch.nn.Module()
    pipe.dit.vision_model = torch.nn.Linear(32, 32)
    pipe.dit.language_model = SenseNovaU1ForCausalLM(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        pad_token_id=0,
    )
    if training_mode == "lora":
        pipe.requires_grad_(False)
    return SenseNovaSFTModel(pipe, training_mode=training_mode, fsdp_branch_groups=True)


def test_sensenova_branch_groups_own_all_decoder_parameters(monkeypatch):
    from relax.backends.fsdp.runtime import fsdp2_wrap, resolve_module_groups
    from relax.models.sensenova.adapter import SenseNovaSFTAdapter

    model = _branch_group_model()
    names = list(model.state_dict())
    SenseNovaSFTAdapter().prepare_for_sharding(model)
    groups = resolve_module_groups(model)
    layer = model.pipe.dit.language_model.model.layers[0]
    understanding = {id(parameter) for module in groups[0] for parameter in module.parameters()}
    generation = {id(parameter) for module in groups[1] for parameter in module.parameters()}
    assert understanding == {id(parameter) for name, parameter in layer.named_parameters() if "_mot_gen" not in name}
    assert generation == {id(parameter) for name, parameter in layer.named_parameters() if "_mot_gen" in name}
    assert understanding.isdisjoint(generation)
    assert list(model.state_dict()) == names
    wrapped = []
    monkeypatch.setattr("torch.distributed.fsdp.fully_shard", lambda module, **kwargs: wrapped.append(module))
    fsdp2_wrap(model)
    assert wrapped[3:5] == groups
    assert wrapped[-2:] == [layer, model]


def test_sensenova_branch_groups_preserve_generation_lora_parameter_ownership():
    pytest.importorskip("peft")
    from relax.backends.fsdp.lora import inject_lora_adapter
    from relax.backends.fsdp.runtime import resolve_module_groups
    from relax.models.sensenova.adapter import SenseNovaSFTAdapter

    model = _branch_group_model()
    inject_lora_adapter(model, rank=2, alpha=2, target_modules=["q_proj_mot_gen"])
    names = list(model.state_dict())
    SenseNovaSFTAdapter().prepare_for_sharding(model)
    groups = resolve_module_groups(model)
    layer = model.pipe.dit.language_model.model.layers[0]
    assert layer.self_attn.q_proj in groups[0]
    assert layer.self_attn.q_proj_mot_gen in groups[1]
    assert not any(parameter.requires_grad for module in groups[0] for parameter in module.parameters())
    assert any(parameter.requires_grad for module in groups[1] for parameter in module.parameters())
    assert model.conditioning_is_frozen()
    assert list(model.state_dict()) == names


@pytest.mark.parametrize("training_mode", ["full", "custom_lora"])
def test_sensenova_branch_groups_reject_trainable_understanding(training_mode):
    from relax.models.sensenova.adapter import SenseNovaSFTAdapter

    model = _branch_group_model("full" if training_mode == "full" else "lora")
    if training_mode == "custom_lora":
        pytest.importorskip("peft")
        from relax.backends.fsdp.lora import inject_lora_adapter

        inject_lora_adapter(model, rank=2, alpha=2, target_modules=["q_proj", "q_proj_mot_gen"])
    assert not model.conditioning_is_frozen()
    with pytest.raises(ValueError, match="requires a frozen understanding branch"):
        SenseNovaSFTAdapter().prepare_for_sharding(model)
    assert not hasattr(model, "_fsdp_module_groups")
    assert not any(getattr(module, "_fsdp_fine_grained", False) for module in model.modules())


def test_sensenova_native_group_hooks_complete_each_branch_at_mlp(monkeypatch):
    """Exercise installed PyTorch's real grouped hooks without CUDA/FSDP state.

    Each branch completes its group at MLP, with every member called; an unused
    branch has no hook calls. GPU tests cover actual collectives and checkpoint
    backward state separately.
    """
    from torch.distributed.fsdp._fully_shard._fsdp_state import _register_group_forward_hooks

    from relax.backends.fsdp.runtime import resolve_module_groups
    from relax.models.sensenova.adapter import SenseNovaSFTAdapter

    monkeypatch.setenv("RELAX_ATTENTION_IMPLEMENTATION", "torch")
    model = _branch_group_model()
    SenseNovaSFTAdapter().prepare_for_sharding(model)
    model.float()
    groups = resolve_module_groups(model)
    calls, complete, pending, handles = [], [], [set(), set()], []
    for group_index, modules in enumerate(groups):

        def pre(module, args, kwargs, index=group_index):
            calls.append((index, module))
            return args, kwargs

        def post(module, args, output, index=group_index):
            complete.append((index, module))
            return output

        handles.append(_register_group_forward_hooks(modules, pre, post, pending[group_index], lambda output: output))
    layer = model.pipe.dit.language_model.model.layers[0]
    hidden = torch.randn(1, 3, 32, requires_grad=True)
    indexes = torch.arange(3).repeat(3, 1)
    try:
        for generation in (False, True, True):
            calls.clear()
            complete.clear()
            output = layer(
                hidden_states=hidden,
                exist_non_image_gen_tokens=not generation,
                exist_image_gen_tokens=generation,
                indexes=indexes,
                attention_mask=None,
                update_cache=False,
            )
            index = int(generation)
            assert {item[0] for item in calls} == {index}
            assert complete == [(index, groups[index][-1])]
            spatial_norm = layer.self_attn.q_norm_hw_mot_gen if generation else layer.self_attn.q_norm_hw
            assert calls.count((index, spatial_norm)) == 1
            assert {item[1] for item in calls} == set(groups[index])
            assert not any(pending)
            output.sum().backward()
    finally:
        for handle in handles:
            handle.remove()
