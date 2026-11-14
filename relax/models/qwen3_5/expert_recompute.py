# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Uninstalled expert-only checkpoint wrapper; caller must enforce actual GPU
gate."""


def install(model: list, checkpoint_function, mask_key: str, *, expert_type=None) -> dict:
    """Keep native experts/backend/wgrad; derive full coverage from actual
    packed metadata."""
    assert len(model) == 1, "Only PP1 single model chunk is supported"
    chunk = model[0]
    while hasattr(chunk, "module"):
        chunk = chunk.module
    decoder = getattr(chunk, "language_model", chunk).decoder
    assert len(decoder.layers) == 40
    if hasattr(decoder, "_qwen_expert_checkpoint_receipt"):
        return decoder._qwen_expert_checkpoint_receipt
    if expert_type is None:
        from megatron.core.transformer.moe.experts import TEGroupedMLP

        expert_type = TEGroupedMLP
    # Validate every target before changing any hook or method.
    experts = [layer.mlp.experts for layer in decoder.layers]
    assert len({id(expert) for expert in experts}) == 40
    for index, (layer, expert) in enumerate(zip(decoder.layers, experts)):
        assert isinstance(expert, expert_type) and expert.training
        assert layer.layer_number == index + 1
        config = expert.config
        assert (
            config.tensor_model_parallel_size,
            config.pipeline_model_parallel_size,
            config.context_parallel_size,
            config.expert_model_parallel_size,
            config.expert_tensor_parallel_size,
        ) == (4, 1, 1, 8, 1)
        assert config.bf16 and not config.fp8 and not config.fp4
        assert config.recompute_granularity == "full" and config.recompute_method == "block"
        assert expert._with_fused_impl and not expert.activation_recompute
        assert not hasattr(expert, "_qwen_expert_checkpoint_state")
    states = []
    for index, layer in enumerate(decoder.layers):
        expert = layer.mlp.experts
        assert not hasattr(expert, "_qwen_expert_checkpoint_state")
        assert expert._with_fused_impl and not expert.activation_recompute
        state = dict(
            layer=index,
            active=None,
            checkpoint_calls=0,
            replay_calls=0,
            full_bypass_calls=0,
            checkpoint=checkpoint_function,
        )
        states.append(state)

        expert._qwen_expert_checkpoint_state = state
        layer._qwen_expert_checkpoint_state = state
        state["mask_key"] = mask_key
    receipt = dict(
        states=states,
        mtp_unchanged=True,
        backend_and_parameters_unchanged=True,
        native_wgrad_owner_unchanged=True,
    )
    decoder._qwen_expert_checkpoint_receipt = receipt
    return receipt


def forward(self, hidden, counts, probs, original, state):
    assert state["active"] is not None, "Expert called outside its transformer layer"
    if state["active"]:
        state["full_bypass_calls"] += 1
        return original(hidden, counts, probs)
    import torch

    checkpoint_function = state["checkpoint"]

    assert torch.is_grad_enabled(), "Non-full training layer must retain gradients"
    state["checkpoint_calls"] += 1

    def body(hidden, counts, probs):
        # Replay captures only the bound expert forward and this fixed state.
        # It never consults current-wave packed metadata or an active layer context.
        if torch.is_grad_enabled():
            state["replay_calls"] += 1
        output, bias = original(hidden, counts, probs)
        assert bias is None
        return output

    return checkpoint_function(body, False, hidden, counts, probs), None
