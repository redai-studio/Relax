# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Experimental decoder-level state plumbing for Qwen3.5.

Consumes already prepared hidden states and absolute rotary embeddings. The
caller owns packed-sequence splitting and restores full-length output layout.
No production actor dispatch is installed by importing this module.
"""

from contextvars import ContextVar
from types import MethodType

import torch
from torch.utils.checkpoint import checkpoint

from relax.utils.training.qwen35_prefix_state import GDNState, attention_forward, gdn_forward


_active_state = ContextVar("qwen35_prefix_layer_state", default=None)
_row_layout = ContextVar("qwen35_prefix_row_layout", default=None)


def install_row_dispatch(module):
    if getattr(module, "parallel_mode", None) != "row" or hasattr(module, "_relax_prefix_row_forward"):
        return
    if not hasattr(module, "te_return_bias"):
        raise NotImplementedError("Prefix row projection requires the current Megatron TE interface")
    if getattr(module, "bias", None) is not None and module.bias.numel():
        raise NotImplementedError("Prefix row projection does not implement fused bias reduction")
    for name in (
        "ub_overlap_ag_fprop",
        "ub_overlap_rs_dgrad",
        "ub_bulk_dgrad",
        "ub_bulk_wgrad",
        "ub_overlap_rs_fprop",
        "ub_overlap_ag_dgrad",
        "gemm_bias_unfused_add",
        "delay_wgrad_compute",
    ):
        if getattr(module, name, False):
            raise NotImplementedError(f"Prefix row projection does not implement {name}")
    original = module.forward
    module._relax_prefix_row_forward = original

    def forward(self, x):
        layout = _row_layout.get()
        if layout is None:
            return original(x)
        from megatron.core import parallel_state as mpu
        from megatron.core import tensor_parallel

        offset, packed_length = layout
        count = x.shape[0]
        if offset + count > packed_length:
            raise ValueError("Prefix projection padding exceeds the original packed layout")
        # Keep original BF16 reduction ownership/order. A changed SP partition
        # otherwise introduces cumulative rounding drift even with zero prefix.
        mode, sp = self.parallel_mode, self.sequence_parallel
        self.parallel_mode, self.sequence_parallel = None, False
        try:
            output, bias = original(x)  # Local TE GEMM; fused wgrad remains intact.
        finally:
            self.parallel_mode, self.sequence_parallel = mode, sp
        output = torch.nn.functional.pad(output, (0, 0, 0, 0, offset, packed_length - offset - count))
        group = mpu.get_tensor_model_parallel_group()
        if sp:
            output = tensor_parallel.reduce_scatter_to_sequence_parallel_region(output, group=group)
            output = tensor_parallel.gather_from_sequence_parallel_region(
                output, tensor_parallel_output_grad=False, group=group
            )
        else:
            output = tensor_parallel.reduce_from_tensor_model_parallel_region(output, group=group)
        output = output[offset : offset + count].contiguous()
        if sp:
            output = tensor_parallel.scatter_to_sequence_parallel_region(output, group=group)
        return output, bias

    module.forward = MethodType(forward, module)


def install_mixer_dispatch(mixer):
    """Instance-local dispatcher; normal forwards retain the original
    method."""
    if hasattr(mixer, "_relax_prefix_original_forward"):
        return
    original = mixer.forward
    mixer._relax_prefix_original_forward = original

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        active = _active_state.get()
        if active is None or active[0] is not self:
            return original(hidden_states, attention_mask, **kwargs)
        _, incoming, outgoing = active
        if hasattr(self, "gated_delta_rule"):
            output, bias, state = gdn_forward(self, hidden_states, incoming, save_state=outgoing is not None)
        else:
            output, bias, state = attention_forward(
                self, hidden_states, kwargs.get("rotary_pos_emb"), incoming, save_state=outgoing is not None
            )
        if outgoing is not None:
            outgoing.append(state)
        return output, bias

    mixer.forward = MethodType(forward, mixer)


def validate_decoder(decoder):
    """Reject interfaces that this decoder executor does not preserve yet."""
    config = decoder.config
    for name in (
        "fp8",
        "fp4",
        "cpu_offloading",
        "enable_hyper_connections",
        "fine_grained_activation_offloading",
        "moe_paged_stash",
        "num_moe_experts",
        "tp_comm_overlap",
    ):
        if getattr(config, name, None):
            raise NotImplementedError(f"Detached prefix decoder does not implement {name}")
    if config.window_size is not None or config.softmax_type != "vanilla":
        raise NotImplementedError("Prefix attention requires full causal vanilla softmax")
    if config.hidden_dropout or config.attention_dropout:
        raise ValueError("Detached prefix validation requires zero dropout")
    if config.context_parallel_size != 1:
        raise NotImplementedError("Detached prefix decoder lacks CP state redistribution")
    if config.pipeline_model_parallel_size != 1:
        raise NotImplementedError("Detached prefix decoder lacks PP state scheduling")
    if getattr(config, "mtp_num_layers", None):
        raise NotImplementedError("Detached prefix decoder does not implement MTP alignment")
    if getattr(config, "recompute_granularity", None) not in (None, "full"):
        raise NotImplementedError("Selective recomputation needs separate state replay validation")
    for layer in decoder.layers:
        mixer = layer.self_attention
        if not (hasattr(mixer, "gated_delta_rule") or hasattr(mixer, "get_query_key_value_tensors")):
            raise NotImplementedError(f"No state adapter for {type(mixer).__name__}")
        install_mixer_dispatch(mixer)
        for module in layer.modules():
            install_row_dispatch(module)


def clone_states(states):
    return tuple(
        state.detached() if isinstance(state, GDNState) else tuple(t.detach().clone() for t in state)
        for state in states
    )


def decoder_chunk(decoder, hidden_states, rotary_pos_emb, states=None, *, build_state=False, row_layout=None):
    """Run an unpacked chunk, preserving original transformer layer weights.

    Prefix construction must run under no_grad. A suffix checkpoint captures
    its own immutable state, so subsequent microbatches cannot change replay.
    Prefix-state writes occur only during no_grad construction, never replay.
    """
    if build_state and torch.is_grad_enabled():
        raise ValueError("Prefix construction must execute under torch.no_grad")
    if states is not None and len(states) != len(decoder.layers):
        raise ValueError("Exactly one state is required per decoder layer")
    snapshots = None if states is None else clone_states(states)
    collected = []
    for index, layer in enumerate(decoder.layers):
        incoming = None if snapshots is None else snapshots[index]
        outgoing = [] if build_state else None

        def run(hidden, layer=layer, incoming=incoming, outgoing=outgoing):
            token = _active_state.set((layer.self_attention, incoming, outgoing))
            layout_token = _row_layout.set(row_layout)
            # Multiple chunks share one Megatron microbatch. TE must accumulate
            # every wgrad, never overwrite an earlier chunk's main_grad. FP8 is
            # rejected above; disabling its transpose-cache hint is safe here.
            hints = [
                (m, m.disable_parameter_transpose_cache)
                for m in layer.modules()
                if getattr(m, "fuse_wgrad_accumulation", False) and hasattr(m, "disable_parameter_transpose_cache")
            ]
            for module, _ in hints:
                module.disable_parameter_transpose_cache = True
            try:
                output, context = layer(
                    hidden_states=hidden,
                    attention_mask=None,
                    rotary_pos_emb=rotary_pos_emb,
                    packed_seq_params=None,
                    inference_context=None,
                )
                if context is not None:
                    raise NotImplementedError("Cross-attention context is not supported")
                return output
            finally:
                for module, previous in hints:
                    module.disable_parameter_transpose_cache = previous
                _row_layout.reset(layout_token)
                _active_state.reset(token)

        if torch.is_grad_enabled() and getattr(decoder.config, "recompute_granularity", None) == "full":
            hidden_states = checkpoint(run, hidden_states, use_reentrant=False)
        else:
            hidden_states = run(hidden_states)
        if build_state:
            if len(outgoing) != 1:
                raise RuntimeError("Expected exactly one state export per decoder layer")
            collected.append(outgoing[0])
    if decoder.final_layernorm is not None:
        hidden_states = decoder.final_layernorm(hidden_states)
    return hidden_states, tuple(collected) if build_state else None
