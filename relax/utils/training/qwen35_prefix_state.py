# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Explicit stateful Qwen3.5 mixer training primitives.

Experimental: not installed into the actor automatically. States are immutable
snapshots; callers must provide authoritative rotary embeddings for each slice.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GDNState:
    convolution: torch.Tensor
    recurrent: torch.Tensor

    def detached(self):
        return GDNState(self.convolution.detach().clone(), self.recurrent.detach().clone())


def gdn_forward(module, hidden_states, state=None, *, save_state=True):
    """Run one unpacked sequence/batch through the actual Megatron GDN weights.

    TP projections and SP collectives remain owned by the module. CP state
    redistribution and packed reset boundaries are not implemented here.
    """
    from fla.modules.convolution import causal_conv1d

    if module.cp_size != 1:
        raise NotImplementedError("GDN prefix state requires CP state redistribution support")
    if module.config.deterministic_mode:
        raise NotImplementedError("GDN deterministic convolution state adapter is not implemented")
    projected, _ = module.in_proj(hidden_states)
    projected = projected.transpose(0, 1)
    batch, length = projected.shape[:2]
    qkv, gate, beta, alpha = torch.split(
        projected,
        [
            2 * module.qk_dim_local_tp + module.v_dim_local_tp,
            module.v_dim_local_tp,
            module.num_value_heads // module.tp_size,
            module.num_value_heads // module.tp_size,
        ],
        dim=-1,
    )
    gate = gate.reshape(batch, length, -1, module.value_head_dim)
    beta = beta.reshape(batch, length, -1)
    alpha = alpha.reshape(batch, length, -1)
    incoming = None if state is None else state.detached()
    qkv, convolution = causal_conv1d(
        x=qkv,
        weight=module.conv1d.weight.squeeze(1),
        bias=module.conv1d.bias,
        activation=module.activation,
        initial_state=None if incoming is None else incoming.convolution,
        output_final_state=save_state,
        cu_seqlens=None,
    )
    with torch._dynamo.config.patch(disable=True):
        query, key, value, gate, beta, alpha = module._prepare_qkv_for_gated_delta_rule(
            qkv,
            gate,
            beta,
            alpha,
            batch,
            length,
        )
    g, beta = module._compute_g_and_beta(module.A_log, module.dt_bias, alpha, beta)
    output, recurrent = module.gated_delta_rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=None if incoming is None else incoming.recurrent,
        output_final_state=save_state,
        use_qk_l2norm_in_kernel=False,
        cu_seqlens=None,
    )
    output = module._apply_gated_norm(output, gate)
    output = output.reshape(batch, length, -1).transpose(0, 1).contiguous()
    output, bias = module.out_proj(output)
    return output, bias, GDNState(convolution, recurrent).detached() if save_state else None


def attention_forward(module, hidden_states, rotary_pos_emb, state=None, *, save_state=True):
    """Unpacked causal attention suffix with absolute mRoPE and detached KV.

    Uses the original TE attention with bottom-right causal alignment.
    CP/attention bias/dropout must be validated by the caller before dispatch.
    """
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.rope import apply_rotary_pos_emb_absolute
    from megatron.core.transformer.enums import AttnMaskType

    if module.config.context_parallel_size != 1:
        raise NotImplementedError("Attention prefix state needs CP KV distribution support")
    if module.config.attention_dropout:
        raise NotImplementedError("Prefix equivalence requires zero attention dropout")
    gate = None
    if module.config.attention_output_gate:
        q, k, v, gate = module.get_query_key_value_tensors(hidden_states, None, output_gate=True)
    else:
        q, k, v = module.get_query_key_value_tensors(hidden_states, None)
    if module.config.no_rope_freq and module.config.no_rope_freq[module.layer_number - 1]:
        rotary_pos_emb = None
    if rotary_pos_emb is not None:
        qr, kr = rotary_pos_emb if isinstance(rotary_pos_emb, tuple) else (rotary_pos_emb, rotary_pos_emb)
        if qr is not None:
            q = apply_rotary_pos_emb_absolute(q, qr, config=module.config, cu_seqlens=None)
        if kr is not None:
            k = apply_rotary_pos_emb_absolute(k, kr, config=module.config, cu_seqlens=None)
    if state is not None:
        pk, pv = (t.detach().clone() for t in state)
        k, v = torch.cat((pk, k)), torch.cat((pv, v))
    saved = (k.detach().clone(), v.detach().clone()) if save_state else None
    output = module.core_attention(
        q, k, v, None, attn_mask_type=AttnMaskType.causal_bottom_right, packed_seq_params=None
    )
    if gate is not None:
        output = module._apply_output_gate(output, gate)
    output, bias = module.linear_proj(output)
    return output, bias, saved
