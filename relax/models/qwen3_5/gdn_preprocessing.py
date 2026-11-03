# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Install fused packed GDN preprocessing for TP4 and CP1."""

_FullPre = None


def pre(
    self,
    qkvzba,
    batch,
    seq_len,
    cp_size_headwise,
    cp_group_headwise,
    cu_seqlens_q=None,
    chunkwise_cp_context=None,
    packed_seq_params=None,
):
    import torch
    from fla.ops.utils import prepare_chunk_indices

    assert batch == cp_size_headwise == 1 and chunkwise_cp_context is None
    assert cp_group_headwise is None or cp_group_headwise.size() == 1
    assert packed_seq_params is not None and packed_seq_params.qkv_format == "thd"
    assert packed_seq_params.cu_seqlens_q_padded is None
    assert cu_seqlens_q is packed_seq_params.cu_seqlens_q
    assert qkvzba.shape == (seq_len, 1, 3088)
    cache = getattr(packed_seq_params, "_qwen_fullpre_metadata", None)
    if cache is None or cache[0] is not cu_seqlens_q:
        boundaries = packed_seq_params.cu_seqlens_q_cpu
        assert boundaries[0] == 0 and boundaries[-1] == seq_len
        cpu_cu = torch.tensor(boundaries, dtype=torch.long, device="cpu")
        chunks = prepare_chunk_indices(cu_seqlens_q, 64, cu_seqlens_cpu=cpu_cu)
        cache = (cu_seqlens_q, cpu_cu, chunks)
        packed_seq_params._qwen_fullpre_metadata = cache
    q, k, v, z, beta_raw, alpha_raw = _FullPre.apply(qkvzba, self.conv1d.weight, self.A_log, self.dt_bias, *cache)
    g, gates = self._compute_gates(self.A_log, self.dt_bias, batch, seq_len, beta_raw, alpha_raw)
    return q, k, v, z, gates["beta"], g


def install(model):
    global _FullPre
    from pathlib import Path

    from megatron.core.ssm.gated_delta_net import GatedDeltaNet

    import relax.models.qwen3_5.ops.gdn_preprocessing as math_module
    from relax.models.qwen3_5.ops.gdn_launch_policy import install as fixed_install

    assert _FullPre is None
    targets = [m for chunk in model for m in chunk.modules() if isinstance(m, GatedDeltaNet)]
    assert len(targets) == 30 and len({id(m) for m in targets}) == 30
    for m in targets:
        assert m.tp_size == 4 and m.config.context_parallel_size == 1
        assert m.num_key_heads == 16 and m.num_value_heads == 32
        assert m.key_head_dim == m.value_head_dim == 128 and m.use_qk_l2norm
        assert m.qk_dim_local_tp == 512 and m.v_dim_local_tp == 1024
        assert not m.conv_bias and not m.gdn_pre_gated_delta_rule_fusion
        assert m.activation in ("silu", "swish")
        assert m.conv1d.weight.shape == (2048, 1, 4) and m.A_log.shape == m.dt_bias.shape == (8,)
    policy = fixed_install(math_module, Path(math_module.__file__).parent)
    _FullPre = math_module.FullPre
    return dict(
        cores=len(targets),
        tp=4,
        cp=1,
        fixed_policy=policy,
        native_compiled_gBeta_preserved=True,
        native_Gate_norm_preserved=True,
    )
