# Modified for Relax integration; see NOTICE.

import torch

from .position import apply_rotary_emb


def fake_quant_fp4(x: torch.Tensor) -> torch.Tensor:
    from .kernels.fake_quant import fake_quant_fp4 as quantize

    return quantize(x)


def fake_quant_compressed_kv(x: torch.Tensor) -> torch.Tensor:
    from .kernels.fake_quant import fake_quant_compressed_kv as quantize

    return quantize(x)


def _fp8_ste(x: torch.Tensor) -> torch.Tensor:
    from .kernels.fake_quant import fake_quant_fp8_block

    return fake_quant_fp8_block(x, 64)


@torch.no_grad()
def _cache_roundtrip(
    values: torch.Tensor, weight: torch.Tensor | None = None, eps: float = 0.0, freqs: torch.Tensor | None = None
) -> torch.Tensor:
    from sglang.kernels.ops.attention.dsv4.dequant_k_cache import dequantize_k_cache_paged

    batch, seqlen, dim = values.shape
    page_size = 64
    page_bytes = ((584 * page_size + 575) // 576) * 576
    pages = (batch * seqlen + page_size) // page_size + 1
    cache = torch.zeros(pages, page_bytes, dtype=torch.uint8, device=values.device)
    flat = values.reshape(batch * seqlen, dim).contiguous()
    if weight is None:
        from sglang.kernels.ops.attention.dsv4.attn import fused_store_cache

        locations = torch.arange(batch * seqlen, dtype=torch.int64, device=values.device)
        fused_store_cache(input=flat, cache=cache, indices=locations, page_size=page_size, type="flashmla")
    else:
        from sglang.kernels.ops.attention.dsv4.elementwise import fused_k_norm_rope_flashmla

        locations = torch.arange(batch * seqlen, dtype=torch.int32, device=values.device)
        positions = torch.arange(seqlen, dtype=torch.int32, device=values.device).repeat(batch)
        fused_k_norm_rope_flashmla(
            kv=flat,
            kv_weight=weight.to(values.dtype).contiguous(),
            eps=eps,
            freqs_cis=freqs.contiguous(),
            positions=positions,
            out_loc=locations,
            kvcache=cache,
            page_size=page_size,
        )
    return dequantize_k_cache_paged(cache, locations, page_size).reshape(batch, seqlen, dim).to(values.dtype)


def kv_norm_rope_fp8(
    values: torch.Tensor, weight: torch.Tensor, eps: float, freqs: torch.Tensor, rope_dim: int
) -> torch.Tensor:
    normalized = values.float()
    normalized = normalized * torch.rsqrt(normalized.square().mean(-1, keepdim=True) + eps) * weight.float()
    apply_rotary_emb(normalized[..., -rope_dim:], freqs)
    soft = normalized.to(values.dtype).clone()
    soft[..., :-rope_dim] = _fp8_ste(soft[..., :-rope_dim])
    hard = _cache_roundtrip(values, weight, eps, freqs)
    return soft + (hard - soft).detach()


def compressed_kv_stored(latent: torch.Tensor, rope_dim: int) -> torch.Tensor:
    soft = latent.clone()
    soft[..., :-rope_dim] = _fp8_ste(soft[..., :-rope_dim])
    hard = _cache_roundtrip(latent)
    return soft + (hard - soft).detach()
