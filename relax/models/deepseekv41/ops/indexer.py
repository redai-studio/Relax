# Modified for Relax integration; see NOTICE.

from typing import Any

import torch
from megatron.core.extensions.transformer_engine import TELinear
from megatron.core.transformer.module import MegatronModule
from torch import nn
from torch.nn import functional as F

from .compressor import RMSNorm
from .position import apply_rotary_emb
from .quant import fake_quant_fp4


def select_candidate_blocks(
    logits: torch.Tensor, lengths: torch.Tensor, topk_blocks: int, block_size: int
) -> torch.Tensor:
    width = logits.shape[-1]
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf).unflatten(-1, (-1, block_size)).amax(-1)
    last = (lengths - 1) // block_size
    scores = scores.masked_fill(torch.arange(scores.shape[-1], device=logits.device) == last, torch.inf)
    top = scores.topk(min(topk_blocks, scores.shape[-1]), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > -torch.inf)
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


def indexer_logits(q: torch.Tensor, keys: torch.Tensor, weights: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    from .kernels.indexer_fwd import indexer_fwd

    batch, seqlen = q.shape[:2]
    starts = torch.zeros(seqlen, dtype=torch.int32, device=q.device)
    ends = lengths.to(torch.int32)
    return torch.stack(
        [
            indexer_fwd(q[b].contiguous(), keys[b].contiguous(), weights[b].float().contiguous(), starts, ends)
            for b in range(batch)
        ]
    )


class DeepSeekV41Indexer(MegatronModule):
    def __init__(self, config: Any, layer_id: int, owns_k: bool) -> None:
        super().__init__(config=config)
        self.layer_id = layer_id
        self.index_n_heads = config.dsa_indexer_n_heads
        self.index_head_dim = config.dsa_indexer_head_dim
        self.topk = config.dsa_indexer_topk
        self.rope_dim = config.qk_pos_emb_head_dim
        self.candidate_source = layer_id == config.v41_candidate_source_layer_id
        self.uses_candidates = 0 <= config.v41_candidate_source_layer_id < layer_id
        self.candidate_topk_blocks = config.v41_candidate_topk_blocks
        self.candidate_block_size = config.v41_candidate_block_size
        self.query_chunk = 8192
        for name, in_dim, out_dim in (
            ("linear_wq_b", config.q_lora_rank, self.index_n_heads * self.index_head_dim),
            ("linear_weights_proj", config.hidden_size, self.index_n_heads),
        ):
            setattr(
                self,
                name,
                TELinear(
                    in_dim,
                    out_dim,
                    config=config,
                    init_method=config.init_method,
                    bias=False,
                    skip_bias_add=False,
                    skip_weight_param_allocation=False,
                    parallel_mode="duplicated",
                ),
            )
        if owns_k:
            self.linear_wk = nn.Linear(config.kv_lora_rank, self.index_head_dim, bias=False, dtype=config.params_dtype)
            self.k_norm = RMSNorm(self.index_head_dim, config.layernorm_epsilon)
        # Freeze the complete retrieval indexer during training.
        self.requires_grad_(False)

    @torch.no_grad()
    def index_keys(self, latent: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        keys = self.k_norm(F.linear(latent, self.linear_wk.weight)).clone()
        apply_rotary_emb(keys[..., -self.rope_dim :], freqs)
        return fake_quant_fp4(keys)

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        keys: torch.Tensor,
        freqs: torch.Tensor,
        lengths: torch.Tensor,
        candidates: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, seqlen, _ = x.shape
        q = self.linear_wq_b(qr)[0].view(batch, seqlen, self.index_n_heads, self.index_head_dim).clone()
        apply_rotary_emb(q[..., -self.rope_dim :], freqs)
        q = fake_quant_fp4(q)
        weights = self.linear_weights_proj(x)[0] * (self.index_head_dim**-0.5 * self.index_n_heads**-0.5)
        n_kv = keys.shape[1]
        indices, masks = [], []
        for start in range(0, seqlen, self.query_chunk):
            end = min(start + self.query_chunk, seqlen)
            scores = indexer_logits(q[:, start:end], keys, weights[:, start:end], lengths[start:end])
            if self.candidate_source:
                masks.append(
                    select_candidate_blocks(
                        scores,
                        lengths[start:end].unsqueeze(-1),
                        self.candidate_topk_blocks,
                        self.candidate_block_size,
                    )
                )
            elif self.uses_candidates:
                if candidates is None:
                    raise RuntimeError("DeepSeek V4.1 candidate source must run before this indexer.")
                scores.masked_fill_(~candidates[:, start:end], -torch.inf)
            if n_kv < self.topk:
                scores = F.pad(scores, (0, self.topk - n_kv), value=-torch.inf)
            values, top = scores.topk(self.topk, dim=-1)
            top = top.masked_fill(values == -torch.inf, -1).sort(dim=-1).values.to(torch.int64)
            indices.append(torch.where(top < lengths[start:end].unsqueeze(-1), top, -1))
        return torch.cat(indices, dim=1), torch.cat(masks, dim=1) if self.candidate_source else candidates
