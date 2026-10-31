# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Keye on SGLang 0.5.21, with paged sparse decode and eager prefill.

The index cache participates in SGLang's existing KV memory saver region.
Overlap scheduling and speculative decoding are unsupported.
"""

from collections.abc import Iterable, Iterator
from typing import Any

import torch
import torch.nn.functional as F
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.managers.mm_utils import MultiModalityDataPaddingPatternMultimodalTokens, general_mm_embed_routine
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import get_req_to_token_pool, get_token_to_kv_pool
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3_moe import Qwen3MoeAttention, Qwen3MoeForCausalLM, Qwen3MoeModel
from sglang.srt.runtime_context import get_model, get_parallel

from relax.models.keye_vl2.indexer import KeyeIndexer, indexer_scores
from relax.models.keye_vl2.norm import RMSNorm as StableRMSNorm
from relax.models.keye_vl2.sparse_attention import compact_indices, sparse_attention
from relax.models.keye_vl2.vision import build_vision, image_features

from .ops import expert_forward, paged_decode, rope


try:
    from sglang.srt.runtime_context import process_model_config
except ImportError:
    # SGLang 0.5.17 keeps the resolved ModelConfig on ServerArgs.
    from sglang.srt.runtime_context import get_server_args

    def process_model_config() -> Any:
        return get_server_args().get_model_config()


def padded_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    rows = x.numel() // x.shape[-1]
    if 0 < rows < 64:
        padded = F.pad(x.reshape(rows, x.shape[-1]), (0, 0, 0, 64 - rows))
        output = F.linear(padded, weight, bias)[:rows]
        return output.reshape(*x.shape[:-1], output.shape[-1])
    return F.linear(x, weight, bias)


class KeyeLinear(torch.nn.Linear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return padded_linear(x, self.weight, self.bias)


class KeyeLinearMethod(UnquantizedLinearMethod):
    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        return padded_linear(x, layer.weight, bias)


class KeyeRowLinearMethod(UnquantizedLinearMethod):
    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        group = get_parallel().attn_tp_group
        if group.world_size > 1:
            x = group.all_gather(x, dim=-1)
            if group.rank_in_group != 0:
                return x.new_zeros((*x.shape[:-1], layer.weight.shape[0]))
        # Round the complete dot once, before the existing layer communicator
        # broadcasts it by summing rank zero's result with the other ranks' zeros.
        return padded_linear(x, layer.weight, bias)


def _replicated_expert_weight_loader(
    parameter: torch.nn.Parameter,
    weight: torch.Tensor,
    name: str = "",
    shard_id: str = "w2",
    expert_id: int | None = None,
) -> None:
    default_weight_loader(parameter if expert_id is None else parameter[expert_id], weight)


def decode_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    index_query: torch.Tensor,
    index_key: torch.Tensor,
    weights: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Batch single-token decode while masking unused cache slots."""
    valid = torch.arange(key.shape[1], device=query.device)[None] < lengths[:, None]
    key = key.masked_fill(~valid[:, :, None, None], 0)
    value = value.masked_fill(~valid[:, :, None, None], 0)
    index_key = index_key.masked_fill(~valid[:, :, None], 0)
    scores = torch.einsum("bhd,bkd->bhk", index_query, index_key) * 0.125
    scores = (scores.relu() * weights[:, :, None]).sum(1)
    scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
    support = torch.zeros_like(valid).scatter_(1, scores.topk(min(2048, key.shape[1]), -1).indices, True) & valid
    repeats = query.shape[1] // key.shape[2]
    output = F.scaled_dot_product_attention(
        query[:, :, None],
        key.repeat_interleave(repeats, 2).transpose(1, 2),
        value.repeat_interleave(repeats, 2).transpose(1, 2),
        attn_mask=support[:, None, None],
    )
    return output.flatten(1)


class KeyeAttention(Qwen3MoeAttention):
    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split((self.q_size, self.kv_size, self.kv_size), -1)
        q = self.q_norm(q.reshape(-1, self.num_heads, self.head_dim))
        k = self.k_norm(k.reshape(-1, self.num_kv_heads, self.head_dim))
        # Decode graph batches always expose an mRoPE scratch buffer, even
        # for language-only models, where replay never updates that buffer.
        rope_positions = getattr(forward_batch, "mrope_positions", None) if self.keye_multimodal_enabled else None
        rope_positions = rope_positions.transpose(0, 1) if rope_positions is not None else positions
        q, k = rope(q, rope_positions, self.rope_theta), rope(k, rope_positions, self.rope_theta)
        v = v.reshape(-1, self.num_kv_heads, self.head_dim)
        from fast_hadamard_transform import hadamard_transform

        projected = self.sa_indexer.q_proj(hidden_states)
        iq, iw = projected.split((1024, 16), -1)
        iq = self.sa_indexer.q_norm(iq.reshape(-1, 16, 64))
        ik = self.sa_indexer.k_norm(self.sa_indexer.k_proj(hidden_states).unsqueeze(-2))
        iq = hadamard_transform(rope(iq, rope_positions, self.rope_theta), scale=0.125)
        ik = hadamard_transform(rope(ik, rope_positions, self.rope_theta), scale=0.125)
        pool = get_token_to_kv_pool()
        pool.set_kv_buffer(self.attn, KVWriteLoc(forward_batch.out_cache_loc, None), k, v)
        if not hasattr(pool, "keye_index_cache"):
            pool.keye_index_cache = {}
        layer_id = self.attn.layer_id
        if layer_id not in pool.keye_index_cache:
            with pool.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
                pool.keye_index_cache[layer_id] = torch.empty(
                    (pool.size + pool.page_size, 64), device=k.device, dtype=k.dtype
                )
        cache = pool.keye_index_cache[layer_id]
        cache[forward_batch.out_cache_loc] = ik.squeeze(1)
        req_to_token = get_req_to_token_pool().req_to_token
        outputs = []
        offset = 0
        lengths = forward_batch.seq_lens_cpu
        if forward_batch.forward_mode.is_decode():
            # A fixed index-score capacity is safe for CUDA graph replay;
            # GPU lengths gate actual cache reads. The shared GQA forward
            # reads only the selected <=2048 slots.
            output = paged_decode(
                q,
                pool.get_key_buffer(layer_id),
                pool.get_value_buffer(layer_id),
                iq,
                cache,
                iw,
                req_to_token,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                min(req_to_token.shape[1], get_model().context_length or req_to_token.shape[1]),
            )
            output, _ = self.o_proj(output)
            return output
        elif forward_batch.forward_mode.is_extend():
            query_lengths = forward_batch.extend_seq_lens_cpu
            req_to_token = req_to_token[forward_batch.req_pool_indices]
        else:
            raise ValueError("Keye reference rollout supports extend/decode only")
        for batch_index, (length, query_length) in enumerate(zip(lengths, query_lengths)):
            length = int(length)  # CPU metadata, never a GPU synchronization.
            slots = req_to_token[batch_index, :length].long()
            all_k = pool.get_key_buffer(layer_id)[slots]
            all_v = pool.get_value_buffer(layer_id)[slots]
            keys = cache[slots].unsqueeze(1)
            for start in range(offset, offset + query_length, 128):
                end = min(start + 128, offset + query_length)
                scores = indexer_scores(iq[start:end], keys, iw[start:end])
                causal = torch.arange(length, device=q.device)[None, :] <= positions[start:end, None]
                scores.masked_fill_(~causal, torch.finfo(scores.dtype).min)
                selected = scores.topk(min(2048, length), -1).indices
                selected_slots, counts = compact_indices(selected, causal.gather(1, selected))
                result = sparse_attention(q[start:end], all_k, all_v, selected_slots, counts)
                outputs.append(result.reshape(end - start, -1))
            offset += query_length
        output, _ = self.o_proj(torch.cat(outputs, 0))
        return output


class KeyeExperts(FusedMoE):
    def forward(self, hidden_states: torch.Tensor, topk_output: Any, pre_quant_input: Any = None) -> torch.Tensor:
        if self.num_local_experts != self.num_experts:
            raise ValueError("Keye rollout currently requires tensor-parallel experts")
        return expert_forward(
            hidden_states,
            self.w13_weight,
            self.w2_weight,
            topk_output.topk_weights,
            topk_output.topk_ids,
            get_parallel().moe_tp_group,
        )


class KeyeRMSNorm(StableRMSNorm):
    @staticmethod
    def convert(norm: RMSNorm) -> None:
        # Keep the module object: SGLang's layer communicator retains it.
        weight, eps = norm.weight, norm.variance_epsilon
        norm.__class__ = KeyeRMSNorm
        StableRMSNorm.__init__(
            norm, normalized_shape=weight.numel(), eps=eps, dtype=weight.dtype, device=weight.device
        )
        norm.weight = weight

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        post_residual_addition: torch.Tensor | None = None,
        quant_linear: torch.nn.Module | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return super().forward(x)
        if post_residual_addition is not None:
            residual = residual + post_residual_addition
        # Megatron rounds the residual add before computing norm statistics.
        # SGLang's fused add/norm instead uses the unrounded FP32 sum.
        residual = x + residual
        return super().forward(residual), residual


class KeyeRouter(ReplicatedLinear):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        # Match Megatron's shape-independent dot, with FP32 routing scores.
        return F.linear(x.double(), self.weight.double()).float(), None


def keye_routing(
    hidden_states: torch.Tensor, gating_output: torch.Tensor, topk: int, renormalize: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    scores, indices = torch.topk(gating_output, k=topk, dim=1, sorted=True)
    return torch.softmax(scores, dim=-1, dtype=torch.float32), indices.int()


class KeyeLanguageModel(Qwen3MoeModel):
    def get_input_embeddings(self) -> torch.nn.Module:
        return self.embed_tokens


class KeyeVL2MoeForConditionalGeneration(Qwen3MoeForCausalLM):
    def __init__(self, config: Any, quant_config: Any = None, prefix: str = "") -> None:
        if quant_config is not None:
            raise ValueError("Keye reference rollout requires BF16 weights")
        torch.backends.cuda.preferred_blas_library("cublaslt")
        super().__init__(config, quant_config, prefix)
        self.lm_head.quant_method = KeyeLinearMethod()
        self.model.__class__ = KeyeLanguageModel
        self.multimodal_enabled = process_model_config().is_multimodal
        if self.multimodal_enabled:
            self.visual, self.mlp_AR = build_vision(config, get_model().model_path)
        KeyeRMSNorm.convert(self.model.norm)
        for layer in self.model.layers:
            KeyeRMSNorm.convert(layer.input_layernorm)
            KeyeRMSNorm.convert(layer.post_attention_layernorm)
            layer.mlp.gate.__class__ = KeyeRouter
            layer.mlp.experts.__class__ = KeyeExperts
            if get_parallel().moe_tp_size > 1:
                weight = layer.mlp.experts.w2_weight
                weight.data = weight.new_empty((*weight.shape[:2], config.moe_intermediate_size))
                weight.weight_loader = _replicated_expert_weight_loader
            layer.mlp.topk.topk_config.custom_routing_function = keye_routing
            attention = layer.self_attn
            KeyeRMSNorm.convert(attention.q_norm)
            KeyeRMSNorm.convert(attention.k_norm)
            attention.__class__ = KeyeAttention
            attention.qkv_proj.quant_method = KeyeLinearMethod()
            attention.o_proj.quant_method = KeyeRowLinearMethod()
            if get_parallel().attn_tp_size > 1:
                weight = attention.o_proj.weight
                weight.data = weight.new_empty((weight.shape[0], config.num_attention_heads * attention.head_dim))
                weight.weight_loader = default_weight_loader
            attention.keye_multimodal_enabled = bool(self.multimodal_enabled)
            attention.sa_indexer = KeyeIndexer(config.hidden_size, config.rms_norm_eps, config.rope_theta)
            attention.sa_indexer.q_proj.__class__ = KeyeLinear
            attention.sa_indexer.k_proj.__class__ = KeyeLinear

    def pad_input_ids(self, input_ids: list[int], mm_inputs: Any) -> list[int]:
        return MultiModalityDataPaddingPatternMultimodalTokens().pad_input_tokens(input_ids, mm_inputs)

    def get_image_feature(self, items: list[Any]) -> torch.Tensor:
        pixels = torch.cat([item.feature for item in items], 0)
        grids = torch.cat([torch.as_tensor(item.image_grid_thw).reshape(-1, 3) for item in items], 0)
        return image_features(self.visual, self.mlp_AR, pixels, grids)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
        pp_proxy_tensors: Any = None,
    ) -> Any:
        if not self.multimodal_enabled:
            return super().forward(input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors)
        hidden = general_mm_embed_routine(
            input_ids,
            forward_batch,
            language_model=self.model,
            multimodal_model=self,
            positions=positions,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        return self.logits_processor(input_ids, hidden, self.lm_head, forward_batch)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]], **kwargs: Any) -> None:
        parameters = dict(self.named_parameters())

        def language_weights() -> Iterator[tuple[str, torch.Tensor]]:
            for name, weight in weights:
                if name.startswith(("visual.", "mlp_AR.")):
                    if self.multimodal_enabled:
                        default_weight_loader(parameters[name], weight)
                    continue
                if ".sa_indexer." in name:
                    default_weight_loader(parameters[name], weight)
                elif name.endswith("experts.gate_up_proj"):
                    # Public Keye stores [experts, hidden, 2*intermediate].
                    if weight.shape[-2:] != (self.config.hidden_size, 2 * self.config.moe_intermediate_size):
                        raise ValueError(f"Unexpected Keye expert layout: {name}: {weight.shape}")
                    gate, up = weight.transpose(-1, -2).chunk(2, -2)
                    for expert in range(self.config.num_experts):
                        yield name.replace("gate_up_proj", f"{expert}.gate_proj.weight"), gate[expert]
                        yield name.replace("gate_up_proj", f"{expert}.up_proj.weight"), up[expert]
                elif name.endswith("experts.down_proj"):
                    for expert in range(self.config.num_experts):
                        yield name.replace("down_proj", f"{expert}.down_proj.weight"), weight[expert].transpose(-1, -2)
                else:
                    yield name, weight

        return super().load_weights(language_weights(), **kwargs)


EntryClass = KeyeVL2MoeForConditionalGeneration
