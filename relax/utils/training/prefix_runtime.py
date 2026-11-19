# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Actor-update-scoped detached text prefixes for Qwen3.5 VCS."""

from contextlib import contextmanager
from math import lcm
from types import MethodType

import torch
import torch.nn.functional as F

from relax.utils.env import Envs
from relax.utils.logging_utils import get_logger
from relax.utils.training.detached_prefix import DetachedPrefixCache, PrefixKey
from relax.utils.training.qwen35_prefix_decoder import decoder_chunk, validate_decoder
from relax.utils.training.qwen35_prefix_state import GDNState


logger = get_logger(__name__)


def record_host_prefix_inputs(rollout_data):
    if Envs.RELAX_TRAIN_PREFIX_MODE == "baseline" or "sample_indices" not in rollout_data:
        return
    records = []
    for tokens, length, response, index in zip(
        rollout_data["tokens"],
        rollout_data["total_lengths"],
        rollout_data["response_lengths"],
        rollout_data["sample_indices"],
        strict=True,
    ):
        if isinstance(tokens, torch.Tensor):
            if tokens.device.type != "cpu":
                raise ValueError("Prefix metadata must be captured before CUDA transfer")
            tokens = tokens.tolist()
        records.append((int(index), tuple(int(t) for t in tokens), max(0, int(length) - int(response) - 1)))
    rollout_data["prefix_inputs"] = records


class PrefixRuntime:
    """One instance per optimizer update; never shared with logprob
    forwards."""

    def __init__(self, mode, max_bytes):
        self.cache = DetachedPrefixCache(mode, max_bytes)
        self.prefix_tokens = self.suffix_tokens = self.windows = 0

    @contextmanager
    def activate(self, model, batch):
        outer = model
        while hasattr(outer, "module"):
            outer = outer.module
        language = getattr(outer, "language_model", None)
        if language is None or language.config.experimental_attention_variant != "gated_delta_net":
            raise NotImplementedError("Detached prefixes require the Qwen3.5 hybrid VLM decoder")
        rope_module = language.rotary_pos_emb
        if type(rope_module).__name__ != "Qwen3VLMultimodalRotaryEmbedding":
            raise NotImplementedError("Prefix positions require the current Qwen VLM get_rope_index contract")
        if getattr(rope_module, "seq_len_interpolation_factor", None) is not None:
            raise NotImplementedError("Interpolated prefix position keys are not implemented")
        if any("lora" in name.lower() or "adapter" in name.lower() for name, _ in language.named_modules()):
            raise NotImplementedError("Prefix adapter identity handling is not implemented")
        semantics = (
            type(language).__name__,
            str(language.config.params_dtype),
            language.config.rotary_base,
            language.config.rotary_percent,
            tuple(language.config.mrope_section),
            language.config.softmax_scale,
            language.config.softmax_type,
        )
        decoder = language.decoder
        validate_decoder(decoder)
        if batch.get("prefix_inputs") is None:
            raise ValueError("Missing authoritative host prefix metadata")
        if batch.get("dynamic_cp_size", 1) != 1:
            raise NotImplementedError("Detached prefixes need dynamic CP state redistribution")
        params = batch["packed_seq_params"]
        if params is None or not hasattr(params, "cu_seqlens_q_cpu"):
            raise NotImplementedError("Prefix actor path requires host THD boundaries")
        bounds = params.cu_seqlens_q_cpu
        records = batch["prefix_inputs"]
        if len(bounds) not in (len(records) + 1, len(records) + 2):
            raise ValueError("Packed prefix metadata does not match microbatch layout")
        media_ids = {getattr(outer, name, None) for name in ("image_token_id", "video_token_id")}
        media_ids.add(getattr(outer.config, "vision_start_token_id", None))
        original = decoder.forward
        runtime = self

        def forward(this, hidden_states, attention_mask=None, rotary_pos_emb=None, **kwargs):
            from megatron.core import parallel_state as mpu
            from megatron.core import tensor_parallel

            if attention_mask is not None or kwargs.get("attention_bias") is not None:
                raise NotImplementedError("Prefix dispatch requires unmodified packed causal attention")
            for key in ("context", "context_mask", "rotary_pos_cos", "rotary_pos_sin", "inference_context"):
                if kwargs.get(key) is not None:
                    raise NotImplementedError(f"Unsupported prefix decoder input: {key}")
            sp = this.config.sequence_parallel
            tp = mpu.get_tensor_model_parallel_world_size() if sp else 1
            tp_group = mpu.get_tensor_model_parallel_group()

            def gather(x):
                return (
                    tensor_parallel.gather_from_sequence_parallel_region(
                        x,
                        tensor_parallel_output_grad=False,
                        group=tp_group,
                    )
                    if sp
                    else x
                )

            def scatter(x):
                return tensor_parallel.scatter_to_sequence_parallel_region(x, group=tp_group) if sp else x

            full = gather(hidden_states)
            if full.shape[1] != 1 or full.shape[0] != bounds[-1]:
                raise ValueError("Prepared VLM embedding layout differs from actor THD boundaries")
            if rotary_pos_emb is None or rotary_pos_emb.shape[0] != full.shape[0]:
                raise ValueError("Expected authoritative full-length mRoPE embeddings")
            if kwargs.get("deepstack_visual_embeds") is not None and len(kwargs["deepstack_visual_embeds"]):
                raise NotImplementedError("Deepstack prefix execution is not implemented")
            outputs = []
            kinds = [hasattr(layer.self_attention, "gated_delta_rule") for layer in this.layers]
            for row, (start, end) in enumerate(zip(bounds, bounds[1:])):
                if row >= len(records):
                    # Trailing pack padding has no supervised tokens.
                    outputs.append(full[start:end].detach())
                    continue
                identity, ids, limit = records[row]
                first_media = next((i for i, token in enumerate(ids) if token in media_ids), len(ids))
                # Installed FLA chunk GDN uses 64-token blocks. Splitting inside
                # a block changes its BF16 WY solve; retain that tail in suffix.
                alignment = lcm(tp, 64)
                prefix = min(limit, first_media) // alignment * alignment
                runtime.prefix_tokens += prefix
                runtime.suffix_tokens += len(ids) - prefix
                runtime.windows += 1
                x, rope = full[start:end], rotary_pos_emb[start:end]

                def chunk(lo, hi, state=None, build=False):
                    value, position = x[lo:hi], rope[lo:hi]
                    pad = (-value.shape[0]) % tp
                    if build and pad:
                        raise ValueError("A prefix cannot contain synthetic SP padding")
                    if pad:
                        value = F.pad(value, (0, 0, 0, 0, 0, pad))
                        position = torch.cat((position, position[-1:].expand(pad, *position.shape[1:])))
                    result, states = decoder_chunk(
                        this,
                        scatter(value),
                        position,
                        state,
                        build_state=build,
                        row_layout=(start + lo, bounds[-1]),
                    )
                    result = gather(result)
                    return result[: hi - lo], states

                if prefix == 0:
                    value, _ = chunk(0, end - start)
                    outputs.append(value)
                    continue

                def unpack(flat):
                    states = tuple(
                        GDNState(*flat[2 * i : 2 * i + 2]) if kind else flat[2 * i : 2 * i + 2]
                        for i, kind in enumerate(kinds)
                    )
                    return states, flat[-1]

                def build(offset, previous):
                    states, history = (None, None) if previous is None else unpack(previous)
                    result, states = chunk(offset, prefix, states, build=True)
                    history = result if history is None else torch.cat((history, result))
                    flat = []
                    for state in states:
                        flat.extend((state.convolution, state.recurrent) if isinstance(state, GDNState) else state)
                    return (*flat, history)

                # Only first packed row has an image-free position origin guaranteed
                # by the actual VLM get_rope_index. Other rows rebuild conservatively.
                cache = runtime.cache if row == 0 else DetachedPrefixCache("detached_prefix", 0)
                if row == 0:
                    # Verify the exact prepared position contract on device. This
                    # does not copy CUDA values to host or synchronize the stream.
                    seq = torch.arange(prefix, device=rope.device, dtype=torch.float32)
                    freq = seq[:, None] * rope_module.inv_freq[None, :]
                    expected = torch.cat((freq, freq), dim=-1)[:, None, None, :]
                    torch._assert_async(
                        (rope[:prefix] == expected).all(),
                        "Prepared text-prefix mRoPE differs from exact cache position key",
                    )
                key = PrefixKey(
                    str(identity), (*semantics, id(this), tp, bounds[-1]), ids[:prefix], tuple(range(prefix))
                )
                flat = cache.prepare(key, build)
                if row != 0:
                    runtime.cache.builds += cache.builds
                    runtime.cache.computed_tokens += cache.computed_tokens
                state, history = unpack(flat)
                suffix, _ = chunk(prefix, end - start, state)
                outputs.append(torch.cat((history, suffix)))
            return scatter(torch.cat(outputs).contiguous())

        decoder.forward = MethodType(forward, decoder)
        try:
            yield
        finally:
            decoder.forward = original

    def close(self):
        logger.info(
            "prefix_update mode=%s windows=%d prefix_tokens=%d suffix_tokens=%d computed_prefix_tokens=%d "
            "hits=%d builds=%d extensions=%d cache_bytes=%d",
            self.cache.mode,
            self.windows,
            self.prefix_tokens,
            self.suffix_tokens,
            self.cache.computed_tokens,
            self.cache.hits,
            self.cache.builds,
            self.cache.extensions,
            self.cache.bytes,
        )
        self.cache.bump_version()


def create_prefix_runtime():
    mode = Envs.RELAX_TRAIN_PREFIX_MODE
    if mode == "baseline":
        return None
    return PrefixRuntime(mode, Envs.RELAX_TRAIN_PREFIX_MAX_BYTES)


def actor_forward(model, forward_kwargs, batch, runtime):
    if runtime is None:
        return model(**forward_kwargs)
    with runtime.activate(model, batch):
        return model(**forward_kwargs)
