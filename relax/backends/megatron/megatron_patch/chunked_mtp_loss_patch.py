# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# Patch ``process_mtp_loss`` (in ``multi_token_prediction`` and its copy in
# ``gpt_model``) to compute each MTP depth's head + cross-entropy in SEQUENCE
# CHUNKS, so the full ``[S, V/TP]`` per-depth logits never materialize.
#
# WHY: the native branch runs each MTP head on the full sequence -> ``[S, V/TP]``
# logits (7.6 GiB bf16 per head at S=65536); with 3 MTP depths this dominates the
# last PP stage and OOMs (worse under CP=2). ``--sft-chunked-logits`` chunks the
# MAIN head but not the MTP heads, and the fused-linear CE that avoids logits is
# Blackwell-only.
#
# EQUIVALENCE: CE reduces per token, so head+CE per sequence slice and
# concatenating the ``[b, chunk]`` losses rebuilds the ``[b, s]`` loss
# bit-for-bit. Everything after the loss is copied verbatim from upstream.
#
# _passthrough (load-bearing): the chunked forward installs ``_passthrough`` on the
# instance ``output_layer.forward`` and counts MTP head calls. To avoid corrupting
# that counter, the head is called via the CLASS-level forward (invisible to the
# instance override), so MTP consumes ZERO intercepted calls. The paired
# ``mtp_output_layer_calls=0`` in relax model.py uses the SAME gate — keep them in
# sync.
#
# GATE: chunks only when ``enable_mtp_training`` AND ``sft_chunked_logits``;
# otherwise delegates to the captured original. Chunk size = ``--sft-logits-chunk-size``.

import inspect
import sys

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

try:
    import torch
    from megatron.core import parallel_state
    from megatron.core.transformer import multi_token_prediction as _mtp_mod
    from megatron.core.transformer.multi_token_prediction import (
        MTPLossAutoScaler,
        MTPLossLoggingHelper,
        roll_tensor,
    )

    # Capture the genuine upstream implementation BEFORE we overwrite the bindings,
    # so the delegate path (fuse-linear / gate-off) runs the original verbatim.
    _ORIG_PROCESS_MTP_LOSS = _mtp_mod.process_mtp_loss
    _ORIG_PROCESS_MTP_LOSS_PARAMETERS = set(inspect.signature(_ORIG_PROCESS_MTP_LOSS).parameters)
    _MTP_TRACKER_PARAMETERS = set(inspect.signature(MTPLossLoggingHelper.save_loss_to_tracker).parameters)
    _COMPUTE_MTP_ACCEPTANCE_COUNTS = getattr(_mtp_mod, "_compute_mtp_acceptance_counts", None)

    def _call_original_process_mtp_loss(*args, tp_group=None, input_ids=None, **kwargs):
        """Delegate across Megatron versions that added TP group/input IDs."""
        if "tp_group" in _ORIG_PROCESS_MTP_LOSS_PARAMETERS:
            kwargs["tp_group"] = tp_group
        if "input_ids" in _ORIG_PROCESS_MTP_LOSS_PARAMETERS:
            kwargs["input_ids"] = input_ids
        return _ORIG_PROCESS_MTP_LOSS(*args, **kwargs)

    @torch._dynamo.disable
    def _chunked_head_ce(
        mtp_hidden,
        mtp_labels,
        output_layer,
        output_weight,
        runtime_gather_output,
        compute_language_model_loss,
        scale_logits_fn,
        chunk_size,
        tp_group=None,
        loss_mask=None,
    ):
        """Per-depth head+CE in sequence chunks -> per-token loss [b, s].

        SP alignment (load-bearing): with ``--sequence-parallel`` the hidden
        arrives SP-scattered ``[S/TP, b, H]`` while labels are full-sequence.
        Chunking the scattered input and letting the head all-gather each chunk
        misaligns with the label slice (the ``[2048] vs [1024]`` IndexError), so
        we gather the hidden to full sequence ONCE (cheap — only the logits are
        large and stay chunked), then run the head with ``sequence_parallel=False``
        per chunk, mapping ``[chunk, b, H] -> [chunk, b, V/TP]`` 1:1 with the
        label slice.

        Runs inside ``torch._dynamo.config.patch(disable=True)``, not just the
        decorator: the MTP loss runs nested inside a live torch.compile region
        (the router's ``@torch.compile`` on every MoE layer), and dynamo still
        fake-traces the call boundary and chokes on the logits getitem unless the
        outer trace is globally disabled. Pure memory orchestration; no numeric
        effect.
        """
        head_forward = type(output_layer).forward  # bypass _passthrough (instance-level)
        sp_enabled = bool(getattr(output_layer, "sequence_parallel", False))
        with torch._dynamo.config.patch(disable=True):
            # Gather SP-scattered hidden to full sequence once; head runs with SP
            # disabled per chunk. When SP is off it is already full-sequence.
            if sp_enabled:
                from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region

                tp_group = (
                    tp_group
                    or getattr(output_layer, "tp_group", None)
                    or parallel_state.get_tensor_model_parallel_group()
                )
                mtp_hidden = gather_from_sequence_parallel_region(
                    mtp_hidden, tensor_parallel_output_grad=False, group=tp_group
                )  # [S/TP, b, H] -> [S, b, H]
                output_layer.sequence_parallel = False

            seq_len = mtp_hidden.shape[0]
            per_chunk_losses = []
            correct = None
            total = None
            try:
                for start in range(0, seq_len, chunk_size):
                    end = min(start + chunk_size, seq_len)
                    logits_slice, _ = head_forward(
                        output_layer,
                        mtp_hidden[start:end],  # [chunk, b, H]
                        weight=output_weight,
                        runtime_gather_output=runtime_gather_output,
                    )  # [chunk, b, V/TP] (no re-gather: SP disabled)
                    if scale_logits_fn is not None:
                        logits_slice = scale_logits_fn(logits_slice)
                    # compute_language_model_loss(labels[b,s], logits[s,b,V]) -> [b, s];
                    # slice labels on the seq axis (dim=1) to match the logits slice.
                    loss_slice = compute_language_model_loss(mtp_labels[:, start:end], logits_slice)
                    per_chunk_losses.append(loss_slice)
                    if _COMPUTE_MTP_ACCEPTANCE_COUNTS is not None and loss_mask is not None:
                        chunk_correct, chunk_total = _COMPUTE_MTP_ACCEPTANCE_COUNTS(
                            logits_slice,
                            mtp_labels[:, start:end],
                            loss_mask[:, start:end],
                            output_layer,
                            runtime_gather_output,
                            tp_group,
                        )
                        correct = chunk_correct if correct is None else correct + chunk_correct
                        total = chunk_total if total is None else total + chunk_total
                    del logits_slice
            finally:
                if sp_enabled:
                    output_layer.sequence_parallel = True
        return torch.cat(per_chunk_losses, dim=1), correct, total  # [b, s], rebuilt exactly

    def _save_mtp_loss_to_tracker(mtp_loss, num_tokens, layer_number, config, correct=None, total=None):
        """Log MTP loss across the legacy and token-aware tracker APIs."""
        avg_group = parallel_state.get_data_parallel_group(with_context_parallel=True)
        if "num_tokens" in _MTP_TRACKER_PARAMETERS:
            mtp_loss_scale = config.mtp_loss_scaling_factor / config.mtp_num_layers
            kwargs = {
                "avg_group": avg_group,
                "calculate_per_token_loss": config.calculate_per_token_loss,
            }
            if "correct" in _MTP_TRACKER_PARAMETERS:
                kwargs.update(correct=correct, total=total)
            MTPLossLoggingHelper.save_loss_to_tracker(
                mtp_loss_scale * torch.sum(mtp_loss),
                num_tokens,
                layer_number,
                config.mtp_num_layers,
                **kwargs,
            )
            return

        mtp_loss_for_log = (torch.sum(mtp_loss) * (num_tokens > 0).to(mtp_loss.dtype)) / num_tokens.clamp(min=1)
        MTPLossLoggingHelper.save_loss_to_tracker(
            mtp_loss_for_log,
            layer_number,
            config.mtp_num_layers,
            avg_group=avg_group,
        )

    def _resolve_chunk_size() -> int:
        """MTP head chunk size = --sft-logits-chunk-size (default 1024)."""
        try:
            from megatron.training.global_vars import get_args

            args = get_args()
        except Exception:
            args = None
        cs = int(getattr(args, "sft_logits_chunk_size", 1024) or 1024) if args is not None else 1024
        return cs if cs > 0 else 1024

    def _chunked_mtp_enabled() -> bool:
        """Gate: chunk MTP heads only for the MTP + sft-chunked-logits SFT run.

        MUST match the mtp_output_layer_calls==0 gate in relax/backends/megatron/
        model.py (single source of truth for the _passthrough accounting).
        """
        try:
            from megatron.training.global_vars import get_args

            args = get_args()
        except Exception:
            return False
        if args is None:
            return False
        return bool(getattr(args, "enable_mtp_training", False)) and bool(getattr(args, "sft_chunked_logits", False))

    @torch._dynamo.disable
    def process_mtp_loss(
        hidden_states,
        labels,
        loss_mask,
        output_layer,
        output_weight,
        runtime_gather_output,
        is_training,
        compute_language_model_loss,
        config,
        cp_group=None,
        tp_group=None,
        packed_seq_params=None,
        scale_logits_fn=None,
        input_ids=None,
    ):
        """Chunked drop-in for ``process_mtp_loss`` (native CE branch only).

        Numerically identical to upstream; only the head+CE is chunked.
        Delegates to the original when the gate is off or the fused-linear CE
        path is used.
        """
        fuse_linear_cross_entropy = config.cross_entropy_loss_fusion and config.cross_entropy_fusion_impl == "linear"
        # Delegate verbatim when: fused-linear CE (Blackwell path — don't chunk), or
        # the chunked-MTP gate is off (preserve exact upstream behavior).
        if fuse_linear_cross_entropy or not _chunked_mtp_enabled() or labels is None:
            return _call_original_process_mtp_loss(
                hidden_states,
                labels,
                loss_mask,
                output_layer,
                output_weight,
                runtime_gather_output,
                is_training,
                compute_language_model_loss,
                config,
                cp_group=cp_group,
                tp_group=tp_group,
                packed_seq_params=packed_seq_params,
                scale_logits_fn=scale_logits_fn,
                input_ids=input_ids,
            )

        # ---- chunked native path (mirrors upstream lines 650-723 exactly) ----
        hidden_states_list = torch.chunk(hidden_states, 1 + config.mtp_num_layers, dim=0)
        hidden_states = hidden_states_list[0]

        if labels is None:
            return hidden_states

        mtp_labels = labels.clone()
        if loss_mask is None:
            loss_mask = torch.ones_like(mtp_labels)

        # per-token count BEFORE rolling (used by the per-token-loss normalization).
        original_num_tokens = loss_mask.sum()

        chunk_size = _resolve_chunk_size()

        for mtp_layer_number in range(config.mtp_num_layers):
            mtp_labels, _ = roll_tensor(
                mtp_labels, shifts=-1, dims=-1, cp_group=cp_group, packed_seq_params=packed_seq_params
            )
            loss_mask, num_tokens = roll_tensor(
                loss_mask, shifts=-1, dims=-1, cp_group=cp_group, packed_seq_params=packed_seq_params
            )

            # Chunked head+CE (dynamo-disabled) -> per-token loss [b, s].
            mtp_loss, correct, total = _chunked_head_ce(
                hidden_states_list[mtp_layer_number + 1],
                mtp_labels,
                output_layer,
                output_weight,
                runtime_gather_output,
                compute_language_model_loss,
                scale_logits_fn,
                chunk_size,
                tp_group=tp_group,
                loss_mask=loss_mask if is_training else None,
            )

            # ---- identical to upstream from here (operates on [b, s], no vocab dim) ----
            mtp_loss = loss_mask * mtp_loss
            # Upstream logs the UNSCALED per-depth loss (before mtp_loss_scale) and
            # only folds the scale into the gradient path — mirror that exactly.
            if is_training:
                _save_mtp_loss_to_tracker(mtp_loss, num_tokens, mtp_layer_number, config, correct, total)
            mtp_loss_scale = config.mtp_loss_scaling_factor / config.mtp_num_layers
            if config.calculate_per_token_loss:
                num_tokens_safe = torch.clamp(num_tokens, min=1)
                mtp_loss_normalized = mtp_loss_scale * mtp_loss * (original_num_tokens / num_tokens_safe)
                hidden_states = MTPLossAutoScaler.apply(hidden_states, mtp_loss_normalized)
            else:
                safe_num_tokens = num_tokens.clamp(min=1)
                hidden_states = MTPLossAutoScaler.apply(hidden_states, mtp_loss_scale * mtp_loss / safe_num_tokens)

        return hidden_states

    # Force gpt_model into sys.modules so we can replace its (separately-bound) copy.
    import megatron.core.models.gpt.gpt_model as _gpt_mod  # noqa: E402

    _mtp_mod.process_mtp_loss = process_mtp_loss
    if hasattr(_gpt_mod, "process_mtp_loss"):
        _gpt_mod.process_mtp_loss = process_mtp_loss
    _mamba_mod = sys.modules.get("megatron.core.models.mamba.mamba_model")
    if _mamba_mod is not None and hasattr(_mamba_mod, "process_mtp_loss"):
        _mamba_mod.process_mtp_loss = process_mtp_loss

    logger.info(
        "Relax chunked MTP loss patch applied to process_mtp_loss "
        "(multi_token_prediction + gpt_model bindings; native CE branch chunked by "
        "--sft-logits-chunk-size, gated on enable_mtp_training AND sft_chunked_logits)."
    )

except ImportError as exc:  # pragma: no cover - Megatron not importable in some envs
    logger.warning("Relax chunked MTP loss patch not applied — Megatron import failed (%r).", exc)
