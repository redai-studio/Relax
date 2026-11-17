# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Qwen3.8 QSA capture and replay of document-relative complete R4 blocks."""

from dataclasses import replace
from functools import wraps
from types import SimpleNamespace
from typing import Any

import torch

from ..codecs import QSA_BITMAP


class QSABitmapCapturer:
    """Owner-only stable encoded scratch with SGLang's async-copy protocol."""

    publication_codec = QSA_BITMAP
    topk_size = 512

    @staticmethod
    def encoded_width(stop: int) -> int:
        blocks = stop // 4
        return 1 if blocks <= 512 else 1 + (blocks + 31) // 32

    def __init__(self, config: Any, num_tokens: int, batch: int, context_length: int, device: str) -> None:
        from ..validation import indexer_topk

        if indexer_topk(config) != self.topk_size or not 0 < context_length <= 262144:
            raise ValueError("QSA bitmap capture requires 512 R4 blocks and context length <= 262144")
        layers = getattr(config, "layer_types", None)
        count = config.num_hidden_layers
        if not isinstance(layers, (tuple, list)) or len(layers) < count:
            raise ValueError("QSA capture requires the explicit HF attention layer map")
        sparse = (i for i in range(count) if layers[i] in {"full_attention", "qwen_sparse_attention"})
        self.layer_slots = {layer: slot for slot, layer in enumerate(sparse)}
        if not self.layer_slots:
            raise ValueError("QSA capture layer map is empty")
        blocks = max(1, context_length // 4)
        self.words = 1 << (((blocks + 31) // 32 - 1).bit_length())
        self.num_layers = len(self.layer_slots)
        self.device_cache = SimpleNamespace(
            buffer=torch.empty((batch, self.num_layers, self.words + 1), dtype=torch.int32, device=device)
        )
        self.host_cache = SimpleNamespace(
            buffer=torch.empty((num_tokens, self.num_layers, self.words + 1), dtype=torch.int32, device="cpu")
        )
        self.d2h_bytes = 0

    def capture_tokens(self, layer: int, tokens: torch.Tensor, positions: torch.Tensor) -> None:
        from .._bitmap import encode_qsa_tokens_into

        encode_qsa_tokens_into(tokens, positions, self.device_cache.buffer[: len(tokens), self.layer_slots[layer]])

    def on_forward_end(
        self, forward_batch: Any, can_run_graph: bool, cuda_graph_batch: Any, no_copy_to_cpu: bool = False
    ) -> Any:
        from sglang.srt.state_capturer.base import TopkCaptureOutput

        slots = forward_batch.out_cache_loc_virtual
        if slots is None:
            slots = forward_batch.out_cache_loc
        width = self.encoded_width(int(max(forward_batch.seq_lens_cpu)))
        if width > self.words + 1:
            raise ValueError("QSA capture exceeds its configured context capacity")
        values = self.device_cache.buffer[: len(slots), :, :width]
        self.d2h_bytes += values.numel() * values.element_size()
        if no_copy_to_cpu:
            return TopkCaptureOutput(
                out_cache_loc=slots.clone(),
                topk=values.clone(),
                host_cache=SimpleNamespace(buffer=self.host_cache.buffer[:, :, :width]),
            )
        self.host_cache.buffer[slots.cpu(), :, :width] = values.cpu()
        return None

    def snapshot_layer(self, slots: Any, layer: int, stop: int) -> Any:
        import numpy as np

        width = self.encoded_width(stop)
        result = np.array(self.host_cache.buffer[slots, layer, :width].numpy(), copy=True, order="C")
        # Earlier forwards copied fewer words. Never publish stale capacity
        # beyond the visible complete-block domain of an individual query.
        blocks = (np.arange(stop - len(slots), stop) + 1) // 4
        tail = np.arange(width - 1)[None, :] >= ((blocks + 31) // 32)[:, None]
        result[:, 1:][tail] = 0
        return result

    def destroy(self) -> None:
        self.host_cache.buffer = self.device_cache.buffer = None


def token_ids_to_blocks(token_ids: torch.Tensor, positions: torch.Tensor, budget: int, ratio: int = 4) -> torch.Tensor:
    """Undo SGLang's compacted token expansion, excluding the partial causal
    tail.

    Prefill and decode both return logical document-local token IDs. Invalid
    historical entries are compacted behind the tail, so slicing every fourth
    column before filtering would accidentally turn tail tokens into blocks.
    """
    if ratio != 4 or budget <= 0 or budget % ratio:
        raise ValueError("QSA capture requires a positive R4 token budget")
    if token_ids.ndim != 2 or token_ids.shape[1] != budget + ratio - 1:
        raise ValueError("QSA capture token width differs from the configured budget")
    if positions.ndim != 1 or positions.shape[0] != token_ids.shape[0]:
        raise ValueError("QSA capture positions must match token rows")
    complete = (token_ids >= 0) & (token_ids % ratio == 0) & (token_ids + ratio - 1 <= positions[:, None])
    columns = torch.arange(token_ids.shape[1], device=token_ids.device).expand_as(token_ids)
    order = torch.where(complete, columns, columns + token_ids.shape[1]).argsort(dim=1, stable=True)
    chosen = order[:, : budget // ratio]
    starts = token_ids.gather(1, chosen)
    return torch.where(complete.gather(1, chosen), starts // ratio, -1).to(torch.int32)


def replay_selection(indexer: Any, selection: Any) -> Any:
    """Replace IDs outside the local checkpoint tape so backward consumes
    once."""
    from .megatron import IndexerReplay

    if not IndexerReplay.is_enabled() or IndexerReplay.get_stage() == "fallthrough":
        return selection
    if indexer.config.context_parallel_size != 1 or indexer.config.qsa_indexer_loss_coeff != 0:
        raise ValueError("QSA rollout replay supports frozen CP1 indexers")
    if any(parameter.requires_grad for parameter in indexer.parameters()):
        raise ValueError("QSA rollout replay requires a frozen indexer")
    if selection.selected_ids is None or selection.compress_ratio != 4:
        raise ValueError("QSA rollout replay requires document-relative R4 selected IDs")
    selected = IndexerReplay.replay(indexer.layer_number, selection.selected_ids)
    # Do not preserve a native all-selected fast path: rollout choices now own
    # historical blocks, while the attention backend still adds causal tails.
    return replace(selection, selected_ids=selected, selected_bits=None, all_selected=False)


def install_capture_patch() -> None:
    """Install before model construction; other model capturers are
    unchanged."""
    import sys

    from sglang.srt.state_capturer import indexer_topk

    if getattr(indexer_topk.create_indexer_capturer, "_relax_qsa_capture", False):
        return
    original_create = indexer_topk.create_indexer_capturer

    @wraps(original_create)
    def create(*, model_config: Any, num_tokens: int, max_running_requests: int, device: str) -> Any:
        config = model_config.hf_text_config
        if getattr(config, "model_type", None) not in {"qwen4_exp", "qwen4_exp_text"}:
            return original_create(
                model_config=model_config,
                num_tokens=num_tokens,
                max_running_requests=max_running_requests,
                device=device,
            )
        from sglang.srt.runtime_context import get_exec, get_parallel, get_schedule

        if not get_exec().features.enable_return_indexer_topk:
            return None
        if device != "cuda":
            raise ValueError("QSA replay capture currently supports CUDA only")
        if get_parallel().attn_tp_rank != 0:
            return None
        install_select_patch()
        return QSABitmapCapturer(
            config,
            num_tokens,
            max(get_schedule().chunked_prefill_size, max_running_requests),
            model_config.context_len,
            device,
        )

    create._relax_qsa_capture = True
    indexer_topk.create_indexer_capturer = create
    # Some SGLang versions imported the factory before scheduler hook setup.
    runner = sys.modules.get("sglang.srt.model_executor.model_runner")
    if runner is not None:
        runner.create_indexer_capturer = create

    def wrap_select(original: Any, position_arg: int) -> Any:
        @wraps(original)
        def select(self: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
            selected = original(self, *args, **kwargs)
            capturer = indexer_topk.get_global_indexer_capturer()
            if isinstance(capturer, QSABitmapCapturer):
                positions = kwargs.get("query_positions")
                if positions is None:
                    positions = args[position_arg]
                capturer.capture_tokens(self.layer_id, selected, positions)
            return selected

        return select

    def install_select_patch() -> None:
        from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer

        if getattr(QSAIndexer.select_prefill_tokens, "_relax_qsa_capture", False):
            return
        QSAIndexer.select_prefill_tokens = wrap_select(QSAIndexer.select_prefill_tokens, 4)
        QSAIndexer.select_decode_tokens = wrap_select(QSAIndexer.select_decode_tokens, 5)
        QSAIndexer.select_prefill_tokens._relax_qsa_capture = True
