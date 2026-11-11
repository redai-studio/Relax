# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Use SGLang's existing indexer publication API for Keye TP rollout."""

from functools import wraps
from types import SimpleNamespace
from typing import Any

import torch

from relax.distributed.replay_manager.codecs import KEYE_BITMAP
from relax.models.keye_vl2.replay import INDEXER_TOPK, capture_rows
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def install_indexer_capturer() -> None:
    """Keye has replicated logical choices, not the DSV4 DP-only layout."""
    from sglang.srt.model_executor import model_runner
    from sglang.srt.runtime_context import get_exec, get_parallel, get_schedule

    from relax.distributed.replay_manager.core import replay_data_manager

    original = model_runner.create_indexer_capturer
    if getattr(original, "_relax_keye_indexer", False):
        return

    @wraps(original)
    def create(*, model_config: Any, num_tokens: int, max_running_requests: int, device: str) -> Any:
        config = model_config.hf_text_config
        if getattr(config, "model_type", None) != "KeyeVL2":
            return original(
                model_config=model_config,
                num_tokens=num_tokens,
                max_running_requests=max_running_requests,
                device=device,
            )
        if not get_exec().features.enable_return_indexer_topk:
            return None
        if device != "cuda" or get_parallel().pp_size != 1 or get_parallel().enable_dp_attention:
            raise ValueError("Keye indexer capture supports CUDA TP rollout without PP or DP attention")
        if config.sa_config["topk"] != INDEXER_TOPK:
            raise ValueError("Keye indexer capture requires top-k 2048")
        config.index_topk = INDEXER_TOPK
        config.num_indexer_layers = config.num_hidden_layers
        # Only this rank publishes responses. Avoid duplicating the very large
        # host cache on TP peers; Keye capture itself has no collectives.
        if get_parallel().attn_tp_rank != 0:
            return None
        if not replay_data_manager.uses_references:
            raise ValueError("Keye compressed capture requires reference replay delivery")
        return KeyeBitmapCapturer(
            num_tokens,
            max(get_schedule().chunked_prefill_size, max_running_requests),
            config.num_hidden_layers,
            device,
        )

    create._relax_keye_indexer = True
    model_runner.create_indexer_capturer = create


def capture(layer_id: int, indices: torch.Tensor, valid: torch.Tensor) -> None:
    from sglang.srt.state_capturer.indexer_topk import get_global_indexer_capturer

    capturer = get_global_indexer_capturer()
    if capturer is not None:
        capturer.capture(layer_id=layer_id, topk_indices=capture_rows(indices, valid))


def capture_compacted(layer_id: int, indices: torch.Tensor, counts: torch.Tensor) -> None:
    from sglang.srt.state_capturer.indexer_topk import get_global_indexer_capturer

    capturer = get_global_indexer_capturer()
    if capturer is not None:
        capturer.capture_compacted(layer_id, indices, counts)


class KeyeBitmapCapturer:
    """Capture encoded sets before D2H; no persistent int32 top-k cache.

    Only non-overlap scheduling is supported. The model's graph captures writes
    into stable scratch. Scheduler publication snapshots rows before KV reuse.
    """

    publication_codec = KEYE_BITMAP

    @staticmethod
    def encoded_width(stop: int) -> int:
        # Header-only FULL prefixes are valid for this model's causal text path.
        return 1 if stop <= INDEXER_TOPK else 1 + (stop + 31) // 32

    def __init__(self, num_tokens: int, batch: int, layers: int, device: str) -> None:
        # The recipe's context ceiling is 32768. This is a format capacity, not
        # a guessed runtime limit; refuse out-of-range IDs in the model adapter.
        self.words = 1024
        self.num_layers, self.topk_size = layers, INDEXER_TOPK
        self.device_cache = SimpleNamespace(
            buffer=torch.empty(
                (batch, layers, self.words + 1),
                dtype=torch.int32,
                device=device,
            )
        )
        self.host_cache = SimpleNamespace(
            buffer=torch.empty(
                (num_tokens, layers, self.words + 1),
                dtype=torch.int32,
                device="cpu",
            )
        )

        self.forward_count = self.captured_rows = self.d2h_bytes = 0
        logger.info(
            "Keye bitmap capture enabled: layers=%d topk=%d host_capacity_bytes=%d scratch_bytes=%d",
            layers,
            INDEXER_TOPK,
            self.host_cache.buffer.numel() * 4,
            self.device_cache.buffer.numel() * 4,
        )

    def capture(self, layer_id: int, topk_indices: torch.Tensor) -> None:
        from relax.models.keye_vl2.replay_codec import encode_into

        encode_into(topk_indices, self.device_cache.buffer[: len(topk_indices), layer_id])

    def capture_compacted(
        self, layer_id: int, indices: torch.Tensor, counts: torch.Tensor, *, row_offset: int = 0
    ) -> None:
        from relax.models.keye_vl2.replay_codec import encode_compacted_into

        encode_compacted_into(
            indices, counts, self.device_cache.buffer[row_offset : row_offset + len(indices), layer_id]
        )

    def on_forward_end(
        self, forward_batch: Any, can_run_graph: bool, cuda_graph_batch: Any, no_copy_to_cpu: bool = False
    ) -> None:
        if no_copy_to_cpu:
            raise ValueError("Keye compressed capture requires non-overlap scheduling")
        slots = forward_batch.out_cache_loc_virtual
        if slots is None:
            slots = forward_batch.out_cache_loc
        count = len(slots)
        max_length = max(forward_batch.seq_lens_cpu)
        width = 1 + (int(max_length) + 31) // 32
        if width > self.words + 1:
            raise ValueError("Keye bitmap capture context exceeds 32768")
        # Copy only active bitmap words. FULL rows use a header and zero words.
        self.host_cache.buffer[slots.cpu(), :, :width] = self.device_cache.buffer[:count, :, :width].cpu()
        self.forward_count += 1
        self.captured_rows += count
        self.d2h_bytes += count * self.num_layers * width * 4
        if self.forward_count % 1024 == 0:
            logger.info(
                "Keye bitmap capture: forwards=%d rows=%d d2h_bytes=%d logical_int32_bytes=%d",
                self.forward_count,
                self.captured_rows,
                self.d2h_bytes,
                self.captured_rows * self.num_layers * INDEXER_TOPK * 4,
            )
        return None

    def snapshot_layer(self, slots: Any, layer: int, stop: int) -> Any:
        import numpy as np

        width = self.encoded_width(stop)
        # Unwritten tail words of earlier short prefixes must never be read.
        result = np.array(self.host_cache.buffer[slots, layer, :width].numpy(), copy=True, order="C")
        lengths = np.arange(stop - len(slots), stop) + 1
        tail = np.arange(width - 1)[None, :] >= ((lengths + 31) // 32)[:, None]
        result[:, 1:][tail] = 0
        return result

    def destroy(self) -> None:
        self.host_cache.buffer = None
        self.device_cache.buffer = None
