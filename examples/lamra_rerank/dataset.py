# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Use the shared SFT pipeline with LamRA's assistant-header supervision."""

from typing import Any

import torch

from examples.lamra_rerank.data import load_rank_image
from relax.engine.sft.dataset.streaming import ProcessedSample, SFTStreamingDataset
from relax.utils.data.processor_pool import MediaLoadError, process_sample_from_paths_in_worker


def _process_rank_sample(text: str, media_paths: dict[str, list[Any]], processor_kwargs: dict[str, Any]) -> Any:
    """Decode reference images in the pool; only paths cross the process
    boundary."""
    media = dict(media_paths)
    try:
        media["image"] = [load_rank_image(path) for path in media_paths.get("image", [])]
    except Exception as exc:
        raise MediaLoadError(f"Failed to load LamRA rank image: {exc}") from exc
    return process_sample_from_paths_in_worker(text, media, processor_kwargs)


class _RankExecutor:
    def __init__(self, executor: Any) -> None:
        self._executor = executor

    def submit(self, function: Any, /, *args: Any, **kwargs: Any) -> Any:
        if function is process_sample_from_paths_in_worker:
            function = _process_rank_sample
        return self._executor.submit(function, *args, **kwargs)


class _RankProcessorPool:
    """Keep SFT's pool ownership while adapting both sync and async
    submissions."""

    def __init__(self, pool: Any) -> None:
        self.executor = _RankExecutor(pool.executor)


def reference_loss_mask(tokens: torch.Tensor, im_start_id: int, pad_id: int | None) -> torch.Tensor:
    """Supervise assistant header, answer and terminator; mask system and
    user."""
    starts = torch.where(tokens == im_start_id)[0].tolist()
    if len(starts) != 3 or starts[0] != 0 or starts[1] != 11:
        raise ValueError("LamRA rerank requires the reference system/user/assistant chat template")
    mask = torch.zeros_like(tokens)
    mask[starts[2] :] = 1
    if pad_id is not None:
        mask[tokens == pad_id] = 0
    return mask


class LamraRerankDataset(SFTStreamingDataset):
    """Keep reference preprocessing local to this recipe."""

    def __init__(self, *args: Any, processor_pool: Any = None, **kwargs: Any) -> None:
        super().__init__(
            *args,
            processor_pool=_RankProcessorPool(processor_pool) if processor_pool is not None else None,
            **kwargs,
        )

    @classmethod
    def from_args(
        cls, config: Any, *, tokenizer: Any, processor_pool: Any, pad_token_ids: Any
    ) -> "LamraRerankDataset":
        if config.task_type != "causal_lm" or config.context_parallel_size != 1:
            raise ValueError("The Qwen2-VL rank recipe requires causal_lm and CP=1")
        if config.sft_oversize_strategy != "keep":
            raise ValueError("Keep complete multi-candidate rank examples instead of truncating visual tokens")
        return cls(
            path=config.prompt_data,
            tokenizer=tokenizer,
            processor_pool=processor_pool,
            capacity=config.max_tokens_per_gpu,
            prompt_key=config.input_key,
            label_key=config.label_key,
            multimodal_keys=config.multimodal_keys,
            metadata_key=config.metadata_key,
            system_prompt=config.system_prompt,
            seed=config.seed,
            prefetch_max_cached=config.sft_prefetch_buffer_size,
            prefetch_chunk_size=config.sft_prefetch_chunk_size,
            prefetch_num_workers=config.sft_prefetch_num_workers,
            pad_token_ids=pad_token_ids,
            oversize_strategy="keep",
            invalid_multimodal_strategy="error",
        )

    def _build_processed(
        self, rendered: Any, prompt_ids: Any | None, mm_inputs: dict[str, Any] | None
    ) -> ProcessedSample | None:
        sample = super()._build_processed(rendered, prompt_ids, mm_inputs)
        if sample is not None:
            sample.loss_mask = reference_loss_mask(
                sample.tokens, self.tokenizer.convert_tokens_to_ids("<|im_start|>"), self.tokenizer.pad_token_id
            )
        return sample
