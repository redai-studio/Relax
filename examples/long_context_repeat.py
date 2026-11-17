# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Explicit text-only benchmark datasets, after the native SFT renderer.

All four scenarios share the <=512k source subset. Repeating tokens also
repeats the unshifted supervision mask; MTP and shifted labels remain
framework-owned. Filled records are synthetic performance inputs, not a
quality-training corpus.
"""

from dataclasses import replace
from typing import Any

from relax.engine.sft.dataset.streaming import ProcessedSample, SFTStreamingDataset


class RepeatedSFTDataset(SFTStreamingDataset):
    """Natural <=512k arm, also the implementation shared by repeat arms."""

    repeat_mode = "natural"

    def __init__(self, *, benchmark_sequence_limit: int, **kwargs: Any) -> None:
        self.benchmark_sequence_limit = benchmark_sequence_limit
        super().__init__(**kwargs)

    @classmethod
    def from_args(cls, args: Any, *, tokenizer: Any, processor_pool: Any, pad_token_ids: Any) -> "RepeatedSFTDataset":
        limit = int(args.seq_length)
        if limit not in (524288, 1048576):
            raise ValueError("This benchmark defines only 512k and 1m scenarios")
        if cls.repeat_mode == "double" and limit != 1048576:
            raise ValueError("Doubled real-data arm requires seq_length=1048576")
        if cls.repeat_mode == "natural" and limit != 524288:
            raise ValueError("Natural arm requires seq_length=524288; use doubled data for 1m")
        if (
            getattr(args, "sft_training_mode", "sft") != "sft"
            or getattr(args, "task_type", "causal_lm") != "causal_lm"
        ):
            raise NotImplementedError("Repeat benchmarks require causal-LM SFT")
        dataset = cls(
            benchmark_sequence_limit=limit,
            path=args.prompt_data,
            line_offsets_cache=getattr(args, "_sft_line_offsets_cache", None),
            tokenizer=tokenizer,
            processor_pool=processor_pool,
            capacity=None,
            prompt_key=args.input_key,
            label_key=getattr(args, "label_key", None),
            multimodal_keys=getattr(args, "multimodal_keys", None),
            conversation_key_map=getattr(args, "conversation_key_map", None),
            metadata_key=getattr(args, "metadata_key", "metadata"),
            tool_key=getattr(args, "tool_key", None),
            system_prompt=getattr(args, "system_prompt", None),
            seed=getattr(args, "seed", 42),
            prefetch_max_cached=getattr(args, "sft_prefetch_buffer_size", 256),
            prefetch_chunk_size=getattr(args, "sft_prefetch_chunk_size", 32),
            prefetch_num_workers=getattr(args, "sft_prefetch_num_workers", 4),
            pad_token_ids=pad_token_ids,
            oversize_strategy="keep",
            invalid_multimodal_strategy="error",
            apply_chat_template_kwargs=getattr(args, "apply_chat_template_kwargs", None),
            loss_last_turn_only=getattr(args, "sft_loss_last_turn_only", False),
            loss_ignore_empty_think=getattr(args, "sft_ignore_empty_think", False),
        )
        return dataset

    def _build_processed(self, rendered: Any, prompt_ids: Any, mm_inputs: Any) -> ProcessedSample | None:
        sample = super()._build_processed(rendered, prompt_ids, mm_inputs)
        if sample is None:
            return None
        if sample.multimodal_train_inputs or sample.image_refs_descriptor or sample.pre_shifted_labels is not None:
            raise NotImplementedError("Repeat benchmarks support rendered text and unshifted labels only")
        length = sample.total_length
        if length > 524288:
            return None
        if length < 2 or sample.loss_mask.shape != sample.tokens.shape:
            raise ValueError("Expected nonempty equal-length native token/mask vectors")
        if self.repeat_mode == "natural":
            return sample
        if int(sample.loss_mask[0]) != 0:
            raise NotImplementedError("Repeating a supervised first token would change boundary loss semantics")
        target = length * 2 if self.repeat_mode == "double" else self.benchmark_sequence_limit
        count = (target + length - 1) // length
        return replace(
            sample,
            tokens=sample.tokens.repeat(count)[:target].clone(),
            loss_mask=sample.loss_mask.repeat(count)[:target].clone(),
            total_length=target,
        )


class DoubledSFTDataset(RepeatedSFTDataset):
    repeat_mode = "double"


class FilledSFTDataset(RepeatedSFTDataset):
    repeat_mode = "fill"
