# Copyright (c) 2026 Relax Authors. All Rights Reserved.
from __future__ import annotations

from typing import TYPE_CHECKING

from relax.utils.opd.opd_multimodal import get_image_token_id, image_inputs, validate_image_input_ids


if TYPE_CHECKING:
    from relax.utils.types import Sample


class OpsdWorker:
    def __init__(self, *, is_opsd: bool = False):
        self.is_opsd = is_opsd

    @classmethod
    def from_args(cls, args) -> "OpsdWorker":
        teacher_prompt_key = getattr(args, "opd_teacher_prompt_key", None)
        teacher_image_key = getattr(args, "opd_teacher_image_key", None)
        return cls(is_opsd=teacher_prompt_key is not None or teacher_image_key is not None)

    async def build_teacher_inputs(self, args, sample: "Sample") -> None:
        """Process only the teacher prompt; append the original sampled IDs.

        Keep collapsed IDs for the native SGLang request and expanded IDs for
        checking its processor output and selecting logprob_start_len. The
        student processor is valid here only for compatible teacher processors.
        """
        response_length = int(sample.response_length or 0)
        if not self.is_opsd or not sample.tokens or response_length <= 0:
            return

        teacher_prompt = sample.teacher_prompt
        teacher_mm = sample.teacher_multimodal_inputs
        if teacher_prompt is None and teacher_mm is None:
            return
        if teacher_mm is None:
            teacher_mm = sample.multimodal_inputs
        images = image_inputs(teacher_mm)

        # Lazy import avoids the sglang_rollout -> OpdManager import cycle.
        from relax.engine.rollout.sglang_rollout import GenerateState, _run_image_processor

        state = GenerateState(args)
        prompt = teacher_prompt if teacher_prompt is not None else sample.prompt
        if not isinstance(prompt, str):
            prompt = state.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        collapsed_prompt_ids = state.tokenizer.encode(prompt, add_special_tokens=False)
        if images:
            if state.processor is None:
                raise ValueError("Native image OPSD requires a compatible image processor.")
            expanded_prompt_ids, _, _ = await _run_image_processor(state, args, prompt, teacher_mm)
        else:
            expanded_prompt_ids = collapsed_prompt_ids

        response_ids = list(sample.tokens[-response_length:])
        teacher_tokens = list(expanded_prompt_ids) + response_ids
        teacher_rollout_tokens = list(collapsed_prompt_ids) + response_ids
        if images:
            validate_image_input_ids(
                teacher_rollout_tokens,
                teacher_tokens,
                response_length,
                len(images),
                get_image_token_id(state.processor, state.tokenizer),
            )
        sample.teacher_tokens = teacher_tokens
        sample.teacher_rollout_tokens = teacher_rollout_tokens
        sample.teacher_prompt_length = len(expanded_prompt_ids)

    def teacher_input_ids(self, sample: "Sample", response_length: int) -> list[int]:
        if sample.teacher_rollout_tokens is not None:
            return sample.teacher_rollout_tokens
        return sample.rollout_tokens or sample.tokens

    def teacher_prompt_len(self, sample: "Sample", response_length: int) -> int:
        if sample.teacher_prompt_length is not None:
            return sample.teacher_prompt_length
        teacher_tokens = sample.teacher_tokens if sample.teacher_tokens is not None else sample.tokens
        return len(teacher_tokens) - response_length
