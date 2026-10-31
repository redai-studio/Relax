# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import re
from typing import Any

import torch
from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor, MultimodalSpecialTokens

from relax.models.keye_vl2.positions import image_positions

from .model import KeyeVL2MoeForConditionalGeneration


class KeyeProcessor(BaseMultimodalProcessor):
    models = [KeyeVL2MoeForConditionalGeneration]
    generates_input_ids_from_raw_prompt = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.mm_tokens = MultimodalSpecialTokens(
            image_token="<|vision_start|><|image_pad|><|vision_end|>",
            image_token_id=self.hf_config.image_token_id,
            image_token_regex=re.compile(r"<\|vision_start\|>(?:<\|image_pad\|>)+<\|vision_end\|>"),
        ).build(self._processor)

    async def process_mm_data_async(
        self, image_data: list[Any], input_text: str | list[int], request_obj: Any, **kwargs: Any
    ) -> MultimodalProcessorOutput:
        if request_obj.video_data or request_obj.audio_data:
            raise ValueError("Keye currently supports images only")
        base = await self.load_mm_data(prompt=input_text, image_data=image_data, multimodal_tokens=self.mm_tokens)
        items, ids, ret = await self.process_and_combine_mm_data_async(base, self.mm_tokens)
        ids = torch.as_tensor(ids).reshape(1, -1)
        positions = image_positions(ids, ret.get("image_grid_thw"), self.hf_config.image_token_id)[:, 0]
        return MultimodalProcessorOutput(
            input_ids=ids[0].tolist(),
            mm_items=items,
            im_start_id=self.hf_config.vision_start_token_id,
            im_end_id=self.hf_config.vision_end_token_id,
            im_token_id=self.hf_config.image_token_id,
            mrope_positions=positions,
            mrope_position_delta=(positions.max() + 1 - ids.numel()).reshape(1, 1),
        )
