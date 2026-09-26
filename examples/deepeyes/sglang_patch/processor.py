# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""DeepEyes Qwen-VL processor registered through SGLang's public loader."""

from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor

from examples.deepeyes.processor_patch_utils import process_preexpanded_inputs


class DeepEyesQwenVLImageProcessor(QwenVLImageProcessor):
    """Preserve pre-expanded DeepEyes input IDs without copying upstream
    code."""

    models = QwenVLImageProcessor.models

    async def process_mm_data_async(self, image_data, input_text, request_obj, *args, **kwargs):
        return await process_preexpanded_inputs(
            self,
            super().process_mm_data_async,
            image_data,
            input_text,
            request_obj,
            *args,
            **kwargs,
        )
