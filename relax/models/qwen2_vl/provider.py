# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Qwen2-VL provider for the existing Relax Megatron model/optimizer setup."""

from dataclasses import dataclass, field
from typing import Any

from megatron.bridge.models.gpt_provider import GPTModelProvider, transformer_engine_layer_spec


@dataclass
class Qwen2VLModelProvider(GPTModelProvider):
    position_embedding_type: str = "mrope"
    mrope_section: list[int] = field(default_factory=lambda: [16, 24, 24])
    scatter_embedding_sequence_parallel: bool = False
    transformer_layer_spec: Any = transformer_engine_layer_spec
    vision_config: Any = None
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False

    def provide(self, pre_process: bool = True, post_process: bool = True, vp_stage: int | None = None) -> Any:
        from relax.models.qwen2_vl.model import Qwen2VLModel

        # First validate the DP-only path. Do not silently advertise model
        # parallel support before its forward and round-trip checks exist.
        sizes = (self.tensor_model_parallel_size, self.pipeline_model_parallel_size, self.context_parallel_size)
        if sizes != (1, 1, 1) or self.sequence_parallel or vp_stage not in (None, 0):
            raise ValueError("The initial Qwen2-VL integration requires TP=PP=CP=1 and no sequence parallelism")
        model = Qwen2VLModel(self, pre_process=pre_process, post_process=post_process)
        model.freeze(self.freeze_language_model, self.freeze_vision_model, self.freeze_vision_projection)
        return model

    def provide_language_model(self, pre_process: bool = True, post_process: bool = True) -> Any:
        return super().provide(pre_process=pre_process, post_process=post_process)
