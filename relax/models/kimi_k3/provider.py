# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from dataclasses import dataclass
from typing import Any

from megatron.bridge.models.kimi.kimi_k3_provider import KimiK3ModelProvider
from megatron.core.models.gpt import GPTModel


@dataclass
class KimiK3VLModelProvider(KimiK3ModelProvider):
    """Keep Bridge's K3 language configuration and add its HF vision
    modules."""

    scatter_embedding_sequence_parallel: bool = False
    vision_config: Any = None
    hf_model_path: str | None = None
    trust_remote_code: bool = False
    image_token_id: int | None = None
    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False
    vision_dp_when_tp: bool = False

    def provide(
        self, pre_process: bool | None = None, post_process: bool | None = None, vp_stage: int | None = None
    ) -> Any:
        from .model import KimiK3VLModel

        if self.vision_dp_when_tp and self.tensor_model_parallel_size > 1 and not self.freeze_vision_model:
            raise ValueError("Kimi K3 --vision-dp-when-tp requires --freeze-vision-model.")
        if self.context_parallel_size > 1:
            if self.cp_partition_mode != "zigzag":
                raise ValueError("Kimi K3 context parallelism requires --cp-partition-mode zigzag.")
            if self.linear_cp_mode not in ("headwise", "chunkwise"):
                raise ValueError("Kimi K3 requires --linear-cp-mode headwise or chunkwise.")
            if self.linear_cp_mode == "headwise" and self.kimi_linear_num_heads % (
                self.tensor_model_parallel_size * self.context_parallel_size
            ):
                raise ValueError("Kimi K3 headwise CP requires KDA heads divisible by TP * CP; use chunkwise CP.")
        if self.tensor_model_parallel_size > 1 and not self.sequence_parallel:
            raise ValueError(
                "Kimi K3 requires --sequence-parallel when --tensor-model-parallel-size is greater than 1 "
                "to synchronize replicated language parameter gradients."
            )
        if self.virtual_pipeline_model_parallel_size is not None:
            raise ValueError("Kimi K3 does not support virtual pipeline parallelism yet.")
        if self.mtp_num_layers:
            raise ValueError("Kimi K3 does not support MTP training yet.")
        model = KimiK3VLModel(self, pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
        model.freeze(self.freeze_language_model, self.freeze_vision_model, self.freeze_vision_projection)
        return model

    def provide_language_model(
        self, pre_process: bool | None = None, post_process: bool | None = None, vp_stage: int | None = None
    ) -> GPTModel:
        return super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
