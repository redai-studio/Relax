# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch


_SPEC_TOKEN_COUNT_KEYS = (
    ("spec_num_correct_drafts", "spec_num_proposed_drafts"),
    ("spec_accepted_drafts", "spec_proposed_drafts"),
    ("spec_accept_token_num", "spec_draft_token_num"),
)


def get_spec_token_counts(meta_info: dict[str, Any]) -> tuple[int, int]:
    """Extract speculative decoding counts across SGLang metadata versions."""
    for accept_key, draft_key in _SPEC_TOKEN_COUNT_KEYS:
        if accept_key in meta_info and draft_key in meta_info:
            return int(meta_info.get(accept_key, 0) or 0), int(meta_info.get(draft_key, 0) or 0)
    return 0, 0


@dataclass
class Sample:
    """The sample generated."""

    group_index: int | None = None
    index: int | None = None
    # prompt
    prompt: str | list[dict[str, str]] = ""
    tokens: list[int] = field(default_factory=list)
    rollout_tokens: list[int] = field(default_factory=list)
    multimodal_inputs: dict[str, Any] | None = None  # raw multimodal data, e.g. images, videos, etc.
    multimodal_train_inputs: dict[str, Any] | None = None  # processed multimodal data, e.g. pixel_values, etc.
    # response
    response: str = ""
    response_length: int = 0
    label: str | None = None
    reward: float | dict[str, Any] | None = None
    custom_advantage: float | list[float] | list[list[float]] | None = None
    loss_mask: list[int] | None = None
    weight_versions: list[str] = field(default_factory=list)
    rollout_log_probs: list[float] | None = None  # Log probabilities from rollout engine
    rollout_routed_experts: np.ndarray | None = None  # [seq_len - 1, num_layers, topk]
    remove_sample: bool = False
    abort_count: int = 0  # Number of times this sample has been aborted

    teacher_log_probs: list[float] | None = None
    student_topk_token_ids: np.ndarray | None = None
    student_topk_log_probs: np.ndarray | None = None
    teacher_at_student_topk_log_probs: np.ndarray | None = None
    teacher_topk_token_ids: np.ndarray | None = None
    teacher_topk_log_probs: np.ndarray | None = None
    student_at_teacher_topk_log_probs: np.ndarray | None = None

    opd_topk_token_ids: list[np.ndarray] | None = None  # [R, K] or [R, max_K'](union pad)
    opd_topk_student_log_probs: list[np.ndarray] | None = None
    opd_topk_teacher_log_probs: list[np.ndarray] | None = None
    opd_topk_ksz: np.ndarray | None = None  # [only union]

    teacher_prompt: str | list[dict[str, str]] | None = None
    teacher_multimodal_inputs: dict[str, Any] | None = None
    teacher_tokens: list[int] | None = None  # expanded teacher_prompt_tokens + response_tokens
    teacher_prompt_length: int | None = None  # number of teacher prompt tokens in teacher_tokens (expanded)
    teacher_image_data: dict[str, list] | None = None  # base64-encoded teacher media for SGLang payload
    teacher_image_b64_list: list[str] | None = None  # raw base64-PNG list for OPD pre-expanded teacher path
    teacher_image_grid_thw: Any = None  # HF processor image_grid_thw, [[t, h, w], ...]

    class Status(Enum):
        PENDING = "pending"
        COMPLETED = "completed"
        TRUNCATED = "truncated"
        ABORTED = "aborted"
        # Indicates a recoverable or non-critical failure during generation (e.g., tool call failure,
        # external API error, parsing error). Unlike ABORTED, FAILED samples may still contain partial
        # valid output and can be retried or handled gracefully.
        FAILED = "failed"

    status: Status = Status.PENDING

    metadata: dict = field(default_factory=dict)
    generate_function_path: str | None = None
    # metadata used during training, e.g., what loss to use for this sample.
    train_metadata: dict | None = None

    # Session ID for consistent hashing routing (used when router policy is consistent_hashing)
    session_id: str | None = None

    non_generation_time: float = 0.0  # time spent in non-generation steps

    @dataclass
    class SpecInfo:
        spec_accept_token_num: int = 0
        spec_draft_token_num: int = 0
        spec_verify_ct: int = 0
        completion_token_num: int = 0

        @property
        def spec_accept_rate(self) -> float:
            return self.spec_accept_token_num / self.spec_draft_token_num if self.spec_draft_token_num > 0 else 0.0

        @property
        def spec_accept_length(self) -> float:
            return self.completion_token_num / self.spec_verify_ct if self.spec_verify_ct > 0 else 0.0

        def add(self, meta_info: dict):
            spec_accept_token_num, spec_draft_token_num = get_spec_token_counts(meta_info)
            self.spec_accept_token_num += spec_accept_token_num
            self.spec_draft_token_num += spec_draft_token_num
            self.spec_verify_ct += meta_info.get("spec_verify_ct", 0)
            self.completion_token_num += meta_info.get("completion_tokens", 0)

        def to_dict(self):
            return {
                "spec_accept_token_num": self.spec_accept_token_num,
                "spec_draft_token_num": self.spec_draft_token_num,
                "spec_verify_ct": self.spec_verify_ct,
                "completion_token_num": self.completion_token_num,
            }

        @staticmethod
        def from_dict(data: dict):
            info = Sample.SpecInfo()
            info.spec_accept_token_num = data.get("spec_accept_token_num", 0)
            info.spec_draft_token_num = data.get("spec_draft_token_num", 0)
            info.spec_verify_ct = data.get("spec_verify_ct", 0)
            info.completion_token_num = data.get("completion_token_num", 0)
            return info

    spec_info: SpecInfo = field(default_factory=SpecInfo)

    @dataclass
    class PrefixCacheInfo:
        cached_tokens: int = 0
        total_prompt_tokens: int = 0

        @property
        def prefix_cache_hit_rate(self) -> float:
            return self.cached_tokens / self.total_prompt_tokens if self.total_prompt_tokens > 0 else 0.0

        def add(self, meta_info: dict):
            self.cached_tokens += meta_info.get("cached_tokens", 0)
            # new_tokens = input_tokens - cached_tokens
            self.total_prompt_tokens += meta_info.get("prompt_tokens", 0)

        def to_dict(self):
            return {
                "cached_tokens": self.cached_tokens,
                "total_prompt_tokens": self.total_prompt_tokens,
            }

        @staticmethod
        def from_dict(data: dict):
            info = Sample.PrefixCacheInfo()
            info.cached_tokens = data.get("cached_tokens", 0)
            info.total_prompt_tokens = data.get("total_prompt_tokens", 0)
            return info

    prefix_cache_info: PrefixCacheInfo = field(default_factory=PrefixCacheInfo)

    def to_dict(self):
        value = self.__dict__.copy()
        value["status"] = self.status.value
        value["spec_info"] = self.spec_info.to_dict()
        value["prefix_cache_info"] = self.prefix_cache_info.to_dict()
        return value

    @staticmethod
    def from_dict(data: dict):
        data = dict(data)
        data["status"] = Sample.Status(data["status"])
        data["spec_info"] = Sample.SpecInfo.from_dict(data.get("spec_info", {}))
        data["prefix_cache_info"] = Sample.PrefixCacheInfo.from_dict(data.get("prefix_cache_info", {}))

        field_names = set(Sample.__dataclass_fields__.keys())
        init_data = {k: v for k, v in data.items() if k in field_names}
        sample = Sample(**init_data)

        for key, value in data.items():
            if key not in field_names:
                setattr(sample, key, value)

        return sample

    def get_reward_value(self, args) -> float:
        return self.reward if not args.reward_key else self.reward[args.reward_key]

    @property
    def effective_response_length(self):
        return sum(self.loss_mask) if self.loss_mask is not None else self.response_length

    def update_from_meta_info(self, args, meta_info: dict):
        """Update the sample with new information from meta_info returned by
        the rollout engine.

        And extract
        """
        if args.sglang_speculative_algorithm:
            # cannot directly use spec info from sglang because of partial rollout.
            self.spec_info.add(meta_info=meta_info)

        # Collect prefix cache statistics
        self.prefix_cache_info.add(meta_info=meta_info)

        if "weight_version" in meta_info:
            self.weight_versions.append(meta_info["weight_version"])

        match meta_info["finish_reason"]["type"]:
            case "length":
                self.status = Sample.Status.TRUNCATED
            case "abort":
                self.status = Sample.Status.ABORTED
            case "stop":
                self.status = Sample.Status.COMPLETED


@dataclass(frozen=True)
class ParamInfo:
    name: str
    dtype: torch.dtype
    shape: torch.Size
    attrs: dict
    size: int
    src_rank: int


# A dict-based batch produced along the rollout -> training path
# In Megatron backend, several fields are converted to torch.Tensor lists on GPU
# before being consumed by data iterators (see megatron_utils.actor._get_rollout_data).
RolloutBatch = dict[str, list[torch.Tensor] | list[int] | list[float] | list[str]]

SFTBatch = dict[str, list[torch.Tensor] | list[int] | list[str] | dict | None]
"""SFT 训练 batch 的 dict alias。

Required keys（producer 必须填、consumer 必读）:
    - "input_ids":        list[torch.Tensor]    # packed token IDs，每条样本一个 1D tensor
    - "position_ids":     list[torch.Tensor]    # 每个 sample 的 position id
    - "loss_mask":        list[torch.Tensor]    # 1=参与 loss（assistant token），0=不参与
    - "cu_seqlens":       list[torch.Tensor]    # packing 边界，shape=(num_samples_in_pack+1,) int32
    - "total_lengths":    list[int]             # 每个 pack 的总 token 数（含 padding）
    - "response_lengths": list[int]             # 每个 pack 的 assistant token 数（用于 metric）
    - "metadata":         list[dict]            # sample-level 元信息（source_dataset、原始行号等）

Optional keys:
    - "multimodal_inputs":       list[dict] | None  # 图像/视频/音频特征（非多模态时为 None）
    - "multimodal_train_inputs": list[dict] | None  # processor_pool 处理后的可入模张量
"""


@dataclass
class MultimodalType:
    name: str  # Type identifier used in message content (e.g., "image")
    placeholder: str  # Placeholder token in conversation messages (e.g., "<image>")


class MultimodalTypes:
    IMAGE = MultimodalType(name="image", placeholder="<image>")
    VIDEO = MultimodalType(name="video", placeholder="<video>")
    AUDIO = MultimodalType(name="audio", placeholder="<audio>")

    @classmethod
    def all(cls) -> list[MultimodalType]:
        return [cls.IMAGE, cls.VIDEO, cls.AUDIO]

    @classmethod
    def get(cls, name: str) -> MultimodalType | None:
        return next((m for m in cls.all() if m.name == name), None)
