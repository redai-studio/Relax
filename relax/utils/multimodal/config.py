# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Multimodal processing configuration management.

This module provides a centralized way to manage configuration parameters for
image, video, and audio processing in the multimodal pipeline.
"""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class MultimodalConfig:
    """Configuration dataclass for multimodal data processing.

    This class stores configuration parameters that control how images, videos,
    and audio are processed. It should be passed directly to processing functions
    to support multi-process environments.

    Attributes:
        image_max_token_num: Maximum number of tokens for image processing
        image_min_token_num: Minimum number of tokens for image processing
        video_min_token_num: Minimum number of tokens for video frame processing
        video_max_token_num: Maximum number of tokens for video frame processing
        video_fps: Target FPS for video processing
        video_fps_min_frames: Minimum number of frames for video processing
        video_fps_max_frames: Maximum number of frames for video processing
        audio_sample_rate: Sample rate for audio processing (Hz)
        frame_factor: Frame count alignment factor.
    """

    image_max_token_num: int = 16384
    image_min_token_num: int = 4
    image_resize_scale_factor: Optional[int] = None
    video_min_token_num: int = 128
    video_max_token_num: int = 768
    video_fps: float = 2.0
    video_fps_min_frames: int = 4
    video_fps_max_frames: int = 768
    audio_sample_rate: int = 16000
    frame_factor: int = 2

    @classmethod
    def from_args(cls, args: Any) -> "MultimodalConfig":
        """Create configuration from parsed arguments.

        Args:
            args: Parsed arguments object containing multimodal configuration.

        Returns:
            MultimodalConfig instance with values from args or defaults.
        """
        kwargs = {}

        # Image params
        if args.image_max_token_num is not None:
            kwargs["image_max_token_num"] = args.image_max_token_num
        if args.image_min_token_num is not None:
            kwargs["image_min_token_num"] = args.image_min_token_num
        if args.image_resize_scale_factor is not None:
            kwargs["image_resize_scale_factor"] = args.image_resize_scale_factor

        # Video params
        if args.video_min_token_num is not None:
            kwargs["video_min_token_num"] = args.video_min_token_num
        if args.video_max_token_num is not None:
            kwargs["video_max_token_num"] = args.video_max_token_num
        if args.video_fps is not None:
            kwargs["video_fps"] = args.video_fps
        if args.video_fps_min_frames is not None:
            kwargs["video_fps_min_frames"] = args.video_fps_min_frames
        if args.video_fps_max_frames is not None:
            kwargs["video_fps_max_frames"] = args.video_fps_max_frames
        if args.frame_factor is not None:
            kwargs["frame_factor"] = args.frame_factor

        # Audio params
        if args.audio_sample_rate is not None:
            kwargs["audio_sample_rate"] = args.audio_sample_rate

        return cls(**kwargs)


# Default values (used when config is not set)
# refer to https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py
DEFAULT_IMAGE_MAX_TOKEN_NUM = 16384
DEFAULT_IMAGE_MIN_TOKEN_NUM = 4
DEFAULT_VIDEO_MIN_TOKEN_NUM = 128
DEFAULT_VIDEO_MAX_TOKEN_NUM = 768
DEFAULT_VIDEO_FPS = 2.0
DEFAULT_VIDEO_FPS_MIN_FRAMES = 4
DEFAULT_VIDEO_FPS_MAX_FRAMES = 768
DEFAULT_AUDIO_SAMPLE_RATE = 16000
DEFAULT_FRAME_FACTOR = 2


def get_image_max_token_num(config: Optional[MultimodalConfig] = None) -> int:
    """Get image max token num from config or default."""
    return config.image_max_token_num if config else DEFAULT_IMAGE_MAX_TOKEN_NUM


def get_image_min_token_num(config: Optional[MultimodalConfig] = None) -> int:
    """Get image min token num from config or default."""
    return config.image_min_token_num if config else DEFAULT_IMAGE_MIN_TOKEN_NUM


def get_video_min_token_num(config: Optional[MultimodalConfig] = None) -> int:
    """Get video min token num from config or default."""
    return config.video_min_token_num if config else DEFAULT_VIDEO_MIN_TOKEN_NUM


def get_video_max_token_num(config: Optional[MultimodalConfig] = None) -> int:
    """Get video max token num from config or default."""
    return config.video_max_token_num if config else DEFAULT_VIDEO_MAX_TOKEN_NUM


def get_video_fps(config: Optional[MultimodalConfig] = None) -> float:
    """Get video FPS from config or default."""
    return config.video_fps if config else DEFAULT_VIDEO_FPS


def get_video_fps_min_frames(config: Optional[MultimodalConfig] = None) -> int:
    """Get video FPS min frames from config or default."""
    return config.video_fps_min_frames if config else DEFAULT_VIDEO_FPS_MIN_FRAMES


def get_video_fps_max_frames(config: Optional[MultimodalConfig] = None) -> int:
    """Get video FPS max frames from config or default."""
    return config.video_fps_max_frames if config else DEFAULT_VIDEO_FPS_MAX_FRAMES


def get_audio_sample_rate(config: Optional[MultimodalConfig] = None) -> int:
    """Get audio sample rate from config or default."""
    return config.audio_sample_rate if config else DEFAULT_AUDIO_SAMPLE_RATE


def get_image_resize_scale_factor(config: Optional[MultimodalConfig] = None) -> Optional[int]:
    """Get image resize scale factor from config, or None if not set."""
    return config.image_resize_scale_factor if config else None


def get_frame_factor(config: Optional[MultimodalConfig] = None) -> int:
    """Get audio sample rate from config or default."""
    return config.frame_factor if config else DEFAULT_FRAME_FACTOR
