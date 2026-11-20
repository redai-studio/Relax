# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image

from .audio_utils import fetch_audio
from .config import MultimodalConfig
from .image_utils import fetch_image
from .video_utils import fetch_video


# Keys that indicate a multimodal (non-text) content element
MODALITY_KEYS = {"image", "image_url", "video", "audio"}


def is_multimodal_element(ele: Dict[str, Any]) -> bool:
    """Check whether a content element represents a multimodal input.

    A multimodal element is identified either by:
    - containing modality-specific keys (image / video / audio), or
    - having a "type" field explicitly set to one of the modality types.
    """
    return bool(MODALITY_KEYS & ele.keys()) or ele.get("type") in MODALITY_KEYS


def extract_vision_info(
    conversations: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
) -> List[Dict[str, Any]]:
    """Extract all multimodal content elements from conversations.

    This function traverses conversations in OpenAI-style format and collects
    all image / video / audio elements from message contents.
    """
    vision_infos: List[Dict[str, Any]] = []

    # Normalize to batch format
    if isinstance(conversations[0], dict):
        conversations = [conversations]

    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if is_multimodal_element(ele):
                        vision_infos.append(ele)

    return vision_infos


def process_multimodal_info(
    conversations: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
    image_patch_size: int = 14,
    use_audio_in_video: bool = True,
    config: MultimodalConfig = None,
) -> Tuple[
    Optional[List[Image.Image]],
    Optional[List[Union[torch.Tensor, List[Image.Image]]]],
    Optional[List[Any]],
]:
    """Load and preprocess all multimodal inputs from conversations.

    This function:
    - extracts multimodal elements (image / video / audio),
    - loads them into model-ready tensors or PIL images,
    - optionally extracts audio tracks from videos.
    """
    vision_infos = extract_vision_info(conversations)

    image_inputs: List[Image.Image] = []
    video_inputs: List[Union[torch.Tensor, List[Image.Image]]] = []
    audio_inputs: List[Any] = []

    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info, image_patch_size=image_patch_size, config=config))
        elif "video" in vision_info:
            video_input, _, video_audio, _ = fetch_video(
                vision_info,
                image_patch_size=image_patch_size,
                use_audio_in_video=use_audio_in_video,
                config=config,
            )
            video_inputs.append(video_input)

            if video_audio is not None:
                audio_inputs.append(video_audio)
        elif "audio" in vision_info:
            audio_inputs.append(fetch_audio(vision_info, config=config))
        else:
            raise ValueError("image, image_url or video should in content.")

    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    if len(audio_inputs) == 0:
        audio_inputs = None

    return image_inputs, video_inputs, audio_inputs
