from __future__ import annotations

import json
import logging
import re
from copy import deepcopy
from math import ceil, floor
from typing import Any, Optional

from PIL import Image

from examples.deepeyes.base_env import BaseInteractionEnv
from relax.utils.types import Sample


logger = logging.getLogger(__name__)

# Regular expressions for parsing tool calls and answers
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
# Set of supported tool names
SUPPORTED_TOOL_NAMES = {"image_zoom_in_tool", "image_rotate_tool"}


class DeepeyesEnv(BaseInteractionEnv):
    """Environment for Deepeyes with zoom in and rotate tools."""

    MIN_DIMENSION = 28

    def __init__(self, *, max_turns: int | None = None, image=None, normalize_bbox: bool = True):
        self.max_turns = max_turns
        self.turn = 0
        self.tool_calls: list[dict[str, Any]] = []
        self.current_image = image
        self.origin_image = image
        # Whether to convert bbox coordinates from normalized [0, 1000] to absolute pixels.
        # Qwen-VL / Qwen2-VL / Qwen3-VL output 0-1000 normalized coords → set True (default).
        # Qwen2.5-VL outputs absolute pixel coords → set False.
        self.normalize_bbox = normalize_bbox

    def reset(self):
        self.turn = 0
        self.tool_calls.clear()
        observation: dict[str, Any] = {}
        reset_info = {"has_image": self.current_image is not None}
        return observation, reset_info

    def close(self):
        """No resources to release."""
        return

    def _parse_tool_payload(self, raw_json: str) -> dict[str, Any] | None:
        """Parse JSON payload from tool call string."""
        try:
            return json.loads(raw_json)
        except Exception as exc:
            logger.warning("Failed to decode tool call payload: %s", exc)
            return None

    def _extract_tool_call(self, text: str) -> dict[str, Any] | None:
        """Extract tool call from response text using regex pattern."""
        matches = list(TOOL_CALL_RE.finditer(text))
        if not matches:
            return None
        raw_json = matches[-1].group(1).strip()
        payload = self._parse_tool_payload(raw_json)
        if payload is None:
            return None

        name = payload.get("name") or payload.get("function", {}).get("name")
        arguments = payload.get("arguments") or payload.get("function", {}).get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                logger.warning("Tool call arguments are not valid JSON; rejecting tool call.")
                return None

        if not name:
            return None
        return {"name": name, "arguments": arguments}

    def _build_obs_text(self, *, text: str, role: str = "tool", image: Image.Image | None = None) -> dict[str, Any]:
        """Build observation dictionary with text and optional image."""
        obs: dict[str, Any] = {"obs_str": text, "role": role}
        if image is not None:
            obs["multi_modal_data"] = {"image": [image]}
        return obs

    def _validate_bbox(self, left: float, top: float, right: float, bottom: float) -> bool:
        """Validate bounding box coordinates for proper shape and
        dimensions."""
        try:
            if not (left < right and bottom > top):
                raise ValueError(f"invalid shape for {left=}, {top=}, {right=}, {bottom=}")
            height = bottom - top
            width = right - left
            if max(height, width) / min(height, width) > 100:
                raise ValueError(f"aspect ratio error: {left=}, {top=}, {right=}, {bottom=}")
            if min(height, width) <= 30:
                raise ValueError(f"{height=}, {width=} is too small")
            return True
        except Exception as exc:
            logger.warning("BBox validation failed: %s", exc)
            return False

    def _maybe_resize_bbox(self, bbox_2d: list[float]) -> Optional[list[float]]:
        """
        # Clamp, validate, and resize the bounding box if needed.
        #
        # This function ensures the bounding box stays within image bounds and meets minimum size requirements.
        # If the box is too small, it will be expanded from the center while staying within image boundaries.
        # Finally, the function validates the dimensions before returning.
        # Reference: https://github.com/verl-project/verl/blob/main/verl/tools/image_zoom_in_tool.py#L205

        Returns:
            A valid bounding box as a list of coordinates, or None if validation fails.
        """
        if self.current_image is None:
            return None

        image_width = self.current_image.width
        image_height = self.current_image.height
        left, top, right, bottom = bbox_2d

        # 1. Convert normalized [0, 1000] coordinates to absolute pixel coordinates.
        # Qwen-VL / Qwen2-VL / Qwen3-VL use 0-1000 normalized coords; Qwen2.5-VL uses absolute pixels.
        if self.normalize_bbox:
            left = left / 1000.0 * image_width
            top = top / 1000.0 * image_height
            right = right / 1000.0 * image_width
            bottom = bottom / 1000.0 * image_height

        # 2. Clamp the bounding box to the image dimensions.
        left = max(0.0, float(left))
        top = max(0.0, float(top))
        right = min(float(image_width), float(right))
        bottom = min(float(image_height), float(bottom))

        # 3. If clamped bbox is invalid, return immediately.
        if not self._validate_bbox(left, top, right, bottom):
            return None

        current_bbox = [left, top, right, bottom]
        height = bottom - top
        width = right - left

        # 4. If the box is too small, attempt to resize it.
        if height < self.MIN_DIMENSION or width < self.MIN_DIMENSION:
            logger.info(f"Bbox {width}x{height} is smaller than {self.MIN_DIMENSION}, attempting resize.")
            center_x = (left + right) / 2.0
            center_y = (top + bottom) / 2.0

            min_dim = min(height, width)
            if min_dim == 0:  # Safeguard for zero-area boxes
                return None

            # 1. Calculate the target dimensions to make the smallest side MIN_DIMENSION.
            ratio = self.MIN_DIMENSION / min_dim
            target_width = width * ratio
            target_height = height * ratio

            # 2. If the target size is larger than the image, scale it down to fit.
            #    This preserves the aspect ratio while respecting image boundaries.
            if target_width > image_width:
                scale_down = image_width / target_width
                target_width = image_width
                target_height *= scale_down

            if target_height > image_height:
                scale_down = image_height / target_height
                target_height = image_height
                target_width *= scale_down

            # 3. Determine the coordinates for the box centered on the original center.
            new_half_width = target_width / 2.0
            new_half_height = target_height / 2.0
            new_left = center_x - new_half_width
            new_top = center_y - new_half_height

            # 4. Shift the box if it extends beyond the image boundaries to keep its size.
            if new_left < 0:
                new_left = 0
            if new_top < 0:
                new_top = 0
            if new_left + target_width > image_width:
                new_left = image_width - target_width
            if new_top + target_height > image_height:
                new_top = image_height - target_height

            new_right = new_left + target_width
            new_bottom = new_top + target_height

            # Use floor and ceil for final integer coordinates.
            current_bbox = [floor(new_left), floor(new_top), ceil(new_right), ceil(new_bottom)]

        # 5. Final validation on the resulting bounding box (either original or resized).
        final_left, final_top, final_right, final_bottom = current_bbox
        if not self._validate_bbox(final_left, final_top, final_right, final_bottom):
            logger.warning(f"Final bbox is invalid after processing: {current_bbox}")
            return None

        final_height = floor(final_bottom) - floor(final_top)
        final_width = floor(final_right) - floor(final_left)

        if final_height < self.MIN_DIMENSION or final_width < self.MIN_DIMENSION:
            logger.warning(
                f"Final bbox size ({final_width}x{final_height}) are still smaller than minimum ({self.MIN_DIMENSION})."
                f"Original bbox: {bbox_2d}, original image size: {image_width}x{image_height}"
            )
            return None

        return current_bbox

    def _apply_tool(self, tool_call: dict[str, Any]) -> tuple[dict[str, Any], bool, dict[str, Any]]:
        """Execute the requested tool and return observation, done flag, and
        info."""
        info: dict[str, Any] = {"tool_call": deepcopy(tool_call), "tool_executed": True}
        try:
            name = tool_call["name"]
            args = tool_call["arguments"]
            if name not in SUPPORTED_TOOL_NAMES:
                raise ValueError(f"Unknown tool name: {name}")
            if self.current_image is None:
                raise ValueError(f"[ERROR] current_image is None in {self.__class__.__name__}")

            # Apply zoom in tool
            if name == "image_zoom_in_tool":
                bbox = args["bbox_2d"]
                bbox = self._maybe_resize_bbox(bbox)
                if not bbox:
                    raise ValueError("ZOOM IN ARGUMENTS ARE INVALID")
                img = self.current_image
                self.current_image = img.crop(bbox)
            # Apply rotate tool
            elif name == "image_rotate_tool":
                angle = args["angle"]
                img = self.current_image
                self.current_image = img.rotate(angle)

            info["tool_used"] = name
            info["status"] = "success"
            self.tool_calls.append(info)
            obs = self._build_obs_text(text="<image>", image=self.current_image)
            return obs, False, info
        except Exception as exc:
            obs = self._build_obs_text(text=f"Error: {str(exc)}")
            info["tool_executed"] = False
            info["error"] = str(exc)
            info["status"] = "failed"
            return obs, False, info

    # Called during rollout after receiving a model response
    def step(self, response_text: str):
        """Process agent response and return observation, done flag, and
        info."""
        self.turn += 1
        # Check if answer is provided
        if ANSWER_RE.search(response_text):
            return self._build_obs_text(text="Answer received."), True, {"final_answer": True}

        # Extract and execute tool call
        tool_call = self._extract_tool_call(response_text)
        if not tool_call:
            obs = self._build_obs_text(text="No tool call detected; ending the episode.")
            return obs, True, {"tool_executed": False}

        obs, done, info = self._apply_tool(tool_call)
        # Check if max turns reached
        if self.max_turns is not None and self.turn >= self.max_turns:
            done = True
        return obs, done, info


def _extract_initial_image(sample: Sample | None):
    """Extract initial image from sample's multimodal inputs."""
    if sample is None:
        return None
    multimodal = sample.multimodal_inputs or {}
    if isinstance(multimodal, dict):
        for key in ("images", "image"):
            images = multimodal.get(key)
            if images:
                return images[0]
    metadata = sample.metadata or {}
    image = metadata.get("image")
    if isinstance(image, str) and image:
        try:
            return Image.open(image)
        except Exception as exc:
            logger.warning("Failed to load image from path %s: %s", image, exc)
            return None
    if image is not None:
        return image
    return None


def build_env(sample: Sample | None = None, args: Any | None = None, **_: Any) -> DeepeyesEnv:
    """Construct a DeepeyesEnv."""
    max_turns = args.max_turns
    if max_turns is None:
        raise ValueError("max_turns must be set via --custom-config-path in the custom config file.")
    normalize_bbox = args.normalize_bbox
    image = _extract_initial_image(sample)
    if image is None:
        logger.warning("No image found in sample.multimodal_inputs or metadata.")
    return DeepeyesEnv(max_turns=max_turns, image=image, normalize_bbox=normalize_bbox)
