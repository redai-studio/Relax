# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""DeepEyes external SGLang processor package."""

from importlib.metadata import version as distribution_version

from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor

from examples.deepeyes.processor_patch_utils import validate_sglang_contract


validate_sglang_contract(QwenVLImageProcessor, distribution_version("sglang"))
