# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Reuse SGLang's model implementation after installing the V4.1 runtime
hooks."""

from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM


EntryClass = [DeepseekV4ForCausalLM]
