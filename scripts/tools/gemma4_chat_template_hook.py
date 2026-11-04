# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Persist the Gemma4 SFT chat template in Hugging Face exports."""

import os
from typing import Any

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def patch_gemma4_chat_template(args: Any, hf_path: str, rollout_id: int, *, dtype: str, is_lora: bool) -> None:
    """Persist the training-time Gemma4 template in the HF export.

    Without this step, serving reconstructs tool history with the original
    template and removes the supervised turn boundary introduced during SFT.
    """
    del args, dtype, is_lora
    from relax.engine.sft.dataset.gemma4_chat_template_patch import (
        _patch_template,
        gemma4_sft_patch_options_from_env,
    )

    thinking_enabled, tool_turn_boundary_enabled, ms_swift_compat = gemma4_sft_patch_options_from_env()
    if not (thinking_enabled or tool_turn_boundary_enabled or ms_swift_compat):
        logger.info("[gemma4-chat-template] skipped: Gemma4 SFT template patches are disabled")
        return

    path = os.path.join(hf_path, "chat_template.jinja")
    if not os.path.isfile(path):
        logger.warning(f"[gemma4-chat-template] skipped: chat_template.jinja not found in {hf_path}")
        return

    with open(path) as f:
        template = f.read()
    result = _patch_template(
        template,
        thinking_enabled=thinking_enabled,
        tool_turn_boundary_enabled=tool_turn_boundary_enabled,
        ms_swift_compat=ms_swift_compat,
    )
    if result is None:
        logger.info("[gemma4-chat-template] skipped: exported template is not Gemma4")
        return
    patched, changed = result
    if not changed:
        logger.info("[gemma4-chat-template] already patched; nothing to do")
        return

    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w") as f:
        f.write(patched)
    os.replace(tmp_path, path)
    logger.info(f"[gemma4-chat-template] rollout={rollout_id}: persisted SFT template patches")
