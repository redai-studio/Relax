# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Keep the latest completed HF export beside the native recovery
checkpoint."""

import re
import shutil
from argparse import Namespace
from pathlib import Path

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def keep_latest_hf(_args: Namespace, hf_path: str, rollout_id: int, **_: object) -> None:
    current = Path(hf_path)
    if current.name != f"final-hf-{rollout_id}" or not (current / "config.json").is_file():
        raise ValueError(f"Invalid completed HF export: {current}")
    for previous in current.parent.glob("final-hf-*"):
        match = re.fullmatch(r"final-hf-(\d+)", previous.name)
        if match and int(match[1]) < rollout_id and previous.is_dir() and not previous.is_symlink():
            shutil.rmtree(previous)
            logger.info("Removed previous HF export: %s", previous)
