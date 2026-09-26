"""Normalize non-PEP-440 Linux kernel releases for Poetry marker evaluation."""

import os
import platform
import re


if os.environ.get("NEMO_GYM_SANITIZE_PLATFORM_RELEASE") == "1":
    _original_platform_release = platform.release

    def _normalized_platform_release() -> str:
        release = _original_platform_release()
        match = re.match(r"^\d+(?:\.\d+)*", release)
        return match.group(0) if match else release

    platform.release = _normalized_platform_release
