# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Launch an SGLang sequence classifier."""

import json
import os
import sys
from pathlib import Path

from sglang.launch_server import run_server
from sglang.srt.plugins import load_plugins
from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree


def _is_multimodal_sequence_classifier(model_path: str) -> bool:
    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        return False
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return "Qwen3VLForSequenceClassification" in (config.get("architectures") or [])


def main() -> None:
    """Parse normal SGLang arguments and select the required modality path."""
    load_plugins()
    server_args = prepare_server_args(sys.argv[1:])
    if not _is_multimodal_sequence_classifier(server_args.model_path):
        server_args.enable_multimodal = False
    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
