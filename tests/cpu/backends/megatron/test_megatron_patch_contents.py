# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
LATEST_PATCH = REPO_ROOT / "docker" / "patch" / "latest" / "megatron.patch"


def test_latest_patch_skips_only_empty_bridge_conversion_tasks():
    patch = LATEST_PATCH.read_text()

    assert patch.count("+            if task is None or task.megatron_module is None:") == 2
    assert (
        '         for task in self._with_progress_tracking(megatron_to_hf_tasks, "Converting to HuggingFace", '
        "show_progress):\n"
        "+            if task is None:\n"
        "+                continue\n"
        "             if isinstance(task.param_weight, DTensor):"
    ) in patch
