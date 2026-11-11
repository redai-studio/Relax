# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
LATEST_PATCH = REPO_ROOT / "docker" / "patch" / "latest" / "megatron.patch"


def test_latest_patch_skips_only_empty_bridge_conversion_tasks():
    patch = LATEST_PATCH.read_text()

    # Bridge validates tasks it builds internally. The streaming methods also
    # accept prebuilt conversion tasks and still need the empty-slot guards.
    assert (
        "         for task in conversion_tasks:\n"
        "             # None means megatron module not on current rank, skip if this task is not going to happen\n"
        "-            if task.megatron_module is None:\n"
        "+            if task is None or task.megatron_module is None:\n"
        "                 continue\n"
    ) in patch
    assert (
        '         for task in self._with_progress_tracking(megatron_to_hf_tasks, "Converting to HuggingFace", '
        "show_progress):\n"
        "+            if task is None:\n"
        "+                continue\n"
        "             if isinstance(task.param_weight, DTensor):"
    ) in patch
