# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Geometry of format-1 target-lane coordinates in the acknowledged HF
schema."""

import re


TP_SIZE = 4

_PROJECTION = re.compile(r"model\.(?:language_model\.)?layers\.\d+\.mlp\.experts\.\d+\.(gate|up|down)_proj\.weight")


def split_axis(name: str, shape: tuple[int, ...]) -> int | None:
    projection = _PROJECTION.fullmatch(name)
    if projection is None or len(shape) != 2 or min(shape) <= 0:
        return None
    axis = int(projection[1] == "down")
    return axis if shape[axis] % TP_SIZE == 0 else None
