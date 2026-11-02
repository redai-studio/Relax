# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Paired image metadata and deterministic, resumable SFT sample ordering.

The CPU producer only plans row IDs. Model adapters decode and preprocess the
corresponding images on training workers, where the dataset mount is available.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from pathlib import Path
from typing import Any

import torch


def image_tier_budget(config: dict[str, Any], index: int) -> int | None:
    """CPU-only per-row tier policy shared by decoding and DP assignment."""
    tiers = config.get("fixed_pixels_tiers") or []
    if not tiers:
        return None
    if len(tiers) == 1:
        return int(tiers[0])
    seed = config.get("seed")
    stream = torch.Generator().manual_seed((0 if seed is None else int(seed)) + index)
    return int(tiers[torch.randint(len(tiers), (1,), generator=stream).item()])


def assign_image_batch(
    ids: list[int], config: dict[str, Any], *, step: int | None = None, world: int | None = None
) -> list[int]:
    """Cluster synchronous micro columns without changing batch membership.

    Rank striding is applied afterwards. Stable descending tier order minimizes
    the sum of micro-column maxima for this tier-only cost proxy. Aspect ratio,
    prompt length and condition-image shapes can still affect actual runtime.
    This changes per-rank RNG/sample correspondence, not the producer cursor.
    """
    rotate = config.get("dp_tier_rank_rotation", False)
    if rotate and not config.get("dp_tier_micro_balance", False):
        raise ValueError("DP tier rank rotation requires micro balancing")
    if not config.get("dp_tier_micro_balance", False):
        return ids
    ordered = sorted(ids, key=lambda index: image_tier_budget(config, index) or 0, reverse=True)
    if not rotate:
        return ordered
    if type(step) is not int or step < 0 or type(world) is not int or world <= 0 or len(ids) % world:
        raise ValueError("DP tier rank rotation requires a nonnegative step and complete world-sized micro blocks")
    # Rotate inside each synchronous column, never across micro boundaries.
    # The checkpoint cursor determines assignment without consuming any RNG.
    offset = step % world
    if not offset:
        return ordered
    return [
        index
        for start in range(0, len(ordered), world)
        for index in ordered[start + world - offset : start + world] + ordered[start : start + world - offset]
    ]


class ImageEditDataset:
    def __init__(self, args: Any, adapter: Any) -> None:
        self.config = args.i2i_sft_config
        self.root = Path(self.config["data_root"])
        raw = Path(args.prompt_data).read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.rows = json.loads(raw)
        self.adapter = adapter
        # Resolve once; never catch a TypeError raised inside sample processing.
        try:
            parameters = inspect.signature(adapter.prepare_sample).parameters.values()
            self._accepts_index = any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                or (parameter.name == "index" and parameter.kind != inspect.Parameter.POSITIONAL_ONLY)
                for parameter in parameters
            )
        except (TypeError, ValueError):
            self._accepts_index = False
        if not isinstance(self.rows, list) or len(self.rows) != self.config["dataset_size"]:
            raise ValueError("Image metadata must be a JSON array matching i2i_sft_config.dataset_size")
        count = self.config["conditioning_images"]
        for index, row in enumerate(self.rows):
            if not isinstance(row, dict) or not isinstance(row.get("prompt"), str):
                raise ValueError(f"Invalid prompt at metadata row {index}")
            if not isinstance(row.get("image"), str) or not row["image"]:
                raise ValueError(f"Missing target image at metadata row {index}")
            conditions = row.get("edit_image")
            if not isinstance(conditions, list) or len(conditions) != count:
                raise ValueError(f"Expected {count} ordered condition images at metadata row {index}")
            if not all(isinstance(path, str) and path for path in conditions):
                raise ValueError(f"Invalid condition image path at metadata row {index}")

    def __getitem__(self, index: int) -> Any:
        if self._accepts_index:
            return self.adapter.prepare_sample(self.rows[index], self.root, self.config, index=index)
        return self.adapter.prepare_sample(self.rows[index], self.root, self.config)


class ImageBatchSampler:
    """DistributedSampler ordering, with an explicit completed-step cursor.

    Each epoch is padded to a complete global batch. Global rows are
    interleaved by rank, matching DistributedSampler for the one-sample-per-
    rank baseline. Prefetch never changes the checkpoint cursor or the training
    RNG streams.
    """

    def __init__(self, size: int, batch_size: int, seed: int) -> None:
        if size <= 0 or batch_size <= 0:
            raise ValueError("Dataset and global batch sizes must be positive")
        self.size, self.batch_size, self.seed = size, batch_size, seed
        self.steps_per_epoch = math.ceil(size / batch_size)
        self._epoch = -1
        self._order: list[int] = []

    def batch(self, step: int) -> list[int]:
        if step < 0:
            raise ValueError("Step must be nonnegative")
        epoch, offset = divmod(step, self.steps_per_epoch)
        if epoch != self._epoch:
            generator = torch.Generator().manual_seed(self.seed + epoch)
            order = torch.randperm(self.size, generator=generator).tolist()
            padded_size = self.steps_per_epoch * self.batch_size
            self._order = (order * math.ceil(padded_size / self.size))[:padded_size]
            self._epoch = epoch
        start = offset * self.batch_size
        return self._order[start : start + self.batch_size]
