# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Opt-in publication configuration, independent of inference deployment."""

import argparse
from dataclasses import dataclass
from pathlib import Path

from .deadline import validate_timeout
from .store import load_json, valid_id


@dataclass(frozen=True)
class PublicationConfig:
    store: str
    stream: str
    scratch: str
    disk_budget: int
    max_weight_bytes: int
    anchor_interval: int = 20
    compression: bool = False
    encode_workers: int = 4
    full_only: bool = False
    source_cache_root: str | None = None
    canonical_blocks: bool = True
    canonical_root: str | None = None
    canonical_budget: int | None = None
    source_limits: dict[str, int] | None = None
    source_io_timeout: float = 3600

    def __post_init__(self) -> None:
        valid_id(self.stream)
        if type(self.encode_workers) is not int or self.encode_workers < 1:
            raise ValueError("encode_workers must be a positive integer")
        validate_timeout(self.source_io_timeout)
        if self.source_limits is not None:
            if not isinstance(self.source_limits, dict) or set(self.source_limits) - {
                "ram_bytes",
                "gpu_bytes",
                "scratch_bytes",
            }:
                raise ValueError("Unknown source resource limit")
            if any(type(value) is not int or value <= 0 for value in self.source_limits.values()):
                raise ValueError("Source resource limits must be positive integers")
        for name in ("disk_budget", "max_weight_bytes", "anchor_interval"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("store", "scratch"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a directory")
        if self.source_cache_root is not None and (
            not isinstance(self.source_cache_root, str) or not self.source_cache_root
        ):
            raise ValueError("source_cache_root must be a directory")
        if any(type(getattr(self, name)) is not bool for name in ("compression", "full_only", "canonical_blocks")):
            raise ValueError("Compression, FULL selection and source selection must be boolean")
        if self.canonical_root is not None:
            if not isinstance(self.canonical_root, str) or not self.canonical_root:
                raise ValueError("canonical_root must be a directory")
            if type(self.canonical_budget) is not int or self.canonical_budget <= 0:
                raise ValueError("canonical_root requires a positive canonical_budget")
        elif self.canonical_budget is not None:
            raise ValueError("canonical_budget requires canonical_root")

    @classmethod
    def read(cls, path: str | Path) -> "PublicationConfig":
        return cls(**load_json(Path(path).read_bytes()))


def add_weight_version_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--weight-consumer-config",
        default=None,
        help="JSON transport, cache and subscription settings for node-local rollout consumers.",
    )
    parser.add_argument(
        "--weight-version-config",
        default=None,
        help="JSON publication configuration. Publish independent checkpoint versions instead of pushing rollout weights.",
    )
    return parser


def validate_weight_version_arguments(args, *, offline_training: bool) -> None:
    publication = getattr(args, "weight_version_config", None)
    consumer = getattr(args, "weight_consumer_config", None)
    if publication:
        if args.train_backend != "megatron":
            raise ValueError("--weight-version-config uses the Megatron checkpoint converter")
        if not consumer and not offline_training and not args.debug_train_only:
            raise ValueError(
                "Set --weight-consumer-config for the managed rollout engines; "
                "source-only publication is available in offline training or --debug-train-only"
            )
        source = PublicationConfig.read(publication)
        if consumer and load_json(Path(consumer).read_bytes())["stream"] != source.stream:
            raise ValueError("Training and managed rollout consumers must use the same weight stream")
