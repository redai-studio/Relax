# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Weighted streaming readers for SFT and CPT prompt data."""

import math
from bisect import bisect_right
from pathlib import Path
from typing import Any, Iterator

import yaml

from relax.utils.data.data_utils import resolve_path_plan
from relax.utils.data.streaming_dataset import CompositeStreamingReader, StreamingReader
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def _resolve_relative_path(value: str, *, config_dir: Path) -> str:
    path_spec = value
    slice_suffix = ""
    if "@[" in value and value.endswith("]"):
        path_spec, slice_spec = value.rsplit("@", 1)
        slice_suffix = f"@{slice_spec}"
    path = Path(path_spec).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return f"{path}{slice_suffix}"


def _resolve_source_path(value: Any, *, config_dir: Path) -> str | list[str]:
    if isinstance(value, str):
        return _resolve_relative_path(value, config_dir=config_dir)
    if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
        return [_resolve_relative_path(item, config_dir=config_dir) for item in value]
    raise ValueError("mixture dataset 'path' must be a string or a non-empty list of strings")


def _build_source_reader(path: str | list[str]):
    paths, row_slice = resolve_path_plan(path)
    if len(paths) == 1 and row_slice is None:
        return StreamingReader(paths[0])
    return CompositeStreamingReader(paths, row_slice)


def _allocate_quotas(weights: list[float], total: int) -> list[int]:
    desired = [weight * total for weight in weights]
    quotas = [math.floor(value) for value in desired]
    order = sorted(range(len(weights)), key=lambda index: (-(desired[index] - quotas[index]), index))
    for index in order[: total - sum(quotas)]:
        quotas[index] += 1
    return quotas


class WeightedMixtureStreamingReader:
    """Expose several readers as one weighted, repeatable logical epoch."""

    def __init__(
        self,
        *,
        names: list[str],
        readers: list[Any],
        weights: list[float],
        epoch_size: int | None,
        config_path: str,
    ) -> None:
        if not names:
            raise ValueError(f"Prompt-data mixture {config_path!r} must contain at least one dataset")
        lengths = [len(reader) for reader in readers]
        empty = [name for name, length in zip(names, lengths, strict=True) if length == 0]
        if empty:
            raise ValueError(f"Prompt-data mixture contains empty dataset(s): {empty}")

        total_weight = sum(weights)
        self.names = names
        self.readers = readers
        self.weights = [weight / total_weight for weight in weights]
        minimum_epoch_size = max(
            math.ceil(length / weight) for length, weight in zip(lengths, self.weights, strict=True)
        )
        if epoch_size is None:
            epoch_size = minimum_epoch_size
        if epoch_size < minimum_epoch_size:
            raise ValueError(
                f"Prompt-data mixture epoch_size={epoch_size} is too small to cover every source once; "
                f"expected at least {minimum_epoch_size}"
            )
        self.epoch_size = epoch_size
        self.quotas = _allocate_quotas(self.weights, epoch_size)
        self._cumulative_quotas: list[int] = []
        total = 0
        for quota in self.quotas:
            total += quota
            self._cumulative_quotas.append(total)
        sources = [
            {"name": name, "rows": length, "weight": weight, "quota": quota}
            for name, length, weight, quota in zip(self.names, lengths, self.weights, self.quotas, strict=True)
        ]
        logger.info(f"Loaded prompt-data mixture {config_path}: epoch_size={self.epoch_size}, sources={sources}")

    @classmethod
    def from_yaml(cls, config_path: str) -> "WeightedMixtureStreamingReader":
        config_file = Path(config_path).expanduser().resolve()
        with config_file.open(encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
        if not isinstance(raw, dict):
            raise ValueError(f"Prompt-data mixture {str(config_file)!r} must be a YAML mapping")
        unknown_top_level = set(raw) - {"datasets", "epoch_size"}
        if unknown_top_level:
            raise ValueError(f"Unknown prompt-data mixture fields: {sorted(unknown_top_level)}")
        datasets = raw.get("datasets")
        if not isinstance(datasets, dict) or not datasets:
            raise ValueError("Prompt-data mixture requires a non-empty 'datasets' mapping")

        names: list[str] = []
        readers: list[Any] = []
        weights: list[float | None] = []
        for raw_name, raw_entry in datasets.items():
            name = str(raw_name)
            entry = {"path": raw_entry} if isinstance(raw_entry, (str, list)) else raw_entry
            if not isinstance(entry, dict):
                raise ValueError(f"Prompt-data mixture dataset {name!r} must be a path or mapping")
            unknown_fields = set(entry) - {"path", "weight"}
            if unknown_fields:
                raise ValueError(f"Prompt-data mixture dataset {name!r} has unknown fields: {sorted(unknown_fields)}")
            if "path" not in entry:
                raise ValueError(f"Prompt-data mixture dataset {name!r} is missing 'path'")
            path = _resolve_source_path(entry["path"], config_dir=config_file.parent)
            weight = entry.get("weight")
            if weight is not None:
                if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                    raise ValueError(f"Prompt-data mixture dataset {name!r} weight must be a number")
                weight = float(weight)
                if not math.isfinite(weight) or weight <= 0:
                    raise ValueError(f"Prompt-data mixture dataset {name!r} weight must be finite and > 0")
            names.append(name)
            readers.append(_build_source_reader(path))
            weights.append(weight)

        configured_weights = [weight for weight in weights if weight is not None]
        if configured_weights and len(configured_weights) != len(weights):
            missing = [name for name, weight in zip(names, weights, strict=True) if weight is None]
            raise ValueError(f"Prompt-data mixture must set every weight or none; missing weights for {missing}")
        normalized_weights = [1.0] * len(weights) if not configured_weights else configured_weights

        epoch_size = raw.get("epoch_size")
        if epoch_size is not None and (
            isinstance(epoch_size, bool) or not isinstance(epoch_size, int) or epoch_size <= 0
        ):
            raise ValueError("Prompt-data mixture epoch_size must be a positive integer")
        return cls(
            names=names,
            readers=readers,
            weights=normalized_weights,
            epoch_size=epoch_size,
            config_path=str(config_file),
        )

    def __len__(self) -> int:
        return self.epoch_size

    def _locate(self, idx: int) -> tuple[int, int]:
        if idx < 0 or idx >= self.epoch_size:
            raise IndexError(f"Index {idx} out of range [0, {self.epoch_size})")
        reader_idx = bisect_right(self._cumulative_quotas, idx)
        previous = 0 if reader_idx == 0 else self._cumulative_quotas[reader_idx - 1]
        local_idx = (idx - previous) % len(self.readers[reader_idx])
        return reader_idx, local_idx

    def __getitem__(self, idx: int) -> dict:
        reader_idx, local_idx = self._locate(idx)
        return self.readers[reader_idx][local_idx]

    def iter_batch(self, indices: list[int]) -> Iterator[tuple[int, dict]]:
        groups: dict[int, list[tuple[int, int, int]]] = {}
        for original_pos, idx in enumerate(indices):
            reader_idx, local_idx = self._locate(idx)
            groups.setdefault(reader_idx, []).append((original_pos, idx, local_idx))

        results: list[tuple[int, dict] | None] = [None] * len(indices)
        for reader_idx, entries in groups.items():
            local_indices = [local_idx for _, _, local_idx in entries]
            fetched = self.readers[reader_idx].iter_batch(local_indices)
            for (original_pos, logical_idx, _), (_, row) in zip(entries, fetched, strict=True):
                results[original_pos] = (logical_idx, row)

        for result in results:
            assert result is not None
            yield result
