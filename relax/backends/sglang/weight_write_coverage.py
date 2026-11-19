# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Fail-closed coverage checks for ordinary SGLang parameter writes."""

import re
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from numbers import Number
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode


_EXPERT_PARAMETER = re.compile(r"(?:^|\.)mlp\.experts\.(?:w13_weight|w2_weight)$")


@dataclass
class _ParameterRegion:
    storage_key: tuple[str, int]
    start: int
    end: int
    names: set[str]
    padding_ranges: tuple[tuple[int, int], ...]
    writes: list[tuple[int, int, str]] = field(default_factory=list)
    allowed_aliases: frozenset[str] = frozenset()

    @property
    def canonical_name(self) -> str:
        return min(self.names)


def _storage_key(tensor: torch.Tensor) -> tuple[str, int]:
    storage = tensor.untyped_storage()
    return str(tensor.device), int(storage._cdata)


def _signature(tensor: torch.Tensor) -> tuple[Any, ...]:
    storage = tensor.untyped_storage()
    return (
        str(tensor.device),
        int(storage._cdata),
        int(storage.data_ptr()),
        tensor.dtype,
        tuple(tensor.shape),
        tuple(tensor.stride()),
        int(tensor.storage_offset()),
        tensor.is_contiguous(),
    )


def _byte_interval(tensor: torch.Tensor) -> tuple[int, int]:
    if not tensor.is_contiguous():
        raise ValueError("Protected target writes must address contiguous tensor slices")
    start = int(tensor.storage_offset()) * tensor.element_size()
    return start, start + tensor.numel() * tensor.element_size()


def _storage_byte_bounds(tensor: torch.Tensor) -> tuple[int, int]:
    if tensor.numel() == 0:
        start = int(tensor.storage_offset()) * tensor.element_size()
        return start, start
    minimum = maximum = int(tensor.storage_offset())
    for size, stride in zip(tensor.shape, tensor.stride(), strict=True):
        extent = (size - 1) * stride
        minimum += min(0, extent)
        maximum += max(0, extent)
    width = tensor.element_size()
    return minimum * width, (maximum + 1) * width


class _WeightWriteCoverageRecorder:
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        required_hf_names: Collection[str],
        excluded_expert_names: Collection[str],
        padding_fill_ranges: Mapping[str, Sequence[tuple[int, int]]] | None,
        allowed_hf_aliases: Mapping[str, Collection[str]] | None,
    ) -> None:
        self._model = model
        self._required_hf_names = frozenset(required_hf_names)
        if any(not isinstance(name, str) or not name for name in self._required_hf_names):
            raise ValueError("required_hf_names must contain nonempty parameter names")
        self._excluded_expert_names = frozenset(excluded_expert_names)
        if any(_EXPERT_PARAMETER.search(name) is None for name in self._excluded_expert_names):
            raise ValueError("Only exact routed-expert w13_weight/w2_weight names may be excluded")
        self._padding_fill_ranges = padding_fill_ranges or {}
        self._allowed_hf_aliases = allowed_hf_aliases or {}
        self._regions: list[_ParameterRegion] = []
        self._regions_by_storage: dict[tuple[str, int], list[_ParameterRegion]] = {}
        self._excluded_expert_storage: dict[tuple[str, int], list[tuple[int, int, str]]] = {}
        self._name_to_region: dict[str, _ParameterRegion] = {}
        self._parameter_signatures: dict[str, tuple[Any, ...]] = {}
        self._yielded_hf_names: set[str] = set()
        self._written_hf_names: set[str] = set()
        self._readonly_source_storage: set[tuple[str, int]] = set()
        self._active_hf_name: str | None = None
        self._failed = False
        self._complete = False
        self._build_schema()

    def _build_schema(self) -> None:
        named_parameters = dict(self._model.named_parameters(remove_duplicate=False))
        if not self._excluded_expert_names.issubset(named_parameters):
            missing = sorted(self._excluded_expert_names - named_parameters.keys())
            raise ValueError(f"Excluded expert parameters are not live model parameters: {missing}")
        live_expert_names = {name for name in named_parameters if _EXPERT_PARAMETER.search(name)}
        if live_expert_names != self._excluded_expert_names:
            missing = sorted(live_expert_names - self._excluded_expert_names)
            stale = sorted(self._excluded_expert_names - live_expert_names)
            raise ValueError(f"Expert exclusion set must be exact: missing={missing}, stale={stale}")
        for name in self._excluded_expert_names:
            if not isinstance(named_parameters[name], torch.nn.Parameter):
                raise ValueError(f"Excluded expert target is not a parameter: {name}")

        all_alias_names: dict[tuple[str, int, int, int], list[str]] = {}
        alias_names: dict[tuple[str, int, int, int], list[str]] = {}
        all_intervals: dict[tuple[str, int], list[tuple[int, int, str]]] = {}
        tensor_by_name = {}
        for name, parameter in named_parameters.items():
            if not isinstance(parameter, torch.nn.Parameter):
                raise ValueError(f"Expected a model parameter at {name}")
            signature = _signature(parameter)
            self._parameter_signatures[name] = signature
            start, end = _storage_byte_bounds(parameter)
            if end <= start:
                if name not in self._excluded_expert_names:
                    raise ValueError(f"Empty protected parameter is unsupported: {name}")
                continue
            storage_key = _storage_key(parameter)
            alias_key = (storage_key[0], storage_key[1], start, end)
            all_alias_names.setdefault(alias_key, []).append(name)
            all_intervals.setdefault(storage_key, []).append((start, end, name))
            if name in self._excluded_expert_names:
                self._excluded_expert_storage.setdefault(storage_key, []).append((start, end, name))
            if not parameter.is_contiguous():
                if name not in self._excluded_expert_names:
                    raise ValueError(f"Non-contiguous protected parameter is unsupported: {name}")
                continue
            start, end = _byte_interval(parameter)
            if name in self._excluded_expert_names:
                continue
            alias_names.setdefault(alias_key, []).append(name)
            tensor_by_name[name] = parameter

        for names in all_alias_names.values():
            if any(name in self._excluded_expert_names for name in names) and any(
                name not in self._excluded_expert_names for name in names
            ):
                raise ValueError(f"A storage alias group cannot be partially excluded: {sorted(names)}")

        for storage_key, intervals in all_intervals.items():
            intervals.sort()
            for previous, current in zip(intervals, intervals[1:]):
                if current[0] < previous[1] and (current[0], current[1]) != (previous[0], previous[1]):
                    raise ValueError(f"Partially overlapping parameter storage: {storage_key}")

        for alias_key, names in alias_names.items():
            device, storage_id, start, end = alias_key
            parameters = [tensor_by_name[name] for name in names]
            first = parameters[0]
            if any(
                parameter.dtype != first.dtype or tuple(parameter.shape) != tuple(first.shape)
                for parameter in parameters[1:]
            ):
                raise ValueError(f"Storage aliases have incompatible parameter metadata: {sorted(names)}")
            storage_key = (device, storage_id)
            padding_ranges = self._validate_padding_ranges(names, end - start)
            region = _ParameterRegion(
                storage_key=storage_key,
                start=start,
                end=end,
                names=set(names),
                padding_ranges=padding_ranges,
            )
            self._regions.append(region)
            self._regions_by_storage.setdefault(storage_key, []).append(region)
            for name in names:
                self._name_to_region[name] = region

        for storage_key, regions in self._regions_by_storage.items():
            regions.sort(key=lambda region: (region.start, region.end))
            for previous, current in zip(regions, regions[1:]):
                if current.start < previous.end:
                    raise ValueError(f"Partially overlapping protected parameter storage: {storage_key}")

        self._apply_alias_contracts()
        unknown_padding_names = set(self._padding_fill_ranges) - self._name_to_region.keys()
        if unknown_padding_names:
            raise ValueError(
                f"Padding fill ranges refer to unknown protected parameters: {sorted(unknown_padding_names)}"
            )

    def _validate_padding_ranges(self, names: Sequence[str], parameter_bytes: int) -> tuple[tuple[int, int], ...]:
        ranges = []
        for name in names:
            for interval in self._padding_fill_ranges.get(name, ()):
                if (
                    len(interval) != 2
                    or any(type(value) is not int for value in interval)
                    or not 0 <= interval[0] < interval[1] <= parameter_bytes
                ):
                    raise ValueError(f"Invalid parameter-relative padding byte range for {name}: {interval}")
                ranges.append(tuple(interval))
        ranges.sort()
        for previous, current in zip(ranges, ranges[1:]):
            if current[0] < previous[1]:
                raise ValueError(f"Overlapping declared padding ranges for {sorted(names)}")
        return tuple(ranges)

    def _apply_alias_contracts(self) -> None:
        for target_name, source_names in self._allowed_hf_aliases.items():
            region = self._name_to_region.get(target_name)
            sources = frozenset(source_names)
            if region is None or len(sources) < 2 or not sources.issubset(self._required_hf_names):
                raise ValueError(f"Invalid allowed HF alias contract for target {target_name}")
            if region.allowed_aliases and region.allowed_aliases != sources:
                raise ValueError(f"Conflicting HF alias contracts for target storage {region.canonical_name}")
            region.allowed_aliases = sources

    def tag_weights(self, named_tensors: Iterable[tuple[str, torch.Tensor]]) -> Iterator[tuple[str, torch.Tensor]]:
        """Tag each lazy HF tuple while the synchronous loader consumes it."""
        for name, tensor in named_tensors:
            if name not in self._required_hf_names:
                self._failed = True
                raise ValueError(f"Unexpected HF name supplied to ordinary loader: {name}")
            if self._active_hf_name is not None:
                self._failed = True
                raise ValueError("HF weight iterator was advanced while a previous item was still active")
            self._yielded_hf_names.add(name)
            self._readonly_source_storage.add(_storage_key(tensor))
            self._active_hf_name = name
            try:
                yield name, tensor
            finally:
                self._active_hf_name = None

    def _region_for_write(self, target: torch.Tensor) -> tuple[_ParameterRegion | None, int, int]:
        storage_key = _storage_key(target)
        regions = self._regions_by_storage.get(storage_key)
        if not regions:
            return None, 0, 0
        if target.numel() == 0:
            start = int(target.storage_offset()) * target.element_size()
            matches = [region for region in regions if region.start <= start <= region.end]
            if not matches:
                raise ValueError("Empty write to protected storage is outside declared parameter intervals")
            return matches[0], start, start
        start, end = _byte_interval(target)
        matches = [region for region in regions if region.start <= start and end <= region.end]
        if len(matches) != 1:
            raise ValueError("Write to protected storage is outside one declared parameter interval")
        return matches[0], start, end

    def _reject_excluded_expert_write(self, target: torch.Tensor) -> None:
        if target.numel() == 0:
            return
        storage_key = _storage_key(target)
        regions = self._excluded_expert_storage.get(storage_key)
        if not regions:
            return
        start, end = _storage_byte_bounds(target)
        if any(start < expert_end and expert_start < end for expert_start, expert_end, _ in regions):
            names = sorted(
                name for expert_start, expert_end, name in regions if start < expert_end and expert_start < end
            )
            raise ValueError(f"Ordinary loader attempted to write excluded expert storage: {names}")

    def _record_write(self, target: torch.Tensor, *, is_padding_fill: bool = False, fill_value: Any = None) -> None:
        region, start, end = self._preflight_write(target, is_padding_fill=is_padding_fill, fill_value=fill_value)
        if region is None:
            return
        self._append_write(region, start, end)

    def _preflight_write(
        self, target: torch.Tensor, *, is_padding_fill: bool = False, fill_value: Any = None
    ) -> tuple[_ParameterRegion | None, int, int]:
        if _storage_key(target) in self._readonly_source_storage:
            raise ValueError("Ordinary loader attempted to mutate read-only HF source storage")
        self._reject_excluded_expert_write(target)
        region, start, end = self._region_for_write(target)
        if region is None:
            return region, start, end
        if target.numel() == 0:
            if is_padding_fill and (not isinstance(fill_value, Number) or fill_value != 0):
                raise ValueError("Only zero fill is supported for empty protected vocabulary padding")
            return region, start, end
        if self._active_hf_name is None:
            raise ValueError("Protected parameter write occurred without an active HF source name")
        if is_padding_fill:
            if not isinstance(fill_value, Number) or fill_value != 0:
                raise ValueError("Only zero fill is supported for explicitly declared vocabulary padding")
            relative_start, relative_end = start - region.start, end - region.start
            if not any(left <= relative_start and relative_end <= right for left, right in region.padding_ranges):
                raise ValueError(f"Undeclared padding fill for protected parameter {region.canonical_name}")

        overlaps = [
            (write_start, write_end, source)
            for write_start, write_end, source in region.writes
            if start < write_end and write_start < end
        ]
        if overlaps:
            allowed = region.allowed_aliases
            if (
                not allowed
                or any((write_start, write_end) != (start, end) for write_start, write_end, _ in overlaps)
                or self._active_hf_name not in allowed
                or any(source not in allowed or source == self._active_hf_name for _, _, source in overlaps)
            ):
                raise ValueError(f"Overlapping parameter writes are not declared HF aliases: {region.canonical_name}")
        return region, start, end

    def _append_write(self, region: _ParameterRegion, start: int, end: int) -> None:
        if self._active_hf_name is None:
            raise ValueError("Protected parameter write occurred without an active HF source name")
        region.writes.append((start, end, self._active_hf_name))
        self._written_hf_names.add(self._active_hf_name)

    def _record_unrecognized_inplace(self, target: torch.Tensor) -> None:
        self._reject_excluded_expert_write(target)
        region, _, _ = self._region_for_write(target)
        if region is not None:
            raise ValueError(f"Unsupported in-place write to protected parameter {region.canonical_name}")

    def assert_complete(self) -> None:
        if self._failed:
            raise RuntimeError("Weight-write coverage recorder is poisoned")
        if self._complete:
            return
        try:
            self._assert_complete()
        except BaseException:
            self._failed = True
            raise

    def _assert_complete(self) -> None:
        if self._active_hf_name is not None:
            raise ValueError(f"HF weight iterator still has active item {self._active_hf_name}")
        if self._yielded_hf_names != self._required_hf_names:
            missing = sorted(self._required_hf_names - self._yielded_hf_names)
            raise ValueError(f"Ordinary loader did not consume required HF names: {missing}")
        if self._written_hf_names != self._required_hf_names:
            missing = sorted(self._required_hf_names - self._written_hf_names)
            raise ValueError(f"Required HF names produced no protected parameter writes: {missing}")
        self._assert_parameter_schema_unchanged()

        for region in self._regions:
            cursor = region.start
            for start, end, _source in sorted(region.writes):
                if start > cursor:
                    break
                cursor = max(cursor, end)
            if cursor < region.end:
                missing = (cursor - region.start, region.end - region.start)
                raise ValueError(f"Parameter storage has an uncovered byte range: {region.canonical_name} {missing}")
        self._complete = True

    def _assert_parameter_schema_unchanged(self) -> None:
        current = dict(self._model.named_parameters(remove_duplicate=False))
        if current.keys() != self._parameter_signatures.keys():
            raise ValueError("Model parameter names changed during ordinary weight loading")
        for name, parameter in current.items():
            if _signature(parameter) != self._parameter_signatures[name]:
                raise ValueError(f"Model parameter storage/schema changed during ordinary weight loading: {name}")


class _WeightWriteDispatchMode(TorchDispatchMode):
    def __init__(self, recorder: _WeightWriteCoverageRecorder) -> None:
        super().__init__()
        self._recorder = recorder

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        schema = getattr(func, "_schema", None)
        if schema is None:
            return func(*args, **kwargs)
        write_arguments = []
        for index, argument in enumerate(schema.arguments):
            alias_info = argument.alias_info
            if alias_info is None or not alias_info.is_write:
                continue
            if argument.name in kwargs:
                value = kwargs[argument.name]
            elif index < len(args):
                value = args[index]
            else:
                continue
            write_arguments.extend(_nested_tensors(value))
        if not write_arguments:
            return func(*args, **kwargs)

        try:
            prepared_writes = []
            for target in write_arguments:
                if _storage_key(target) in self._recorder._readonly_source_storage:
                    raise ValueError("Ordinary loader attempted to mutate read-only HF source storage")
                if schema.name == "aten::copy_":
                    prepared_writes.append(self._recorder._preflight_write(target))
                elif schema.name == "aten::fill_":
                    value = kwargs.get("value", args[1] if len(args) > 1 else None)
                    prepared_writes.append(
                        self._recorder._preflight_write(target, is_padding_fill=True, fill_value=value)
                    )
                else:
                    self._recorder._record_unrecognized_inplace(target)
            result = func(*args, **kwargs)
            if schema.name == "aten::copy_":
                region, start, end = prepared_writes[0]
                if region is not None:
                    self._recorder._append_write(region, start, end)
            elif schema.name == "aten::fill_":
                region, start, end = prepared_writes[0]
                if region is not None:
                    self._recorder._append_write(region, start, end)
        except BaseException:
            self._recorder._failed = True
            raise
        return result


def _nested_tensors(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        return [tensor for item in value for tensor in _nested_tensors(item)]
    if isinstance(value, dict):
        return [tensor for item in value.values() for tensor in _nested_tensors(item)]
    return []


@contextmanager
def record_weight_write_coverage(
    model: torch.nn.Module,
    *,
    required_hf_names: Collection[str],
    excluded_expert_names: Collection[str],
    padding_fill_ranges: Mapping[str, Sequence[tuple[int, int]]] | None = None,
    allowed_hf_aliases: Mapping[str, Collection[str]] | None = None,
) -> Iterator[_WeightWriteCoverageRecorder]:
    """Record ordinary loader writes and require complete live-parameter
    coverage.

    Padding byte ranges are relative to the named parameter and must be derived
    by the caller from its actual vocab/shard metadata. The lazy tagged
    iterator assumes the loader consumes input tuples synchronously. Any
    exception during loading or validation makes the context fail closed.
    """
    recorder = _WeightWriteCoverageRecorder(
        model,
        required_hf_names=required_hf_names,
        excluded_expert_names=excluded_expert_names,
        padding_fill_ranges=padding_fill_ranges,
        allowed_hf_aliases=allowed_hf_aliases,
    )
    try:
        with _WeightWriteDispatchMode(recorder):
            try:
                yield recorder
                recorder.assert_complete()
            except BaseException:
                recorder._failed = True
                recorder._active_hf_name = None
                raise
    finally:
        recorder._active_hf_name = None
