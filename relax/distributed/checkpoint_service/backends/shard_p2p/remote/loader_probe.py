# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Adapted from radixark/miles at 286634cc6aacec369938190880bdbd80d9b3769c.
# Licensed under Apache-2.0; see MILES_LICENSE.
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten
from torch.utils.weak import WeakIdKeyDictionary


class HfTensorSpec(NamedTuple):
    """An HF tensor the trainer sends, without its data."""

    shape: tuple[int, ...]
    dtype: torch.dtype


class NativeExportCatalog:
    """Metadata from one successful native reference, never tensor storage."""

    def __init__(self, version: int) -> None:
        self.version = version
        self.completed_version: int | None = None
        self.specs: dict[str, HfTensorSpec] = {}
        self.error: str | None = None

    def record(self, version: int, bucket: Iterable[tuple[str, torch.Tensor]]) -> None:
        # Capture cannot raise between the native path's payload collectives.
        # Resolve latched failures through the Actor group's error agreement.
        try:
            if version != self.version or self.completed_version is not None:
                raise RuntimeError("Native reference catalog version changed")
            for name, value in bucket:
                spec = HfTensorSpec(tuple(value.shape), value.dtype)
                if name in self.specs and self.specs[name] != spec:
                    raise RuntimeError(f"Conflicting native HF specification: {name}")
                self.specs[name] = spec
        except Exception as exc:
            self.error = self.error or f"{type(exc).__name__}: {exc}"

    def complete(self, version: int) -> None:
        self.completed_version = version

    def validate(self, version: int) -> None:
        if self.error or self.version != version or self.completed_version != version:
            raise RuntimeError(f"Native HF reference catalog is incomplete or stale: {self.error}")

    def resolve(self, version: int) -> dict[str, HfTensorSpec]:
        self.validate(version)
        if not self.specs:
            raise RuntimeError("The native reference publisher did not export any HF tensors")
        return dict(self.specs)


@dataclass(frozen=True)
class HfNameMapping:
    """Bidirectional HF-to-parameter dependencies; ignored HF names are
    absent."""

    param_names_by_hf_name: Mapping[str, frozenset[str]]
    hf_names_by_param_name: Mapping[str, frozenset[str]]

    @classmethod
    def from_hf_names_by_param_name(cls, hf_names_by_param_name: Mapping[str, frozenset[str]]) -> "HfNameMapping":
        param_names_by_hf_name = defaultdict(set)
        for param_name, hf_names in hf_names_by_param_name.items():
            for hf_name in hf_names:
                param_names_by_hf_name[hf_name].add(param_name)
        return cls(
            param_names_by_hf_name={name: frozenset(params) for name, params in param_names_by_hf_name.items()},
            hf_names_by_param_name=dict(hf_names_by_param_name),
        )

    def union(self, other: "HfNameMapping") -> "HfNameMapping":
        param_names = self.hf_names_by_param_name.keys() | other.hf_names_by_param_name.keys()
        return HfNameMapping.from_hf_names_by_param_name(
            {
                param_name: self.hf_names_by_param_name.get(param_name, frozenset())
                | other.hf_names_by_param_name.get(param_name, frozenset())
                for param_name in param_names
            }
        )


class LoaderWrites(NamedTuple):
    """Parameter dependencies and buffer names written by the loader."""

    hf_name_mapping: HfNameMapping
    buffer_names: frozenset[str]


def probe_loader_writes(
    model: torch.nn.Module,
    param_layouts: Mapping[str, object],
    hf_tensor_specs: Mapping[str, HfTensorSpec],
    *,
    device: torch.device,
) -> LoaderWrites:
    """Record `model.load_weights` writes using meta parameters and real HF
    tensors on `device`.

    The caller must enter the replica's parallelism context. Original
    parameters are restored on exit.
    """
    from relax.backends.sglang.shard_p2p_reload import parameter_bindings, validate_parameter_bindings

    originals, param_names_by_storage = _swap_params_for_meta(model, param_layouts)
    expected_bindings = parameter_bindings(model)
    recorder = _SourceRecordingMode(param_names_by_storage, _get_buffer_names_by_storage(model))

    def iter_probe_tensors():
        for hf_name, spec in hf_tensor_specs.items():
            # ones pass the loaders' format checks: positive scales, equal repeated scale rows
            tensor = torch.ones(spec.shape, dtype=spec.dtype, device=device)
            recorder.tag(tensor, frozenset({hf_name}))
            yield hf_name, tensor

    try:
        with recorder:
            model.load_weights(iter_probe_tensors())
        validate_parameter_bindings(model, expected_bindings)
    finally:
        for module, local_name, param in originals:
            module._parameters[local_name] = param
    return LoaderWrites(
        hf_name_mapping=HfNameMapping.from_hf_names_by_param_name(
            {param_name: frozenset(hf_names) for param_name, hf_names in recorder.hf_names_by_param_name.items()}
        ),
        buffer_names=frozenset(recorder.written_buffer_names),
    )


def _swap_params_for_meta(
    model: torch.nn.Module, param_layouts: Mapping[str, object]
) -> tuple[list[tuple[torch.nn.Module, str, torch.nn.Parameter]], dict[StorageWeakRef, list[str]]]:
    # preserve parameter ties when replacing CPU parameters with meta ones
    param_names_by_param_id = defaultdict(list)
    for name, param in model.named_parameters(remove_duplicate=False):
        param_names_by_param_id[id(param)].append(name)
    replacements, originals = {}, []
    for module in model.modules():
        for local_name, param in list(module._parameters.items()):
            if param is None:
                continue
            if id(param) not in replacements:
                layout = param_layouts[param_names_by_param_id[id(param)][0]]
                meta_data = torch.empty_strided(layout.shape, layout.stride, dtype=layout.dtype, device="meta")
                replacement = torch.Tensor._make_subclass(type(param), meta_data, param.requires_grad)
                replacement.__dict__.update(param.__dict__)
                replacements[id(param)] = replacement
            originals.append((module, local_name, param))
            module._parameters[local_name] = replacements[id(param)]
    # match the canonical names from named_parameters()
    param_names_by_storage = {
        StorageWeakRef(replacements[param_id].untyped_storage()): [names[0]]
        for param_id, names in param_names_by_param_id.items()
    }
    return originals, param_names_by_storage


def _get_buffer_names_by_storage(model: torch.nn.Module) -> dict[StorageWeakRef, list[str]]:
    buffer_names_by_storage = defaultdict(list)
    for name, buffer in model.named_buffers():
        buffer_names_by_storage[StorageWeakRef(buffer.untyped_storage())].append(name)
    return buffer_names_by_storage


class _SourceRecordingMode(TorchDispatchMode):
    """Propagate HF tensor sources through ops and record parameter and buffer
    writes."""

    def __init__(
        self,
        param_names_by_storage: Mapping[StorageWeakRef, list[str]],
        buffer_names_by_storage: Mapping[StorageWeakRef, list[str]],
    ) -> None:
        super().__init__()
        self._param_names_by_storage = param_names_by_storage
        self._buffer_names_by_storage = buffer_names_by_storage
        self._hf_names_by_tensor: WeakIdKeyDictionary = WeakIdKeyDictionary()
        self._hf_names_of_pending_scalar: frozenset[str] = frozenset()
        self.hf_names_by_param_name: dict[str, set[str]] = defaultdict(set)
        self.written_buffer_names: set[str] = set()

    def tag(self, tensor: torch.Tensor, hf_names: frozenset[str]) -> None:
        self._hf_names_by_tensor[tensor] = hf_names

    def _hf_names_of(self, value: object) -> frozenset[str]:
        return self._hf_names_by_tensor.get(value, frozenset()) if isinstance(value, torch.Tensor) else frozenset()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        hf_names = frozenset().union(*map(self._hf_names_of, tree_flatten((args, kwargs))[0]))
        if func is torch.ops.aten._local_scalar_dense.default:
            # the write that consumes the value takes its HF names
            self._hf_names_of_pending_scalar |= hf_names
            if not args[0].is_meta:
                return func(*args, **kwargs)
            # a param's value; where bytes land never depends on it
            dtype = args[0].dtype
            return False if dtype == torch.bool else (0.0 if dtype.is_floating_point else 0)
        written = [
            args[index] if index < len(args) else kwargs.get(argument.name)
            for index, argument in enumerate(func._schema.arguments)
            if argument.alias_info is not None and argument.alias_info.is_write
        ]
        if written and not hf_names:
            hf_names, self._hf_names_of_pending_scalar = self._hf_names_of_pending_scalar, frozenset()
        output = func(*args, **kwargs)
        for destination in written:
            if not isinstance(destination, torch.Tensor):
                continue
            storage = StorageWeakRef(destination.untyped_storage())
            param_names = self._param_names_by_storage.get(storage)
            if param_names is None:
                self.tag(destination, self._hf_names_of(destination) | hf_names)
                self.written_buffer_names.update(self._buffer_names_by_storage.get(storage, ()))
            else:
                # writes without HF names, such as zeroed padding, are not loaded data
                for param_name in param_names if hf_names else ():
                    self.hf_names_by_param_name[param_name] |= hf_names
        if hf_names:
            for value in tree_flatten(output)[0]:
                if isinstance(value, torch.Tensor) and not any(value is destination for destination in written):
                    self.tag(value, hf_names)
        return output


class ModelParamStager:
    """Hold each parameter group's HF tensors until the group is complete.

    Parameters sharing HF inputs form one group; loading only part would leave
    parameters unbound or incomplete.
    """

    def __init__(self, hf_name_mapping: HfNameMapping) -> None:
        self._param_group_by_hf_name, self._hf_names_by_param_group = _build_param_groups_by_shared_hf_inputs(
            hf_name_mapping
        )
        self._staged_hf_tensors_by_param_group: dict[tuple[str, ...], list[tuple[str, torch.Tensor]]] = {}
        self._missing_hf_names_by_param_group: dict[tuple[str, ...], set[str]] = {}

    @property
    def param_groups(self) -> list[tuple[str, ...]]:
        return list(self._hf_names_by_param_group)

    def stage(
        self, hf_tensors: Iterable[tuple[str, torch.Tensor]]
    ) -> dict[tuple[str, ...], list[tuple[str, torch.Tensor]]]:
        """Accumulate HF tensors across calls and return each group with all
        its required inputs once complete."""
        ready_hf_tensors_by_param_group = {}
        for hf_name, tensor in hf_tensors:
            param_group = self._param_group_by_hf_name[hf_name]
            missing_hf_names = self._missing_hf_names_by_param_group.setdefault(
                param_group, set(self._hf_names_by_param_group[param_group])
            )
            missing_hf_names.discard(hf_name)
            self._staged_hf_tensors_by_param_group.setdefault(param_group, []).append((hf_name, tensor))
            if not missing_hf_names:
                del self._missing_hf_names_by_param_group[param_group]
                ready_hf_tensors_by_param_group[param_group] = self._staged_hf_tensors_by_param_group.pop(param_group)
        return ready_hf_tensors_by_param_group

    def assert_all_done(self) -> None:
        assert not self._missing_hf_names_by_param_group, (
            "p2p update ended with params still missing HF tensors, which would leave their bytes unwritten: "
            + ", ".join(
                f"{param_group} lacks {sorted(missing)[:3]}"
                for param_group, missing in list(self._missing_hf_names_by_param_group.items())[:5]
            )
        )


def _build_param_groups_by_shared_hf_inputs(
    hf_name_mapping: HfNameMapping,
) -> tuple[dict[str, tuple[str, ...]], dict[tuple[str, ...], frozenset[str]]]:
    """Group params connected directly or transitively by shared HF inputs.

    Return mappings from HF name to param group and from param group to its
    required HF names.
    """
    # union-find over params: two params sharing an HF name load together
    root_by_param_name = {param_name: param_name for param_name in hf_name_mapping.hf_names_by_param_name}

    def find_root(param_name: str) -> str:
        while root_by_param_name[param_name] != param_name:
            root_by_param_name[param_name] = root_by_param_name[root_by_param_name[param_name]]
            param_name = root_by_param_name[param_name]
        return param_name

    for param_names in hf_name_mapping.param_names_by_hf_name.values():
        first_param_name, *other_param_names = sorted(param_names)
        for param_name in other_param_names:
            root_by_param_name[find_root(param_name)] = find_root(first_param_name)

    param_names_by_root = defaultdict(list)
    for param_name in root_by_param_name:
        param_names_by_root[find_root(param_name)].append(param_name)
    param_group_by_hf_name, hf_names_by_param_group = {}, {}
    for param_names in param_names_by_root.values():
        param_group = tuple(sorted(param_names))
        hf_names = frozenset().union(*(hf_name_mapping.hf_names_by_param_name[name] for name in param_group))
        hf_names_by_param_group[param_group] = hf_names
        for hf_name in hf_names:
            param_group_by_hf_name[hf_name] = param_group
    return param_group_by_hf_name, hf_names_by_param_group
