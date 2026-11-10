# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Adapted from radixark/miles at 286634cc6aacec369938190880bdbd80d9b3769c.
# Licensed under Apache-2.0; see MILES_LICENSE.
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any, NamedTuple

import torch
from sglang.srt.runtime_context import get_server_args

from relax.backends.sglang.shard_p2p_reload import parameter_bindings, validate_parameter_bindings
from relax.utils.logging_utils import get_logger

from .compat import ParallelismContext
from .compat import RankConfig as RolloutEngineRankConfig
from .compat import publish_server_args as _publish_server_args
from .loader_probe import HfNameMapping, HfTensorSpec, probe_loader_writes


logger = get_logger(__name__)
_PARAM_ALIGNMENT_BYTES = 256
RankParallelismConfig = tuple[int, int]


class TransferBufferParamLayout(NamedTuple):
    """How one param's bytes are laid out in a transfer buffer: as the engine's
    param is after `restore_weights_before_loading`, which is what its loader
    writes into."""

    shape: torch.Size
    stride: tuple[int, ...]
    dtype: torch.dtype
    occupied_nbytes: int

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "TransferBufferParamLayout":
        return cls(tensor.shape, tensor.stride(), tensor.dtype, _compute_occupied_nbytes(tensor))


def _compute_occupied_nbytes(tensor: torch.Tensor) -> int:
    """The bytes a tensor's elements cover in memory, from its first element to
    its last; `numel × element_size` when it is contiguous, more when its
    strides leave gaps."""
    if tensor.numel() == 0:
        return 0
    last_element_offset = sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride(), strict=True))
    return (last_element_offset + 1) * tensor.element_size()


class ModelReplica:
    """Convert HF weights into one rollout rank's reload layout using its
    sglang loader.

    Parameters retain their reload metadata but have storage only while bound
    to a transfer buffer.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        built_shapes_by_name: Mapping[str, torch.Size],
        config: RolloutEngineRankConfig,
        *,
        postprocess_device: torch.device,
        transfer_buffer_device: torch.device,
        reload_adapter: Any = None,
    ) -> None:
        if reload_adapter is not None:
            reload_adapter.validate_model(model)
        # some models call it from their own load_weights; the engine runs it after the writes
        if hasattr(model, "post_load_weights"):  # config-access-exempt: optional inference-model post-load callback
            model.post_load_weights = lambda *args, **kwargs: None
        self._model = model
        self.reload_adapter = reload_adapter
        self._config = config
        # the trainer's HF tensors are on the GPU postprocess runs on
        self._hf_tensor_device = postprocess_device
        self._transfer_buffer_device = transfer_buffer_device
        self.transfer_buffer_param_layouts = _run_engine_postprocess_and_restore(
            model, built_shapes_by_name, config.parallelism, postprocess_device
        )
        self._params_by_name = dict(model.named_parameters())
        self._parameter_bindings = parameter_bindings(model)
        for param in self._params_by_name.values():
            param.data = torch.empty(0, dtype=param.dtype)
        self._fully_written_param_names: set[str] = set()

    def validate_reload_policy(self) -> None:
        if self.reload_adapter is None:
            raise NotImplementedError("Native layout replica has no registered reload policy")
        self.reload_adapter.validate_written_buffers(self._model, self.loader_written_buffer_names)

    def map_hf_names(self, hf_tensor_specs: Mapping[str, HfTensorSpec]) -> HfNameMapping:
        """Probe HF-to-parameter dependencies and allocate scratch storage for
        buffers the loader writes."""
        validate_parameter_bindings(self._model, self._parameter_bindings)
        # This image has no storage-less model constructor. Make its CPU buffers
        # meta before probing (GemmaRMSNorm derives a buffer from a meta param).
        for module in self._model.modules():
            for name, buffer in list(module._buffers.items()):
                if buffer is not None:
                    module._buffers[name] = buffer.to("meta")
        if get_server_args() is not self._config.server_args:
            _publish_server_args(self._config.server_args)
        with ParallelismContext(self._config.parallelism):
            loader_writes = probe_loader_writes(
                self._model, self.transfer_buffer_param_layouts, hf_tensor_specs, device=self._hf_tensor_device
            )
        self.loader_written_buffer_names = loader_writes.buffer_names
        validate_parameter_bindings(self._model, self._parameter_bindings)
        if loader_writes.buffer_names:
            logger.info(
                f"the {self._config.runner_role} loader also fills {len(loader_writes.buffer_names)} buffers, e.g. "
                f"{sorted(loader_writes.buffer_names)[:3]}; p2p writes only params, so the rollout engine derives "
                "these itself"
            )
        _materialize_loader_written_buffers(self._model, loader_writes.buffer_names, self._transfer_buffer_device)
        return loader_writes.hf_name_mapping

    def load_into(
        self, buffer: torch.Tensor, param_names: Sequence[str], hf_tensors: list[tuple[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        """Loads `hf_tensors`, the HF tensors of exactly `param_names`, into
        the uint8 `buffer`; returns the bytes of each param in `buffer`, by
        name.

        `pack_into_buffers` gives groups of params that fit one buffer. A write
        carries a param's bytes, all of them, to the rollout engine, so this
        raises if loading changed a param in any way but its bytes, or, the
        first time a param is loaded, left any of its bytes unwritten.
        """
        # sglang keeps one live config per process; load under the one this replica was built with
        validate_parameter_bindings(self._model, self._parameter_bindings)
        if get_server_args() is not self._config.server_args:
            _publish_server_args(self._config.server_args)
        param_bytes_by_name = _slice_buffer_by_param(buffer, param_names, self.transfer_buffer_param_layouts)
        if self._fully_written_param_names.issuperset(param_names):
            self._load(param_bytes_by_name, hf_tensors)
        else:
            self._load_asserting_every_byte_is_written(param_bytes_by_name, hf_tensors)
        return param_bytes_by_name

    def _load(
        self, param_bytes_by_name: Mapping[str, torch.Tensor], hf_tensors: list[tuple[str, torch.Tensor]]
    ) -> None:
        with self._bind_params_to_bytes(param_bytes_by_name) as bound_params_by_name:
            state_before_load_by_name = {
                name: _get_param_state_besides_bytes(param) for name, param in bound_params_by_name.items()
            }
            with ParallelismContext(self._config.parallelism):
                self._model.load_weights(hf_tensors)
            self._assert_params_changed_only_bytes(bound_params_by_name, state_before_load_by_name)

    def _load_asserting_every_byte_is_written(
        self, param_bytes_by_name: Mapping[str, torch.Tensor], hf_tensors: list[tuple[str, torch.Tensor]]
    ) -> None:
        for param_bytes in param_bytes_by_name.values():
            param_bytes.zero_()
        self._load(param_bytes_by_name, hf_tensors)
        bytes_loaded_over_zeros_by_name = {
            name: param_bytes.clone() for name, param_bytes in param_bytes_by_name.items()
        }
        for param_bytes in param_bytes_by_name.values():
            param_bytes.fill_(0xFF)
        self._load(param_bytes_by_name, hf_tensors)
        unwritten_nbytes_by_name = {
            name: int(torch.count_nonzero(param_bytes != bytes_loaded_over_zeros_by_name[name]))
            for name, param_bytes in param_bytes_by_name.items()
        }
        partly_written_param_names = [name for name, nbytes in unwritten_nbytes_by_name.items() if nbytes]
        if partly_written_param_names:
            raise NotImplementedError(
                f"the rollout engine's loader leaves bytes of {len(partly_written_param_names)} params unwritten, e.g. "
                + ", ".join(
                    f"{name} ({unwritten_nbytes_by_name[name]} of {param_bytes_by_name[name].numel()} bytes)"
                    for name in partly_written_param_names[:5]
                )
                + "; p2p would send what an earlier load left there in their place"
            )
        self._fully_written_param_names.update(param_bytes_by_name)

    @contextmanager
    def _bind_params_to_bytes(
        self, param_bytes_by_name: Mapping[str, torch.Tensor]
    ) -> Iterator[dict[str, torch.nn.Parameter]]:
        bound_params_by_name = {name: self._params_by_name[name] for name in param_bytes_by_name}
        try:
            for name, param in bound_params_by_name.items():
                param.data = _view_bytes_as_param(param_bytes_by_name[name], self.transfer_buffer_param_layouts[name])
            yield bound_params_by_name
        finally:
            for param in bound_params_by_name.values():
                param.data = torch.empty(0, dtype=param.dtype)

    def _assert_params_changed_only_bytes(
        self, bound_params_by_name: Mapping[str, torch.nn.Parameter], state_before_load_by_name: Mapping[str, tuple]
    ) -> None:
        validate_parameter_bindings(self._model, self._parameter_bindings)
        params_after_load_by_name = dict(self._model.named_parameters())
        changed_param_names = [
            name
            for name, param in bound_params_by_name.items()
            if params_after_load_by_name[name] is not param
            or _get_param_state_besides_bytes(param) != state_before_load_by_name[name]
        ]
        assert not changed_param_names, (
            f"loading changed more than the bytes of {', '.join(changed_param_names[:5])} "
            f"({len(changed_param_names)} in all); the rollout engine receives only bytes, so its param would keep "
            "the old shape, storage or attributes"
        )


def _materialize_loader_written_buffers(
    model: torch.nn.Module, buffer_names: frozenset[str], device: torch.device
) -> None:
    # preserve aliases for buffers registered under multiple names
    meta_buffer_ids = {id(buffer) for name, buffer in model.named_buffers() if name in buffer_names and buffer.is_meta}
    scratch_buffers_by_id = {}
    for module in model.modules():
        for local_name, buffer in module._buffers.items():
            if buffer is None or id(buffer) not in meta_buffer_ids:
                continue
            if id(buffer) not in scratch_buffers_by_id:
                scratch_buffers_by_id[id(buffer)] = torch.empty_strided(
                    buffer.shape, buffer.stride(), dtype=buffer.dtype, device=device
                )
            module._buffers[local_name] = scratch_buffers_by_id[id(buffer)]


def _run_engine_postprocess_and_restore(
    model: torch.nn.Module,
    built_shapes_by_name: Mapping[str, torch.Size],
    parallelism: RankParallelismConfig,
    postprocess_device: torch.device,
) -> dict[str, TransferBufferParamLayout]:
    """Runs the engine's startup postprocess and session restore on `model`,
    one module at a time on zero tensors, and returns the transfer buffer
    layout of each param afterwards, by name.

    Every param ends 0-size on the CPU.
    """
    built_shapes_by_param_id = {id(param): built_shapes_by_name[name] for name, param in model.named_parameters()}
    reload_layouts_by_param_id = {}
    for module_name, module in model.named_modules():
        # modules without one keep their params as built
        if (
            getattr(module, "quant_method", None) is not None
        ):  # config-access-exempt: model traversal includes modules without quantization methods
            reload_layouts_by_param_id |= _run_module_postprocess_and_restore(
                module_name, module, built_shapes_by_param_id, parallelism, postprocess_device
            )

    param_layouts = {}
    for name, param in model.named_parameters():
        if id(param) in reload_layouts_by_param_id:
            param_layouts[name] = reload_layouts_by_param_id[id(param)]
        else:
            built_param = torch.empty(built_shapes_by_name[name], dtype=param.dtype, device="meta")
            param_layouts[name] = TransferBufferParamLayout.from_tensor(built_param)
    return param_layouts


def _run_module_postprocess_and_restore(
    module_name: str,
    module: torch.nn.Module,
    built_shapes_by_param_id: Mapping[int, torch.Size],
    parallelism: RankParallelismConfig,
    postprocess_device: torch.device,
) -> dict[int, TransferBufferParamLayout]:
    for param in module.parameters(recurse=False):
        param.data = torch.zeros(built_shapes_by_param_id[id(param)], dtype=param.dtype, device=postprocess_device)
    attributes_before_postprocess = [*vars(module).values(), *module._buffers.values()]
    with ParallelismContext(parallelism):
        module.quant_method.process_weights_after_loading(module)
        # Match ModelRunner.post_process_weights: older SGLang images have no
        # restore method on the base class, and leave that module's native
        # postprocessed layout unchanged when beginning a weight update.
        restore = getattr(module.quant_method, "restore_weights_before_loading", None)
        if callable(restore):
            restore(module)

    reload_layouts_by_param_id = {}
    # one entry per Parameter, though postprocess may register one under two names
    for param_name, param in module.named_parameters(recurse=False):
        assert param.device.type != "meta", (
            f"the postprocess of {module_name} left {param_name} on meta; it must have read a tensor that "
            "construction without storage does not allocate"
        )
        reload_layouts_by_param_id[id(param)] = TransferBufferParamLayout.from_tensor(param)
        param.data = torch.empty(0, dtype=param.dtype)
    _free_postprocess_tensors(module, attributes_before_postprocess)
    return reload_layouts_by_param_id


def _free_postprocess_tensors(module: torch.nn.Module, attributes_before_postprocess: Sequence[object]) -> None:
    """Postprocess attaches what the engine's kernels read in forward; a
    replica never runs forward."""

    def without_storage(value: object) -> object:
        if isinstance(value, torch.Tensor) and not isinstance(value, torch.nn.Parameter):
            return value.to("meta")
        if type(value) in (tuple, list):
            return type(value)(without_storage(item) for item in value)
        return value

    ids_before_postprocess = {id(value) for value in attributes_before_postprocess}
    for attributes in (vars(module), module._buffers):
        for name, value in list(attributes.items()):
            if id(value) not in ids_before_postprocess:
                attributes[name] = without_storage(value)


def _slice_buffer_by_param(
    buffer: torch.Tensor, param_names: Sequence[str], param_layouts: Mapping[str, TransferBufferParamLayout]
) -> dict[str, torch.Tensor]:
    """Lays `param_names` out one after another in the uint8 `buffer`, each
    start aligned; returns the bytes each param occupies, by name."""
    param_bytes_by_name, param_end_offset = {}, 0
    for name in param_names:
        param_start_offset = _align_param_start(param_end_offset)
        param_end_offset = param_start_offset + param_layouts[name].occupied_nbytes
        assert param_end_offset <= buffer.numel(), (
            f"{', '.join(param_names)} do not fit a {buffer.numel()}-byte transfer buffer; group them with "
            "pack_into_buffers"
        )
        param_bytes_by_name[name] = buffer[param_start_offset:param_end_offset]
    return param_bytes_by_name


def _align_param_start(offset: int) -> int:
    return -(-offset // _PARAM_ALIGNMENT_BYTES) * _PARAM_ALIGNMENT_BYTES


def _view_bytes_as_param(param_bytes: torch.Tensor, param_layout: TransferBufferParamLayout) -> torch.Tensor:
    return torch.as_strided(param_bytes.view(param_layout.dtype), param_layout.shape, param_layout.stride)


def _get_param_state_besides_bytes(param: torch.nn.Parameter) -> tuple:
    attributes = {
        key: value if isinstance(value, bool | int | float | str | None) else id(value)
        for key, value in vars(param).items()
    }
    return param.shape, param.stride(), param.dtype, param.data_ptr(), attributes


def pack_into_buffers(
    param_groups: Iterable[tuple[str, ...]],
    param_layouts: Mapping[str, TransferBufferParamLayout],
    buffer_nbytes: int,
) -> Iterator[list[tuple[str, ...]]]:
    """Pack complete parameter groups into buffers, preserving order and
    parameter alignment."""
    packed_groups, end_offset = [], 0
    for param_group in param_groups:
        group_nbytes = compute_param_group_nbytes(param_group, param_layouts)
        assert group_nbytes <= buffer_nbytes, f"{param_group} need {group_nbytes} bytes, a buffer has {buffer_nbytes}"
        group_end_offset = _end_offset_after(param_group, param_layouts, start_offset=end_offset)
        if packed_groups and group_end_offset > buffer_nbytes:
            yield packed_groups
            packed_groups, group_end_offset = [], group_nbytes
        packed_groups.append(param_group)
        end_offset = group_end_offset
    if packed_groups:
        yield packed_groups


def compute_param_group_nbytes(
    param_group: Iterable[str], param_layouts: Mapping[str, TransferBufferParamLayout]
) -> int:
    """Return the group's byte size, including alignment between parameters."""
    return _end_offset_after(param_group, param_layouts, start_offset=0)


def _end_offset_after(
    param_names: Iterable[str], param_layouts: Mapping[str, TransferBufferParamLayout], start_offset: int
) -> int:
    end_offset = start_offset
    for name in param_names:
        end_offset = _align_param_start(end_offset) + param_layouts[name].occupied_nbytes
    return end_offset
