# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Native reload adapters own model-specific post-load cache semantics."""

from dataclasses import dataclass
from importlib import import_module
from typing import Any

import torch


def parameter_bindings(model: torch.nn.Module) -> tuple[tuple[str, torch.nn.Parameter], ...]:
    """Keep every native path, including names hidden by canonical
    iteration."""
    return tuple(model.named_parameters(remove_duplicate=False))


def validate_parameter_bindings(model: torch.nn.Module, expected: tuple) -> None:
    actual = dict(parameter_bindings(model))
    if len(actual) != len(expected) or any(actual.get(name) is not value for name, value in expected):
        raise RuntimeError("Native parameter alias binding changed")


def validate_parameter_storage(model: torch.nn.Module) -> None:
    """Identical Parameter aliases have one owner; distinct overlapping views
    do not."""
    intervals = []
    for name, value in model.named_parameters():
        if not value.numel():
            continue
        if value.is_meta or any(stride < 0 for stride in value.stride()):
            raise NotImplementedError("Native parameter storage cannot be proved")
        size = (
            1 + sum((n - 1) * stride for n, stride in zip(value.shape, value.stride(), strict=True))
        ) * value.element_size()
        intervals.append((str(value.device), value.data_ptr(), value.data_ptr() + size, name))
    previous_device, previous_end, previous_name = None, 0, None
    for device, start, end, name in sorted(intervals):
        if device == previous_device and start < previous_end:
            raise NotImplementedError(f"Distinct native Parameters have overlapping storage: {previous_name}, {name}")
        previous_device, previous_end, previous_name = device, end, name


def _registered_models() -> tuple[type, ...]:
    # Register native constructors rather than interpreting a wrapper's config:
    # a multimodal root may legitimately expose only its text config.
    models = []
    for name in ("qwen3_5", "qwen3_5_text"):
        try:
            entry = import_module(f"sglang.srt.models.{name}").EntryClass
        except ModuleNotFoundError:
            continue
        models.extend(entry if isinstance(entry, list) else [entry])
    return tuple(models)


def _buffer_schema(model: torch.nn.Module) -> dict[str, tuple]:
    return {
        name: (value.data_ptr(), tuple(value.shape), tuple(value.stride()), value.dtype, value.device)
        for name, value in model.named_buffers()
    }


@dataclass(frozen=True)
class NativeReloadState:
    """One guarded model's static policy and live cache allocation epoch."""

    adapter: "Qwen35ReloadAdapter"
    model: torch.nn.Module
    buffer_schema: dict[str, tuple]
    derived: tuple[tuple[str, torch.nn.Module], ...]
    parameters: tuple[tuple[str, torch.nn.Parameter], ...]

    def validate(self) -> None:
        validate_parameter_bindings(self.model, self.parameters)
        actual = _buffer_schema(self.model)
        derived = {f"{name}.gemma_weight" for name, _ in self.derived}
        if (
            type(self.model) is not self.adapter.model_class
            or actual.keys() != self.buffer_schema.keys()
            or any(actual[name] != self.buffer_schema[name] for name in derived)
            or any(actual[name][-2:] != self.buffer_schema[name][-2:] for name in actual.keys() - derived)
        ):
            raise RuntimeError("Native reload cache allocation epoch changed")
        for name in ("load_weights", "post_load_weights"):
            actual = getattr(self.model, name, None)
            if getattr(actual, "__func__", actual) is not getattr(self.adapter.model_class, name, None):
                raise RuntimeError("Native reload model hook changed")
        for name, module in self.derived:
            if self.model.get_submodule(name) is not module:
                raise RuntimeError("Native reload cache owner changed")

    def capture_epoch(self) -> dict[str, tuple]:
        self.validate()
        # Rotary tables may grow between rollouts. They must remain stable
        # throughout this paused update; Gemma graph storage stays fixed.
        return _buffer_schema(self.model)

    def refresh(self, expected: set[str], *, epoch: dict[str, tuple] | None = None) -> set[str]:
        from relax.distributed.checkpoint_service.backends.shard_p2p.remote.compat import post_load_weights

        self.validate()
        epoch = self.capture_epoch() if epoch is None else epoch
        if _buffer_schema(self.model) != epoch:
            raise RuntimeError("Native cache changed after update prepare")
        refreshed = {f"{name}.gemma_weight" for name, _ in self.derived}
        if refreshed != set(expected):
            raise RuntimeError("Native loader cache completion is incomplete")
        for _, module in self.derived:
            module._weight_loader(module.weight, module.weight.data)
        post_load_weights(self.model)
        self.validate()
        if _buffer_schema(self.model) != epoch:
            raise RuntimeError("Native post-load changed graph-captured cache storage")
        return refreshed


@dataclass(frozen=True)
class Qwen35ReloadAdapter:
    """Registered Qwen3.5 loaders use native Gemma caches and rotary tables.

    Registration selects these cache semantics; actual expert layout, bound
    loaders, allocation aliases and cache ownership still require proof.
    """

    model_class: type

    @property
    def identity(self) -> str:
        return f"gemma-rotary-v1:{self.model_class.__module__}.{self.model_class.__qualname__}"

    def validate_model(self, model: torch.nn.Module) -> None:
        if type(model) is not self.model_class:
            raise NotImplementedError("Replica and actual scheduler use different native model constructors")
        for name in ("load_weights", "post_load_weights"):
            actual, expected = getattr(model, name, None), getattr(self.model_class, name, None)
            if getattr(actual, "__func__", actual) is not expected:
                raise NotImplementedError(f"Unrecognized native model {name} hook")

    def buffers(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        from sglang.srt.layers.layernorm import GemmaRMSNorm
        from sglang.srt.layers.rotary_embedding import RotaryEmbedding
        from sglang.srt.models.qwen2_vl import Qwen2VisionRotaryEmbedding

        self.validate_model(model)
        buffers = {}
        for prefix, module in model.named_modules():
            for name, value in module.named_buffers(recurse=False):
                if type(module) is GemmaRMSNorm:
                    if (
                        name != "gemma_weight"
                        or getattr(module._weight_loader, "__func__", None) is not GemmaRMSNorm._weight_loader
                    ):
                        raise NotImplementedError("Unrecognized native Gemma cache loader")
                elif isinstance(module, (RotaryEmbedding, Qwen2VisionRotaryEmbedding)):
                    if name not in ("cos_sin_cache", "inv_freq"):
                        raise NotImplementedError("Unrecognized native rotary cache")
                else:
                    raise NotImplementedError(f"No reload cache policy for {type(module).__qualname__}.{name}")
                buffers[f"{prefix}.{name}" if prefix else name] = value
        return buffers

    def validate_native(self, model: torch.nn.Module, args: Any, tp_rank: int) -> None:
        from relax.distributed.checkpoint_service.backends.shard_p2p.remote.layout import validate_native_model
        from relax.distributed.checkpoint_service.backends.shard_p2p.remote.replica_overlap import (
            validate_disjoint_ranges,
        )

        experts = validate_native_model(model, args, tp_rank)
        values = dict(model.named_parameters())
        values.update({f"buffer:{name}": value for name, value in self.buffers(model).items()})
        ranges = {
            name: dict(
                address=value.data_ptr(),
                nbytes=(1 + sum((n - 1) * stride for n, stride in zip(value.shape, value.stride(), strict=True)))
                * value.element_size(),
            )
            for name, value in values.items()
            if value.numel()
        }
        validate_disjoint_ranges(ranges, set(experts), set(ranges) - experts.keys())

    def validate_written_buffers(self, model: torch.nn.Module, names: set[str]) -> None:
        from sglang.srt.layers.layernorm import GemmaRMSNorm

        # The replica intentionally suppresses model post-load while staging
        # partial buckets. Its native loader and cache ownership must still match.
        actual = getattr(model, "load_weights", None)
        if (
            type(model) is not self.model_class
            or getattr(actual, "__func__", actual) is not self.model_class.load_weights
        ):
            raise NotImplementedError("Replica native loader changed")
        allowed = {f"{name}.gemma_weight" for name, module in model.named_modules() if type(module) is GemmaRMSNorm}
        if any(
            getattr(module._weight_loader, "__func__", None) is not GemmaRMSNorm._weight_loader
            for module in model.modules()
            if type(module) is GemmaRMSNorm
        ):
            raise NotImplementedError("Replica native cache loader changed")
        if set(names) != allowed:
            raise NotImplementedError("Native loader writes caches outside its registered reload policy")

    def bind(self, model: torch.nn.Module) -> NativeReloadState:
        from sglang.srt.layers.layernorm import GemmaRMSNorm

        validate_parameter_storage(model)
        self.buffers(model)
        return NativeReloadState(
            self,
            model,
            _buffer_schema(model),
            tuple((name, module) for name, module in model.named_modules() if type(module) is GemmaRMSNorm),
            parameter_bindings(model),
        )


def reload_adapter(model_config: Any, model: torch.nn.Module | None = None) -> Qwen35ReloadAdapter:
    """Resolve the installed native factory, then require a cache adapter."""
    from sglang.srt.model_loader.utils import get_model_architecture

    native_class, _ = get_model_architecture(model_config)
    if native_class not in _registered_models():
        raise NotImplementedError("No reload/post-load adapter for the installed native model constructor")
    adapter = Qwen35ReloadAdapter(native_class)
    if model is not None:
        adapter.validate_model(model)
    return adapter


def receiver_capability(model: torch.nn.Module, args: Any, tp_rank: int) -> dict:
    """Return observed predicates; unsupported adapters remain native."""
    from sglang.srt.configs.model_config import ModelConfig

    values = dict(
        pp_size=args.pp_size,
        dp_size=args.dp_size,
        ep_size=args.ep_size,
        quantization=args.quantization,
        enable_lora=args.enable_lora,
        speculative_algorithm=args.speculative_algorithm,
        enable_dp_attention=args.enable_dp_attention,
        enable_memory_saver=getattr(args, "enable_memory_saver", None),
    )
    mode = dict(values)
    # SGLang leaves the disabled default as None; configured adapter paths can
    # still enable LoRA even when that optional flag is unset.
    mode["enable_lora"] = bool(args.enable_lora or args.lora_paths)
    values["lora_paths"] = args.lora_paths
    checks = {
        "receiver_mode": mode
        == dict(
            pp_size=1,
            dp_size=1,
            ep_size=1,
            quantization=None,
            enable_lora=False,
            speculative_algorithm=None,
            enable_dp_attention=False,
            enable_memory_saver=False,
        )
    }
    result = dict(
        observed=values,
        checks=checks,
        native_class=f"{type(model).__module__}.{type(model).__qualname__}",
        actual_model_type=getattr(getattr(model, "config", None), "model_type", None),
        adapter=None,
        rejection=None,
    )
    if checks["receiver_mode"]:
        try:
            config = ModelConfig.from_server_args(args)
            text = getattr(config.hf_config, "text_config", None) or config.hf_config
            result.update(
                outer_model_type=getattr(config.hf_config, "model_type", None),
                text_model_type=getattr(text, "model_type", None),
            )
            adapter = reload_adapter(config)
            failures = []
            for name, validate in (
                ("native_constructor_hooks", lambda: adapter.validate_model(model)),
                ("native_parameter_storage", lambda: validate_parameter_storage(model)),
                ("native_expert_layout_cache", lambda: adapter.validate_native(model, args, tp_rank)),
            ):
                try:
                    validate()
                    checks[name] = True
                except NotImplementedError as exc:
                    checks[name] = False
                    failures.append(str(exc))
            checks["native_loader_layout_cache"] = not failures
            if failures:
                result["rejection"] = "; ".join(failures)
            else:
                model._relax_native_weight_guard.native_reload = adapter.bind(model)
            result["adapter"] = adapter.identity
        except NotImplementedError as exc:
            checks["native_loader_layout_cache"] = False
            result["rejection"] = str(exc)
    return result
