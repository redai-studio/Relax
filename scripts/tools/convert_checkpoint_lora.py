# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Plan native LoRA merges without importing Megatron or constructing a
model."""

from __future__ import annotations

import io
import math
from argparse import Namespace
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from relax.utils.lora_checkpoint_identity import hf_checkpoint_identity, verified_lora_base_identity
from scripts.tools.convert_checkpoint_mapping import Conversion, TensorSlice, build_conversions, normalize_key


class _MetadataEnum:
    """Retain unused enum values without importing their training runtime."""

    def __init__(self, value: Any) -> None:
        self.value = value


def _load_common(source: Any) -> Any:
    import torch

    enum_names = {
        "megatron.core.enums": ("ModelType", "Fp8Recipe", "Fp4Recipe"),
        "megatron.core.transformer.enums": (
            "ModelType",
            "LayerType",
            "AttnType",
            "AttnMaskType",
            "AttnBackend",
            "CudaGraphScope",
            "CudaGraphModule",
            "InferenceCudaGraphScope",
        ),
        "transfer_queue.utils.enum_utils": ("Role", "TransferQueueRole"),
        "megatron.core.rerun_state_machine": ("RerunMode", "RerunState"),
        "signal": ("Signals",),
        "relax.core.registry": ("ROLES_SFT_ONLY",),
    }
    allowed = [Namespace, io.BytesIO, dict, list, int, defaultdict, Any]
    # Old streaming-data checkpoints retain connection metadata. Export only
    # needs training args; preserve this record as inert data, without Ray/ZMQ.
    allowed.append((Namespace, "transfer_queue.utils.zmq_utils.ZMQServerInfo"))
    allowed.extend(
        (Namespace, f"relax.utils.training.eval_config.{name}")
        for name in ("EvalDatasetConfig", "NamedPromptDataConfig")
    )
    # SFT args retain OmegaConf and sampler state. Their behavior is irrelevant
    # to tensor export; decode only their serialized attributes, without imports.
    allowed.extend(
        (Namespace, name)
        for name in (
            "omegaconf.nodes.AnyNode",
            "omegaconf.base.Metadata",
            "omegaconf.base.ContainerMetadata",
            "omegaconf.dictconfig.DictConfig",
            "omegaconf.listconfig.ListConfig",
            "relax.utils.data.identity_window_sampler.IdentityWindowSampler",
            "relax.utils.model_source.ModelSource",
            "transfer_queue.sampler.seqlen_balanced_sampler.SeqlenBalancedSampler",
        )
    )
    allowed.extend((_MetadataEnum, f"{module}.{name}") for module, names in enum_names.items() for name in names)
    with torch.serialization.safe_globals(allowed):
        # Unknown runtime objects fail closed instead of importing Megatron/CUDA.
        return torch.load(source, map_location="cpu", weights_only=True)


def read_lora_args(checkpoint: Path, reader: Any) -> dict[str, Any]:
    import torch.distributed.checkpoint as dcp

    if (checkpoint / "common.pt").is_file():
        common = _load_common(checkpoint / "common.pt")
    else:
        key = "common_state/shard_0_1"
        if key not in reader.read_metadata().state_dict_metadata:
            raise ValueError("LoRA conversion requires saved training args in common.pt or DCP common_state")

        class CommonPlanner(dcp.DefaultLoadPlanner):
            def load_bytes(self, read_item: Any, read_io: Any) -> None:
                self.state_dict[read_item.dest_index.fqn] = _load_common(read_io)

        state = {key: None}
        dcp.load(
            state,
            storage_reader=reader,
            planner=CommonPlanner(flatten_state_dict=False, flatten_sharded_tensors=False),
            no_dist=True,
        )
        common = state[key]
        if isinstance(common, io.BytesIO):
            common.seek(0)
            common = _load_common(common)
        if isinstance(common, list) and len(common) == 1:
            common = common[0]
    args = common.get("args") if isinstance(common, dict) else None
    if isinstance(args, Namespace):
        args = vars(args)
    if not isinstance(args, dict):
        raise ValueError("LoRA checkpoint has no saved training args")
    return args


def validate_lora_base(args: dict[str, Any], model: Path) -> bool:
    """Require the same base identity as native adapter-only resume."""
    marker = args.get("relax_lora_checkpoint_metadata")
    if marker is None:
        return False
    expected = verified_lora_base_identity(args, marker)
    if hf_checkpoint_identity(model) != expected:
        raise ValueError("LoRA-only checkpoint base HF identity mismatch; use the exact training base")
    for key in ("lora_rank", "lora_alpha", "tensor_model_parallel_size"):
        if key in marker and marker[key] != args.get(key):
            raise ValueError(f"LoRA-only checkpoint metadata disagrees with saved args: {key}")
    return True


@dataclass(frozen=True)
class MergeInput:
    target: TensorSlice
    base: TensorSlice | None
    linear_in: TensorSlice
    linear_out: TensorSlice
    scale: float
    # GDN B is TP-rank-major; the base's factory keys are component-major.
    section: tuple[int, int, int] | None = None

    def apply(self, state: dict[str, Any]) -> Any:
        import torch

        a = state[self.linear_in.key].to(torch.float32)
        b = state[self.linear_out.key].to(torch.float32)
        if self.section is not None:
            tp, start, length = self.section
            b = b.reshape(tp, -1, b.shape[-1])[:, start : start + length].reshape(-1, b.shape[-1])
        delta = (b @ a) * self.scale
        if tuple(delta.shape) != self.target.shape:
            delta = delta.expand(self.target.shape)
        if self.base is None:
            return delta
        base = state[self.base.key]
        return (base.float() + delta).to(base.dtype)


@dataclass(frozen=True)
class LoRAConversion:
    conversion: Conversion
    merges: tuple[MergeInput, ...]
    inputs: tuple[TensorSlice, ...]
    add_hf_base: bool
    workspace_bytes: int

    @property
    def outputs(self) -> tuple[str, ...]:
        return self.conversion.outputs

    def apply(self, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        values = {item.key: state[item.key] for item in self.conversion.inputs if item.key in state}
        for merge in self.merges:
            values[merge.target.key] = merge.apply(state)
        return self.conversion.apply(values, config)


def _factor_slice(key: str, shape: tuple[int, ...], target: TensorSlice, *, is_a: bool, gdn: bool) -> TensorSlice:
    offset = [0] * len(shape)
    size = list(shape)
    if len(shape) == 3 and shape[0] != 1:
        offset[0], size[0] = target.offset[0], target.shape[0]
    if is_a:
        offset[-1], size[-1] = target.offset[-1], target.shape[-1]
    elif not gdn:
        offset[-2], size[-2] = target.offset[-2], target.shape[-2]
    return TensorSlice(key, tuple(size), tuple(offset))


def build_lora_conversions(
    tensors: dict[str, Any],
    config: dict[str, Any],
    schema: dict[str, Any],
    args: dict[str, Any],
    *,
    adapter_only: bool,
) -> list[Conversion | LoRAConversion]:
    """Merge before the existing tensor mapping, retaining range-sized
    loads."""
    import torch

    rank, alpha = args.get("lora_rank"), args.get("lora_alpha")
    if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
        raise ValueError("Saved lora_rank must be a positive integer")
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or not math.isfinite(alpha):
        raise ValueError("Saved lora_alpha must be a finite number")
    pairs: dict[str, dict[str, str]] = {}
    base_tensors = {}
    for key, value in tensors.items():
        if ".adapter." not in key:
            base_tensors[key] = value
            continue
        for side in ("in", "out"):
            suffix = f".adapter.linear_{side}.weight"
            if key.endswith(suffix):
                pairs.setdefault(key[: -len(suffix)], {})[side] = key
                break
        else:
            raise ValueError(f"Unsupported LoRA adapter tensor: {key}")
    if adapter_only and base_tensors:
        raise ValueError("LoRA-only checkpoint unexpectedly contains base model tensors")
    names = {normalize_key(key): key for key in base_tensors}
    bindings: dict[str, tuple[str, str, float, tuple[int, int, int] | None]] = {}
    text = config.get("text_config", config)
    for prefix, pair in pairs.items():
        if set(pair) != {"in", "out"}:
            raise ValueError(f"Incomplete LoRA adapter pair: {prefix}")
        a_key, b_key = pair["in"], pair["out"]
        a, b = tuple(tensors[a_key].size), tuple(tensors[b_key].size)
        if len(a) not in (2, 3) or len(b) not in (2, 3) or a[-2] != b[-1] or a[-2] <= 0:
            raise ValueError(f"Invalid LoRA factor shapes: {prefix}: {a}, {b}")
        for key in (a_key, b_key):
            if not tensors[key].properties.dtype.is_floating_point:
                raise ValueError(f"LoRA factors must be floating point: {key}")
        expert = ".mlp.experts." in prefix
        expected_rank = rank
        if expert and args.get("normalize_moe_lora", False):
            topk = args.get("moe_router_topk", text.get("num_experts_per_tok"))
            if not isinstance(topk, int) or topk <= 0 or rank % topk:
                raise ValueError("Invalid normalized MoE LoRA rank/top-k")
            expected_rank = rank // topk
            etp = args.get("expert_tensor_parallel_size", 1) or 1
            if "linear_fc1" in prefix:
                expected_rank = math.ceil(expected_rank / etp) * etp
        if a[-2] != expected_rank:
            raise ValueError(f"LoRA rank mismatch: {prefix}: tensors={a[-2]}, expected={expected_rank}")
        leading = torch.broadcast_shapes(a[:-2], b[:-2])
        if a[:-2] != b[:-2] and not expert:
            raise ValueError(f"Only expert LoRA may share factors across a leading axis: {prefix}")
        shape = (*leading, b[-2], a[-1])
        scale = alpha / a[-2]
        targets: list[tuple[str, tuple[int, ...], tuple[int, int, int] | None]] = []
        if prefix.endswith(".self_attention.in_proj"):
            tp = args.get("tensor_model_parallel_size")
            if not isinstance(tp, int) or tp <= 0 or len(a) != 2 or len(b) != 2:
                raise ValueError("GDN LoRA requires saved tensor_model_parallel_size and 2-D factors")
            qk = text["linear_key_head_dim"] * text["linear_num_key_heads"]
            v = text["linear_value_head_dim"] * text["linear_num_value_heads"]
            sections = (qk, qk, v, v, text["linear_num_value_heads"], text["linear_num_value_heads"])
            if sum(sections) != b[-2] or any(size <= 0 or size % tp for size in sections):
                raise ValueError(f"Invalid GDN LoRA output layout for saved TP={tp}: {prefix}")
            start = 0
            for field, size in zip(("query", "key", "value", "z", "beta", "alpha"), sections):
                targets.append((prefix + ".weight." + field, (size, a[-1]), (tp, start, size // tp)))
                start += size // tp
        else:
            target = prefix + ".weight"
            if expert and (len(shape) == 2 or shape[0] == 1):
                base_key = names.get(normalize_key(target))
                experts = base_tensors[base_key].size[0] if base_key else text.get("num_experts")
                if not isinstance(experts, int) or experts <= 0:
                    raise ValueError(f"Shared expert LoRA requires the global expert count: {prefix}")
                shape = (experts, *shape[-2:])
            targets.append((target, shape, None))
        for target, shape, section in targets:
            key = names.get(normalize_key(target), target)
            if key not in base_tensors:
                if not adapter_only:
                    raise ValueError(f"LoRA base tensor is missing from full checkpoint: {target}")
                base_tensors[key] = Namespace(size=shape)
            if tuple(base_tensors[key].size) != shape:
                raise ValueError(f"LoRA/base shape mismatch: {target}: {shape} vs {tuple(base_tensors[key].size)}")
            if key in bindings:
                raise ValueError(f"Multiple adapters target the same base tensor: {target}")
            bindings[key] = (a_key, b_key, scale, section)
    jobs = build_conversions(base_tensors, config, schema)
    result: list[Conversion | LoRAConversion] = []
    used = set()
    for job in jobs:
        merges = []
        inputs = {}
        workspace = 0
        for item in job.inputs:
            if item.key not in bindings:
                inputs[item.key] = item
                continue
            used.add(item.key)
            a_key, b_key, scale, section = bindings[item.key]
            a = _factor_slice(a_key, tuple(tensors[a_key].size), item, is_a=True, gdn=section is not None)
            b = _factor_slice(b_key, tuple(tensors[b_key].size), item, is_a=False, gdn=section is not None)
            base = None if adapter_only else item
            merges.append(MergeInput(item, base, a, b, scale, section))
            for source in (a, b, base):
                if source is not None:
                    if source.key in inputs and inputs[source.key] != source:
                        raise ValueError(f"Conflicting LoRA input slices: {source.key}")
                    inputs[source.key] = source
            # Includes FP32 factor casts, GDN reorder, product, addition and
            # optional HF base reads. Preflight's storage count excludes temporaries.
            workspace += 4 * (math.prod(a.shape) + 2 * math.prod(b.shape) + 4 * math.prod(item.shape))
        if merges:
            if adapter_only and len(merges) != len(job.inputs):
                raise ValueError("A LoRA-only mapping cannot mix base tensors and adapter deltas")
            result.append(LoRAConversion(job, tuple(merges), tuple(inputs.values()), adapter_only, workspace))
        else:
            result.append(job)
    if used != bindings.keys():
        raise ValueError(f"Unmapped LoRA base tensors: {sorted(bindings.keys() - used)}")
    return result
