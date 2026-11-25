# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Plan native LoRA merges without importing Megatron or constructing a
model."""

from __future__ import annotations

import io
import math
import re
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
        "relax.core.registry": (
            "ROLES",
            "ROLES_TRAIN_ONLY",
            "ROLES_COLOCATE",
            "ROLES_SFT_ONLY",
            "ROLES_FULLY_ASYNC_ON_POLICY",
            "ROLES_PPO_COLOCATE",
            "ROLES_PPO_FULLY_ASYNC",
            "ROLES_PPO_FULLY_ASYNC_ON_POLICY",
        ),
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
            "relax.utils.data.wholepack.WholepackSampler",
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
    expected = verified_lora_base_identity(args, marker, model)
    if hf_checkpoint_identity(model, version=marker["format_version"]) != expected:
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


def _pipeline_local_layers(args: dict[str, Any], config: dict[str, Any]) -> list[int]:
    """Translate global decoder indices to the indices used by
    chunk.named_parameters()."""
    pp = args.get("pipeline_model_parallel_size", 1)
    if pp == 1:
        return []
    total = args.get("num_layers") or config.get("text_config", config).get("num_hidden_layers")
    if not isinstance(total, int) or total <= 0:
        raise ValueError("Missing decoder layer count in LoRA checkpoint pipeline layout")
    vp = args.get("virtual_pipeline_model_parallel_size") or 1
    layout = args.get("pipeline_model_parallel_layout")
    if layout and layout != "None":  # Native LoRA markers stringify an unset layout.
        if isinstance(layout, str):
            # Megatron's CLI layout grammar, e.g. Et*12|(tt|)*2t*8mL.
            layout = layout.replace(",", "").replace(" ", "")
            for pattern in (r"\(([^)]+)\)\*(\d+)", r"([EtmL|])\*(\d+)"):
                layout = re.sub(pattern, lambda m: m[1] * int(m[2]), layout)
            counts = [stage.count("t") for stage in layout.split("|")]
        else:
            counts = [
                sum(layer == "decoder" or getattr(layer, "value", None) == 4 for layer in stage) for stage in layout
            ]
    else:
        first, last = args.get("decoder_first_pipeline_num_layers"), args.get("decoder_last_pipeline_num_layers")
        embedding = int(bool(args.get("account_for_embedding_in_pipeline_split")))
        loss = int(bool(args.get("account_for_loss_in_pipeline_split")))
        per_stage = [None] * pp
        per_stage[0], per_stage[-1] = first, last
        remaining = total + embedding + loss - (first or 0) - (last or 0)
        middle = per_stage.count(None)
        counts = [(n if n is not None else remaining // middle) // vp for n in per_stage] * vp
        counts[0] -= embedding
        counts[-1] -= loss
    if min(counts) < 0 or sum(counts) != total:
        raise ValueError("Cannot verify trained LoRA adapters against the saved pipeline layout")
    return [index for count in counts for index in range(count)]


def _validate_trained_factors(
    tensors: dict[str, Any], jobs: list[Conversion | LoRAConversion], schema: dict, config: dict, args: dict
) -> None:
    marker = args["relax_lora_checkpoint_metadata"]
    declared = marker.get("trainable_parameter_names")
    if not isinstance(declared, (list, tuple)) or not declared or not all(isinstance(key, str) for key in declared):
        raise ValueError("Missing trained LoRA parameter names in checkpoint metadata")

    def factor_name(key: str) -> str:
        name = normalize_key(key).replace(".mlp.experts.experts.", ".mlp.experts.")
        return re.sub(r"^mtp\.layers\.0\.mtp_model_layer\.(?:layers\.0\.)?", "mtp.layers.0.transformer_layer.", name)

    required = set()
    min_local_experts: dict[str, int] = {}
    for key in declared:
        name = factor_name(key)
        if match := re.fullmatch(r"(.+\.mlp\.experts\.)local_experts\.(\d+)\.(.+)", name):
            name = match[1] + match[3]
            min_local_experts[name] = max(min_local_experts.get(name, 0), int(match[2]) + 1)
        required.add(name)

    local_layers = _pipeline_local_layers(args | marker, config)
    num_experts = config.get("text_config", config).get("num_experts") or args.get("num_moe_experts")
    ep = marker.get("expert_model_parallel_size", args.get("expert_model_parallel_size", 1))
    found = set()
    for key, metadata in tensors.items():
        name = factor_name(key)
        if name.startswith("vision_model.decoder.layers.") and not name.split(".")[3].isdigit():
            found.update(name.replace("decoder.layers.", f"decoder.layers.{i}.", 1) for i in range(metadata.size[0]))
            continue
        if local_layers and (match := re.fullmatch(r"decoder\.layers\.(\d+)\.(.+)", name)):
            name = f"decoder.layers.{local_layers[int(match[1])]}.{match[2]}"
        if name in min_local_experts:
            # SequentialMLP saves every EP rank's local matrices on one global DCP expert axis.
            if len(metadata.size) != 3 or metadata.size[0] != num_experts:
                raise ValueError(f"Incomplete sequential expert axis for trained LoRA tensor: {key}")
            if not isinstance(ep, int) or ep <= 0 or num_experts % ep or min_local_experts[name] > num_experts // ep:
                raise ValueError(f"Invalid local expert index or EP size for trained LoRA tensor: {key}")
        found.add(name)
    missing = required - found
    if local_layers:
        # Different PP stages can declare the same local name. Check each HF layer
        # that actually owns that projection (e.g. full attention vs GDN).
        hf_layer = re.compile(r"model\.(?:language_model\.)?layers\.(\d+)\.(.+)")

        def hf_family(suffix: str) -> str:
            # HF may pack experts in one layer and store individual projections in another.
            expert = re.fullmatch(r"mlp\.experts\.\d+\.(gate|up|down)_proj\.weight", suffix)
            if expert:
                return "mlp.experts.down_proj" if expert[1] == "down" else "mlp.experts.gate_up_proj"
            return suffix

        merged = set()
        sources = {}
        for job in jobs:
            if not isinstance(job, LoRAConversion):
                continue
            for output in job.outputs:
                if match := hf_layer.fullmatch(output):
                    merged.add(output)
                    source = factor_name(job.merges[0].linear_in.key)
                    sources[hf_family(match[2])] = source.split(".", 3)[3].removesuffix(".adapter.linear_in.weight")
        for output in schema.keys() - merged:
            if (match := hf_layer.fullmatch(output)) and (source := sources.get(hf_family(match[2]))):
                prefix = f"decoder.layers.{local_layers[int(match[1])]}.{source}.adapter."
                if any(prefix + side in required for side in ("linear_in.weight", "linear_out.weight")):
                    missing.add(output)
    if missing:
        raise ValueError(f"Missing trained LoRA tensors in checkpoint: {sorted(missing)[:8]}")


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
        leading = a[:-2]
        if expert:
            base_key = names.get(normalize_key(prefix + ".weight"))
            experts = text.get("num_experts") or args.get("num_moe_experts")
            if experts is None and base_key is not None:
                experts = base_tensors[base_key].size[0]
            if not isinstance(experts, int) or experts <= 0:
                raise ValueError(f"Missing global expert count for LoRA tensor: {prefix}")
            leading = (experts,)
            # Native grouped adapters save both global axes. A compact factor
            # needs a saved sharing mode: its shape alone cannot establish one.
            grouped = args.get("moe_grouped_gemm") is not False
            shared_outer = grouped and args.get("experts_shared_outer_loras") is True
            legacy_shared = (
                grouped
                and not adapter_only
                and not shared_outer
                and args.get("share_expert_adapters") is True
                and len(a) == len(b) == 2
            )
            for key, factor, outer in ((a_key, a, ".linear_fc1"), (b_key, b, ".linear_fc2")):
                if factor[:-2] == leading:
                    continue
                can_share = legacy_shared or (shared_outer and prefix.endswith(outer))
                if factor[:-2] in ((), (1,)) and (experts == 1 or can_share):
                    continue
                raise ValueError(f"Incomplete expert axis for LoRA tensor {key}: expected {experts}, got {factor}")
        elif a[:-2] != b[:-2]:
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
    if adapter_only:
        _validate_trained_factors(tensors, result, schema, config, args)
    return result
