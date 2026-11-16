# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU tensor mappings for supported torch_dist checkpoint layouts.

Mappings handle persistent tensor names, GDN split factories and leading
expert/vision-layer axes. The HF configuration selects the architecture;
unsupported layouts fail explicitly. No model construction is needed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TensorSlice:
    key: str
    shape: tuple[int, ...]
    offset: tuple[int, ...]
    squeeze: bool = False


@dataclass(frozen=True)
class Conversion:
    inputs: tuple[TensorSlice, ...]
    outputs: tuple[str, ...]
    operation: str = "copy"

    def apply(self, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        import torch

        values = [state[item.key].squeeze(0) if item.squeeze else state[item.key] for item in self.inputs]
        text = config.get("text_config", config)
        if self.operation == "concat":
            values = [torch.cat(values, dim=0)]
        elif self.operation == "norm":
            values = [values[0] + 1]
        elif self.operation == "split_mlp":
            values = values[0].chunk(2, dim=0)
        elif self.operation in ("qkv", "vision_qkv"):
            vision = self.operation == "vision_qkv"
            cfg = config["vision_config"] if vision else text
            heads = cfg["num_heads"] if vision else cfg["num_attention_heads"]
            groups = heads if vision else cfg["num_key_value_heads"]
            dim = cfg.get("head_dim", cfg["hidden_size"] // heads)
            per_group = heads // groups
            tensor = values[0]
            tail = tensor.shape[1:]
            fused = tensor.reshape(groups, (per_group if vision else 2 * per_group) + 2, dim, *tail)
            q, k, v = fused[:, :per_group], fused[:, -2], fused[:, -1]
            if not vision:
                # HF interleaves Q and output gate per head; MCore groups them.
                q = torch.cat((q, fused[:, per_group : 2 * per_group]), dim=2)
            values = [part.reshape(-1, *tail) for part in (q, k, v)]
            if vision:
                values = [torch.cat(values, dim=0)]
        if len(values) != len(self.outputs):
            raise ValueError(f"Invalid conversion: {self}")
        return dict(zip(self.outputs, values))


def normalize_key(key: str) -> str:
    while key.startswith("module."):
        key = key[len("module.") :]
    return key.removeprefix("language_model.")


def model_key(key: str) -> bool:
    return normalize_key(key).startswith(("embedding.", "output_layer.", "decoder.", "mtp.", "vision_model."))


def build_conversions(
    tensors: dict[str, Any], config: dict[str, Any], reference: dict[str, dict[str, Any]]
) -> list[Conversion]:
    """Build a metadata-only plan; every supported model tensor is consumed."""
    standard_norm = config.get("model_type") in ("qwen3_5_moe", "qwen3_5_moe_text")
    if not standard_norm and config.get("model_type") not in ("qwen4_exp", "qwen4_exp_text"):
        raise ValueError(f"No CPU checkpoint mapping for model_type={config.get('model_type')!r}")
    names = {normalize_key(key): key for key in tensors}
    if len(names) != len(tensors):
        raise ValueError("Duplicate normalized checkpoint keys")
    prefix = "model.language_model." if "text_config" in config else "model."
    consumed: set[str] = set()
    jobs: list[Conversion] = []

    def selection(name: str) -> TensorSlice:
        key = names[name]
        shape = tuple(tensors[key].size)
        consumed.add(name)
        return TensorSlice(key, shape, (0,) * len(shape))

    def emit(item: TensorSlice, target: str, operation: str = "copy") -> None:
        jobs.append(Conversion((item,), (target,), operation))

    for name in sorted(names):
        if name in consumed:
            continue
        if ".adapter." in name:
            raise ValueError("CPU export requires a full checkpoint; LoRA adapters are not supported")
        item = selection(name)
        if name in ("embedding.word_embeddings.weight", "output_layer.weight"):
            target = prefix + "embed_tokens.weight" if name.startswith("embedding.") else "lm_head.weight"
            shape = tuple(reference[target]["shape"])
            if len(item.shape) != 2 or item.shape[0] < shape[0] or item.shape[1:] != shape[1:]:
                raise ValueError(f"Invalid padded vocabulary: {name}: {item.shape}")
            emit(TensorSlice(item.key, shape, (0, 0)), target)
            continue
        if name.startswith("vision_model."):
            match = re.fullmatch(r"vision_model\.decoder\.layers\.(?:(\d+)\.)?(.+)", name)
            if match:
                layer, suffix = match.groups()
                if layer is None:
                    if item.shape[0] != config["vision_config"]["depth"]:
                        raise ValueError(f"Invalid stacked vision layers: {name}")
                    for index in range(item.shape[0]):
                        part = TensorSlice(item.key, (1, *item.shape[1:]), (index, *item.offset[1:]), True)
                        target, op = _vision_target(suffix)
                        emit(part, f"model.visual.blocks.{index}.{target}", op)
                else:
                    target, op = _vision_target(suffix)
                    emit(item, f"model.visual.blocks.{layer}.{target}", op)
            else:
                suffix = name.removeprefix("vision_model.").replace("merger.patch_norm.", "merger.norm.")
                emit(item, "model.visual." + suffix)
            continue
        if name.startswith("decoder.final_layernorm."):
            target = "norm." if standard_norm else "hyper_connection_mixer."
            emit(item, prefix + target + name.removeprefix("decoder.final_layernorm."))
            continue
        if not standard_norm and name.startswith("decoder.hc_head_contraction."):
            suffix = name.removeprefix("decoder.hc_head_contraction.")
            emit(item, prefix + "hyper_connection_mixer." + _hc_field(suffix))
            continue
        match = re.fullmatch(r"decoder\.layers\.(\d+)\.(.+)", name)
        if match:
            layer, suffix = match.groups()
            target_prefix = f"{prefix}layers.{layer}."
        else:
            match = re.fullmatch(r"mtp\.layers\.0\.(.+)", name)
            if not match:
                raise ValueError(f"Unsupported Relax checkpoint tensor: {name}")
            suffix = match[1]
            direct = {
                "enorm.weight": "pre_fc_norm_embedding.weight",
                "hnorm.weight": "pre_fc_norm_hidden.weight",
                "e_proj.weight": "fc_embedding.weight",
                "h_proj.weight": "fc_hidden.weight",
            }
            if standard_norm:
                direct["eh_proj.weight"] = "fc.weight"
            if suffix in direct:
                emit(item, "mtp." + direct[suffix])
                continue
            if suffix.startswith("final_layernorm."):
                target = "mtp.norm." if standard_norm else "mtp.hyper_connection_mixer."
                emit(item, target + suffix.removeprefix("final_layernorm."))
                continue
            if not standard_norm and suffix.startswith("hc_head_contraction."):
                emit(item, "mtp.hyper_connection_mixer." + _hc_field(suffix.removeprefix("hc_head_contraction.")))
                continue
            inner_prefixes = (
                ("transformer_layer.", "mtp_model_layer.")
                if standard_norm
                else ("mtp_model_layer.layers.0.", "transformer_layer.")
            )
            inner = next((part for part in inner_prefixes if suffix.startswith(part)), None)
            if inner is None:
                raise ValueError(f"Unsupported Relax MTP tensor: {name}")
            suffix = suffix.removeprefix(inner)
            target_prefix = "mtp.layers.0."

        # GDN factories save components under independent persistent keys.
        for source, target in (("in_proj.weight", "in_proj_qkv.weight"), ("conv1d.weight", "conv1d.weight")):
            stem = "self_attention." + source
            if suffix.startswith(stem + "."):
                component = suffix.removeprefix(stem + ".")
                if component in ("query", "key", "value"):
                    base = name[: -len(component)]
                    inputs = tuple(selection(base + part) for part in ("query", "key", "value"))
                    jobs.append(Conversion(inputs, (target_prefix + "linear_attn." + target,), "concat"))
                elif source == "in_proj.weight" and component in ("z", "beta", "alpha"):
                    field = {"z": "z", "beta": "b", "alpha": "a"}[component]
                    emit(item, target_prefix + f"linear_attn.in_proj_{field}.weight")
                else:
                    raise ValueError(f"Unsupported GDN component: {name}")
                break
        else:
            if suffix == "per_layer_embedding.ple_embedding.ngram_embedding.weight":
                target = target_prefix + "ple.ple_embedding.ngram_embedding."
                shards = sorted(
                    (key for key in reference if key.startswith(target + "shard_")),
                    key=lambda key: int(key.split("shard_")[-1].split(".")[0]),
                )
                offset = 0
                for index, key in enumerate(shards):
                    if key != target + f"shard_{index}.weight":
                        raise ValueError("Non-contiguous PLE shard indices")
                    shape = tuple(reference[key]["shape"])
                    emit(TensorSlice(item.key, shape, (offset, 0)), key)
                    offset += shape[0]
                if not shards or item.shape[1:] != shape[1:] or offset > item.shape[0]:
                    raise ValueError(f"Invalid PLE table shape: {name}")
                # MCore pads the frozen table to ple_ngram_vocab_divisible_by.
                divisor = config.get("text_config", config).get("make_ngram_vocab_size_divisible_by", 1)
                if item.shape[0] - offset >= divisor:
                    raise ValueError(f"Unexpected PLE padding: {name}")
            elif suffix in ("mlp.experts.experts.linear_fc1.weight", "mlp.experts.experts.linear_fc2.weight"):
                fc1 = suffix.endswith("linear_fc1.weight")
                packed = target_prefix + "mlp.experts." + ("gate_up_proj" if fc1 else "down_proj")
                if packed in reference:
                    emit(item, packed)
                else:
                    # HF can store separate expert matrices. Read one global
                    # expert slice at a time, preserving DCP EP offsets.
                    if len(item.shape) != 3:
                        raise ValueError(f"Invalid global expert shape: {name}: {item.shape}")
                    for expert in range(item.shape[0]):
                        part = TensorSlice(item.key, (1, *item.shape[1:]), (expert, 0, 0), True)
                        fields = ("gate", "up") if fc1 else ("down",)
                        targets = tuple(
                            target_prefix + f"mlp.experts.{expert}.{field}_proj.weight" for field in fields
                        )
                        jobs.append(Conversion((part,), targets, "split_mlp" if fc1 else "copy"))
            elif suffix == "self_attention.linear_qkv.weight":
                jobs.append(
                    Conversion((item,), tuple(target_prefix + f"self_attn.{p}_proj.weight" for p in "qkv"), "qkv")
                )
            elif suffix == "mlp.shared_experts.linear_fc1.weight":
                jobs.append(
                    Conversion(
                        (item,),
                        tuple(target_prefix + f"mlp.shared_expert.{p}_proj.weight" for p in ("gate", "up")),
                        "split_mlp",
                    )
                )
            else:
                target, operation = _language_target(suffix)
                emit(item, target_prefix + target, operation)
    return jobs


def _hc_field(suffix: str) -> str:
    # Map bare Parameter names to HF module weight names.
    if suffix == "hc_norm_weight":
        return "hc_norm.weight"
    if suffix in ("input_mix_weight_down", "input_mix_weight_up", "block_inject_weight"):
        return suffix + ".weight"
    return suffix


def _language_target(suffix: str) -> tuple[str, str]:
    direct = {
        "pre_mlp_layernorm.weight": "post_attention_layernorm.weight",
        "self_attention.linear_qkv.layer_norm_weight": "input_layernorm.weight",
        "self_attention.in_proj.layer_norm_weight": "input_layernorm.weight",
        "mlp.router.weight": "mlp.gate.weight",
        "mlp.shared_experts.linear_fc2.weight": "mlp.shared_expert.down_proj.weight",
        "mlp.shared_experts.gate_weight": "mlp.shared_expert_gate.weight",
        "self_attention.q_layernorm.weight": "self_attn.q_norm.weight",
        "self_attention.k_layernorm.weight": "self_attn.k_norm.weight",
        "self_attention.linear_proj.weight": "self_attn.o_proj.weight",
        "self_attention.out_proj.weight": "linear_attn.out_proj.weight",
        "self_attention.A_log": "linear_attn.A_log",
        "self_attention.dt_bias": "linear_attn.dt_bias",
    }
    if suffix in direct:
        return direct[suffix], "copy"
    if suffix == "self_attention.out_norm.weight":
        return "linear_attn.norm.weight", "norm"
    if suffix.startswith("self_attention.linear_attn."):
        # Nested GDN modules already store HF-style, one-centered norms.
        return suffix.removeprefix("self_attention."), "copy"
    if suffix.startswith("self_attention_hyper_connection.ple."):
        field = suffix.removeprefix("self_attention_hyper_connection.ple.")
        field = {
            "norm_key": "norm_key.weight",
            "norm_query": "norm_query.weight",
            "norm_conv": "norm_conv.weight",
            "conv1d_weight": "conv1d.weight",
        }.get(field, field)
        return "ple." + field, "copy"
    for source, target in (
        ("self_attention.indexer.", "self_attn.indexer."),
        ("self_attention_hyper_connection.", "attn_hyper_connection."),
        ("mlp_hyper_connection.", "mlp_hyper_connection."),
        ("per_layer_embedding.", "ple."),
    ):
        if suffix.startswith(source):
            field = suffix.removeprefix(source)
            if "hyper_connection" in source:
                field = _hc_field(field)
            elif source == "self_attention.indexer." and field in ("q_layernorm", "k_layernorm"):
                field += ".weight"
            return target + field, "copy"
    raise ValueError(f"Unsupported checkpoint tensor: {suffix}")


def _vision_target(suffix: str) -> tuple[str, str]:
    for field in ("weight", "bias"):
        if suffix == f"self_attention.linear_qkv.{field}":
            return f"attn.qkv.{field}", "vision_qkv"
        for source, target in (
            ("self_attention.linear_proj.", "attn.proj."),
            ("self_attention.linear_qkv.layer_norm_", "norm1."),
            ("mlp.linear_fc1.layer_norm_", "norm2."),
            ("mlp.linear_fc1.", "mlp.linear_fc1."),
            ("mlp.linear_fc2.", "mlp.linear_fc2."),
        ):
            if suffix == source + field:
                return target + field, "copy"
    raise ValueError(f"Unsupported vision tensor: {suffix}")
