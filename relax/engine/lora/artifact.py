# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Versioned execution semantics and producer provenance for plain dense
LoRA."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CONTRACT = "relax.dense-lora.v1"
PROVENANCE = "producer_manifest.json"
_DTYPES = {"F16": 2, "BF16": 2, "F32": 4}
_MODULES = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
# These defaults do not change inference. All other PEFT options are rejected.
_DEFAULT_ONLY = {
    "auto_mapping": None,
    "use_rslora": False,
    "use_dora": False,
    "lora_bias": False,
    "fan_in_fan_out": False,
    "rank_pattern": {},
    "alpha_pattern": {},
    "modules_to_save": None,
    "target_parameters": None,
    "layers_to_transform": None,
    "layers_pattern": None,
    "layer_replication": None,
    "trainable_token_indices": None,
    "exclude_modules": None,
    "megatron_config": None,
    "loftq_config": {},
    "eva_config": None,
    "corda_config": None,
    "qalora_group_size": 16,
    "use_qalora": False,
    "arrow_config": None,
    "ensure_weight_tying": False,
}
_METADATA_FIELDS = {"base_model_name_or_path", "revision", "peft_version", "inference_mode"}


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def read_object(path: Path, *, limit: int = 1024 * 1024) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError(f"invalid or oversized manifest: {path.name}")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must be a JSON object")
    return value


@dataclass(frozen=True)
class ModelContract:
    base_digest: str
    config: dict[str, Any]
    max_rank: int
    target_modules: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.config.get("model_type") not in {"qwen2", "qwen3", "llama"}:
            raise ValueError("contract v1 supports dense Qwen2/Qwen3/Llama text models only")
        architecture = {"qwen2": "Qwen2ForCausalLM", "qwen3": "Qwen3ForCausalLM", "llama": "LlamaForCausalLM"}
        if self.config.get("architectures", [architecture[self.config["model_type"]]]) != [
            architecture[self.config["model_type"]]
        ]:
            raise ValueError("custom model architectures are outside contract v1")
        if (
            self.config.get("num_experts")
            or self.config.get("num_local_experts")
            or self.config.get("quantization_config")
        ):
            raise ValueError("contract v1 excludes MoE and quantized base models")
        if self.max_rank <= 0 or not set(self.target_modules) <= _MODULES or not self.target_modules:
            raise ValueError("explicit supported target modules and a positive maximum rank are required")
        for name in ("hidden_size", "num_attention_heads", "num_hidden_layers", "intermediate_size"):
            if type(self.config.get(name)) is not int or self.config[name] <= 0:
                raise ValueError(f"invalid base model dimension: {name}")

        for name in ("head_dim", "num_key_value_heads"):
            if name in self.config and (type(self.config[name]) is not int or self.config[name] <= 0):
                raise ValueError(f"invalid base model dimension: {name}")
        if "head_dim" not in self.config and self.config["hidden_size"] % self.config["num_attention_heads"]:
            raise ValueError("hidden_size is not divisible by attention heads")

    @property
    def config_digest(self) -> str:
        return hashlib.sha256(canonical(self.config)).hexdigest()

    def validate(self, directory: Path, *, require_provenance: bool = True) -> dict[str, Any]:
        cfg = read_object(directory / "adapter_config.json")
        supported = {
            "r",
            "lora_alpha",
            "target_modules",
            "lora_dropout",
            "bias",
            "peft_type",
            "task_type",
            "init_lora_weights",
        }
        unknown = set(cfg) - supported - _METADATA_FIELDS - set(_DEFAULT_ONLY)
        if unknown:
            raise ValueError(f"unsupported adapter config fields: {sorted(unknown)}")
        for name, default in _DEFAULT_ONLY.items():
            if name in cfg and cfg[name] != default:
                raise ValueError(f"unsupported adapter semantics: {name}={cfg[name]!r}")
        if cfg.get("bias", "none") != "none" or cfg.get("peft_type") != "LORA" or cfg.get("task_type") != "CAUSAL_LM":
            raise ValueError("only bias-free causal-LM LoRA is supported")
        # PiSSA/OLoRA/LoftQ may modify the base: do not accept initialization
        # metadata whose source/base relationship this contract cannot prove.
        if cfg.get("init_lora_weights", True) not in (True, False, "gaussian"):
            raise ValueError("unsupported base-modifying LoRA initialization")
        rank = cfg.get("r")
        alpha = cfg.get("lora_alpha")
        if type(rank) is not int or not 0 < rank <= self.max_rank:
            raise ValueError("adapter rank exceeds the engine contract")
        if type(alpha) not in (int, float) or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("lora_alpha must be positive and finite")
        dropout = cfg.get("lora_dropout", 0.0)
        if type(dropout) not in (int, float) or not 0 <= dropout < 1:
            raise ValueError("invalid lora_dropout")
        modules = cfg.get("target_modules")
        if not isinstance(modules, list) or not modules or any(not isinstance(m, str) for m in modules):
            raise ValueError("target_modules must be an explicit nonempty list")
        if len(set(modules)) != len(modules) or not set(modules) <= set(self.target_modules):
            raise ValueError("adapter target modules are outside the engine contract")
        shapes = self._shapes(rank, modules)
        self._validate_tensors(directory / "adapter_model.safetensors", shapes)
        if require_provenance:
            source = read_object(directory / PROVENANCE)
            if (
                source.get("contract") != CONTRACT
                or source.get("base_model_digest") != self.base_digest
                or source.get("base_config_digest") != self.config_digest
                or not isinstance(source.get("producer"), str)
                or not source["producer"]
                or "export_step" not in source
                or (
                    source["export_step"] is not None
                    and (type(source["export_step"]) is not int or source["export_step"] < 0)
                )
            ):
                raise ValueError("adapter producer provenance does not match the target base/format")
            from relax.engine.lora.snapshot import _file_digest

            if source.get("weights_digest") != _file_digest(directory / "adapter_model.safetensors"):
                raise ValueError("producer manifest does not describe these weights")
            if source.get("adapter_config_digest") != hashlib.sha256(canonical(cfg)).hexdigest():
                raise ValueError("producer manifest does not describe this adapter configuration")
        return {"contract": CONTRACT, "rank": rank, "scaling": alpha / rank, "target_modules": modules}

    def _shapes(self, rank: int, modules: list[str]) -> dict[str, list[int]]:
        cfg = self.config
        hidden, intermediate = cfg["hidden_size"], cfg["intermediate_size"]
        heads = cfg["num_attention_heads"]
        dim = cfg.get("head_dim", hidden // heads)
        kv = cfg.get("num_key_value_heads", heads) * dim
        widths = {
            "q_proj": (hidden, heads * dim),
            "k_proj": (hidden, kv),
            "v_proj": (hidden, kv),
            "o_proj": (heads * dim, hidden),
            "gate_proj": (hidden, intermediate),
            "up_proj": (hidden, intermediate),
            "down_proj": (intermediate, hidden),
        }
        result = {}
        for layer in range(cfg["num_hidden_layers"]):
            for module in modules:
                parent = "self_attn" if module in {"q_proj", "k_proj", "v_proj", "o_proj"} else "mlp"
                prefix = f"base_model.model.model.layers.{layer}.{parent}.{module}"
                input_width, output_width = widths[module]
                result[f"{prefix}.lora_A.weight"] = [rank, input_width]
                result[f"{prefix}.lora_B.weight"] = [output_width, rank]
        return result

    @staticmethod
    def _validate_tensors(path: Path, expected: dict[str, list[int]]) -> None:
        if path.is_symlink() or not path.is_file():
            raise ValueError("adapter weights must be a regular safetensors file")
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError("invalid safetensors header")
            header_size = struct.unpack("<Q", prefix)[0]
            if not 2 <= header_size <= 16 * 1024 * 1024:
                raise ValueError("invalid safetensors header size")
            header = json.loads(stream.read(header_size))
        if not isinstance(header, dict):
            raise ValueError("invalid tensor index")
        header.pop("__metadata__", None)
        if set(header) != set(expected):
            raise ValueError("adapter tensors do not exactly match the supported target modules/layers")
        segments = []
        dtypes = set()
        for name, shape in expected.items():
            tensor = header[name]
            dtype = tensor.get("dtype")
            if dtype not in _DTYPES or tensor.get("shape") != shape:
                raise ValueError(f"unsupported tensor shape/dtype: {name}")
            offsets = tensor.get("data_offsets")
            if not isinstance(offsets, list) or len(offsets) != 2 or any(type(i) is not int for i in offsets):
                raise ValueError(f"invalid tensor offsets: {name}")
            start, end = offsets
            if start < 0 or end - start != math.prod(shape) * _DTYPES[dtype]:
                raise ValueError(f"invalid tensor byte length: {name}")
            dtypes.add(dtype)
            segments.append((start, end))
        if len(dtypes) != 1:
            raise ValueError("mixed adapter tensor dtypes are not supported")
        position = 0
        for start, end in sorted(segments):
            if start != position:
                raise ValueError("overlapping or missing safetensors bytes")
            position = end
        if 8 + header_size + position != path.stat().st_size:
            raise ValueError("safetensors length does not match tensor index")


def provenance(contract: ModelContract, directory: Path, *, producer: str, export_step: int | None) -> dict[str, Any]:
    """Called by the producer before handing off a sealed export, never by the
    target."""
    contract.validate(directory, require_provenance=False)
    from relax.engine.lora.snapshot import _file_digest

    return {
        "weights_digest": _file_digest(directory / "adapter_model.safetensors"),
        "contract": CONTRACT,
        "producer": producer,
        "export_step": export_step,
        "base_model_digest": contract.base_digest,
        "base_config_digest": contract.config_digest,
        "adapter_config_digest": hashlib.sha256(canonical(read_object(directory / "adapter_config.json"))).hexdigest(),
    }
