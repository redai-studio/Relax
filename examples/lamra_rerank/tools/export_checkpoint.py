#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Export a native Qwen2-VL CanonicalLoRA checkpoint and verify an FP32
merge."""

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def hf_suffix(name: str) -> str:
    layer = re.search(r"\.layers\.(\d+)\.", name)
    if layer is None:
        raise ValueError(f"Missing transformer layer in adapter parameter: {name}")
    if ".linear_qkv.adapter.adapter_" in name:
        projection = re.search(r"\.adapter_([qkv])\.", name).group(1) + "_proj"
        block = "self_attn"
    elif ".linear_fc1.adapter.adapter_" in name:
        projection = re.search(r"\.adapter_(gate|up)\.", name).group(1) + "_proj"
        block = "mlp"
    elif ".linear_proj.adapter." in name:
        projection, block = "o_proj", "self_attn"
    elif ".linear_fc2.adapter." in name:
        projection, block = "down_proj", "mlp"
    else:
        raise ValueError(f"Unsupported canonical adapter parameter: {name}")
    if ".linear_in." in name:
        side = "A"
    elif ".linear_out." in name:
        side = "B"
    else:
        raise ValueError(f"Missing adapter A/B side: {name}")
    return f".layers.{layer.group(1)}.{block}.{projection}.lora_{side}.weight"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-updates", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.adapter_output.exists():
        raise FileExistsError("Partial or completed exports require inspection before reuse")
    import torch
    import torch.distributed.checkpoint as dcp
    from megatron.core.dist_checkpointing import load_common_state_dict, load_tensors_metadata
    from peft import PeftModel
    from transformers import AutoConfig, AutoProcessor, Qwen2VLForConditionalGeneration

    from examples.lamra_rerank.tools.prepare_data import sha256
    from relax.models.qwen2_vl.configuration import checkpoint_prefixes
    from relax.utils.megatron_peft_utils import is_lora_adapter_param, write_hf_peft_adapter

    if torch.cuda.is_initialized():
        raise RuntimeError("Export must run on CPU")
    common = load_common_state_dict(str(args.checkpoint))
    saved = common["args"]
    if common["iteration"] + 1 != args.expected_updates:
        raise ValueError("Checkpoint does not contain the requested number of updates")
    if common["opt_param_scheduler"]["num_steps"] != args.expected_updates * saved.global_batch_size:
        raise ValueError("Checkpoint scheduler and optimizer update counts disagree")
    config = AutoConfig.from_pretrained(args.base)
    num_layers = getattr(config, "text_config", config).num_hidden_layers
    if config.model_type != "qwen2_vl" or saved.task_type != "causal_lm":
        raise ValueError("This exporter requires a Qwen2-VL causal-LM rank checkpoint")
    with torch.device("meta"):
        reference = Qwen2VLForConditionalGeneration(config)
    prefix, _ = checkpoint_prefixes(reference.state_dict())
    metadata = {
        key: value
        for key, value in load_tensors_metadata(str(args.checkpoint)).items()
        if is_lora_adapter_param(key) and not re.match(r"^(?:chained_\d+\.)*optimizer(?:[./]|$)", key)
    }
    if not metadata:
        raise ValueError("Native checkpoint contains no adapters")
    payload = {key: torch.empty(tuple(value.global_shape), dtype=value.dtype) for key, value in metadata.items()}
    dcp.load(payload, storage_reader=dcp.FileSystemReader(args.checkpoint), no_dist=True)
    converted = {}
    for name, tensor in payload.items():
        if tensor.ndim == 3 and tensor.shape[0] == num_layers and ".layers." in name:
            entries = [
                (name.replace(".layers.", f".layers.{index}.", 1), tensor[index]) for index in range(num_layers)
            ]
        elif tensor.ndim == 2:
            entries = [(name, tensor)]
        else:
            raise ValueError(f"Unexpected adapter shape: {name} {tuple(tensor.shape)}")
        for key, value in entries:
            target = prefix + hf_suffix(key)
            if target in converted or not bool(torch.isfinite(value).all()):
                raise ValueError(f"Duplicate or non-finite adapter: {target}")
            converted[target] = value.contiguous()
    if len(converted) != num_layers * 7 * 2:
        raise ValueError("Expected independent Q/K/V/O/gate/up/down adapters on every layer")
    write_hf_peft_adapter(
        converted,
        args.adapter_output,
        lora_rank=saved.lora_rank,
        lora_alpha=saved.lora_alpha,
        lora_dropout=saved.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    def tensor_sha(tensor: torch.Tensor) -> str:
        return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

    base, info = Qwen2VLForConditionalGeneration.from_pretrained(
        args.base, torch_dtype=torch.float32, device_map="cpu", output_loading_info=True
    )
    if any(info.get(key) for key in ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"]):
        raise ValueError(f"Incomplete base load: {info}")
    adapted = {key.rsplit(".lora_", 1)[0] + ".weight" for key in converted}
    unchanged = {key: tensor_sha(value) for key, value in base.state_dict().items() if key not in adapted}
    model = PeftModel.from_pretrained(base, args.adapter_output, is_trainable=False)
    expected = {}
    for name, layer in model.named_modules():
        if hasattr(layer, "lora_A") and "default" in layer.lora_A:
            expected[name.removeprefix("base_model.model.") + ".weight"] = (
                (
                    layer.base_layer.weight
                    + layer.lora_B["default"].weight @ layer.lora_A["default"].weight * layer.scaling["default"]
                )
                .detach()
                .clone()
            )
    if set(expected) != adapted:
        raise ValueError("HF adapter pairing does not cover all trained projections")
    merged = model.merge_and_unload(safe_merge=True).float()
    state = merged.state_dict()
    if any(not torch.equal(state[key], value) for key, value in expected.items()):
        raise ValueError("Merged parameters do not satisfy the FP32 LoRA equation")
    if any(tensor_sha(state[key]) != digest for key, digest in unchanged.items()):
        raise ValueError("A frozen base parameter changed during merge")
    merged.save_pretrained(args.output, safe_serialization=True, max_shard_size="10GB", save_original_format=False)
    AutoProcessor.from_pretrained(args.base, use_fast=False).save_pretrained(args.output)
    reloaded, info = Qwen2VLForConditionalGeneration.from_pretrained(
        args.output, torch_dtype=torch.float32, device_map="cpu", output_loading_info=True
    )
    if any(info.get(key) for key in ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"]):
        raise ValueError(f"Incomplete merged reload: {info}")
    if any(not torch.equal(value, state[key]) for key, value in reloaded.state_dict().items()):
        raise ValueError("Merged HF reload differs from the verified parameters")
    proof = {
        "status": "passed",
        "checkpoint": str(args.checkpoint),
        "updates": args.expected_updates,
        "adapter_tensors": len(converted),
        "adapter_parameters": sum(math.prod(tensor.shape) for tensor in converted.values()),
        "merged_projections": len(expected),
        "frozen_parameters_unchanged": True,
        "FP32_merge_equation_exact": True,
        "HF_reload_exact": True,
        "CUDA_initialized": torch.cuda.is_initialized(),
        "model_files_sha256": {path.name: sha256(path) for path in args.output.glob("*.safetensors")},
        "adapter_sha256": sha256(args.adapter_output / "adapter_model.safetensors"),
    }
    (args.output / "merge_verification.json").write_text(json.dumps(proof, indent=2))
    print(json.dumps(proof), flush=True)


if __name__ == "__main__":
    main()
