# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Reconstruct gathered source parameters before the existing Bridge
converter."""

import math
import os
from pathlib import Path

from relax.distributed.weight_sync.checkpoint import snapshot_tensors
from relax.distributed.weight_sync.source_map import TensorMapping
from relax.distributed.weight_sync.source_patch import file_digest
from relax.distributed.weight_sync.store import digest


class BridgeSourceAdapter:
    """Keep Bridge conversion separate from lossless source-shard replacement.

    Cached inputs use the layout produced by the existing TP gather. Bridge
    still performs every model-specific transformation, including expert and
    visual mappings. Only the publication owner retains gathered inputs.
    """

    def __init__(self, iterator):
        if iterator.lora_merge_mode or iterator.lora_adapter_mode or iterator.quantization_config:
            raise ValueError("Bridge source replacement requires unquantized, non-LoRA weights")
        self.iterator = iterator
        self.args, self.model_name = iterator.args, iterator.model_name
        self.megatron_local_param_info_buckets = iterator._expert_buckets + iterator._non_expert_buckets
        iterator._bridge_converter.init_tasks()
        iterator._bridge_converter.broadcast_and_apply_configs()

    def normalize_weights(self, weights):
        return {name: weights[key] for name, key in self.iterator._vanilla_key_map.items() if key in weights}

    def source_mapping(self, info, tp_size):
        name = info.name.replace(".to_wrap.", ".")
        # These specialized gather kernels rearrange several components.
        # Their full route remains available until registered independently.
        if any(part in name for part in ("self_attention.conv1d.", "self_attention.in_proj.", "expert_bias")):
            return None
        sharded = bool(info.attrs.get("tensor_model_parallel")) and info.attrs.get("parallel_mode") != "duplicated"
        dimension = info.attrs.get("partition_dim") if sharded else None
        if "linear_fc2.weight" in name and "vision_model" not in name and dimension == 0:
            dimension = 1
        gated = "linear_fc1.weight" in name and "vision_model" not in name
        if not gated and info.attrs.get("partition_stride", 1) != 1:
            return None
        full_shape = list(info.shape)
        if dimension is not None:
            full_shape[dimension] *= tp_size
        return TensorMapping(
            shape=tuple(info.shape),
            tp_size=tp_size,
            partition_dim=dimension,
            outputs=((info.name, tuple(full_shape)),),
            kind="gather_gate_up" if gated else "identity",
        )

    def _convert_to_hf_named_tensors(self, params, infos):
        # AdaptiveSourceExporter caches the gathered source, then calls
        # finish_group on the owner for both complete and patched inputs.
        return [(info.name, param) for info, param in zip(infos, params, strict=True)]

    snapshot_hashes = True

    def prepare_resident_group(self, info, tensors, directory: Path):
        """Convert the just-gathered CPU value without a disk round trip.

        Keep a private clone because a converter may mutate its input. This
        protects both frozen rank tensors and the canonical cache write view.
        Only one parameter group is resident, as in the previous file clone.
        """
        destination = directory / "logical"
        destination.mkdir(exist_ok=True)
        values = dict(tensors)
        if len(values) != 1 or info.name not in values:
            raise ValueError("Bridge resident group differs from gathered source")
        tensor = values[info.name]
        if tensor.device.type != "cpu":
            raise ValueError("Bridge resident group must already be on CPU")
        return {"tensor": tensor.clone(), "destination": destination}

    def prepare_group(self, info, table, directory: Path, route, previous):
        import torch

        destination = directory / "logical"
        destination.mkdir(exist_ok=True)
        reusable = route == "REUSE" and previous and info.name in previous["logical_groups"]
        if reusable:
            try:
                reusable = all(
                    file_digest(previous["logical"][name][2]) == previous["logical_hashes"][name]
                    for name in previous["logical_groups"][info.name]
                )
            except OSError:
                reusable = False
        if reusable:
            exported = {}
            for name in previous["logical_groups"][info.name]:
                dtype, shape, source = previous["logical"][name]
                target = destination / (digest(name.encode()) + ".bin")
                os.link(source, target)
                exported[name] = (dtype, shape, target)
            return {
                "reused": exported,
                "destination": destination,
                "hashes": {name: previous["logical_hashes"][name] for name in exported},
            }
        dtype, shape, source = table[info.name]
        names = {
            "BF16": "bfloat16",
            "F16": "float16",
            "F32": "float32",
            "F64": "float64",
            "I64": "int64",
            "I32": "int32",
            "I16": "int16",
            "I8": "int8",
            "U8": "uint8",
            "BOOL": "bool",
        }
        tensor = (
            torch.from_file(str(source), shared=False, size=math.prod(shape), dtype=getattr(torch, names[dtype]))
            .reshape(shape)
            .clone()
        )
        return {"tensor": tensor, "destination": destination}

    def convert_group(self, info, prepared):
        """Run Bridge on the training thread with an already resident input."""
        if "reused" in prepared:
            return None
        converted = self.iterator._bridge_converter.convert(info.name, prepared["tensor"])
        return [(name, tensor.detach().contiguous().cpu()) for name, tensor in converted]

    def write_group(self, prepared, converted, *, hashes=None, hash_executor=None):
        if "reused" in prepared:
            if hashes is not None:
                hashes.update(prepared["hashes"])
            return prepared["reused"]
        return snapshot_tensors(converted, prepared["destination"], hashes=hashes, hash_executor=hash_executor)

    def finish_group(self, info, table, directory: Path, route, previous):
        prepared = self.prepare_group(info, table, directory, route, previous)
        return self.write_group(prepared, self.convert_group(info, prepared))
