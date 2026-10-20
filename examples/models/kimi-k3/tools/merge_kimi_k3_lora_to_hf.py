# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Merge an adapter-only K3 torch_dist checkpoint into native HF shards on one
Ray node.

Never materializes the full BF16 model. Each worker holds only its adapter
subset and one output shard; untouched source tensors are copied exactly.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any


def _worker(rank: int, args: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    sys.path.insert(0, args["tools_dir"])
    sys.path.insert(0, args["repo_root"])
    import resource

    import torch
    import torch.distributed.checkpoint as dcp
    from convert_kimi_k3_torch_dist_to_hf import SourceLayout, _native_tensor, _validate_shards
    from kimi_k3_streaming_lora import merge_weight
    from megatron.bridge.models.conversion.quantization_utils import (
        dequantize_mxfp4_e2m1_packed,
        quantize_mxfp4_e2m1_like_scale,
    )
    from safetensors import safe_open
    from safetensors.torch import save_file

    from relax.utils.logging_utils import get_logger

    logger = get_logger(__name__)
    torch.set_num_threads(args["cpus_per_worker"])
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    layout = SourceLayout(args["origin_hf_dir"], rank, args["workers"])
    files = [name for name in layout.filenames if layout.owners[name] == rank]
    mappings = spec["mappings"]
    selected = {
        name: entry
        for name, entry in mappings.items()
        if layout.owns(name + "_packed" if name + "_packed" in layout.weight_map else name)
    }
    needed = {entry[key] for entry in selected.values() for key in ("a_key", "b_key")}
    reader = dcp.FileSystemReader(args["input_dir"])
    metadata = reader.read_metadata()
    adapters = {
        key: torch.empty(
            metadata.state_dict_metadata[key].size, dtype=metadata.state_dict_metadata[key].properties.dtype
        )
        for key in needed
    }
    logger.info(
        "STREAM_LOAD rank=%d adapter_tensors=%d adapter_bytes=%d",
        rank,
        len(adapters),
        sum(t.numel() * t.element_size() for t in adapters.values()),
    )
    if adapters:
        dcp.load(adapters, storage_reader=reader, no_dist=True)
    started = time.monotonic()
    changed = 0
    for filename in files:
        output = {}
        with safe_open(Path(args["origin_hf_dir"]) / filename, framework="pt", device="cpu") as source:
            for key in source.keys():
                if key in output:
                    continue
                logical = key.removesuffix("_packed") if key.endswith("_packed") else key
                if key.endswith("_scale") and key.removesuffix("_scale") in selected:
                    continue
                if logical not in selected:
                    output[key] = source.get_tensor(key)
                    continue
                entry = selected[logical]
                base = source.get_tensor(key).to("cuda")
                if key.endswith("_packed"):
                    scale_key = logical + "_scale"
                    if layout.weight_map[scale_key] != filename:
                        raise ValueError(f"Packed weight and scale must share a shard: {logical}")
                    scale = source.get_tensor(scale_key).to("cuda")
                    base = dequantize_mxfp4_e2m1_packed(base, scale)
                merged = merge_weight(
                    base, adapters[entry["a_key"]], adapters[entry["b_key"]], entry, spec["alpha"], spec["rank"]
                )
                if key.endswith("_packed"):
                    packed, scale = quantize_mxfp4_e2m1_like_scale(merged, scale)
                    packed_dtype = source.get_tensor(key).dtype
                    output[key] = _native_tensor(layout, key, packed.view(packed_dtype))
                    output[scale_key] = _native_tensor(layout, scale_key, scale)
                else:
                    output[key] = _native_tensor(layout, key, merged)
                changed += 1
                del base, merged
        expected = {key for key, file in layout.weight_map.items() if file == filename}
        if set(output) != expected:
            raise ValueError(f"Output shard keys differ: {filename}")
        target = Path(args["staging_dir"]) / filename
        temporary = target.with_suffix(".safetensors.tmp")
        save_file(output, str(temporary), metadata={"format": "pt"})
        temporary.replace(target)
        del output
        progress = {
            "rank": rank,
            "last_shard": filename,
            "completed_shards": files.index(filename) + 1,
            "total_shards": len(files),
            "merged_weights": changed,
            "elapsed_seconds": time.monotonic() - started,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        }
        (Path(args["staging_dir"]) / f"stream_rank_{rank:02d}.json").write_text(json.dumps(progress, indent=2) + "\n")
        logger.info("STREAM_PROGRESS %s", progress)
    if changed != len(selected):
        raise ValueError(f"Adapter mapping coverage mismatch: {changed} != {len(selected)}")
    return {
        **_validate_shards(Path(args["staging_dir"]), layout),
        "merged_weights": changed,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--origin-hf-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cpus-per-worker", type=int, default=4)
    parser.add_argument("--node-id", help="Pin all workers to this Ray node; default is the driver node")
    parser.add_argument("--ray-address", default="auto")
    args = vars(parser.parse_args())
    args["tools_dir"] = str(Path(__file__).resolve().parent)
    args["repo_root"] = str(Path(__file__).resolve().parents[4])
    sys.path.insert(0, args["repo_root"])
    from kimi_k3_streaming_lora import read_checkpoint_spec

    from relax.utils.logging_utils import get_logger

    logger = get_logger(__name__)
    for key in ("input_dir", "origin_hf_dir", "output_dir"):
        args[key] = str(Path(args[key]).resolve())
    if args["workers"] < 1 or args["cpus_per_worker"] < 1:
        parser.error("Worker and CPU counts must be positive")
    output = Path(args["output_dir"])
    if output.exists():
        raise FileExistsError(output)
    spec = read_checkpoint_spec(args["input_dir"], args["origin_hf_dir"])
    args["staging_dir"] = str(output.with_name(output.name + ".exporting-" + uuid.uuid4().hex[:12]))
    staging = Path(args["staging_dir"])
    staging.mkdir()
    (staging / "streaming_lora_plan.json").write_text(json.dumps({"args": args, "spec": spec}, indent=2) + "\n")
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    ray.init(address=args["ray_address"])
    node_id = args["node_id"] or ray.get_runtime_context().get_node_id()
    node = next(n for n in ray.nodes() if n["NodeID"] == node_id and n["Alive"])
    if node["Resources"].get("GPU", 0) < args["workers"]:
        raise ValueError("Selected node has fewer GPUs than workers")
    started = time.monotonic()
    worker = ray.remote(num_gpus=1, num_cpus=args["cpus_per_worker"], max_retries=0)(_worker)
    futures = [
        worker.options(scheduling_strategy=NodeAffinitySchedulingStrategy(node_id, soft=False)).remote(i, args, spec)
        for i in range(args["workers"])
    ]
    logger.info("STREAM_STARTED node=%s workers=%d staging=%s", node["NodeManagerAddress"], args["workers"], staging)
    try:
        summaries = ray.get(futures)
        for source in Path(args["origin_hf_dir"]).iterdir():
            if source.is_file() and not source.name.endswith(".safetensors") and not source.name.startswith("."):
                shutil.copy2(source, staging / source.name)
        report = {
            "input_dir": args["input_dir"],
            "origin_hf_dir": args["origin_hf_dir"],
            "strategy": "single-node HF shard streaming; FP32 LoRA merge then native MXFP4 quantization",
            "node": node["NodeManagerAddress"],
            "workers": args["workers"],
            "validation": summaries,
            "elapsed_seconds": time.monotonic() - started,
            "lora_rank": spec["rank"],
            "lora_alpha": spec["alpha"],
        }
        index = json.loads((staging / "model.safetensors.index.json").read_text())
        if sum(s["tensor_bytes"] for s in summaries) != index["metadata"]["total_size"]:
            raise ValueError("Output index payload size mismatch")
        if sum(s["merged_weights"] for s in summaries) != len(spec["mappings"]):
            raise ValueError("Incomplete LoRA merge coverage")
        (staging / "relax_export_report.json").write_text(json.dumps(report, indent=2) + "\n")
        staging.rename(output)
        logger.info("STREAM_EXPORT_PUBLISHED %s report=%s", output, report)
    except BaseException:
        for future in futures:
            ray.cancel(future, force=True)
        raise
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
