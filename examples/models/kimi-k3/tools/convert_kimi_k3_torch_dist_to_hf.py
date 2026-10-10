# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Distributed, offline Kimi K3 torch_dist -> native MXFP4 Hugging Face export.

Run this script as a Ray job on a cluster with a shared checkout and filesystem.
Model shards stay in host memory; each saver uses its GPU for one expert matrix
at a time. All ranks participate in Bridge mapping collectives, but only the
owner of an HF file quantizes and buffers its tensors. The original safetensors
headers provide dtype/shape information without reading original weight values.

--after-job-id waits for another Ray job to SUCCEED before acquiring GPUs.
--replace-output preserves the existing output as a timestamped backup; the new
output is published only after every shard and the index have been validated.
Accepts full checkpoints, or language LoRA checkpoints with --merge-lora.
In LoRA mode --origin-hf-dir must be the exact base used for adapter training.
Architecture-reduced checkpoints and HF/PEFT adapter files are not supported.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import os
import shutil
import socket
import struct
import sys
import time
import uuid
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    import torch


def _logger() -> Any:
    from relax.utils.logging_utils import get_logger

    return get_logger(__name__)


class SourceLayout:
    """HF file ownership and lazy header metadata; never loads source
    weights."""

    def __init__(self, origin: str | Path, rank: int, world_size: int):
        self.origin = Path(origin)
        self.rank = rank
        with (self.origin / "model.safetensors.index.json").open() as handle:
            self.index = json.load(handle)
        self.weight_map = self.index["weight_map"]
        if not self.weight_map or world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("Invalid source weight map or export rank")
        self.filenames = sorted(set(self.weight_map.values()))
        for filename in self.filenames:
            if Path(filename).name != filename:
                raise ValueError(f"Expected flat HF shard filenames, got {filename!r}")
        self.owners = {name: index % world_size for index, name in enumerate(self.filenames)}
        self.headers: dict[str, dict[str, Any]] = {}

    def owns(self, name: str) -> bool:
        return self.owners[self.weight_map[name]] == self.rank

    def spec(self, name: str) -> dict[str, Any]:
        filename = self.weight_map[name]
        if filename not in self.headers:
            with (self.origin / filename).open("rb") as handle:
                header_size = struct.unpack("<Q", handle.read(8))[0]
                if not 0 < header_size < 100_000_000:
                    raise ValueError(f"Invalid safetensors header size: {filename}")
                self.headers[filename] = json.loads(handle.read(header_size))
        return self.headers[filename][name]

    def assigned_keys(self) -> list[str]:
        return [name for name in self.weight_map if self.owns(name)]


def _dtype(name: str) -> torch.dtype:
    import torch

    return {
        "BF16": torch.bfloat16,
        "F16": torch.float16,
        "F32": torch.float32,
        "F64": torch.float64,
        "U8": torch.uint8,
        "I8": torch.int8,
        "I16": torch.int16,
        "I32": torch.int32,
        "I64": torch.int64,
        "BOOL": torch.bool,
    }[name]


def _native_tensor(layout: SourceLayout, name: str, tensor: torch.Tensor) -> torch.Tensor:
    spec = layout.spec(name)
    if list(tensor.shape) != spec["shape"]:
        raise ValueError(f"HF shape mismatch for {name}: {tuple(tensor.shape)} != {spec['shape']}")
    # Casting BF16 checkpoint values to F32 restores the source file schema,
    # not precision already lost during training.
    return tensor.detach().to(device="cpu", dtype=_dtype(spec["dtype"])).contiguous()


class NativeK3Transform:
    def __init__(self, layout: SourceLayout, device: str, quantize: Callable[..., Any]):
        self.layout = layout
        self.device = device
        self.quantize = quantize
        self.quantized = 0
        self.started = time.monotonic()

    def __call__(self, task: Any, converted: dict[str, torch.Tensor], hf_state: Any) -> dict[str, torch.Tensor]:
        import torch

        if getattr(task, "weight_dtype", None) is not None:
            raise ValueError("Native MXFP4 export must not override weight_dtype")
        result = {}
        for name, weight in converted.items():
            packed_key, scale_key = name + "_packed", name + "_scale"
            if packed_key in self.layout.weight_map:
                if scale_key not in self.layout.weight_map:
                    raise ValueError(f"MXFP4 scale is missing: {scale_key}")
                if (
                    self.layout.owners[self.layout.weight_map[packed_key]]
                    != self.layout.owners[self.layout.weight_map[scale_key]]
                ):
                    raise ValueError(f"MXFP4 pair spans different saver ranks: {name}")
                if not self.layout.owns(packed_key):
                    continue
                scale_spec = self.layout.spec(scale_key)
                if scale_spec["dtype"] != "U8":
                    raise ValueError(f"K3 native MXFP4 requires U8 E8M0 scales: {scale_key}")
                # Bridge's quantizer uses only the reference scale's shape/dtype;
                # it recomputes all scale values from the trained weight.
                source_scale = torch.empty(scale_spec["shape"], dtype=torch.uint8, device=self.device)
                packed, scale = self.quantize(weight.to(self.device), source_scale, name=name)
                result[packed_key] = _native_tensor(
                    self.layout, packed_key, packed.view(_dtype(self.layout.spec(packed_key)["dtype"]))
                )
                result[scale_key] = _native_tensor(self.layout, scale_key, scale)
                self.quantized += 1
                if self.quantized % 256 == 0:
                    _logger().info(
                        "rank=%d quantized=%d elapsed=%.1fs last=%s",
                        self.layout.rank,
                        self.quantized,
                        time.monotonic() - self.started,
                        name,
                    )
            elif self.layout.owns(name):
                if name.endswith(".self_attn.A_log"):
                    shape = self.layout.spec(name)["shape"]
                    if weight.ndim != 1 or len(shape) != 1 or weight.shape[0] > shape[0]:
                        raise ValueError(f"Cannot restore A_log padding: {name}")
                    weight = torch.cat((weight, weight.new_zeros(shape[0] - weight.shape[0])))
                result[name] = _native_tensor(self.layout, name, weight)
        return result


def _checkpoint_vision(
    input_dir: str, layout: SourceLayout, *, allow_base_fallback: bool = False
) -> Iterator[tuple[str, torch.Tensor]]:
    """Load saved vision parameters, including TP shards, directly from DCP."""
    import torch
    import torch.distributed.checkpoint as dcp

    keys = [k for k in layout.assigned_keys() if k.startswith(("vision_tower.", "mm_projector."))]
    if not keys:
        return
    reader = dcp.FileSystemReader(input_dir)
    metadata = reader.read_metadata()
    unexpected = [
        name
        for name in metadata.state_dict_metadata
        if name.startswith(("vision_tower.", "mm_projector."))
        and name not in layout.weight_map
        and hasattr(metadata.state_dict_metadata[name], "size")
    ]
    if unexpected:
        raise ValueError(f"Unmapped checkpoint vision/projector tensors: {unexpected[:5]}")
    tensors = {}
    fallback = []
    for name in keys:
        saved = metadata.state_dict_metadata.get(name)
        if saved is None or not hasattr(saved, "properties"):
            if allow_base_fallback and saved is None:
                fallback.append(name)
                continue
            raise ValueError(f"Checkpoint is missing vision tensor {name}; refusing original-weight fallback")
        if list(saved.size) != layout.spec(name)["shape"]:
            raise ValueError(f"Checkpoint vision shape mismatch: {name}")
        tensors[name] = torch.empty(tuple(saved.size), dtype=saved.properties.dtype, device="cpu")
    if tensors:
        dcp.load(tensors, storage_reader=reader, no_dist=True)
    for name in list(tensors):
        yield name, _native_tensor(layout, name, tensors.pop(name))
    if fallback:
        from safetensors import safe_open

        for name in fallback:
            with safe_open(layout.origin / layout.weight_map[name], framework="pt", device="cpu") as handle:
                yield name, _native_tensor(layout, name, handle.get_tensor(name))
    _logger().info("rank=%d exported %d vision/projector tensors from checkpoint", layout.rank, len(keys))


def _install_load_patches(input_dir: str, *, merge_lora: bool = False) -> None:
    import torch
    from megatron.bridge.training import checkpointing
    from megatron.core.dist_checkpointing.dict_utils import nested_values
    from megatron.core.dist_checkpointing.mapping import ShardedBase, ShardedTensor
    from megatron.core.dist_checkpointing.strategies import torch as mcore_torch
    from torch.distributed.checkpoint import FileSystemReader

    metadata = FileSystemReader(input_dir).read_metadata()
    keys = set(metadata.state_dict_metadata)
    if not merge_lora and any(".adapter." in name for name in keys):
        raise ValueError("This exporter requires a full checkpoint without unmerged LoRA adapters")
    original_replace = mcore_torch._replace_sharded_keys_with_state_dict_keys

    def decode_objects(state: dict[str, Any], flat_mapping: Any, rename_mapping: Any) -> Any:
        for name, value in state.items():
            if isinstance(value, io.BytesIO):
                value.seek(0)
                state[name] = torch.load(value, map_location="cpu", weights_only=False)
        return original_replace(state, flat_mapping, rename_mapping)

    mcore_torch._replace_sharded_keys_with_state_dict_keys = decode_objects
    original_generate = checkpointing._generate_model_state_dict

    def prefix_and_validate(*args: Any, **kwargs: Any) -> Any:
        state = original_generate(*args, **kwargs)
        if merge_lora:
            from kimi_k3_lora_export import prepare_overlay

            tensor_keys = {key for key, value in metadata.state_dict_metadata.items() if hasattr(value, "size")}
            return prepare_overlay(state, tensor_keys)
        missing = []
        for value in nested_values(state):
            if isinstance(value, ShardedBase) and value.key.startswith(("decoder.", "embedding.", "output_layer.")):
                value.key = "language_model." + value.key
            if isinstance(value, ShardedTensor) and value.key not in keys:
                missing.append(value.key)
        if missing:
            raise ValueError(f"Model weights missing in checkpoint ({len(missing)}): {missing[:10]}")
        return state

    checkpointing._generate_model_state_dict = prefix_and_validate


def _install_export_hooks(bridge: Any, input_dir: str, transform: NativeK3Transform) -> None:
    from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge

    model_bridge = bridge._model_bridge
    if type(model_bridge).__name__ != "KimiK3Bridge":
        raise ValueError(f"Expected KimiK3Bridge, found {type(model_bridge).__name__}")

    def modify(self: Any, task: Any, weights: Any, state: Any) -> Any:
        return transform(task, weights, state)

    def stream(self: Any, *args: Any, **kwargs: Any) -> Iterator[Any]:
        if getattr(transform, "merge_lora", False):
            if not kwargs.get("merge_adapter_weights", True):
                raise ValueError("LoRA export must merge adapter weights")
            models, pretrained = args[:2]
            if not isinstance(models, list):
                models = [models]
            tasks = kwargs.get("conversion_tasks")
            if tasks is None:
                tasks = self.build_conversion_tasks(pretrained, models)
                kwargs["conversion_tasks"] = tasks
            adapters = self.build_adapter_conversion_tasks(models)
            _validate_lora_export_tasks(tasks, adapters)
        # Call the parent to exclude K3's original-HF vision passthrough.
        # Every rank still executes every mapping collective before filtering.
        yield from MegatronModelBridge.stream_weights_megatron_to_hf(self, *args, **kwargs)
        yield from _checkpoint_vision(
            input_dir, transform.layout, allow_base_fallback=getattr(transform, "merge_lora", False)
        )

    # AutoBridge._model_bridge is a factory property, not a cached instance.
    # Patch the K3 class inside this dedicated worker so future instances use
    # the same rank-local transform and checkpoint vision reader.
    bridge_type = type(model_bridge)
    bridge_type.maybe_modify_converted_hf_weight = modify
    bridge_type.stream_weights_megatron_to_hf = stream
    fresh_bridge = bridge._model_bridge
    if fresh_bridge.maybe_modify_converted_hf_weight.__func__ is not modify:
        raise RuntimeError("Native K3 export hook did not survive Bridge recreation")
    if fresh_bridge.stream_weights_megatron_to_hf.__func__ is not stream:
        raise RuntimeError("Checkpoint vision hook did not survive Bridge recreation")


def _validate_lora_export_tasks(tasks: list[Any], adapters: dict[str, Any]) -> None:
    """Every reconstructed adapter must be paired with a base export task."""
    prefixes = {
        task.global_param_name.partition(".to_wrap.weight")[0]
        for task in tasks
        if task is not None and ".to_wrap.weight" in task.global_param_name
    }
    if not adapters or set(adapters) != prefixes:
        raise ValueError(
            f"LoRA export mapping coverage mismatch: missing={sorted(set(adapters) - prefixes)[:5]}, "
            f"unpaired={sorted(prefixes - set(adapters))[:5]}"
        )


def _validate_shards(output: Path, layout: SourceLayout) -> dict[str, int]:
    from safetensors import safe_open

    count, total_bytes = 0, 0
    assigned_files = [name for name in layout.filenames if layout.owners[name] == layout.rank]
    expected_by_file: dict[str, set[str]] = {name: set() for name in assigned_files}
    for name in layout.assigned_keys():
        expected_by_file[layout.weight_map[name]].add(name)
    for filename, expected in expected_by_file.items():
        with safe_open(output / filename, framework="pt", device="cpu") as handle:
            if set(handle.keys()) != expected:
                raise ValueError(f"Incomplete or unexpected output keys in {filename}")
            for name in expected:
                view, spec = handle.get_slice(name), layout.spec(name)
                if view.get_shape() != spec["shape"] or view.get_dtype() != spec["dtype"]:
                    raise ValueError(f"Output dtype/shape mismatch for {name}")
                total_bytes += spec["data_offsets"][1] - spec["data_offsets"][0]
                count += 1
    return {"tensors": count, "tensor_bytes": total_bytes, "shards": len(assigned_files)}


def _worker(rank: int, master: str, port: int, args: argparse.Namespace) -> dict[str, int]:
    import torch
    import torch.distributed as dist

    sys.path.insert(0, args.repo_root)
    os.environ.update(
        MASTER_ADDR=master, MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(args.world_size), LOCAL_RANK="0"
    )
    torch.set_num_threads(args.cpus_per_worker)
    torch.cuda.set_device(0)
    dist.init_process_group("gloo", rank=rank, world_size=args.world_size, timeout=timedelta(hours=24))
    try:
        merge_lora = getattr(args, "merge_lora", False)
        sys.path.insert(0, str(Path(args.repo_root) / "examples/models/kimi-k3/tools"))
        _install_load_patches(args.input_dir, merge_lora=merge_lora)
        path = Path(args.repo_root) / "scripts/tools/convert_torch_dist_to_hf_bridge.py"
        spec = importlib.util.spec_from_file_location("relax_native_k3_checkpoint_patch", path)
        patch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(patch)
        if merge_lora:
            from kimi_k3_lora_export import build_spec, configure_patch

            lora_spec = build_spec(args.input_dir, patch)
            if lora_spec is None:
                raise ValueError("--merge-lora requires a Megatron checkpoint containing LoRA adapter tensors")
            configure_patch(patch, lora_spec)
        patch._initialize_bridge_patches()
        from megatron.bridge import AutoBridge
        from megatron.bridge.models.conversion.quantization_utils import quantize_mxfp4_e2m1_like_scale

        bridge = AutoBridge.from_hf_pretrained(args.origin_hf_dir, trust_remote_code=True)
        provider = bridge.to_megatron_provider(load_weights=merge_lora)
        patch._provider_override["provider"] = provider
        layout = SourceLayout(args.origin_hf_dir, rank, args.world_size)
        transform = NativeK3Transform(layout, "cuda:0", quantize_mxfp4_e2m1_like_scale)
        transform.merge_lora = merge_lora
        _install_export_hooks(bridge, args.input_dir, transform)
        _logger().info("rank=%d loading CPU model shard", rank)
        model = bridge.load_megatron_model(
            args.input_dir,
            use_cpu_initialization=True,
            wrap_with_ddp=False,
            mp_overrides={
                "tensor_model_parallel_size": args.tp,
                "pipeline_model_parallel_size": args.pp,
                "expert_model_parallel_size": args.ep,
                "expert_tensor_parallel_size": args.expert_tp,
                "context_parallel_size": 1,
                "sequence_parallel": args.tp > 1,
                "pipeline_model_parallel_layout": None,
            },
        )
        _logger().info("rank=%d loaded; exporting assigned native MXFP4 shards", rank)
        bridge.save_hf_pretrained(
            model,
            args.staging_dir,
            strict=True,
            show_progress=False,
            merge_adapter_weights=merge_lora,
            distributed_save=True,
            save_every_n_ranks=1,
        )
        summary = _validate_shards(Path(args.staging_dir), layout)
        _logger().info("rank=%d verified output: %s", rank, summary)
        return summary
    finally:
        dist.destroy_process_group()


def _wait_for_job(job_address: str, job_id: str) -> None:
    from ray.job_submission import JobStatus, JobSubmissionClient

    client = JobSubmissionClient(job_address)
    while True:
        status = client.get_job_status(job_id)
        if status == JobStatus.SUCCEEDED:
            _logger().info("Previous job %s succeeded; starting replacement export", job_id)
            return
        if status in (JobStatus.FAILED, JobStatus.STOPPED):
            raise RuntimeError(f"Previous job {job_id} ended with {status}; replacement export was not started")
        _logger().info("Waiting for job %s (%s); no export GPU workers allocated", job_id, status)
        time.sleep(30)


def _publish(args: argparse.Namespace, summaries: list[dict[str, int]]) -> None:
    staging, output = Path(args.staging_dir), Path(args.output_dir)
    source_index = json.loads((Path(args.origin_hf_dir) / "model.safetensors.index.json").read_text())
    index = json.loads((staging / "model.safetensors.index.json").read_text())
    if index["weight_map"] != source_index["weight_map"]:
        raise ValueError("Export index does not cover exactly the original HF keys and shards")
    expected_bytes = sum(s["tensor_bytes"] for s in summaries)
    if index["metadata"]["total_size"] != expected_bytes:
        raise ValueError("Export index total_size differs from verified tensor bytes")
    shutil.copyfile(Path(args.origin_hf_dir) / "config.json", staging / "config.json")
    report = {
        "input_dir": args.input_dir,
        "origin_hf_dir": args.origin_hf_dir,
        "world_size": args.world_size,
        "parallelism": {"tp": args.tp, "pp": args.pp, "ep": args.ep, "expert_tp": args.expert_tp},
        "quantization": "native MXFP4 E2M1, E8M0 U8 scales, group size 32; GPU per saver",
        "vision_source": "checkpoint when present; otherwise exact HF base"
        if getattr(args, "merge_lora", False)
        else "training checkpoint",
        "lora_merge": getattr(args, "merge_lora", False),
        "lora_base": args.origin_hf_dir if getattr(args, "merge_lora", False) else None,
        "config": "verbatim origin config.json",
        "validation": summaries,
        "completed_at": datetime.now().isoformat(),
    }
    (staging / "relax_export_report.json").write_text(json.dumps(report, indent=2) + "\n")
    backup = None
    if output.exists():
        if not args.replace_output:
            raise FileExistsError(f"Output exists: {output}; use --replace-output to retain it as a backup")
        backup = output.with_name(output.name + f".before-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}")
        output.rename(backup)
    try:
        staging.rename(output)
    except BaseException:
        if backup is not None:
            backup.rename(output)
        raise
    _logger().info("Validated export published: %s; previous output backup: %s", output, backup)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="A single torch_dist iter_* checkpoint directory")
    parser.add_argument("--origin-hf-dir", required=True, help="Native Kimi K3 MXFP4 HF checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--merge-lora",
        action="store_true",
        help="Merge Megatron language LoRA; origin-hf-dir must be the exact training base (PP=1)",
    )
    parser.add_argument("--world-size", type=int, default=32)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=32)
    parser.add_argument("--expert-tp", type=int, default=1)
    parser.add_argument("--cpus-per-worker", type=int, default=2)
    parser.add_argument(
        "--gpus-per-node",
        type=int,
        help="Parallel exporter only: pack workers onto eligible GPU nodes, preferring the driver node when eligible",
    )
    parser.add_argument(
        "--expert-gather-backend",
        choices=("gloo", "nccl"),
        default="nccl",
        help="Parallel exporter only: routed-expert EP all-gather transport; nccl stages payloads on the worker GPU",
    )
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument("--job-address", help="Ray dashboard URL, required with --after-job-id")
    parser.add_argument("--after-job-id")
    parser.add_argument("--replace-output", action="store_true")
    args = parser.parse_args()
    args.repo_root = str(Path(__file__).resolve().parents[4])
    for name in ("input_dir", "origin_hf_dir", "output_dir"):
        setattr(args, name, str(Path(getattr(args, name)).resolve()))
    if args.after_job_id and not args.job_address:
        parser.error("--after-job-id requires --job-address")
    if any(getattr(args, k) < 1 for k in ("world_size", "tp", "pp", "ep", "expert_tp", "cpus_per_worker")):
        parser.error("Parallel sizes and CPU allocation must be positive")
    if args.world_size % (args.tp * args.pp) or args.world_size % (args.ep * args.expert_tp * args.pp):
        parser.error("world-size must be divisible by TP*PP and EP*expert-TP*PP")
    if args.merge_lora and args.pp != 1:
        parser.error("LoRA export currently requires PP=1")
    if args.output_dir in (args.input_dir, args.origin_hf_dir):
        parser.error("Output must differ from the training checkpoint and original HF checkpoint")
    if not (Path(args.input_dir) / ".metadata").is_file():
        parser.error("--input-dir must contain a torch_dist .metadata file")
    config = json.loads((Path(args.origin_hf_dir) / "config.json").read_text())
    if config.get("model_type") != "kimi_k3":
        parser.error("--origin-hf-dir must be a native kimi_k3 checkpoint")
    if config.get("text_config", {}).get("quantization_config", {}).get("format") != "mxfp4-pack-quantized":
        parser.error("Origin must use native mxfp4-pack-quantized weights")
    return args


def _eligible_nodes(ray: Any, driver_node_id: str, *, gpus: int, cpus: int) -> list[str]:
    """Prefer the driver only if it can host the requested export workers."""
    nodes = [
        node["NodeID"]
        for node in ray.nodes()
        if node.get("Alive")
        and node.get("Resources", {}).get("GPU", 0) >= gpus
        and node.get("Resources", {}).get("CPU", 0) >= cpus
    ]
    if not nodes:
        raise RuntimeError(f"No alive export node has at least {gpus} GPUs and {cpus} CPUs")
    if driver_node_id in nodes:
        nodes.remove(driver_node_id)
        nodes.insert(0, driver_node_id)
    return nodes


def _rendezvous_address() -> tuple[str, int]:
    import ray

    with socket.socket() as listener:
        listener.bind(("", 0))
        return ray.util.get_node_ip_address(), listener.getsockname()[1]


def _rendezvous_on_node(ray: Any, node_id: str) -> tuple[str, int]:
    """Choose the TCP store address on the node that will host rank zero."""
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    probe = ray.remote(num_cpus=0, max_retries=0)(_rendezvous_address)
    return ray.get(probe.options(scheduling_strategy=NodeAffinitySchedulingStrategy(node_id, soft=False)).remote())


def main() -> None:
    args = _parse_args()
    sys.path.insert(0, args.repo_root)
    if args.after_job_id:
        _wait_for_job(args.job_address, args.after_job_id)
    output = Path(args.output_dir)
    if output.exists() and not args.replace_output:
        raise FileExistsError(f"Output exists: {output}; pass --replace-output to preserve it as a backup")
    args.staging_dir = str(output.with_name(output.name + f".exporting-{uuid.uuid4().hex[:12]}"))
    import ray

    ray.init(address=args.ray_address)
    if ray.cluster_resources().get("GPU", 0) < args.world_size:
        raise RuntimeError(f"Cluster has fewer than {args.world_size} GPUs")
    node_id = _eligible_nodes(ray, ray.get_runtime_context().get_node_id(), gpus=1, cpus=args.cpus_per_worker)[0]
    master, port = _rendezvous_on_node(ray, node_id)
    worker = ray.remote(num_gpus=1, num_cpus=args.cpus_per_worker, max_retries=0)(_worker)
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    # The Gloo TCP store is hosted by rank zero at MASTER_ADDR.
    rank_zero = worker.options(scheduling_strategy=NodeAffinitySchedulingStrategy(node_id, soft=False))
    futures = [rank_zero.remote(0, master, port, args)]
    futures.extend(worker.remote(rank, master, port, args) for rank in range(1, args.world_size))
    _logger().info("Scheduled %d export workers; staging=%s", args.world_size, args.staging_dir)
    try:
        summaries = ray.get(futures)
        _publish(args, summaries)
    except BaseException:
        for future in futures:
            ray.cancel(future, force=True)
        _logger().exception("Export failed; partial files remain in %s", args.staging_dir)
        raise
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
