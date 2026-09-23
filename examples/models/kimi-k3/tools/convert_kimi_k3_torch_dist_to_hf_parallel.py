# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Native K3 export with expert quantization on every EP rank before gathering.

Accepts the baseline exporter's CLI. Requires PP=1, expert-TP=1 and EP=world
size. Non-expert mappings, source checkpoint loading, native schema validation
and atomic publication reuse the baseline exporter. Install patches only inside
this export's dedicated workers; never modify the installed Megatron package.
--gpus-per-node packs workers onto whole nodes with the driver node first, so
rank 0 keeps hosting the Gloo store. --expert-gather-backend nccl stages the
quantized expert payloads on the worker GPU for an NCCL all-gather instead of
CPU Gloo; gathered values stay bit-identical.
"""

from __future__ import annotations

import argparse
import json
import re
import socket
import sys
import time
import uuid
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable


if TYPE_CHECKING:
    import torch


def _logger() -> Any:
    from relax.utils.logging_utils import get_logger

    return get_logger(__name__)


def _expert_names(mapping: Any) -> list[str]:
    if not mapping.is_expert or getattr(mapping, "is_adapter", False):
        return []
    names = list(mapping.hf_param.values()) if isinstance(mapping.hf_param, dict) else [mapping.hf_param]
    if not all(
        re.fullmatch(r"language_model\.model\.layers\.\d+\.block_sparse_moe\.experts\.\d+\.w[123]\.weight", n)
        for n in names
    ):
        return []
    return names


class LocalExpertQuantizer:
    """Keep Gloo payloads on CPU and reuse the exact reference MXFP4
    arithmetic."""

    def __init__(self, device: str, quantize: Callable[..., Any]):
        self.device = device
        self.quantize = quantize
        self.calls = 0
        self.seconds = 0.0
        self.input_bytes = 0
        self.output_bytes = 0

    def __call__(self, weight: torch.Tensor, block_size: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        import torch

        if block_size != (1, 32) or weight.ndim != 2 or weight.shape[1] % 32:
            raise ValueError(f"Unsupported local MXFP4 geometry: {tuple(weight.shape)}, {block_size}")
        if weight.dtype != torch.bfloat16 or weight.device.type != "cpu":
            raise ValueError("Parallel export requires plain BF16 CPU expert weights")
        started = time.monotonic()
        scale_template = torch.empty((weight.shape[0], weight.shape[1] // 32), dtype=torch.uint8, device=self.device)
        packed, scale = self.quantize(weight.to(self.device), scale_template)
        if packed.dtype != torch.int8 or tuple(packed.shape) != (weight.shape[0], weight.shape[1] // 2):
            raise ValueError("Invalid packed MXFP4 output from quantizer")
        if scale.dtype != torch.uint8 or tuple(scale.shape) != tuple(scale_template.shape):
            raise ValueError("Invalid E8M0 scale output from quantizer")
        packed, scale = packed.cpu().contiguous(), scale.cpu().contiguous()
        self.seconds += time.monotonic() - started
        self.calls += 1
        self.input_bytes += weight.numel() * weight.element_size()
        self.output_bytes += packed.numel() * packed.element_size() + scale.numel() * scale.element_size()
        return packed, scale


def convert_expert(
    mapping: Any,
    weight: torch.Tensor,
    module: Any,
    quantizer: LocalExpertQuantizer,
    layout: Any,
) -> dict[str, torch.Tensor]:
    """Use Bridge's local-quantization mapping, then restore native key
    names."""
    if mapping.tp_size != 1 or mapping.pp_size != 1:
        raise ValueError("Local K3 expert quantization requires expert-TP=1 and PP=1")
    names = _expert_names(mapping)
    if not names:
        raise ValueError(f"Not a supported routed expert: {mapping.hf_param}")
    for name in names:
        if name + "_packed" not in layout.weight_map or name + "_scale" not in layout.weight_map:
            raise ValueError(f"Missing native MXFP4 pair: {name}")
        if layout.spec(name + "_scale")["dtype"] != "U8":
            raise ValueError(f"Expected native E8M0 scales: {name}")
        if layout.owners[layout.weight_map[name + "_packed"]] != layout.owners[layout.weight_map[name + "_scale"]]:
            raise ValueError(f"MXFP4 pair spans different owners: {name}")
    converted = mapping.megatron_to_hf_quant(weight, module, lambda _: True, quantizer, (1, 32))
    weights = {name for name in converted if not name.endswith("_scale_inv")}
    scales = {name.removesuffix("_scale_inv") for name in converted if name.endswith("_scale_inv")}
    if weights != scales:
        raise ValueError("Quantized mapping returned unpaired weights/scales")
    result = {}
    for name, tensor in converted.items():
        key = name.removesuffix("_scale_inv") + "_scale" if name.endswith("_scale_inv") else name + "_packed"
        if key not in layout.weight_map:
            raise ValueError(f"Unexpected quantized expert key: {key}")
        result[key] = tensor
    return result


class NcclExpertGather:
    """All-gather quantized expert payloads over NCCL on the worker GPU.

    Bridge's expert path all-gathers CPU tensors over Gloo TCP, which dominates
    export wall time. Each payload is staged on the worker's single GPU for the
    collective and copied back to CPU, so gathered values stay bit-identical
    while the transport uses NVLink/IB. Group creation is itself a collective:
    every rank must construct this class once, in the same order.
    """

    def __init__(self) -> None:
        import torch
        import torch.distributed as dist

        if not torch.cuda.is_available():
            raise RuntimeError("NCCL expert gather requires CUDA")
        self._dist = dist
        self.group = dist.new_group(backend="nccl", timeout=timedelta(hours=24))
        self.calls = 0
        self.bytes = 0

    def all_gather(self, tensor: Any) -> list[Any]:
        import torch

        if tensor.device.type != "cpu":
            raise ValueError("Expert gather expects the quantized payload on CPU")
        device = torch.device("cuda", torch.cuda.current_device())
        local = tensor.to(device)
        world = self._dist.get_world_size(group=self.group)
        gathered = [torch.empty_like(local) for _ in range(world)]
        self._dist.all_gather(gathered, local, group=self.group)
        chunks = [chunk.cpu() for chunk in gathered]
        self.calls += 1
        self.bytes += local.numel() * local.element_size() * world
        return chunks


def _ep_gather_on_device(
    mapping: Any, megatron_weights: Any, megatron_module: Any, hf_param_name: Any, gather: NcclExpertGather
) -> dict[str, Any]:
    """Mirror MegatronParamMapping.gather_from_ep_ranks with the payload
    gathered by ``gather``.

    Name reconstruction, grouping and squeezing follow Bridge's implementation
    exactly; only the transport differs. Requires EP=world-size, which the
    parallel export enforces.
    """
    import torch
    import torch.distributed as dist
    from megatron.bridge.utils.common_utils import extract_expert_number_from_param

    if mapping.ep_size == 1:
        return {str(hf_param_name): megatron_weights}
    if mapping.ep_size != dist.get_world_size():
        raise ValueError("On-device expert gather requires EP=world-size")
    if megatron_module is None:
        num_experts_per_rank = mapping.broadcast_obj_from_pp_rank(None, "num_experts_per_rank")
    else:
        model_config = mapping._get_config(megatron_module)
        num_experts_per_rank = model_config.num_moe_experts // mapping.ep_size
        num_experts_per_rank = mapping.broadcast_obj_from_pp_rank(num_experts_per_rank, "num_experts_per_rank")
    global_expert_number = extract_expert_number_from_param(mapping.megatron_param)
    local_expert_number = global_expert_number % num_experts_per_rank
    gathered_expert_param_names = [
        re.sub(r"experts\.(\d+)", f"experts.{int(local_expert_number) + num_experts_per_rank * i}", str(hf_param_name))
        for i in range(mapping.ep_size)
    ]
    if str(hf_param_name) not in gathered_expert_param_names:
        raise ValueError(f"HF param {hf_param_name} missing from gathered expert names")
    chunks = gather.all_gather(megatron_weights)
    weights_dict = {}
    for i, param_name in enumerate(gathered_expert_param_names):
        if param_name in weights_dict:
            weights_dict[param_name] = torch.cat([weights_dict[param_name], chunks[i].unsqueeze(0)], dim=0)
        else:
            weights_dict[param_name] = chunks[i].unsqueeze(0)
    for param_name in weights_dict:
        weights_dict[param_name] = weights_dict[param_name].squeeze(0)
    return weights_dict


def install_parallel_mappings(
    layout: Any, quantizer: LocalExpertQuantizer, gather: NcclExpertGather | None = None
) -> Callable[[], None]:
    """Patch only the two routed expert mapping classes, within a worker.

    With ``gather``, the routed-expert EP all-gather runs over the on-device
    backend instead of CPU Gloo; name handling stays byte-identical to Bridge.
    """
    from megatron.bridge.models.conversion.param_mapping import GatedMLPMapping, RowParallelMapping

    originals: dict[type, dict[str, Any]] = {}
    for cls in (GatedMLPMapping, RowParallelMapping):
        cls_originals = {"megatron_to_hf": cls.megatron_to_hf}
        original = cls_originals["megatron_to_hf"]

        def convert(self: Any, weight: Any, module: Any, _original: Any = original) -> Any:
            if _expert_names(self):
                return convert_expert(self, weight, module, quantizer, layout)
            return _original(self, weight, module)

        cls.megatron_to_hf = convert
        if gather is not None:

            def gather_ep(self: Any, megatron_weights: Any, megatron_module: Any, hf_param_name: Any) -> Any:
                return _ep_gather_on_device(self, megatron_weights, megatron_module, hf_param_name, gather)

            cls_originals["gather_from_ep_ranks"] = cls.gather_from_ep_ranks
            cls_originals["gather_from_ep_ranks_scale"] = cls.gather_from_ep_ranks_scale
            cls.gather_from_ep_ranks = gather_ep
            cls.gather_from_ep_ranks_scale = gather_ep
        originals[cls] = cls_originals

    def restore() -> None:
        for cls, cls_originals in originals.items():
            for name, cls_original in cls_originals.items():
                setattr(cls, name, cls_original)

    return restore


def _worker(rank: int, master: str, port: int, args: argparse.Namespace) -> dict[str, int]:
    sys.path.insert(0, str(Path(args.repo_root) / "examples/models/kimi-k3/tools"))
    import convert_kimi_k3_torch_dist_to_hf as baseline
    from convert_kimi_k3_torch_dist_to_hf_parallel import LocalExpertQuantizer, install_parallel_mappings
    from megatron.bridge.models.conversion.quantization_utils import quantize_mxfp4_e2m1_like_scale

    original_hooks = baseline._install_export_hooks
    quantizer = LocalExpertQuantizer("cuda:0", quantize_mxfp4_e2m1_like_scale)
    restore = None
    gather = None

    def install(bridge: Any, input_dir: str, transform: Any) -> None:
        nonlocal restore, gather
        original_hooks(bridge, input_dir, transform)
        # Bridge merges LoRA after mapping. Quantizing inside mapping would
        # therefore attempt to add BF16 deltas to packed MXFP4 tensors.
        if not getattr(args, "merge_lora", False):
            if getattr(args, "expert_gather_backend", "gloo") == "nccl":
                gather = NcclExpertGather()
                _logger().info("Routed-expert EP all-gather backend: nccl")
            restore = install_parallel_mappings(transform.layout, quantizer, gather)
        else:
            _logger().info("LoRA export: BF16 mapping and merge precede owner-rank MXFP4 quantization")

    baseline._install_export_hooks = install
    started = time.monotonic()
    try:
        summary = baseline._worker(rank, master, port, args)
        if quantizer.calls == 0 and not getattr(args, "merge_lora", False):
            raise RuntimeError("Parallel expert quantization was never invoked")
        stats = {
            "rank": rank,
            "local_quantization_calls": quantizer.calls,
            "quantization_seconds": quantizer.seconds,
            "input_bytes": quantizer.input_bytes,
            "output_bytes": quantizer.output_bytes,
            "gather_calls": gather.calls if gather is not None else 0,
            "gathered_bytes": gather.bytes if gather is not None else 0,
            "worker_seconds": time.monotonic() - started,
        }
        (Path(args.staging_dir) / f"parallel_export_rank_{rank:03d}.json").write_text(
            json.dumps(stats, indent=2) + "\n"
        )
        _logger().info("PARALLEL_EXPORT_WORKER_DONE %s", stats)
        return summary
    finally:
        baseline._install_export_hooks = original_hooks
        if restore is not None:
            restore()


def _pin_nodes(ray: Any, driver_node_id: str, args: argparse.Namespace) -> list[str | None]:
    """Return the target node id per rank; None leaves a rank to ordinary Ray
    scheduling.

    With --gpus-per-node, ranks are packed onto whole nodes, driver node first,
    so rank 0 keeps hosting the Gloo store. Node GPU totals decide eligibility;
    free capacity is still resolved by Ray at scheduling time, and workers stay
    pending when a chosen node runs out.
    """
    per_node = args.gpus_per_node
    if per_node is None:
        return [driver_node_id] + [None] * (args.world_size - 1)
    candidates = [
        node["NodeID"]
        for node in ray.nodes()
        if node.get("Alive") and node.get("Resources", {}).get("GPU", 0.0) >= per_node
    ]
    if driver_node_id not in candidates:
        raise RuntimeError(
            f"Driver node reports fewer than {per_node} GPUs; rank 0 hosts the Gloo store and must stay there"
        )
    candidates.remove(driver_node_id)
    chunks = args.world_size // per_node
    if len(candidates) < chunks - 1:
        raise RuntimeError(
            f"Need {chunks} alive nodes with at least {per_node} GPUs each; found {len(candidates) + 1} eligible"
        )
    nodes = [driver_node_id] + candidates[: chunks - 1]
    return [nodes[rank // per_node] for rank in range(args.world_size)]


def main() -> None:
    import convert_kimi_k3_torch_dist_to_hf as baseline

    args = baseline._parse_args()
    if args.pp != 1 or args.expert_tp != 1 or args.ep != args.world_size:
        raise ValueError("Parallel export requires PP=1, expert-TP=1 and EP=world-size")
    if args.gpus_per_node is not None and (args.gpus_per_node < 1 or args.world_size % args.gpus_per_node):
        raise ValueError("--gpus-per-node must be positive and divide --world-size")
    sys.path.insert(0, args.repo_root)
    if args.after_job_id:
        baseline._wait_for_job(args.job_address, args.after_job_id)
    output = Path(args.output_dir)
    if output.exists() and not args.replace_output:
        raise FileExistsError(f"Output exists: {output}")
    args.staging_dir = str(output.with_name(output.name + f".exporting-{uuid.uuid4().hex[:12]}"))
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    ray.init(address=args.ray_address)
    if ray.cluster_resources().get("GPU", 0) < args.world_size:
        raise RuntimeError(f"Cluster has fewer than {args.world_size} GPUs")
    with socket.socket() as listener:
        listener.bind(("", 0))
        port = listener.getsockname()[1]
    worker = ray.remote(num_gpus=1, num_cpus=args.cpus_per_worker, max_retries=0)(_worker)
    master = ray.util.get_node_ip_address()
    nodes = _pin_nodes(ray, ray.get_runtime_context().get_node_id(), args)
    futures = []
    for rank, node_id in enumerate(nodes):
        task = (
            worker
            if node_id is None
            else worker.options(scheduling_strategy=NodeAffinitySchedulingStrategy(node_id, soft=False))
        )
        futures.append(task.remote(rank, master, port, args))
    _logger().info("Scheduled %d parallel export workers; staging=%s", args.world_size, args.staging_dir)
    started = time.monotonic()
    try:
        summaries = ray.get(futures)
        stats = [
            json.loads((Path(args.staging_dir) / f"parallel_export_rank_{rank:03d}.json").read_text())
            for rank in range(args.world_size)
        ]
        # Save optimization provenance before publication; baseline report describes the schema.
        (Path(args.staging_dir) / "parallel_export_report.json").write_text(
            json.dumps(
                {
                    "strategy": "BF16 gather, LoRA merge, owner-rank MXFP4 quantization"
                    if args.merge_lora
                    else "local BF16 expert MXFP4 quantization before EP all-gather",
                    "expert_gather_backend": getattr(args, "expert_gather_backend", "gloo"),
                    "export_seconds": time.monotonic() - started,
                    "expert_tp": args.expert_tp,
                    "workers": stats,
                },
                indent=2,
            )
            + "\n"
        )
        baseline._publish(args, summaries)
        report_path = output / "relax_export_report.json"
        report = json.loads(report_path.read_text())
        if not args.merge_lora:
            report["quantization"] = (
                "native MXFP4 E2M1, E8M0 U8 scales, group size 32; GPU per EP rank before all-gather"
            )
        report["optimization_report"] = "parallel_export_report.json"
        temporary = output / "relax_export_report.json.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(report_path)
    except BaseException:
        for future in futures:
            ray.cancel(future, force=True)
        _logger().exception("Parallel export failed; partial files remain in %s", args.staging_dir)
        raise
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
