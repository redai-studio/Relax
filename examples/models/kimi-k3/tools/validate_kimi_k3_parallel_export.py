# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Compare locally quantized checkpoint experts with an existing native export.

Reads only selected DCP storage chunks; never constructs the full model. The
reference must come from the same checkpoint. A single GPU is sufficient.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def load_expert(checkpoint: Path, metadata: Any, layer: int, expert: int, fc: int) -> Any:
    import torch

    key = f"language_model.decoder.layers.{layer}.mlp.experts.experts.linear_fc{fc}.weight"
    spec = metadata.state_dict_metadata[key]
    output = torch.empty(tuple(spec.size[1:]), dtype=spec.properties.dtype)
    covered = 0
    # Each storage entry is one torch.save payload, not an entire distcp file.
    for chunk in spec.chunks:
        if chunk.offsets[0] != expert:
            continue
        if chunk.sizes[0] != 1:
            raise ValueError("Probe expects one expert per saved DCP chunk")
        matches = [
            info
            for index, info in metadata.storage_data.items()
            if index.fqn == key and tuple(index.offset) == tuple(chunk.offsets)
        ]
        if len(matches) != 1 or matches[0].transform_descriptors:
            raise ValueError(f"Unsupported DCP storage for {key}, {chunk.offsets}")
        info = matches[0]
        with (checkpoint / info.relative_path).open("rb") as handle:
            handle.seek(info.offset)
            raw = handle.read(info.length)
        tensor = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        tensor = tensor.reshape(tuple(chunk.sizes[1:]))
        row, col = chunk.offsets[1:]
        rows, cols = chunk.sizes[1:]
        output[row : row + rows, col : col + cols].copy_(tensor)
        covered += rows * cols
    if covered != output.numel():
        raise ValueError(f"Incomplete expert coverage for {key}: {covered}/{output.numel()}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--reference-hf-dir", required=True)
    parser.add_argument("--origin-hf-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[1])
    parser.add_argument("--experts", type=int, nargs="+", default=[0, 895])
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

    import convert_kimi_k3_torch_dist_to_hf as baseline
    import convert_kimi_k3_torch_dist_to_hf_parallel as parallel
    import torch
    import torch.distributed as dist
    from megatron.bridge.models.conversion.param_mapping import GatedMLPMapping, RowParallelMapping
    from megatron.bridge.models.conversion.quantization_utils import quantize_mxfp4_e2m1_like_scale
    from safetensors import safe_open
    from torch.distributed.checkpoint import FileSystemReader

    from relax.utils.logging_utils import get_logger

    logger = get_logger(__name__)
    torch.set_num_threads(2)
    checkpoint = Path(args.checkpoint_dir)
    metadata = FileSystemReader(checkpoint).read_metadata()
    layout = baseline.SourceLayout(args.origin_hf_dir, 0, 1)
    transform = baseline.NativeK3Transform(layout, args.device, quantize_mxfp4_e2m1_like_scale)
    quantizer = parallel.LocalExpertQuantizer(args.device, quantize_mxfp4_e2m1_like_scale)
    cases = []
    with tempfile.TemporaryDirectory() as temporary:
        dist.init_process_group("gloo", init_method="file://" + temporary + "/gloo", rank=0, world_size=1)
        try:
            for layer in args.layers:
                for expert in args.experts:
                    for fc in (1, 2):
                        weight = load_expert(checkpoint, metadata, layer, expert, fc)
                        prefix = f"language_model.model.layers.{layer}.block_sparse_moe.experts.{expert}"
                        megatron = f"decoder.layers.{layer}.mlp.experts.linear_fc{fc}.weight{expert}"
                        mapping = (
                            GatedMLPMapping(megatron, prefix + ".w1.weight", prefix + ".w3.weight")
                            if fc == 1
                            else RowParallelMapping(megatron, prefix + ".w2.weight")
                        )
                        module = SimpleNamespace(config=SimpleNamespace(num_moe_experts=896))
                        started = time.monotonic()
                        reference = transform(
                            SimpleNamespace(weight_dtype=None), mapping.megatron_to_hf(weight, module), None
                        )
                        reference_seconds = time.monotonic() - started
                        started = time.monotonic()
                        converted = parallel.convert_expert(mapping, weight, module, quantizer, layout)
                        actual = transform(SimpleNamespace(weight_dtype=None), converted, None)
                        parallel_seconds = time.monotonic() - started
                        if actual.keys() != reference.keys():
                            raise ValueError("Output key mismatch")
                        for key, tensor in actual.items():
                            if not torch.equal(tensor, reference[key]):
                                raise ValueError(f"Local quantization changed bytes: {key}")
                            filename = layout.weight_map[key]
                            with safe_open(
                                Path(args.reference_hf_dir) / filename, framework="pt", device="cpu"
                            ) as handle:
                                gold = handle.get_tensor(key)
                                if tensor.dtype != gold.dtype or not torch.equal(tensor, gold):
                                    raise ValueError(f"Mismatch against baseline checkpoint export: {key}")
                        case = {
                            "layer": layer,
                            "expert": expert,
                            "fc": fc,
                            "matched_keys": len(actual),
                            "reference_seconds": reference_seconds,
                            "parallel_seconds": parallel_seconds,
                        }
                        cases.append(case)
                        logger.info("EXPERT_PARITY_PASSED %s", case)
        finally:
            dist.destroy_process_group()
    report = {
        "status": "passed",
        "checkpoint": str(checkpoint),
        "reference": args.reference_hf_dir,
        "device": args.device,
        "cases": cases,
        "matched_keys": sum(c["matched_keys"] for c in cases),
        "timing_note": "Single-rank correctness probe; timings are not a distributed throughput benchmark",
    }
    if args.device.startswith("cuda"):
        report["peak_allocated_gpu_bytes"] = torch.cuda.max_memory_allocated(args.device)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    logger.info("PARALLEL_QUANTIZATION_VALIDATION_PASSED report=%s", args.report)


if __name__ == "__main__":
    main()
