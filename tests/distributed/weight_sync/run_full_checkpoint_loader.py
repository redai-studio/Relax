# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Start the existing SGLang Loader from a reconstructed FULL checkpoint."""

import argparse
import json
from pathlib import Path


def run(store_path, stream, version, output, tokenizer_path, max_weight_bytes, disk_budget, probe):
    import sglang as sgl

    from relax.distributed.weight_sync.checkpoint import validate_checkpoint, write_checkpoint
    from relax.distributed.weight_sync.store import VersionStore

    store = VersionStore(store_path, disk_budget=disk_budget, max_weight_bytes=max_weight_bytes)
    checkpoint = write_checkpoint(store, stream, version, output / "checkpoint")
    manifest = store.manifest(stream, version)
    validate_checkpoint(checkpoint, manifest)
    engine = sgl.Engine(
        model_path=str(checkpoint),
        tokenizer_path=str(tokenizer_path),
        dtype="bfloat16",
        disable_cuda_graph=True,
        disable_overlap_schedule=True,
        attention_backend="torch_native",
        skip_tokenizer_init=True,
        mm_feature_transport="cpu",
        disable_fast_image_processor=True,
        max_total_tokens=512,
        mem_fraction_static=0.2,
        context_length=256,
        log_level="warning",
    )
    try:
        result = engine.generate(**probe)
        assert result["meta_info"]["completion_tokens"] > 0, result
        (output / "result.json").write_text(
            json.dumps(
                {
                    "passed": True,
                    "stream": stream,
                    "version": version,
                    "root": manifest["root"],
                    "logical_tensors": len(manifest["tensors"]),
                    "generation": result,
                },
                indent=2,
            )
        )
    finally:
        engine.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--stream", required=True)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--max-weight-bytes", type=int, required=True)
    parser.add_argument("--disk-budget", type=int, required=True)
    parser.add_argument("--probe", type=Path, help="JSON kwargs for Engine.generate")
    options = parser.parse_args()
    probe = (
        json.loads(options.probe.read_text())
        if options.probe
        else {
            "input_ids": [1, 2, 3],
            "sampling_params": {"temperature": 0, "max_new_tokens": 2},
        }
    )
    run(
        options.store,
        options.stream,
        options.version,
        options.output,
        options.tokenizer_path,
        options.max_weight_bytes,
        options.disk_budget,
        probe,
    )
