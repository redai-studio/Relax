# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Check a frozen workload and record the exact CLI before a training run."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import torch

from examples.flops_balancing.prepare_workload import sha256_file


def hash_model_weight_shards(checkpoint: Path) -> dict[str, str]:
    index_path = checkpoint / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    filenames = sorted(set(index["weight_map"].values()))
    if not filenames or any(not (checkpoint / filename).is_file() for filename in filenames):
        raise ValueError("All pinned model weight shards must be downloaded before GPU training")
    return {
        index_path.name: sha256_file(index_path),
        **{filename: sha256_file(checkpoint / filename) for filename in filenames},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=list("ABCD"), required=True)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    manifest = json.loads((args.workload / "manifest.json").read_text())
    if sha256_file(args.workload / "train.jsonl") != manifest["train_sha256"]:
        raise ValueError("Frozen dataset changed")
    for filename, digest in manifest["tokenizer_files"].items():
        if sha256_file(args.checkpoint / filename) != digest:
            raise ValueError(f"Model configuration/tokenizer changed: {filename}")
    weight_shards = None
    if not args.dry_run:
        weight_shards = hash_model_weight_shards(args.checkpoint)
    if not args.command or args.command[0] != "--":
        parser.error("Separate the recorded training command with --")
    command = args.command[1:]
    invariant_command = [arg for arg in command if arg not in ("--flops-microbatch-packing", "--flops-dp-balancing")]
    versions = {}
    for package in ("torch", "transformers", "ray", "TransferQueue", "megatron-core", "transformer-engine"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    root = Path(__file__).resolve().parents[2]
    source = hashlib.sha256()
    for path in sorted(
        [
            *root.joinpath("relax").rglob("*.py"),
            *root.joinpath("scripts/models").glob("*.sh"),
            *root.joinpath("scripts/training/sft").glob("*flops-task16.sh"),
            *root.joinpath("examples/flops_balancing").glob("*.py"),
        ]
    ):
        source.update(str(path.relative_to(root)).encode())
        source.update(bytes.fromhex(sha256_file(path)))
    native_root = Path(os.environ.get("MEGATRON", "/root/Megatron-LM")) / "megatron"
    native_files = sorted(native_root.rglob("*.py")) if native_root.is_dir() else []
    native_digest = hashlib.sha256()
    for path in native_files:
        native_digest.update(str(path.relative_to(native_root)).encode())
        native_digest.update(bytes.fromhex(sha256_file(path)))
    tq_spec = importlib.util.find_spec("transfer_queue")
    tq_root = Path(tq_spec.origin).parent if tq_spec and tq_spec.origin else None
    tq_files = sorted(tq_root.rglob("*.py")) if tq_root else []
    tq_digest = hashlib.sha256()
    for path in tq_files:
        tq_digest.update(str(path.relative_to(tq_root)).encode())
        tq_digest.update(bytes.fromhex(sha256_file(path)))
    if not args.dry_run and (not native_files or not tq_files):
        raise ValueError("Actual training requires identifiable Megatron and TransferQueue sources")
    result = dict(
        mode=args.mode,
        capture=args.capture,
        dry_run=args.dry_run,
        command=command,
        invariant_command=invariant_command,
        versions=versions,
        launch_environment={
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NCCL_ALGO",
                "NCCL_PROTO",
                "CUDA_DEVICE_MAX_CONNECTIONS",
                "CUDA_MODULE_LOADING",
                "PYTORCH_CUDA_ALLOC_CONF",
                "CUBLAS_WORKSPACE_CONFIG",
                "NVTE_ALLOW_NONDETERMINISTIC_ALGO",
                "RELAX_SFT_RESIDENT_PARAM_OVERLAP",
            )
        },
        source_sha256=source.hexdigest(),
        workload_sha256=sha256_file(args.workload / "manifest.json"),
        model_revision=manifest["model_revision"],
        model_weight_sha256=weight_shards,
        data_revision=manifest["source_revision"],
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        native_source_sha256=native_digest.hexdigest() if native_files else None,
        transfer_queue_source_sha256=tq_digest.hexdigest() if tq_files else None,
        devices=[
            {
                "name": torch.cuda.get_device_name(index),
                "memory_bytes": torch.cuda.get_device_properties(index).total_memory,
                "capability": list(torch.cuda.get_device_capability(index)),
            }
            for index in range(torch.cuda.device_count())
        ],
    )
    args.run_dir.mkdir(parents=True, exist_ok=False)
    (args.run_dir / "run.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
