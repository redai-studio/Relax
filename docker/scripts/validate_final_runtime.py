# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Validate the final image after every install, including the NCCL
override."""

import argparse
import hashlib
import importlib.metadata as md
import json
import os
import subprocess
import sys
from pathlib import Path

from install_legacy_fla import SOURCE_SHA256, source_hash
from record_qwen35_runtime import validate_pip_check


def validate_cudnn_runtime(expected_version: str) -> dict[str, int]:
    """Check the libraries used by Torch and TE in both import orders."""
    from packaging.version import Version

    major, minor, patch, *_ = Version(expected_version).release
    runtime_version = major * 10000 + minor * 100 + patch
    expected = {"torch": runtime_version, "transformer_engine": runtime_version}
    probe = """
import json
import sys
from importlib import import_module

import_module(sys.argv[1])
import torch
import transformer_engine.pytorch
import transformer_engine_torch

print(json.dumps({
    "torch": torch.backends.cudnn.version(),
    "transformer_engine": transformer_engine_torch.get_cudnn_version(),
}))
"""
    for first_import in ("torch", "transformer_engine.pytorch"):
        result = subprocess.check_output([sys.executable, "-c", probe, first_import], text=True)
        runtimes = json.loads(result.splitlines()[-1])
        if runtimes != expected:
            raise ValueError(f"cuDNN {expected_version} expected after importing {first_import} first, got {runtimes}")
    return expected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("cu12", "cu13"), default="cu13")
    parser.add_argument("--cudnn-version", help="Expected cuDNN release for the CUDA 12 image")
    args = parser.parse_args()
    cudnn_runtime = None
    expected = {"ray": "2.56.1", "numpy": "2.2.6", "nvidia-nccl-cu13": "2.30.7", "fla-core": "0.5.2"}
    if args.profile == "cu12":
        expected.pop("numpy")
        expected.pop("nvidia-nccl-cu13")
    for package, version in expected.items():
        if md.version(package) != version:
            raise ValueError(f"{package}: expected {version}, got {md.version(package)}")
    import torch
    from flash_mla import flash_mla_sparse_fwd

    if not callable(flash_mla_sparse_fwd):
        raise ValueError("FlashMLA is missing the sparse forward API required by cuDNN DSA")

    if not torch.version.cuda or not torch.version.cuda.startswith("12." if args.profile == "cu12" else "13."):
        raise ValueError(f"CUDA runtime does not match {args.profile}: {torch.version.cuda}")
    expected["torch"] = md.version("torch")
    expected["transformer-engine"] = md.version("transformer-engine")
    hybrid = md.version("relax-hybrid-ep")
    if "+" not in hybrid or not os.environ["RELAX_HYBRID_EP_COMMIT"].startswith(hybrid.split("+", 1)[1]):
        raise ValueError(f"HybridEP revision mismatch: {hybrid}")
    expected["relax-hybrid-ep"] = hybrid
    check = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    if args.profile == "cu13":
        validate_pip_check(check)
    elif check.returncode:
        raise ValueError(check.stdout + check.stderr)
    if args.profile == "cu12":
        from packaging.version import Version

        if Version(md.version("numpy")) >= Version("2"):
            raise ValueError("The CUDA 12 image requires numpy<2")
        expected["numpy"] = md.version("numpy")
        try:
            wheel_version = md.version("nvidia-cudnn-cu12")
        except md.PackageNotFoundError:
            wheel_version = None
        cudnn_version = args.cudnn_version or wheel_version
        if cudnn_version is None:
            raise ValueError("Pass --cudnn-version when using a system-only cuDNN install")
        if wheel_version is not None:
            if wheel_version != cudnn_version:
                raise ValueError(f"cuDNN wheel {wheel_version} does not match {cudnn_version}")
            expected["nvidia-cudnn-cu12"] = wheel_version
        cudnn_runtime = validate_cudnn_runtime(cudnn_version)
    if args.profile == "cu12" or os.environ["RELAX_GPU_ARCH"] == "blackwell":
        manifest = json.loads(Path("/etc/relax-fla042-runtime.json").read_text())
        if manifest["source_sha256"] != SOURCE_SHA256 or source_hash(Path(manifest["root"]) / "fla") != SOURCE_SHA256:
            raise ValueError("Legacy FLA source closure is incomplete or changed")
    artifacts = {}
    for path in sorted(Path("/etc").glob("relax-*.json")):
        if path.name != "relax-final-runtime.json":
            artifacts[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = {
        "packages": expected,
        "manifests": artifacts,
        "profile": args.profile,
        "gpu_arch": os.environ.get("RELAX_GPU_ARCH"),
        "gpu_validated": False,
    }
    if cudnn_runtime is not None:
        report["cudnn_runtime"] = cudnn_runtime
    Path("/etc/relax-final-runtime.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
