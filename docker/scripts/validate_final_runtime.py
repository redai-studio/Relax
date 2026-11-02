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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("cu12", "cu13"), default="cu13")
    args = parser.parse_args()
    expected = {"ray": "2.56.1", "numpy": "2.2.6", "nvidia-nccl-cu13": "2.30.7", "fla-core": "0.5.2"}
    if args.profile == "cu12":
        expected.pop("numpy")
        expected.pop("nvidia-nccl-cu13")
    for package, version in expected.items():
        if md.version(package) != version:
            raise ValueError(f"{package}: expected {version}, got {md.version(package)}")
    import torch

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
    Path("/etc/relax-final-runtime.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
