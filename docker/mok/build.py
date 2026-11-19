# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Build the pinned Blackwell MoK wheel."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


PACKAGE = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_files(root: Path, expected: dict[str, str]) -> None:
    failures = [
        name for name, checksum in expected.items() if not (root / name).is_file() or digest(root / name) != checksum
    ]
    if failures:
        raise RuntimeError(f"Source fingerprint mismatch under {root}: {failures[:12]}; total={len(failures)}")


def run(*command: str, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, cwd=cwd, env=env, check=True)


def apply_patches(root: Path, names: list[str], manifest: dict) -> None:
    for name in names:
        patch = PACKAGE / "patches" / name
        if digest(patch) != manifest["patch_sha256"][name]:
            raise RuntimeError(f"Patch checksum mismatch: {name}")
        with patch.open("rb") as source:
            subprocess.run(["patch", "--batch", "--forward", "--fuzz=0", "-p1"], cwd=root, stdin=source, check=True)


def checkout(root: Path, spec: dict) -> None:
    if root.exists():
        raise RuntimeError(f"Refusing to reuse an existing source tree: {root}")
    root.mkdir(parents=True)
    run("git", "init", "--quiet", str(root))
    run("git", "remote", "add", "origin", spec["url"], cwd=root)
    run("git", "fetch", "--depth=1", "origin", spec["commit"], cwd=root)
    run("git", "checkout", "--quiet", "--detach", "FETCH_HEAD", cwd=root)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if head != spec["commit"]:
        raise RuntimeError(f"Unexpected checkout: {head}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mok-root", type=Path, required=True)
    parser.add_argument("--arch", choices=["SM103"], default="SM103")
    parser.add_argument("--wheel-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = PACKAGE / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for executable in ("git", "patch", "make", "nvcc"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"The existing training image must provide {executable}")
    if sys.version_info[:2] != tuple(manifest["validated_environment"]["python_major_minor"]):
        raise RuntimeError("Use the existing Python 3.12 training image")
    import torch

    expected = manifest["validated_environment"]
    if str(torch.__version__) != expected["torch"] or torch.version.cuda != expected["torch_cuda"]:
        raise RuntimeError(
            f"Expected Torch {expected['torch']}/CUDA {expected['torch_cuda']}; found {torch.__version__}/{torch.version.cuda}"
        )
    compiler = shutil.which("nvcc")
    nvcc = subprocess.check_output([compiler, "--version"], text=True)
    if re.search(r"release\s+13\.0\b", nvcc) is None:
        raise RuntimeError("The compiler must match Torch's CUDA 13.0; do not rebuild or replace the CUDA stack")

    # Fetch and verify the pinned sources before compiling the wheel.
    checkout(args.mok_root, manifest["mok"])
    thunderkittens = args.mok_root / "third_party/ThunderKittens"
    # Git creates an empty directory for the uninitialized gitlink.
    if thunderkittens.is_dir() and not any(thunderkittens.iterdir()):
        thunderkittens.rmdir()
    checkout(thunderkittens, manifest["thunderkittens"])
    apply_patches(args.mok_root, manifest["patch_order"]["mok"], manifest)
    apply_patches(thunderkittens, manifest["patch_order"]["thunderkittens"], manifest)
    check_files(args.mok_root, manifest["mok"]["after_sha256"])
    check_files(thunderkittens, manifest["thunderkittens"]["after_sha256"])

    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        MOK_ARCH=args.arch,
        PIP_DISABLE_PIP_VERSION_CHECK="1",
        NVCC=compiler,
        MOK_NVCC=compiler,
    )
    cuda = Path(env.get("CUDA_HOME", "/usr/local/cuda"))
    # Driver stubs are for link time only. Never add them to runtime LD_LIBRARY_PATH.
    env["LIBRARY_PATH"] = str(cuda / "lib64/stubs") + os.pathsep + env.get("LIBRARY_PATH", "")
    args.wheel_dir.mkdir(parents=True, exist_ok=True)
    run(
        sys.executable,
        "-m",
        "pip",
        "wheel",
        "--no-deps",
        "--no-build-isolation",
        "--no-cache-dir",
        "--wheel-dir",
        str(args.wheel_dir),
        str(args.mok_root),
        env=env,
    )
    artifacts = list(args.wheel_dir.glob("mixture_of_kittens-*.whl"))
    if len(artifacts) != 1:
        raise RuntimeError(f"Expected one MoK wheel: {artifacts}")
    (args.wheel_dir / "build-result.json").write_text(
        json.dumps(
            {
                "status": "SOURCE_AND_WHEEL_BUILT",
                "arch": args.arch,
                "wheel_sha256": digest(artifacts[0]),
                "torch": str(torch.__version__),
                "cuda": torch.version.cuda,
                "nvcc": nvcc.strip(),
                "mok_root": str(args.mok_root),
                "manifest_sha256": digest(manifest_path),
                "runtime_gpu_validation": "PENDING",
            },
            indent=2,
        )
        + "\n"
    )
    print("MoK wheel built from source; GPU runtime validation remains separate.", flush=True)


if __name__ == "__main__":
    main()
