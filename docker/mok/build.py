# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Build the pinned MoK wheel and apply the Megatron adapter."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from importlib import metadata
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


def check_megatron(root: Path, spec: dict, phase: str) -> None:
    check_files(root, spec[phase + "_sha256"])


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


def check_transformer_engine(manifest: dict) -> None:
    spec = manifest["transformer_engine"]
    distribution = metadata.distribution(spec["distribution"])
    if distribution.version != spec["version"]:
        raise RuntimeError(f"Expected Transformer Engine {spec['version']}")
    root = Path(distribution.locate_file(""))
    check_files(root, {spec["path"]: spec["after_sha256"]})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--megatron-root", type=Path, required=True)
    parser.add_argument("--mok-root", type=Path)
    parser.add_argument("--arch", choices=["SM103"], default="SM103")
    parser.add_argument("--check-base-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((PACKAGE / "manifest.json").read_text())
    check_megatron(args.megatron_root, manifest["megatron"], "before")
    print("Megatron patch-target fingerprints verified.", flush=True)
    if args.check_base_only:
        return
    if args.mok_root is None:
        parser.error("--mok-root is required unless --check-base-only is used")
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
    check_transformer_engine(manifest)

    # Fetch/verify all sources before modifying the image's Megatron tree.
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
    apply_patches(args.megatron_root, manifest["patch_order"]["megatron"], manifest)
    check_megatron(args.megatron_root, manifest["megatron"], "after")

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
    with tempfile.TemporaryDirectory(prefix="relax-mok-wheels-") as wheels:
        run(
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--no-cache-dir",
            "--wheel-dir",
            wheels,
            str(args.mok_root),
            env=env,
        )
        artifacts = list(Path(wheels).glob("mixture_of_kittens-*.whl"))
        if len(artifacts) != 1:
            raise RuntimeError(f"Expected one MoK wheel: {artifacts}")
        wheel_hash = digest(artifacts[0])
        run(
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-index",
            "--force-reinstall",
            str(artifacts[0]),
            env=env,
        )
    run(sys.executable, "-m", "compileall", "-q", str(args.megatron_root / "megatron"))
    (PACKAGE / "build-result.json").write_text(
        json.dumps(
            {
                "status": "SOURCE_AND_WHEEL_BUILT",
                "arch": args.arch,
                "wheel_sha256": wheel_hash,
                "torch": str(torch.__version__),
                "cuda": torch.version.cuda,
                "nvcc": nvcc.strip(),
                "megatron_root": str(args.megatron_root),
                "mok_root": str(args.mok_root),
                "manifest_sha256": digest(PACKAGE / "manifest.json"),
                "runtime_gpu_validation": "PENDING",
            },
            indent=2,
        )
        + "\n"
    )
    print("MoK wheel built from source and installed; GPU runtime validation remains separate.", flush=True)


if __name__ == "__main__":
    main()
