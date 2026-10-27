# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Build and install Relax's patched SGLang Rust cache against the active
Torch."""

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import runpy
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path


MODULE = "sglang.srt.mem_cache.rust_tree_core.mem_cache"
RELATIVE_PACKAGE = Path("srt/mem_cache/rust_tree_core")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_binding(path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib.util, sys, torch; "
            "spec = importlib.util.spec_from_file_location(sys.argv[2], sys.argv[1]); "
            "module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
            "assert all(hasattr(getattr(module, name), 'finish_deferred_chunked_backup') "
            "for name in ('RustUnifiedTreeCoreBinding', 'RustBigramUnifiedTreeCoreBinding')); "
            "assert module.__file__ == sys.argv[1]",
            str(path.resolve()),
            MODULE,
        ],
        check=True,
    )


def build(source_root: Path, output: Path) -> None:
    source_root = source_root.resolve()
    crate = source_root / "rust/sglang-radix-tree"
    helper = source_root / "python/sglang/srt/rust_extensions/torch_build.py"
    configuration = runpy.run_path(str(helper))["torch_build_configuration"](
        compat_header=crate / "torch_2_13_compat.h",
        python_module=MODULE,
        include_absolute_rpath=False,
    )
    environment = configuration.environment
    target = Path(environment.get("CARGO_TARGET_DIR", str(source_root / "target/relax-tree-core"))).resolve()
    environment["CARGO_TARGET_DIR"] = str(target)
    subprocess.run(
        [
            "cargo",
            "build",
            "--locked",
            "--release",
            "--features",
            "python-extension",
            "--manifest-path",
            str(crate / "Cargo.toml"),
        ],
        env=environment,
        check=True,
    )
    output.mkdir(parents=True, exist_ok=True)
    artifact = output / ("mem_cache" + sysconfig.get_config_var("EXT_SUFFIX"))
    shutil.copy2(target / "release/libmem_cache.so", artifact)
    validate_binding(artifact)
    sources = [p for p in crate.rglob("*") if p.is_file() and "target" not in p.relative_to(crate).parts]
    manifest = {
        "module": MODULE,
        "artifact": artifact.name,
        "artifact_sha256": digest(artifact),
        "torch": configuration.fingerprint,
        "sources": {str(p.relative_to(source_root)): digest(p) for p in sorted(sources)},
        "production_binding_validated": True,
    }
    (output / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def install(output: Path) -> None:
    import torch

    manifest = json.loads((output / "build-manifest.json").read_text())
    expected = manifest["torch"]
    if importlib.metadata.version("sglang") != "0.5.21":
        raise RuntimeError("patched tree core requires SGLang 0.5.21")
    if (
        str(torch.__version__) != expected["torch_version"]
        or bool(torch.compiled_with_cxx11_abi()) != expected["torch_cxx11_abi"]
        or torch.version.cuda != expected["torch_cuda"]
        or torch.version.hip != expected["torch_hip"]
    ):
        raise RuntimeError("tree core build and runtime Torch ABIs differ")
    artifact = output / manifest["artifact"]
    if digest(artifact) != manifest["artifact_sha256"]:
        raise RuntimeError("tree core artifact checksum differs")
    if artifact.name != "mem_cache" + sysconfig.get_config_var("EXT_SUFFIX"):
        raise RuntimeError("tree core build and runtime Python ABIs differ")
    package = Path(importlib.util.find_spec("sglang").origin).parent
    destination = package / RELATIVE_PACKAGE / artifact.name
    shutil.copy2(artifact, destination)
    validate_binding(destination)
    # A source checkout normally selects a fingerprinted JIT artifact. Images
    # explicitly trust the extension built above, and verify that actual loader.
    environment = dict(os.environ, SGLANG_RUST_BUILD_MODE="never")
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import pathlib, sys; "
            "from sglang.srt.mem_cache.rust_tree_core.extension import bindings; "
            "assert pathlib.Path(bindings.__file__).resolve() == pathlib.Path(sys.argv[1]).resolve(); "
            "assert hasattr(bindings.RustUnifiedTreeCoreBinding, 'finish_deferred_chunked_backup')",
            str(destination),
        ],
        env=environment,
        check=True,
    )


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "build":
        build(Path(sys.argv[2]), Path(sys.argv[3]))
    elif len(sys.argv) == 3 and sys.argv[1] == "install":
        install(Path(sys.argv[2]))
    else:
        raise SystemExit("usage: build_sglang_tree_core.py build SOURCE OUTPUT | install OUTPUT")
