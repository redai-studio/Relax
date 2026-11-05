# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Record the image-built optimized runtime and reject incomplete
installations."""

import ast
import hashlib
import importlib.metadata as md
import json
import os
import subprocess
import sys
import sysconfig
from pathlib import Path


try:
    import tomllib
except ModuleNotFoundError:
    # Python 3.10 CI already receives this backport through pytest.
    import tomli as tomllib


def hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }


def python_ast_hashes(root: Path) -> dict[str, str]:
    """Compare delivered implementations without pinning cosmetic
    formatting."""
    return {
        str(path.relative_to(root)): hashlib.sha256(ast.unparse(ast.parse(path.read_text())).encode()).hexdigest()
        for path in sorted(root.rglob("*.py"))
    }


def validate_source_dependencies(pyproject: Path) -> None:
    """Check dependencies of copied JIT sources that have no wheel metadata."""
    from packaging.requirements import Requirement

    project = tomllib.loads(pyproject.read_text())["project"]
    for entry in project["dependencies"]:
        requirement = Requirement(entry)
        if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
            continue
        try:
            installed = md.version(requirement.name)
        except md.PackageNotFoundError as exc:
            raise RuntimeError(f"{pyproject}: missing {requirement}") from exc
        if not requirement.specifier.contains(installed, prereleases=True):
            raise RuntimeError(f"{pyproject}: requires {requirement}, found {installed}")


def validate_pip_check(check: subprocess.CompletedProcess[str]) -> None:
    """Accept only Torch's known NCCL metadata override, including markers."""
    from packaging.requirements import InvalidRequirement, Requirement
    from packaging.utils import canonicalize_name

    if check.returncode == 0:
        return
    issues = check.stdout.strip().splitlines()
    allowed = check.returncode == 1 and bool(issues)
    for line in issues:
        owner, separator, detail = line.partition(" has requirement ")
        requirement, mismatch, installed = detail.partition(", but you have ")
        try:
            parsed = Requirement(requirement)
            known = (
                owner.split()[0] == "torch"
                and bool(separator)
                and bool(mismatch)
                and canonicalize_name(parsed.name) == "nvidia-nccl-cu13"
                and str(parsed.specifier) == "==2.29.7"
                and installed == "nvidia-nccl-cu13 2.30.7."
            )
        except (InvalidRequirement, IndexError):
            known = False
        allowed = allowed and known
    if not allowed:
        raise RuntimeError(check.stdout + check.stderr)


def main() -> None:
    site = Path(sysconfig.get_paths()["purelib"])
    sources = Path("/opt/relax/qwen35/sources")
    fa4 = {name: hashes(sources / name) for name in ("fa4-upstream", "fa4-isolated")}
    expected = json.loads(Path("/tmp/fa4-source-pins.json").read_text())
    assert fa4["fa4-upstream"] == expected["fa4-upstream"]
    assert python_ast_hashes(sources / "fa4-isolated") == expected["fa4-isolated-python-ast"]
    assert hashes(site / "_relax_fa4_pr2968/cute") == fa4["fa4-isolated"]
    validate_source_dependencies(sources / "fa4-upstream/pyproject.toml")
    cublas = Path("/opt/relax/cublas13.3/nvidia/cu13/lib")
    cublas_pins = json.loads(Path("/tmp/cublas-library-pins.json").read_text())
    for name, digest in cublas_pins.items():
        assert hashlib.sha256((cublas / name).read_bytes()).hexdigest() == digest, name
    versions = {
        "relax-deep-ep-v2": "2.1.0",
        "nvidia-cutlass-dsl": "4.6.2",
        "tilelang": "0.1.12",
        "flash-qla": "0.1.3",
        "nvidia-nccl-cu13": "2.30.7",
    }
    for name, version in versions.items():
        assert md.version(name).split("+")[0] == version, (name, md.version(name))
    assert list((site / "_relax_deep_ep_v2").glob("_C*.so"))
    assert (site / "_relax_hybrid_ep/hybrid_ep_buffer.py").is_file()
    # Torch's metadata pins the older NCCL; accept only this documented base-image
    # override. A missing dependency or any other version conflict still fails.
    check = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    validate_pip_check(check)
    protected_sources = (
        "megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py",
        "megatron/core/optimizer/distrib_optimizer.py",
        "megatron/core/optimizer/optimizer.py",
        "megatron/core/transformer/moe/token_dispatcher.py",
        "megatron/core/transformer/moe/moe_layer.py",
        "megatron/core/transformer/moe/experts.py",
        "megatron/core/transformer/moe/fused_a2a.py",
        "megatron/core/transformer/moe/shared_experts.py",
        "megatron/core/tensor_parallel/random.py",
        "megatron/core/extensions/transformer_engine.py",
        "megatron/core/transformer/multi_token_prediction.py",
        "megatron/core/transformer/attention.py",
        "megatron/core/transformer/transformer_block.py",
        "megatron/core/transformer/transformer_config.py",
        "megatron/training/training.py",
        "megatron/core/fusions/linear_cross_entropy/blackwell/fwd_mainloop.py",
        "megatron/core/fusions/linear_cross_entropy/blackwell/bwd_partial_dlogits.py",
    )
    operator_patch = Path("/root/Megatron-LM/20261020-model-operators.patch")
    protected_sources = sorted(
        set(protected_sources)
        | {line[6:] for line in operator_patch.read_text().splitlines() if line.startswith("+++ b/")}
    )
    report = {
        "deepep_commit": os.environ["DEEP_EP_V2_COMMIT"],
        "fa4_commit": os.environ["FA4_COMMIT"],
        "flashqla_commit": os.environ["FLASHQLA_COMMIT"],
        "delivery_commit": "3fb0859060e51700cba8d30f8d00dac4b9989cbd",
        "flashqla_upstream_archive_source_verified": os.environ["FLASHQLA_COMMIT"]
        == "5538c35f0df33ef804eacd90c77d9d9087c1d4e4",
        "fa4_isolated_archive_ast_verified": True,
        "fa4_sources": fa4,
        "megatron_sources": {
            name: hashlib.sha256((Path("/root/Megatron-LM") / name).read_bytes()).hexdigest()
            for name in protected_sources
        },
        "cublas_library_root": str(cublas),
        "cublas_library_sha256": cublas_pins,
        "packages": versions,
        "deep_ep_extension_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (site / "_relax_deep_ep_v2").glob("_C*.so")
        },
        "patch_sha256": {
            name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for name, path in {
                "deepep": "/tmp/qwen35-build/2.1.0-relax-namespace.patch",
                "fa4": "/tmp/qwen35-build/56e48e3-fa4-isolated.patch",
                "flashqla": "/tmp/qwen35-build/0.1.3-kkt-no-tma.patch",
                "megatron": "/root/Megatron-LM/20261020-qwen35-optimized.patch",
                "model_operators": "/root/Megatron-LM/20261020-model-operators.patch",
                "checkpoint": "/root/Megatron-LM/20261028-dp-reshardable-checkpoint.patch",
            }.items()
        },
        "gpu_validated": False,
    }
    Path("/etc/relax-qwen35-runtime.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
