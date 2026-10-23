# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Check the 280G Blackwell image's source contracts without importing CUDA
extensions."""

import ast
import hashlib
import importlib.metadata as md
import importlib.util
import json
import os
from importlib.machinery import PathFinder
from pathlib import Path


def check_fla_conv(package: Path, mode: str = "all") -> dict[str, str]:
    """Verify all installed entries without importing Torch or initializing
    CUDA."""
    if mode not in {"all", "original", "pow2", "none"}:
        raise ValueError(f"Unknown FLA convolution mode: {mode}")
    root = package / "modules/conv/triton"
    paths = [root / name for name in ("kernels.py", "kernels_pow2.py", "kernels_none.py", "ops.py", "conv_policy.py")]
    names = {"causal_conv1d_fwd_kernel", "causal_conv1d_bwd_kernel"}
    bodies = {}
    for entry, path in zip(("original", "pow2", "none"), paths[:3], strict=True):
        suffix = "" if entry == "original" else f"_{entry}"
        functions = [
            n
            for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name.removesuffix(suffix) in names
        ]
        if len(functions) != 2:
            raise ValueError(f"Missing FLA forward/backward convolution kernels: {entry}")
        for node in functions:
            if any(isinstance(n, ast.Name) and n.id == "NB" for stmt in node.body for n in ast.walk(stmt)):
                raise ValueError("FLA NB participates in kernel arithmetic; cache-only patch is unsafe")
            keys = [
                k.value for d in node.decorator_list if isinstance(d, ast.Call) for k in d.keywords if k.arg == "key"
            ]
            expected = ["D", "W"] if entry == "none" else ["D", "W", "NB"]
            if len(keys) != 1 or ast.literal_eval(keys[0]) != expected:
                raise ValueError(f"Unexpected FLA convolution autotune key: {entry}")
            if ("NB" in {a.arg for a in node.args.args}) != (entry != "none"):
                raise ValueError("FLA NB signature disagrees with entry")
            body = [ast.dump(stmt) for stmt in node.body]
            if entry == "original":
                bodies[node.name] = body
            elif body != bodies[node.name.removesuffix(suffix)]:
                raise ValueError(f"FLA entry changed kernel arithmetic: {entry}")
    ops = ast.parse(paths[3].read_text())
    policies = [
        n
        for n in ast.walk(ops)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "conv_nb_kwargs"
    ]
    expansions = [
        n
        for n in ast.walk(ops)
        if isinstance(n, ast.keyword) and n.arg is None and isinstance(n.value, ast.Name) and n.value.id == "nb_kwargs"
    ]
    if len(policies) != 2 or len(expansions) != 2:
        raise ValueError("FLA forward/backward must both use the selected NB policy")
    for path in paths:
        compile(path.read_text(), str(path), "exec")
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def check_mcore(root: Path) -> None:
    config = ast.parse((root / "core/transformer/transformer_config.py").read_text())
    fields = {n.target.id for n in ast.walk(config) if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)}
    required = {
        "moe_hybridep_pad_variable_tokens",
        "linear_cp_mode",
        "gdn_pre_gated_delta_rule_fusion",
        "gdn_conv_pad_alignment",
    }
    if missing := required - fields:
        raise ValueError(f"MCore is missing 280G Blackwell configuration fields: {sorted(missing)}")
    gpt = (root / "core/models/gpt/gpt_model.py").read_text()
    if "labels_are_shifted" not in gpt:
        raise ValueError("MCore is missing Relax's shifted-label MTP patch")


def check_hybridep(package: Path) -> None:
    tree = ast.parse((package / "hybrid_ep_buffer.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HybridEPBuffer")
    methods = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
    if not {"__init__", "dispatch_with_permute", "combine_with_unpermute", "update_template_config"} <= methods:
        raise ValueError("HybridEPBuffer lacks the MCore dispatch/combine API")
    exported = ast.parse((package / "__init__.py").read_text())
    symbols = {a.name for n in exported.body if isinstance(n, ast.ImportFrom) for a in n.names}
    if not {"Buffer", "HybridEPBuffer", "Config"} <= symbols:
        raise ValueError("HybridEP installation must retain the ordinary DeepEP API")


def main() -> None:
    expected = {
        "fla-core": "0.5.2",
        "flash-linear-attention": "0.5.2",
        "audioread": "3.1.0",
        "nvidia-nvshmem-cu13": "3.4.5",
        "pynvml": "13.0.1",
    }
    for name, version in expected.items():
        if md.version(name) != version:
            raise ValueError(f"{name}: expected {version}, got {md.version(name)}")
    revision = os.environ["RELAX_HYBRID_EP_COMMIT"]
    installed = md.version("relax-hybrid-ep")
    if "+" not in installed or not revision.startswith(installed.split("+", 1)[1]):
        raise ValueError(f"DeepEP package {installed} does not match {revision}")
    deep_ep = Path(importlib.util.find_spec("_relax_hybrid_ep").origin).parent
    for extension in ("deep_ep_cpp", "hybrid_ep_cpp"):
        if PathFinder.find_spec(f"_relax_hybrid_ep.{extension}", [str(deep_ep)]) is None:
            raise ValueError(f"Missing compiled extension: {extension}")
    fla = Path(importlib.util.find_spec("fla").origin).parent
    check_hybridep(deep_ep)
    check_mcore(Path("/root/Megatron-LM/megatron"))
    hashes = check_fla_conv(fla)
    report = {
        "hybridep_commit": revision,
        "deep_ep_version": installed,
        "fla_conv_nb_modes": ["original", "pow2", "none"],
        "fla_conv_selection": "process-import",
        "fla_conv_sha256": hashes,
        "packages": expected,
        "gpu_validated": False,
    }
    Path("/etc/relax-blackwell-runtime.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
