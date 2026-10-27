# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Validate the shared Qwen and legacy model source contracts during image
build."""

import ast
import importlib.metadata as metadata
import importlib.util
from pathlib import Path


def require_functions(path: Path, names: set[str]) -> None:
    tree = ast.parse(path.read_text())
    defined = {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    if missing := names - defined:
        raise ValueError(f"Missing unified runtime definitions in {path}: {sorted(missing)}")


def require_mtp_stream_norm(path: Path) -> None:
    tree = ast.parse(path.read_text())
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GatedResidualMTPHiddenNorm"
    )
    forward = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
    calls = [node for node in ast.walk(forward) if isinstance(node, ast.Call)]
    if not any(
        isinstance(node.func, ast.Name)
        and node.func.id == "gated_residual_group_rmsnorm"
        and len(node.args) == 4
        and isinstance(node.args[2], ast.Attribute)
        and isinstance(node.args[2].value, ast.Name)
        and node.args[2].value.id == "self"
        and node.args[2].attr == "num_streams"
        for node in calls
    ):
        raise ValueError(f"Expected independent-stream MTP hidden norm in {path}")


def main() -> None:
    for name, version in {
        "sglang": "0.5.21",
        "transformers": "5.16.0",
        "tokenizers": "0.23.2",
        "tilelang": "0.1.12",
        "quack-kernels": "0.6.4",
    }.items():
        if metadata.version(name) != version:
            raise ValueError(f"Expected {name}=={version}, found {metadata.version(name)}")
    core = Path(importlib.util.find_spec("megatron.core").origin).parent
    require_functions(
        core / "transformer/hyper_connection.py", {"GatedResidualHyperConnection", "GatedResidualMTPHiddenNorm"}
    )
    require_mtp_stream_norm(core / "transformer/hyper_connection.py")
    require_functions(core / "transformer/per_layer_embedding.py", {"PerLayerEmbedding"})
    require_functions(core / "transformer/experimental_attention_variant/qsa.py", {"QwenSparseSelfAttention"})
    for name in ("qwen4_exp_bridge.py", "qwen4_exp_vl_bridge.py"):
        path = core.parent / "bridge/models/qwen" / name
        ast.parse(path.read_text())
    sglang = Path(importlib.util.find_spec("sglang").origin).parent
    require_functions(
        sglang / "srt/layers/quantization/mxfp4.py", {"restore_weights_before_loading", "_copy_mxfp4_runtime_tensor"}
    )
    require_functions(sglang / "srt/models/kimi_k3.py", {"begin_weight_update", "end_weight_update"})
    ast.parse((sglang / "srt/models/qwen4_exp.py").read_text())


if __name__ == "__main__":
    main()
