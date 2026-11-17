# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Check installed source fingerprints; optionally load CUDA extensions at
runtime."""

import argparse
import importlib.util
import inspect
import json
import os
import subprocess
from importlib import metadata
from pathlib import Path

from build import PACKAGE, check_files, check_megatron, check_transformer_engine, digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime", action="store_true", help="Load the compiled module; use docker run --gpus, after building"
    )
    args = parser.parse_args()
    manifest = json.loads((PACKAGE / "manifest.json").read_text())
    result = json.loads((PACKAGE / "build-result.json").read_text())
    if digest(PACKAGE / "manifest.json") != result["manifest_sha256"]:
        raise RuntimeError("The build manifest has changed")
    megatron = Path(os.environ.get("RELAX_MEGATRON_ROOT", result["megatron_root"]))
    mok_source = Path(os.environ.get("RELAX_MOK_SOURCE_ROOT", result["mok_root"]))
    check_megatron(megatron, manifest["megatron"], "after")
    check_files(mok_source, manifest["mok"]["after_sha256"])
    check_files(mok_source / "third_party/ThunderKittens", manifest["thunderkittens"]["after_sha256"])
    check_transformer_engine(manifest)
    distribution = metadata.distribution("mixture-of-kittens")
    installed = Path(distribution.locate_file("mok")).resolve()
    python_files = {
        name.removeprefix("mok/"): checksum
        for name, checksum in manifest["mok"]["after_sha256"].items()
        if name.startswith("mok/") and name.endswith(".py")
    }
    check_files(installed, python_files)
    extensions = list(installed.glob("_C*.so"))
    if len(extensions) != 1:
        raise RuntimeError(f"Expected exactly one installed CUDA extension: {extensions}")
    cubins = subprocess.check_output(["cuobjdump", "--list-elf", str(extensions[0])], text=True)
    if "sm_103" not in cubins:
        raise RuntimeError("The wheel is missing the SM103 cubin")
    located = importlib.util.find_spec("mok")
    if located is None or Path(located.origin).resolve().parent != installed:
        raise RuntimeError("PYTHONPATH shadows the installed MoK wheel; remove the old experiment overlay")
    if args.runtime:
        import mok
        import mok._C
        import torch
        from megatron.core.transformer.moe.megakernel.mok.backend import MoKMegakernel
        from megatron.core.transformer.transformer_config import TransformerConfig

        for imported_class in (TransformerConfig, MoKMegakernel):
            definition = Path(inspect.getfile(imported_class)).resolve()
            if not definition.is_relative_to(megatron.resolve()):
                raise RuntimeError(f"PYTHONPATH shadows the image's Megatron: {definition}")
        if Path(mok._C.__file__).resolve() != extensions[0].resolve():
            raise RuntimeError("An old MoK CUDA extension shadows the installed wheel")

        if str(torch.__version__) != manifest["validated_environment"]["torch"]:
            raise RuntimeError("Torch changed after the wheel was built")
        config = TransformerConfig(
            num_layers=1,
            hidden_size=2560,
            num_attention_heads=20,
            num_moe_experts=512,
            moe_ffn_hidden_size=640,
            moe_shared_expert_intermediate_size=640,
            moe_router_topk=10,
            moe_grouped_gemm=True,
            moe_router_dtype="fp32",
            tensor_model_parallel_size=2,
            expert_model_parallel_size=8,
            expert_tensor_parallel_size=1,
            sequence_parallel=True,
            gated_linear_unit=True,
            activation_func=torch.nn.functional.silu,
            add_bias_linear=False,
            gradient_accumulation_fusion=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            moe_megakernel_backend="mok",
            moe_megakernel_backend_config={"shared_expert_mode": "native"},
        )
        if config.moe_megakernel_backend != "mok" or MoKMegakernel is None:
            raise RuntimeError("Megatron MoK adapter is unavailable")
        if Path(mok.__file__).resolve().parent != installed:
            raise RuntimeError("Unexpected imported MoK package")
    print(
        json.dumps(
            {
                "source_and_wheel_fingerprints": "PASS",
                "runtime_import": args.runtime,
                "extension_sha256": digest(extensions[0]),
                "installed_mok": str(installed),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
