# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Check GPU imports inside the built image on a host with a GPU driver."""

import itertools
import os
import subprocess
import sys
from importlib import import_module
from importlib import metadata as md
from pathlib import Path


def check_deepep_coexistence() -> None:
    # Fresh interpreters exercise both registration orders, not just cached modules.
    code = """
import importlib, sys, torch
assert torch.cuda.is_available()
for name in sys.argv[1:]:
    importlib.import_module(name)
modules = ['deep_ep._C', '_relax_hybrid_ep.deep_ep_cpp', '_relax_deep_ep_v2._C']
events = [importlib.import_module(name).EventHandle for name in modules]
assert len(set(events)) == 3
for event, module in zip(events, modules):
    assert event.__module__ == module
    event().current_stream_wait()
torch.cuda.synchronize()
"""
    for order in itertools.permutations(("deep_ep", "_relax_hybrid_ep", "_relax_deep_ep_v2")):
        subprocess.run([sys.executable, "-c", code, *order], check=True, timeout=180)


def main() -> None:
    import torch

    assert torch.cuda.is_available(), "GPU/driver unavailable"
    import_module("transformer_engine.pytorch")
    import_module("sgl_kernel")
    expected_sglang = os.environ.get("RELAX_SGLANG_VERSION")
    if expected_sglang:
        sglang = import_module("sglang")
        assert md.version("sglang") == sglang.__version__ == expected_sglang
        import_module("sglang.srt.server_args")
    if os.environ.get("RELAX_GPU_ARCH") in {"blackwell", "hopper"} or os.environ.get("RELAX_HYBRID_EP_COMMIT"):
        deep_ep = import_module("_relax_hybrid_ep")
        import_module("_relax_hybrid_ep.deep_ep_cpp")
        import_module("_relax_hybrid_ep.hybrid_ep_cpp")
        assert hasattr(deep_ep, "Buffer") and hasattr(deep_ep, "HybridEPBuffer")
    if os.environ.get("RELAX_GPU_ARCH") == "blackwell":
        from transformer_engine.pytorch.cross_entropy import parallel_cross_entropy

        assert callable(parallel_cross_entropy)
        if Path("/etc/relax-qwen35-runtime.json").is_file():
            check_deepep_coexistence()
            optimized_ep = import_module("_relax_deep_ep_v2")
            assert hasattr(optimized_ep, "ElasticBuffer")
            fa4 = import_module("_relax_fa4_pr2968.cute.interface")
            assert callable(fa4.flash_attn_varlen_func)
            flashqla = import_module("flash_qla")
            assert callable(flashqla.chunk_gated_delta_rule)
            expected_nccl = Path(
                md.distribution("nvidia-nccl-cu13").locate_file("nvidia/nccl/lib/libnccl.so.2")
            ).resolve()
            mapped_nccl = {
                Path(line.split()[-1]).resolve()
                for line in Path("/proc/self/maps").read_text().splitlines()
                if "libnccl.so" in line
            }
            assert mapped_nccl == {expected_nccl}, f"Unexpected NCCL libraries: {mapped_nccl}"
    sys.stdout.write(f"GPU imports OK: {torch.cuda.get_device_name()} {torch.cuda.get_device_capability()}\n")


if __name__ == "__main__":
    main()
