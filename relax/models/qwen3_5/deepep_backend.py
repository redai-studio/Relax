# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Select the image-built DeepEP v2 without replacing the legacy package."""

import importlib.metadata as md
import os
from pathlib import Path


def ensure_backend(args) -> None:
    from .recipe import enabled

    if not enabled(args):
        return
    from .dependency_checks import runtime_manifest, verify_megatron_sources

    manifest = runtime_manifest()
    assert manifest["deepep_commit"] == "099d5f2bad488b9c534ea785062b12f2e91d1d41"
    assert md.version("relax-deep-ep-v2").split("+")[0] == "2.1.0"
    assert args.moe_flex_dispatcher_backend == "deepepv2"
    assert os.environ.get("EP_DISABLE_GIN") == "1"
    expected = Path(md.distribution("nvidia-nccl-cu13").locate_file("nvidia/nccl/lib/libnccl.so.2")).resolve()
    mapped = {
        Path(line.split()[-1]).resolve()
        for line in Path("/proc/self/maps").read_text().splitlines()
        if "libnccl.so" in line
    }
    assert mapped == {expected}, f"NCCL mappings differ from the unified image: {mapped}"
    import _relax_deep_ep_v2
    from megatron.core.transformer.moe import fused_a2a

    assert _relax_deep_ep_v2.__version__ == "2.1.0"
    assert fused_a2a.HAVE_DEEP_EP_V2 and fused_a2a._elastic_buffer is None
    verify_megatron_sources()
