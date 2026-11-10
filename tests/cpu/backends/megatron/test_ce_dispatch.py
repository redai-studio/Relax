# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CE dispatch preserves import identity for every initial backend flag."""

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor


def _check_ce_dispatch_entrypoint(enabled: bool, backport_first: bool) -> None:
    script = f"""
import importlib
if {backport_first!r}:
    importlib.import_module('relax.backends.megatron.kernels.linear_cross_entropy')
from relax.backends.megatron import fused_linear_ce
original = fused_linear_ce.linear_cross_entropy
from relax.backends.megatron.kernels import linear_cross_entropy as kernel
assert fused_linear_ce.linear_cross_entropy is original
backport_result, fallback_result = object(), object()
kernel.linear_cross_entropy = lambda *args, **kwargs: backport_result
fused_linear_ce._LinearCE.apply = lambda *args: fallback_result
fused_linear_ce._log_enabled = lambda *args: None
import torch
hidden = torch.ones(1)
expected = backport_result if {enabled!r} else fallback_result
assert fused_linear_ce.linear_cross_entropy(hidden, hidden, hidden, None) is expected
assert fused_linear_ce.linear_cross_entropy is original
"""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", RELAX_SFT_FUSED_LINEAR_CE=str(int(enabled)))
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, (
        f"enabled={enabled}, backport_first={backport_first}\n{result.stdout}{result.stderr}"
    )


def test_ce_dispatch_preserves_entrypoint_identity() -> None:
    # Each import order and initial environment needs a fresh interpreter.
    # Bound CPU-only subprocess concurrency without sharing import state.
    with ThreadPoolExecutor(max_workers=4) as executor:
        cases = [
            executor.submit(_check_ce_dispatch_entrypoint, enabled, backport_first)
            for backport_first in (False, True)
            for enabled in (False, True)
        ]
        for case in cases:
            case.result()
