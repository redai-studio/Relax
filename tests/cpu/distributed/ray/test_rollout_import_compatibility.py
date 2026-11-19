# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Legacy runtime exports; run where the deployment dependencies are
available."""

import importlib
import pickle

import pytest

from relax.distributed.ray import rollout_scale_types
from relax.utils.metrics import rollout_metrics


def test_rollout_legacy_exports_and_pickle_globals_resolve_to_moved_definitions():
    for dependency in ("ray", "sglang", "sglang_router", "transfer_queue"):
        pytest.importorskip(dependency)
    # Top-level sglang can import even when its installed torchao/torch pair
    # cannot load ServerArgs. Check that external prerequisite separately so
    # errors in Relax's actual re-exports are never treated as a skip.
    try:
        importlib.import_module("sglang.srt.server_args")
    except ImportError as exc:
        pytest.skip(f"SGLang server dependencies are unavailable: {exc}")
    pytest.importorskip("megatron.core")
    runtime = importlib.import_module("relax.distributed.ray.rollout")

    for name in ("ScaleOutStatus", "ScaleOutMode", "ScaleOutRequest", "ScaleInStatus", "ScaleInRequest"):
        expected = getattr(rollout_scale_types, name)
        assert getattr(runtime, name) is expected
        # Protocol-0 GLOBALs represent the class lookup used by older pickles.
        assert pickle.loads(f"crelax.distributed.ray.rollout\n{name}\n.".encode()) is expected

    for name in ("compute_metrics_from_samples", "compute_perf_metrics_from_samples", "compute_mopd_metrics"):
        assert getattr(runtime, name) is getattr(rollout_metrics, name)
