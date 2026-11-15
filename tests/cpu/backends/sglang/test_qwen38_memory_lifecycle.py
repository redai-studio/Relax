# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""GPU-free execution of lifecycle methods from a patched SGLang checkout.

Set SGLANG_SOURCE_DIR to the checkout root used to validate/build the image.
Only method ASTs are loaded, so importing CUDA extensions is unnecessary.
"""

import ast
import os
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest


def _load_method(path: Path, class_name: str, method_name: str, namespace: dict):
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    method.decorator_list = []
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[method_name]


@pytest.fixture
def lifecycle():
    source = os.environ.get("SGLANG_SOURCE_DIR")
    if not source:
        pytest.skip("Set SGLANG_SOURCE_DIR to the patched SGLang checkout")
    root = Path(source) / "python/sglang/srt/managers"
    events = []
    updater = SimpleNamespace(offload_tags=set(), scheduler=None, tp_cpu_group=object())

    def clear_pool():
        assert "kv_cache" not in updater.offload_tags, "Pool clear happened after KV was marked offloaded"
        events.append("clear_pool")

    def pause(tag):
        assert tag in updater.offload_tags
        events.append(f"pause:{tag}")

    logger = SimpleNamespace(info=lambda *_args: None, warning=lambda *_args: None)
    namespace = {
        "GPU_MEMORY_TYPE_KV_CACHE": "kv_cache",
        "GPU_MEMORY_TYPE_WEIGHTS": "weights",
        "GPU_MEMORY_TYPE_CUDA_GRAPH": "cuda_graph",
        "GPU_MEMORY_ALL_TYPES": ["kv_cache", "weights", "cuda_graph"],
        "ReleaseMemoryOccupationReqOutput": SimpleNamespace,
        "logger": logger,
        "logging": logger,
        "current_platform": SimpleNamespace(empty_cache=lambda: events.append("empty_cache")),
        "torch": SimpleNamespace(
            distributed=SimpleNamespace(barrier=lambda _group: events.append("barrier")),
            get_device_module=lambda: SimpleNamespace(synchronize=lambda: events.append("synchronize")),
        ),
        "_export_static_state": lambda _model: events.append("export_static_state"),
    }
    scheduler = SimpleNamespace(
        weight_updater=updater,
        is_fully_idle=lambda **_kwargs: True,
        _engine_paused=False,
        tree_cache=SimpleNamespace(reset=clear_pool),
        req_to_token_pool=SimpleNamespace(clear=clear_pool, reset_aux_cache_allocator=clear_pool),
        token_to_kv_pool_allocator=SimpleNamespace(clear=clear_pool),
        grammar_manager=SimpleNamespace(clear=clear_pool),
        metrics_reporter=SimpleNamespace(reset_metrics=clear_pool, is_stats_logging_rank=False),
        draft_worker=None,
        waiting_queue=[],
        running_batch=SimpleNamespace(reqs=[]),
    )
    flush = _load_method(root / "scheduler.py", "Scheduler", "flush_cache", namespace)
    scheduler.flush_cache = MethodType(flush, scheduler)
    updater.flush_cache = scheduler.flush_cache
    updater.is_fully_idle = lambda **_kwargs: True
    updater.memory_saver_adapter = SimpleNamespace(pause=pause)
    updater._assert_weight_cache_inactive = lambda _op: None
    updater.tp_worker = SimpleNamespace(model_runner=SimpleNamespace(model=object()))
    release = _load_method(
        root / "scheduler_components/weight_updater.py",
        "SchedulerWeightUpdaterManager",
        "release_memory_occupation",
        namespace,
    )
    updater.release_memory_occupation = MethodType(release, updater)
    return SimpleNamespace(scheduler=scheduler, updater=updater, events=events)


def test_qwen38_flush_skips_unmapped_kv_pools(lifecycle):
    lifecycle.updater.offload_tags.add("kv_cache")
    assert lifecycle.scheduler.flush_cache()
    assert lifecycle.scheduler.flush_cache()
    assert lifecycle.events == []


def test_qwen38_live_flush_clears_pools_and_busy_flush_preserves_them(lifecycle):
    assert lifecycle.scheduler.flush_cache(empty_cache=False)
    assert lifecycle.events == ["clear_pool"] * 6
    lifecycle.events.clear()
    lifecycle.scheduler.is_fully_idle = lambda **_kwargs: False
    assert not lifecycle.scheduler.flush_cache()
    assert lifecycle.events == []


@pytest.mark.parametrize("tags", [["kv_cache"], None, []])
def test_qwen38_release_flushes_before_marking_and_unmapping_kv(lifecycle, tags):
    lifecycle.updater.release_memory_occupation(SimpleNamespace(tags=tags))
    assert lifecycle.events[:7] == ["clear_pool"] * 6 + ["empty_cache"]
    assert lifecycle.events[7] == "pause:kv_cache"
    assert lifecycle.events[-1] == "synchronize"
    if not tags:
        assert lifecycle.events[8:-1] == ["export_static_state", "barrier", "pause:weights", "pause:cuda_graph"]
    before = list(lifecycle.events)
    assert lifecycle.scheduler.flush_cache()
    assert lifecycle.events == before


def test_qwen38_failed_flush_does_not_publish_offload_state_or_unmap(lifecycle):
    lifecycle.updater.flush_cache = lambda: False
    with pytest.raises(AssertionError, match="Cache flush failed"):
        lifecycle.updater.release_memory_occupation(SimpleNamespace(tags=["kv_cache"]))
    assert lifecycle.updater.offload_tags == set()
    assert lifecycle.events == []


def test_qwen38_weights_only_release_does_not_touch_kv_pools(lifecycle):
    lifecycle.updater.release_memory_occupation(SimpleNamespace(tags=["weights"]))
    assert lifecycle.events == ["export_static_state", "barrier", "pause:weights", "synchronize"]
    assert lifecycle.updater.offload_tags == {"weights"}
    assert lifecycle.scheduler.flush_cache()
