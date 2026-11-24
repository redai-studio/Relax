# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for the configurable media-encoding thread pool in processing_utils.

Imports are deferred to a fixture because processing_utils pulls in the heavy
imageio / soundfile / transformers / torch stack at module level.
"""

import importlib
import os
import sys
from types import ModuleType

import pytest


@pytest.fixture()
def pu(monkeypatch):
    """processing_utils with encode-executor + env state reset around each
    test."""
    transformers = ModuleType("transformers")

    class AutoProcessor:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

    class AutoTokenizer:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

    class PreTrainedTokenizerBase:
        pass

    class ProcessorMixin:
        pass

    transformers.AutoProcessor = AutoProcessor
    transformers.AutoTokenizer = AutoTokenizer
    transformers.PreTrainedTokenizerBase = PreTrainedTokenizerBase
    transformers.ProcessorMixin = ProcessorMixin
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    module_name = "relax.utils.data.processing_utils"
    original_module = sys.modules.get(module_name)
    module_was_loaded = module_name in sys.modules
    data_module = sys.modules.get("relax.utils.data")
    processing_attr_was_set = data_module is not None and hasattr(data_module, "processing_utils")
    original_processing_attr = getattr(data_module, "processing_utils", None) if processing_attr_was_set else None

    sys.modules.pop(module_name, None)
    if data_module is not None and processing_attr_was_set:
        delattr(data_module, "processing_utils")
    mod = importlib.import_module(module_name)

    saved_env = os.environ.get(mod._ENCODE_MAX_WORKERS_ENV)
    saved_executor = mod._encode_executor
    saved_workers = mod._encode_executor_workers
    os.environ.pop(mod._ENCODE_MAX_WORKERS_ENV, None)
    mod._encode_executor = None
    mod._encode_executor_workers = None
    try:
        yield mod
    finally:
        if mod._encode_executor is not None and mod._encode_executor is not saved_executor:
            mod._encode_executor.shutdown(wait=False)
        mod._encode_executor = saved_executor
        mod._encode_executor_workers = saved_workers
        if saved_env is None:
            os.environ.pop(mod._ENCODE_MAX_WORKERS_ENV, None)
        else:
            os.environ[mod._ENCODE_MAX_WORKERS_ENV] = saved_env
        sys.modules.pop(module_name, None)
        if module_was_loaded:
            sys.modules[module_name] = original_module
        if data_module is not None:
            if processing_attr_was_set:
                setattr(data_module, "processing_utils", original_processing_attr)
            elif hasattr(data_module, "processing_utils"):
                delattr(data_module, "processing_utils")


def test_default_uses_affinity_aware_cpu_count(pu):
    workers, source = pu.resolve_encode_max_workers(None)
    assert source == "default"
    assert workers == min(32, pu._usable_cpu_count())
    assert 1 <= workers <= 32


def test_explicit_flag_takes_priority_over_env(pu):
    os.environ[pu._ENCODE_MAX_WORKERS_ENV] = "9"
    workers, source = pu.resolve_encode_max_workers(4)
    assert (workers, source) == (4, "flag")


def test_env_var_used_when_flag_unset(pu):
    os.environ[pu._ENCODE_MAX_WORKERS_ENV] = "7"
    workers, source = pu.resolve_encode_max_workers(None)
    assert workers == 7
    assert source == f"${pu._ENCODE_MAX_WORKERS_ENV}"


@pytest.mark.parametrize("bad", [0, -1, -32])
def test_non_positive_raises_value_error(pu, bad):
    with pytest.raises(ValueError):
        pu.resolve_encode_max_workers(bad)


@pytest.mark.parametrize("bad", [3.5, "8", True])
def test_non_int_raises_type_error(pu, bad):
    # bool is rejected explicitly even though it subclasses int.
    with pytest.raises(TypeError):
        pu.resolve_encode_max_workers(bad)


def test_illegal_env_value_raises(pu):
    os.environ[pu._ENCODE_MAX_WORKERS_ENV] = "0"
    with pytest.raises(ValueError):
        pu.resolve_encode_max_workers(None)
    os.environ[pu._ENCODE_MAX_WORKERS_ENV] = "not-an-int"
    with pytest.raises(ValueError):
        pu.resolve_encode_max_workers(None)


def test_configure_creates_pool_with_requested_size(pu):
    executor = pu.configure_encode_executor(5)
    assert executor._max_workers == 5
    assert pu.get_encode_executor() is executor


def test_configure_is_idempotent_for_same_size(pu):
    first = pu.configure_encode_executor(6)
    second = pu.configure_encode_executor(6)
    assert first is second


def test_reconfigure_to_different_size_raises(pu):
    pu.configure_encode_executor(4)
    with pytest.raises(RuntimeError, match="already initialized"):
        pu.configure_encode_executor(8)


def test_get_encode_executor_lazily_creates_default(pu):
    assert pu._encode_executor is None
    executor = pu.get_encode_executor()
    assert executor is not None
    assert executor._max_workers == min(32, pu._usable_cpu_count())


def test_shutdown_resets_state_and_allows_new_size(pu):
    pu.configure_encode_executor(4)
    pu.shutdown_encode_executor()
    assert pu._encode_executor is None
    assert pu._encode_executor_workers is None
    executor = pu.configure_encode_executor(8)
    assert executor._max_workers == 8
