# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Regression tests for S3 model paths in agentic compiler and SFT consumers.

Each site must resolve an S3-selected ``hf_checkpoint`` to a shm path before
handing it to tokenizer and processor loaders.
"""

import importlib
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

from relax.agentic.pipeline import runtime
from relax.engine.sft.predict import loop as predict_loop
from relax.utils import s3_model_loader


def _model_source(uri: str, *, enabled: bool = True):
    if not enabled:
        return None
    return s3_model_loader.ModelSource(uri, "http://s3.example")


@pytest.fixture()
def processor_pool_module(monkeypatch):
    module = ModuleType("relax.utils.data.processor_pool")
    module.ProcessorPool = object
    monkeypatch.setitem(sys.modules, "relax.utils.data.processor_pool", module)
    return module


@pytest.fixture()
def processing_utils_module(monkeypatch):
    module = ModuleType("relax.utils.data.processing_utils")
    module.configure_encode_executor = MagicMock(return_value="EXECUTOR")
    module.load_processor = MagicMock(return_value="PROCESSOR")
    module.load_tokenizer = MagicMock(return_value="TOKENIZER")
    monkeypatch.setitem(sys.modules, "relax.utils.data.processing_utils", module)
    return module


@pytest.fixture()
def sft_module(monkeypatch, processor_pool_module):
    module_name = "relax.components.sft"
    original_module = sys.modules.get(module_name)
    module_was_loaded = module_name in sys.modules
    components_module = sys.modules.get("relax.components")
    sft_attr_was_set = components_module is not None and hasattr(components_module, "sft")
    original_sft_attr = getattr(components_module, "sft", None) if sft_attr_was_set else None

    sys.modules.pop(module_name, None)
    if components_module is not None and sft_attr_was_set:
        delattr(components_module, "sft")
    module = importlib.import_module(module_name)
    try:
        yield module
    finally:
        sys.modules.pop(module_name, None)
        if module_was_loaded:
            sys.modules[module_name] = original_module
        if components_module is not None:
            if sft_attr_was_set:
                setattr(components_module, "sft", original_sft_attr)
            elif hasattr(components_module, "sft"):
                delattr(components_module, "sft")


def test_load_agentic_compiler_resources_resolves_s3_path(monkeypatch, processing_utils_module, processor_pool_module):
    captured = {}

    def fake_update(obj, **kwargs):
        captured["resolve_args"] = obj
        captured["resolve_kwargs"] = kwargs
        obj.hf_checkpoint = "/dev/shm/resolved_model"

    def fake_pool(**kwargs):
        captured["pool_kwargs"] = kwargs
        return "POOL"

    monkeypatch.setattr(runtime, "prepare_model_maybe_update_args", fake_update)
    monkeypatch.setattr(processor_pool_module, "ProcessorPool", fake_pool)

    args = SimpleNamespace(
        mm_processor_pool_size=2,
        encode_max_workers=4,
        hf_checkpoint="s3://bkt/model/",
        model_source=_model_source("s3://bkt/model/"),
    )
    resources = runtime.load_agentic_compiler_resources(args)

    assert resources.processor_pool == "POOL"
    assert captured["resolve_args"] is args
    assert captured["resolve_kwargs"] == {"completeness": "metadata"}
    processing_utils_module.load_tokenizer.assert_called_once_with("/dev/shm/resolved_model", trust_remote_code=True)
    processing_utils_module.load_processor.assert_called_once_with("/dev/shm/resolved_model", trust_remote_code=True)
    assert captured["pool_kwargs"]["model_path"] == "/dev/shm/resolved_model"
    assert args.hf_checkpoint == "/dev/shm/resolved_model"


def test_load_agentic_compiler_resources_noop_when_S3_disabled(
    monkeypatch, processing_utils_module, processor_pool_module
):
    captured = {}

    def fake_update(obj, **kwargs):
        captured["resolve_args"] = obj
        captured["resolve_kwargs"] = kwargs

    def fake_pool(**kwargs):
        captured["pool_kwargs"] = kwargs
        return "POOL"

    monkeypatch.setattr(runtime, "prepare_model_maybe_update_args", fake_update)
    monkeypatch.setattr(processor_pool_module, "ProcessorPool", fake_pool)

    args = SimpleNamespace(
        mm_processor_pool_size=1,
        encode_max_workers=4,
        hf_checkpoint="s3://bkt/model/",
        model_source=None,
    )
    runtime.load_agentic_compiler_resources(args)

    assert captured["resolve_args"] is args
    assert captured["resolve_kwargs"] == {"completeness": "metadata"}
    processing_utils_module.load_tokenizer.assert_called_once_with("s3://bkt/model/", trust_remote_code=True)
    processing_utils_module.load_processor.assert_called_once_with("s3://bkt/model/", trust_remote_code=True)
    assert captured["pool_kwargs"]["model_path"] == "s3://bkt/model/"


class _FakeDataset:
    def __len__(self):
        return 4

    def shuffle(self, *args, **kwargs):
        return None


def _make_sft_instance(config, sft_module):
    # SFT is a @serve.deployment; grab the underlying class and bypass __init__
    # so we can exercise _init_data_pipeline in isolation.
    cls = sft_module.SFT.func_or_class
    obj = object.__new__(cls)
    obj._dataset = None
    obj.config = config
    obj.step = 0
    # ``_logger`` is a read-only cached property on Base; seed its backing field.
    obj._logger_instance = MagicMock()
    return obj


def test_sft_init_data_pipeline_resolves_s3_path(monkeypatch, sft_module):
    captured = {}

    def fake_update(obj, **kwargs):
        captured["resolve_args"] = obj
        captured["resolve_kwargs"] = kwargs
        obj.hf_checkpoint = "/dev/shm/resolved_sft"

    monkeypatch.setattr(sft_module, "prepare_model_maybe_update_args", fake_update)
    monkeypatch.setattr(sft_module, "AutoTokenizer", MagicMock())
    monkeypatch.setattr(sft_module, "ProcessorPool", MagicMock())
    monkeypatch.setattr(sft_module, "_resolve_pad_token_ids_from_config", MagicMock(return_value=frozenset()))
    monkeypatch.setattr(sft_module, "build_named_prompt_data_configs", MagicMock(return_value=[]))

    fake_cls = MagicMock()
    fake_cls.from_args = MagicMock(return_value=_FakeDataset())
    monkeypatch.setattr(sft_module, "_load_custom_dataset_class", MagicMock(return_value=fake_cls))

    config = SimpleNamespace(
        hf_checkpoint="s3://bkt/sftmodel/",
        model_source=_model_source("s3://bkt/sftmodel/"),
        context_parallel_size=1,
        max_tokens_per_gpu=8,
        sft_prefetch_buffer_size=1,
        sft_prefetch_chunk_size=1,
        sft_prefetch_num_workers=1,
        seed=0,
        sft_oversize_strategy="keep",
        custom_dataset_class_path="pkg.Custom",
        eval_prompt_data=None,
        eval_size=None,
        global_batch_size=1,
    )
    obj = _make_sft_instance(config, sft_module)

    obj._init_data_pipeline()

    resolved = "/dev/shm/resolved_sft"
    assert captured["resolve_args"] is config
    assert captured["resolve_kwargs"] == {"completeness": "metadata"}
    sft_module.AutoTokenizer.from_pretrained.assert_called_once_with(resolved, trust_remote_code=True)
    sft_module.ProcessorPool.assert_called_once_with(
        resolved,
        pool_size=None,
        trust_remote_code=True,
        multimodal_config=sft_module.MultimodalConfig.from_args(config),
    )
    sft_module._resolve_pad_token_ids_from_config.assert_called_once_with(resolved)
    assert config.hf_checkpoint == resolved


def test_sft_init_data_pipeline_noop_when_S3_disabled(monkeypatch, sft_module):
    captured = {}

    def fake_update(obj, **kwargs):
        captured["resolve_args"] = obj
        captured["resolve_kwargs"] = kwargs

    monkeypatch.setattr(sft_module, "prepare_model_maybe_update_args", fake_update)
    monkeypatch.setattr(sft_module, "AutoTokenizer", MagicMock())
    monkeypatch.setattr(sft_module, "ProcessorPool", MagicMock())
    monkeypatch.setattr(sft_module, "_resolve_pad_token_ids_from_config", MagicMock(return_value=frozenset()))
    monkeypatch.setattr(sft_module, "build_named_prompt_data_configs", MagicMock(return_value=[]))

    fake_cls = MagicMock()
    fake_cls.from_args = MagicMock(return_value=_FakeDataset())
    monkeypatch.setattr(sft_module, "_load_custom_dataset_class", MagicMock(return_value=fake_cls))

    config = SimpleNamespace(
        hf_checkpoint="s3://bkt/sftmodel/",
        model_source=None,
        context_parallel_size=1,
        max_tokens_per_gpu=8,
        sft_prefetch_buffer_size=1,
        sft_prefetch_chunk_size=1,
        sft_prefetch_num_workers=1,
        seed=0,
        sft_oversize_strategy="keep",
        custom_dataset_class_path="pkg.Custom",
        eval_prompt_data=None,
        eval_size=None,
        global_batch_size=1,
    )
    obj = _make_sft_instance(config, sft_module)

    obj._init_data_pipeline()

    assert captured["resolve_args"] is config
    assert captured["resolve_kwargs"] == {"completeness": "metadata"}
    sft_module.AutoTokenizer.from_pretrained.assert_called_once_with("s3://bkt/sftmodel/", trust_remote_code=True)


class _EmptyEvalDataset:
    """Stub SFTStreamingDataset that renders zero eval prompts (clean early
    exit)."""

    def __init__(self, *args, **kwargs):
        pass

    def __len__(self):
        return 0

    def stop(self):
        return None


def _predict_config(*, enabled: bool):
    uri = "s3://bkt/predictmodel/"
    return SimpleNamespace(
        hf_checkpoint=uri,
        model_source=_model_source(uri, enabled=enabled),
        context_parallel_size=1,
        max_tokens_per_gpu=8,
        seed=0,
        prompt_data="/data/train.jsonl",
        eval_prompt_data=None,
        eval_size=1,
        input_key="messages",
        label_key="label",
        multimodal_keys=None,
        conversation_key_map=None,
        metadata_key="metadata",
        tool_key="tools",
        system_prompt=None,
        apply_chat_template_kwargs=None,
    )


def test_predict_render_eval_prompts_resolves_s3_path(monkeypatch):
    captured = {}

    def fake_update(obj, **kwargs):
        captured["resolve_args"] = obj
        captured["resolve_kwargs"] = kwargs
        obj.hf_checkpoint = "/dev/shm/resolved_predict"

    fake_tok_cls = MagicMock()
    monkeypatch.setattr(predict_loop, "prepare_model_maybe_update_args", fake_update)
    monkeypatch.setattr("transformers.AutoTokenizer", fake_tok_cls)
    monkeypatch.setattr("relax.engine.sft.dataset.streaming.SFTStreamingDataset", _EmptyEvalDataset)
    monkeypatch.setattr(predict_loop, "build_named_prompt_data_configs", MagicMock(return_value=[]))

    config = _predict_config(enabled=True)
    out = predict_loop.render_eval_prompts(config)

    assert out == []
    assert captured["resolve_args"] is config
    assert captured["resolve_kwargs"] == {"completeness": "metadata"}
    fake_tok_cls.from_pretrained.assert_called_once_with("/dev/shm/resolved_predict", trust_remote_code=True)
    assert config.hf_checkpoint == "/dev/shm/resolved_predict"


def test_predict_render_eval_prompts_noop_when_S3_disabled(monkeypatch):
    captured = {}

    def fake_update(obj, **kwargs):
        captured["resolve_args"] = obj
        captured["resolve_kwargs"] = kwargs

    fake_tok_cls = MagicMock()
    monkeypatch.setattr(predict_loop, "prepare_model_maybe_update_args", fake_update)
    monkeypatch.setattr("transformers.AutoTokenizer", fake_tok_cls)
    monkeypatch.setattr("relax.engine.sft.dataset.streaming.SFTStreamingDataset", _EmptyEvalDataset)
    monkeypatch.setattr(predict_loop, "build_named_prompt_data_configs", MagicMock(return_value=[]))

    config = _predict_config(enabled=False)
    out = predict_loop.render_eval_prompts(config)

    assert out == []
    assert captured["resolve_args"] is config
    assert captured["resolve_kwargs"] == {"completeness": "metadata"}
    fake_tok_cls.from_pretrained.assert_called_once_with("s3://bkt/predictmodel/", trust_remote_code=True)
