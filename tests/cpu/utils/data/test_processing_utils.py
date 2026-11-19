# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for relax.utils.data.processing_utils.adapt_processor_kwargs.

Imports are deferred to fixtures because processing_utils pulls in the heavy
imageio / soundfile / transformers / torch stack at module level, which trips a
numpy ABI mismatch in this CI image during pytest collection.
"""

import importlib
import sys
from types import ModuleType

import pytest


@pytest.fixture()
def processing_utils_module(monkeypatch):
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
    module = importlib.import_module(module_name)
    try:
        yield module
    finally:
        sys.modules.pop(module_name, None)
        if module_was_loaded:
            sys.modules[module_name] = original_module
        if data_module is not None:
            if processing_attr_was_set:
                setattr(data_module, "processing_utils", original_processing_attr)
            elif hasattr(data_module, "processing_utils"):
                delattr(data_module, "processing_utils")


@pytest.fixture()
def adapt_processor_kwargs(processing_utils_module):
    fn = processing_utils_module.adapt_processor_kwargs

    return fn


class _FakeQwenVLProcessor:
    """Stand-in for a standard HF VLM processor (Qwen-VL / Qwen-Omni shape)."""

    def __call__(self, text=None, images=None, videos=None, audio=None, **kwargs):
        raise AssertionError("not invoked in tests")


class KimiK25Processor:
    """Class-name match for the K2.x adapter branch — body is irrelevant."""

    def __call__(self, messages=None, medias=None, text=None, return_tensors="pt", **kwargs):
        raise AssertionError("not invoked in tests")


class KimiK26Processor(KimiK25Processor):
    """Future K2.x variants must keep getting the K2 adapter via class-name
    prefix."""


def test_adapt_processor_kwargs_default_passthrough(adapt_processor_kwargs):
    proc = _FakeQwenVLProcessor()
    mm = {"images": ["pil_img_1"], "videos": [], "audio": []}
    extra = {"return_tensors": None, "images_kwargs": {"return_tensors": "pt"}}

    out = adapt_processor_kwargs(proc, mm, extra)

    assert out == {**mm, **extra}, "Non-K2 processors must see the original shape unchanged"


def test_adapt_processor_kwargs_default_handles_none_inputs(adapt_processor_kwargs):
    proc = _FakeQwenVLProcessor()
    assert adapt_processor_kwargs(proc, None, None) == {}
    assert adapt_processor_kwargs(proc, None, {"foo": 1}) == {"foo": 1}
    assert adapt_processor_kwargs(proc, {"images": ["x"]}, None) == {"images": ["x"]}


def test_adapt_processor_kwargs_kimi_k25_translates_images_to_medias(adapt_processor_kwargs):
    proc = KimiK25Processor()
    mm = {"images": ["pil_a", "pil_b"], "videos": [], "audio": []}

    out = adapt_processor_kwargs(proc, mm, extra_kwargs={"return_tensors": None})

    assert out == {
        "medias": [
            {"type": "image", "image": "pil_a"},
            {"type": "image", "image": "pil_b"},
        ],
        # Forced "pt" wins over the None coming from build_processor_kwargs;
        # the K2.x processor's tokenizer call needs real tensors.
        "return_tensors": "pt",
    }
    assert "images" not in out and "videos" not in out and "audio" not in out


def test_adapt_processor_kwargs_kimi_future_variant_uses_same_branch(adapt_processor_kwargs):
    proc = KimiK26Processor()
    out = adapt_processor_kwargs(proc, {"images": ["x"]}, None)
    assert out["medias"] == [{"type": "image", "image": "x"}]


def test_adapt_processor_kwargs_kimi_no_images_returns_only_return_tensors(adapt_processor_kwargs):
    proc = KimiK25Processor()
    # No medias → caller relies on text-only K2 branch (still legal as long as text is provided).
    out = adapt_processor_kwargs(proc, {"images": [], "videos": [], "audio": []}, None)
    assert out == {"return_tensors": "pt"}


def test_adapt_processor_kwargs_kimi_warns_on_unsupported_modalities(
    monkeypatch, processing_utils_module, adapt_processor_kwargs
):
    proc = KimiK25Processor()
    mm = {"images": ["x"], "videos": ["v"], "audio": ["a"]}
    warnings = []
    monkeypatch.setattr(processing_utils_module.logger, "warning", lambda msg: warnings.append(msg))
    out = adapt_processor_kwargs(proc, mm, None)

    msgs = " ".join(warnings)
    assert "video" in msgs.lower()
    assert "audio" in msgs.lower()
    assert "videos" not in out
    assert "audio" not in out
    assert out["medias"] == [{"type": "image", "image": "x"}]


def test_adapt_processor_kwargs_kimi_drops_conflicting_extra_kwargs(adapt_processor_kwargs):
    """build_processor_kwargs adds images_kwargs / videos_kwargs /
    audio_kwargs.

    + return_tensors=None.

    K2.x ignores the per-modality dicts via **kwargs but would crash on
    duplicate return_tensors; the adapter must own the kwarg space.
    """
    proc = KimiK25Processor()
    extra = {
        "return_tensors": None,
        "images_kwargs": {"return_tensors": "pt"},
        "videos_kwargs": {"return_tensors": "pt"},
        "audio_kwargs": {"return_tensors": "pt"},
    }
    out = adapt_processor_kwargs(proc, {"images": ["x"]}, extra)
    assert out == {
        "medias": [{"type": "image", "image": "x"}],
        "return_tensors": "pt",
    }


@pytest.mark.parametrize(
    "cls_name, expected_kimi",
    [
        ("KimiK25Processor", True),
        ("KimiK26Processor", True),
        ("KimiK2Processor", True),
        ("Qwen2VLProcessor", False),
        ("Qwen3OmniProcessor", False),
        ("KimiAudio", False),  # doesn't end with Processor
        ("SomeOtherKimiK2Tokenizer", False),  # doesn't end with Processor
    ],
)
def test_adapt_processor_kwargs_class_name_match(cls_name, expected_kimi, adapt_processor_kwargs):
    fake_cls = type(cls_name, (), {"__call__": lambda self, **kw: None})
    proc = fake_cls()
    out = adapt_processor_kwargs(proc, {"images": ["x"]}, None)
    if expected_kimi:
        assert "medias" in out
        assert "images" not in out
    else:
        assert "images" in out
        assert "medias" not in out
