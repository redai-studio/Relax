# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Live cache identity must stay exact while avoiding discarded deep copies."""

from types import MethodType

from transformers.processing_utils import ProcessorMixin

from relax.models.qwen_image.sft_adapter import _processor_cache_identity


class Component:
    copies = 0

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"Component({self.value!r})"

    def __deepcopy__(self, memo):
        type(self).copies += 1
        return Component(self.value)


class Processor(ProcessorMixin):
    attributes = ["tokenizer", "image_processor"]

    def __init__(self, tokenizer=None, image_processor=None, settings=None):
        self.tokenizer = tokenizer or Component("vocab")
        self.image_processor = image_processor or Component("resize")
        self.settings = settings or {"nested": [1, 2]}
        self.chat_template = "template"


def test_identity_exact_live_and_does_not_copy_or_modify_components():
    processor = Processor()
    for change in (
        lambda: None,
        lambda: setattr(processor.tokenizer, "value", "changed"),
        lambda: processor.settings["nested"].append(3),
        lambda: setattr(processor, "image_processor", Component("replacement")),
    ):
        change()
        expected = str(processor)
        before = Component.copies
        original_dict = processor.__dict__.copy()
        assert _processor_cache_identity(processor) == expected
        assert Component.copies == before
        assert processor.__dict__ == original_dict


def test_custom_serializers_and_str_keep_original_behavior():
    class CustomStr(Processor):
        def __str__(self):
            return "custom-str"

    class CustomDict(Processor):
        def to_dict(self, legacy_serialization=True):
            return {"custom": self.tokenizer.value}

    for processor in (CustomStr(), CustomDict(), "stub", None):
        assert _processor_cache_identity(processor) == str(processor)
    processor = Processor()
    processor.to_dict = MethodType(lambda self, **kwargs: {"instance": "override"}, processor)
    assert _processor_cache_identity(processor) == str(processor)
    processor = Processor()
    processor.to_json_string = MethodType(lambda self: "instance-json", processor)
    assert _processor_cache_identity(processor) == str(processor)
