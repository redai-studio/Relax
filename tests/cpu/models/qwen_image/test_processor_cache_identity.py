# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Live cache identity must stay exact while avoiding discarded deep copies."""

from copy import deepcopy
from inspect import signature
from types import MethodType

from transformers.image_processing_utils import BaseImageProcessor
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


class ImageComponent(BaseImageProcessor):
    copies = 0

    def __init__(self, value):
        super().__init__()
        self.value = value

    def __repr__(self):
        return f"ImageComponent({self.value!r})"

    def __deepcopy__(self, memo):
        type(self).copies += 1
        return ImageComponent(self.value)


class Processor(ProcessorMixin):
    attributes = ["tokenizer", "image_processor"]

    def __init__(self, tokenizer=None, image_processor=None, settings=None):
        self.tokenizer = tokenizer or Component("vocab")
        self.image_processor = image_processor or ImageComponent("resize")
        self.settings = settings or {"nested": [1, 2]}
        self.chat_template = "template"


def test_identity_exact_live_and_does_not_copy_or_modify_components():
    processor = Processor()
    for change in (
        lambda: None,
        lambda: setattr(processor.tokenizer, "value", "changed"),
        lambda: processor.settings["nested"].append(3),
        lambda: setattr(processor, "image_processor", ImageComponent("replacement")),
    ):
        change()
        native_before = ImageComponent.copies
        expected = str(processor)
        native_image_copies = ImageComponent.copies - native_before
        before = Component.copies, ImageComponent.copies
        original_dict = processor.__dict__.copy()
        component_states = {name: deepcopy(vars(getattr(processor, name))) for name in processor.attributes}
        settings = deepcopy(processor.settings)
        assert _processor_cache_identity(processor) == expected
        assert Component.copies == before[0]
        if "legacy_serialization" in signature(ProcessorMixin.to_dict).parameters:
            assert ImageComponent.copies == before[1]
        else:
            # New serializers retain the image config; the native fallback may
            # copy it, but must not add work beyond one native serialization.
            assert ImageComponent.copies - before[1] == native_image_copies
        assert processor.__dict__ == original_dict
        assert processor.settings == settings
        for name, state in component_states.items():
            assert getattr(processor, name) is original_dict[name]
            assert vars(getattr(processor, name)) == state


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
