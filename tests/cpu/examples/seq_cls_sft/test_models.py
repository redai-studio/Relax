# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest
import torch


try:
    import sglang.srt.models.qwen3_5  # noqa: F401
    from sglang.srt.layers.pooler import EmbeddingPoolerOutput, score_and_pool  # noqa: F401
    from sglang.srt.managers.multimodal_processor import PROCESSOR_MAPPING, import_processors
    from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration
except Exception as exc:
    pytest.skip(f"requires a Qwen3.5-compatible SGLang >= 0.5.12 image: {exc}", allow_module_level=True)

from examples.seq_cls_sft.models.sglang.model import (  # noqa: E402
    EntryClass,
    Qwen3_5ForSequenceClassification,
    Qwen3_5MoeForSequenceClassification,
    Qwen3VLForSequenceClassification,
    Qwen3VLSequenceClassificationProcessor,
)
from examples.seq_cls_sft.tools.launch_sglang_text_classification import (  # noqa: E402
    _is_multimodal_sequence_classifier,
)


def test_qwen3_5_sequence_classification_external_registry_names():
    assert EntryClass == [
        Qwen3_5ForSequenceClassification,
        Qwen3_5MoeForSequenceClassification,
        Qwen3VLForSequenceClassification,
    ]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("model.language_model.layers.0.self_attn.q_proj.weight", "layers.0.self_attn.q_proj.weight"),
        ("model.layers.0.mlp.down_proj.weight", "layers.0.mlp.down_proj.weight"),
        ("model.visual.blocks.0.weight", None),
        ("lm_head.weight", None),
        ("mtp.layers.0.weight", None),
    ],
)
def test_qwen3_5_sequence_classification_normalizes_backbone_weight_names(name, expected):
    assert Qwen3_5ForSequenceClassification._backbone_weight_name(name) == expected


def test_qwen3_vl_sequence_classifier_enables_multimodal_launcher(tmp_path):
    (tmp_path / "config.json").write_text(
        '{"architectures": ["Qwen3VLForSequenceClassification"]}',
        encoding="utf-8",
    )

    assert _is_multimodal_sequence_classifier(str(tmp_path)) is True


def test_qwen3_vl_sequence_classifier_registers_native_mm_processor():
    import_processors("examples.seq_cls_sft.models.sglang", overwrite=True)

    assert PROCESSOR_MAPPING[Qwen3VLForSequenceClassification] is Qwen3VLSequenceClassificationProcessor


def test_qwen3_5_sequence_classifier_keeps_text_only_launcher(tmp_path):
    (tmp_path / "config.json").write_text(
        '{"architectures": ["Qwen3_5ForSequenceClassification"]}',
        encoding="utf-8",
    )

    assert _is_multimodal_sequence_classifier(str(tmp_path)) is False


def test_qwen3_vl_sequence_classifier_scores_pooled_hidden_state(monkeypatch):
    model = Qwen3VLForSequenceClassification.__new__(Qwen3VLForSequenceClassification)
    torch.nn.Module.__init__(model)
    model.score = torch.nn.Linear(4, 2, bias=False)
    model.score.weight.data.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]))
    pooled_hidden = torch.tensor([[2.0, 3.0, 4.0, 5.0]])
    monkeypatch.setattr(
        Qwen3VLForConditionalGeneration,
        "forward",
        lambda *args, **kwargs: EmbeddingPoolerOutput(embeddings=pooled_hidden),
    )

    output = model.forward(
        input_ids=torch.tensor([1]),
        positions=torch.tensor([0]),
        forward_batch=object(),
    )

    assert torch.equal(output.embeddings, torch.tensor([[2.0, 3.0]]))


def test_qwen3_vl_sequence_classifier_loads_score_weight(monkeypatch):
    model = Qwen3VLForSequenceClassification.__new__(Qwen3VLForSequenceClassification)
    torch.nn.Module.__init__(model)
    model.score = torch.nn.Linear(4, 2, bias=False)
    backbone_weights = []
    monkeypatch.setattr(
        Qwen3VLForConditionalGeneration,
        "load_weights",
        lambda _self, weights: backbone_weights.extend(weights),
    )
    score = torch.arange(8, dtype=torch.float32).reshape(2, 4)

    loaded = model.load_weights(iter([("model.layer.weight", torch.ones(1)), ("score.weight", score)]))

    assert loaded == {"score.weight"}
    assert torch.equal(model.score.weight, score)
    assert [name for name, _weight in backbone_weights] == ["model.layer.weight"]
