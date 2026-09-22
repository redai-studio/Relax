# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from examples.seq_cls_sft.tools.convert_sequence_classification_checkpoint import (
    reconcile_export_index,
    resolve_iteration_dir,
    rewrite_task_head,
    update_classification_config,
    validate_export,
    validate_path_separation,
)
from examples.seq_cls_sft.tools.request_sequence_classification import sigmoid


def test_resolve_iteration_dir_uses_checkpoint_marker(tmp_path):
    iteration_dir = tmp_path / "iter_0000100"
    iteration_dir.mkdir()
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("100\n", encoding="utf-8")

    assert resolve_iteration_dir(tmp_path) == iteration_dir.resolve()
    assert resolve_iteration_dir(iteration_dir) == iteration_dir.resolve()


def test_rewrite_task_head_updates_shard_index_and_config(tmp_path):
    shard_name = "model-00001-of-00001.safetensors"
    score = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    save_file(
        {"model.language_model.norm.weight": torch.ones(4), "lm_head.weight": score},
        str(tmp_path / shard_name),
    )
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps(
            {
                "metadata": {"total_size": 64},
                "weight_map": {
                    "model.language_model.norm.weight": shard_name,
                    "lm_head.weight": shard_name,
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"]}),
        encoding="utf-8",
    )

    assert rewrite_task_head(tmp_path, num_labels=3) == (3, 4)
    architecture = update_classification_config(
        tmp_path,
        num_labels=3,
        problem_type="single_label_classification",
        label_names=["a", "b", "c"],
    )
    validate_export(tmp_path, num_labels=3)

    tensors = load_file(str(tmp_path / shard_name), device="cpu")
    assert "lm_head.weight" not in tensors
    assert torch.equal(tensors["score.weight"], score)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert "lm_head.weight" not in index["weight_map"]
    assert index["weight_map"]["score.weight"] == shard_name
    config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert architecture == "Qwen3_5ForSequenceClassification"
    assert config["architectures"] == [architecture]
    assert config["id2label"] == {"0": "a", "1": "b", "2": "c"}
    assert config["label2id"] == {"a": 0, "b": 1, "c": 2}
    assert config["normalize"] is False


def test_rewrite_task_head_adds_captured_head_when_tied_export_omits_lm_head(tmp_path):
    shard_name = "model-00001-of-00001.safetensors"
    save_file({"model.language_model.norm.weight": torch.ones(4)}, str(tmp_path / shard_name))
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps(
            {
                "metadata": {"total_size": 16},
                "weight_map": {"model.language_model.norm.weight": shard_name},
            }
        ),
        encoding="utf-8",
    )
    score = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    assert rewrite_task_head(tmp_path, num_labels=3, task_head=score) == (3, 4)
    validate_export(tmp_path, num_labels=3)

    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert index["weight_map"]["score.weight"] == "sequence-classification-head.safetensors"
    assert index["metadata"]["total_size"] == 64
    exported_head = load_file(str(tmp_path / "sequence-classification-head.safetensors"), device="cpu")
    assert torch.equal(exported_head["score.weight"], score)


def test_reconcile_export_index_drops_absent_mtp_ghosts(tmp_path):
    shard_name = "model-00001-of-00001.safetensors"
    save_file({"model.language_model.norm.weight": torch.ones(4)}, str(tmp_path / shard_name))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 16},
                "weight_map": {
                    "model.language_model.norm.weight": shard_name,
                    "mtp.layers.0.mlp.down_proj.weight": shard_name,
                },
            }
        ),
        encoding="utf-8",
    )

    summary = reconcile_export_index(tmp_path)

    assert summary["ghosts"] == ["mtp.layers.0.mlp.down_proj.weight"]
    assert summary["dropped"] == ["mtp.layers.0.mlp.down_proj.weight"]


def test_update_classification_config_detects_moe_and_validates_labels(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5", "text_config": {"model_type": "qwen3_5_moe_text"}}),
        encoding="utf-8",
    )

    architecture = update_classification_config(
        tmp_path,
        num_labels=2,
        problem_type="multi_label_classification",
        label_names=None,
    )
    assert architecture == "Qwen3_5MoeForSequenceClassification"

    with pytest.raises(ValueError, match="expected 2"):
        update_classification_config(
            tmp_path,
            num_labels=2,
            problem_type="multi_label_classification",
            label_names=["only-one"],
        )


def test_update_classification_config_supports_qwen3_vl(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_vl",
                "architectures": ["Qwen3VLForConditionalGeneration"],
                "text_config": {"model_type": "qwen3_vl_text"},
            }
        ),
        encoding="utf-8",
    )

    architecture = update_classification_config(
        tmp_path,
        num_labels=3,
        problem_type="single_label_classification",
        label_names=["a", "b", "c"],
    )

    config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert architecture == "Qwen3VLForSequenceClassification"
    assert config["architectures"] == [architecture]
    assert config["text_config"]["id2label"] == config["id2label"]
    assert config["text_config"]["label2id"] == config["label2id"]


def test_validate_path_separation_rejects_checkpoint_and_origin_overlap(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint" / "iter_0000100"
    origin_dir = tmp_path / "origin"
    checkpoint_dir.mkdir(parents=True)
    origin_dir.mkdir()

    with pytest.raises(ValueError, match="checkpoint"):
        validate_path_separation(checkpoint_dir / "export", checkpoint_dir, origin_dir)
    with pytest.raises(ValueError, match="origin HF"):
        validate_path_separation(origin_dir / "export", checkpoint_dir, origin_dir)


def test_sigmoid_is_stable_for_extreme_logits():
    assert sigmoid(-1000.0) == 0.0
    assert sigmoid(0.0) == 0.5
    assert sigmoid(1000.0) == 1.0
