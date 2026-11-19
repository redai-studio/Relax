# Copyright (c) 2026 Relax Authors. All Rights Reserved.
import pytest
import torch

from relax.utils.training.prefix_runtime import create_prefix_runtime, record_host_prefix_inputs


def test_baseline_does_not_add_prefix_metadata(monkeypatch):
    monkeypatch.setenv("RELAX_TRAIN_PREFIX_MODE", "baseline")
    data = {"tokens": [[1, 2, 3]]}
    record_host_prefix_inputs(data)
    assert data == {"tokens": [[1, 2, 3]]}
    assert create_prefix_runtime() is None


def test_host_metadata_preserves_loss_predecessor_and_identity(monkeypatch):
    monkeypatch.setenv("RELAX_TRAIN_PREFIX_MODE", "detached_prefix")
    tokens = torch.arange(10)
    data = {
        "tokens": [tokens],
        "total_lengths": [10],
        "response_lengths": [4],
        "sample_indices": [17],
        "loss_masks": [[1, 0, 1, 1]],
        "rewards": [0.5],
    }
    record_host_prefix_inputs(data)
    assert data["prefix_inputs"] == [(17, tuple(range(10)), 5)]
    assert data["tokens"][0] is tokens
    assert data["loss_masks"] == [[1, 0, 1, 1]]
    assert data["rewards"] == [0.5]


def test_unknown_mode_fails(monkeypatch):
    monkeypatch.setenv("RELAX_TRAIN_PREFIX_MODE", "reuse_typo")
    with pytest.raises(ValueError, match="Unknown prefix mode"):
        create_prefix_runtime()


def test_logprob_only_fetch_does_not_require_training_identity(monkeypatch):
    monkeypatch.setenv("RELAX_TRAIN_PREFIX_MODE", "detached_prefix_reuse")
    data = {"tokens": [[1, 2, 3]], "total_lengths": [3], "response_lengths": [1]}
    record_host_prefix_inputs(data)
    assert "prefix_inputs" not in data


@pytest.mark.parametrize("cp,pp,message", [(2, 1, "CP state redistribution"), (1, 2, "PP state scheduling")])
def test_unsupported_state_distribution_fails_explicitly(cp, pp, message):
    from types import SimpleNamespace

    from relax.utils.training.qwen35_prefix_decoder import validate_decoder

    config = SimpleNamespace(
        window_size=None,
        softmax_type="vanilla",
        hidden_dropout=0,
        attention_dropout=0,
        context_parallel_size=cp,
        pipeline_model_parallel_size=pp,
    )
    with pytest.raises(NotImplementedError, match=message):
        validate_decoder(SimpleNamespace(config=config, layers=[]))
