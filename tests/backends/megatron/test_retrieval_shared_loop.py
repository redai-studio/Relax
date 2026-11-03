# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Production iterator/loss dispatch with CPU stubs for MCore and TQ
transport."""

from argparse import Namespace

import pytest
import torch
import torch.nn.functional as F

from relax.engine.sft.runtime import should_bypass_main_output_layer
from relax.utils.training.data_fields import build_data_fields
from relax.utils.training.retrieval_utils import freeze_retrieval_unused_heads, validate_sequence_contrast_args
from tests.backends.megatron.test_data_vpp import _load_data_module
from tests.backends.megatron.test_rloo_policy_loss_dispatch import loss_module
from tests.utils.data.test_streaming_tq_iterator import _load_stream_module


def _args(**overrides):
    values = dict(
        loss_type="sft",
        task_type="seq_contrast",
        train_backend="megatron",
        custom_dataset_class_path="relax.engine.sft.dataset.retrieval.NLIPairDataset",
        global_batch_size=4,
        n_samples_per_prompt=1,
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=128,
        data_pad_size_multiplier=8,
        use_gloo_process_groups=True,
        contrastive_temperature=0.05,
        retrieval_emb_token_id=99,
        qkv_format="thd",
        calculate_per_token_loss=False,
        allgather_cp=False,
        recompute_loss_function=False,
        use_opd=False,
    )
    values.update(overrides)
    return Namespace(**values)


@pytest.mark.parametrize(
    "overrides",
    [
        {"loss_type": "dpo"},
        {"train_backend": "fsdp"},
        {"megatron_to_hf_mode": "raw"},
        {"context_parallel_size": 2},
        {"sequence_parallel": True},
        {"sft_oversize_strategy": "drop"},
        {"eval_size": 0},
        {"eval_interval": 1},
        {"recompute_loss_function": True},
        {"num_steps_per_rollout": 2},
        {"n_samples_per_prompt": 2},
        {"sft_training_mode": "cpt"},
        {"use_dynamic_batch_size": False},
        {"contrastive_temperature": float("nan")},
    ],
)
def test_unsupported_contracts_fail_before_job_creation(overrides):
    with pytest.raises(ValueError):
        validate_sequence_contrast_args(_args(**overrides))


def test_runtime_reuses_sft_fields_and_hidden_state_bypass():
    args = _args()
    validate_sequence_contrast_args(args)
    assert should_bypass_main_output_layer(args)
    assert set(build_data_fields(args)) == {
        "tokens",
        "loss_masks",
        "total_lengths",
        "response_lengths",
        "multimodal_train_inputs",
    }
    assert not should_bypass_main_output_layer(_args(task_type="causal_lm"))
    assert not should_bypass_main_output_layer(_args(loss_type="policy_loss"))


@pytest.mark.parametrize("peer", [None, (1, 4, 16, 128), (2, 4, 136, 128)])
def test_shared_iterator_keeps_whole_negative_set_or_rejects(monkeypatch, peer):
    data = _load_data_module(monkeypatch)
    monkeypatch.setattr(data.mpu, "get_data_parallel_world_size", lambda **_kw: 2, raising=False)
    group = object()
    monkeypatch.setattr(data.mpu, "get_data_parallel_group_gloo", lambda **_kw: group, raising=False)

    def gather(output, value, *, group):
        output[:] = [value, value if peer is None else peer]

    monkeypatch.setattr(data.dist, "all_gather_object", gather)
    rows = {"retrieval_pair_count": 2, "tokens": [torch.tensor([1, 99])] * 4, "total_lengths": [2] * 4}
    if peer is not None:
        with pytest.raises(ValueError, match="seq_contrast|contrastive batch"):
            data.get_data_iterator(_args(), None, rows)
        return
    iterators, microbatches = data.get_data_iterator(_args(), None, rows)
    assert microbatches == [1]
    assert len(iterators) == 1 and isinstance(iterators[0], data.DataIterator)
    assert len(iterators[0].get_next(["tokens"])["tokens"]) == 4
    assert rows["dynamic_global_batch_size"] == 4  # pairs, not 8 expanded sequences


def test_production_loss_dispatch_preserves_mean_and_hidden_gradients(monkeypatch):
    # Exercise the actual outer Megatron loss wrapper and CP=1 reducers. These
    # stubs supply CPU process-group metadata; no MCore model is simulated.
    cp_mpu = loss_module.get_sum_of_sample_mean.__globals__["mpu"]
    for mpu in (loss_module.mpu, cp_mpu):
        monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: 1, raising=False)
        monkeypatch.setattr(mpu, "get_data_parallel_world_size", lambda **_kw: 1, raising=False)
        monkeypatch.setattr(mpu, "get_data_parallel_group", lambda **_kw: object(), raising=False)
    monkeypatch.setattr(loss_module.dist, "get_world_size", lambda group: 1)
    hidden = torch.randn(1, 16, 5, dtype=torch.float64, requires_grad=True)
    tokens = [torch.tensor([index, 99]) for index in range(8)]
    batch = dict(
        unconcat_tokens=tokens, total_lengths=[2] * 8, response_lengths=[2] * 8, loss_masks=[torch.ones(2)] * 8
    )
    loss, normalizer, metrics = loss_module.loss_function(_args(), batch, 1, hidden)
    features = hidden[0, ::2]
    query, positive = features[::2], features[1::2]
    reference = F.cross_entropy(
        F.cosine_similarity(F.normalize(query, dim=-1)[:, None], F.normalize(positive, dim=-1)[None], dim=-1) / 0.05,
        torch.arange(4),
    )
    torch.testing.assert_close(loss, reference, rtol=0, atol=0)
    assert normalizer.item() == 1
    assert metrics["keys"] == ["loss"]
    torch.testing.assert_close(metrics["values"][1] / 4, reference.detach(), rtol=0, atol=0)
    actual_gradient = torch.autograd.grad(loss, hidden, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(reference, hidden)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)
    with pytest.raises(ValueError, match="complete contrastive microbatch"):
        loss_module.loss_function(_args(), batch, 2, hidden)


def test_retrieval_freezes_unused_vocab_but_retains_language_transformer():
    language = torch.nn.Module()
    language.embedding = torch.nn.Embedding(8, 4)
    language.output_layer = torch.nn.Linear(4, 8, bias=False)
    language.decoder = torch.nn.Linear(4, 4)
    wrapper = torch.nn.Module()
    wrapper.language_model = language
    freeze_retrieval_unused_heads(wrapper)
    assert not any(param.requires_grad for param in language.embedding.parameters())
    assert not any(param.requires_grad for param in language.output_layer.parameters())
    assert all(param.requires_grad for param in language.decoder.parameters())


def test_pair_postprocessing_reaches_existing_packed_batch(monkeypatch):
    data = _load_data_module(monkeypatch)
    stream_dataloader = _load_stream_module(monkeypatch)

    args = _args(global_batch_size=2)
    for name, value in {
        "get_context_parallel_world_size": lambda: 1,
        "get_context_parallel_rank": lambda: 0,
        "get_tensor_model_parallel_world_size": lambda: 1,
        "get_data_parallel_world_size": lambda **_kw: 1,
        "get_data_parallel_group_gloo": lambda **_kw: object(),
    }.items():
        monkeypatch.setattr(data.mpu, name, value, raising=False)
        monkeypatch.setattr(stream_dataloader.mpu, name, value, raising=False)
    monkeypatch.setattr(data, "get_args", lambda: args)
    monkeypatch.setattr(data, "PackedSeqParams", Namespace)
    monkeypatch.setattr(data.dist, "all_gather_object", lambda output, value, **_kw: output.__setitem__(0, value))
    monkeypatch.setattr(stream_dataloader.device_utils, "make_current_torch_device", lambda: torch.device("cpu"))
    rows = {
        "multimodal_train_inputs": [
            {"query": {"input_ids": [1, 2, 99]}, "positive": {"input_ids": [3, 99]}},
            {"query": {"input_ids": [4, 99]}, "positive": {"input_ids": [5, 6, 99]}},
        ]
    }
    stream_dataloader.post_process_rollout_data(args, rows)
    iterators, microbatches = data.get_data_iterator(args, None, rows)
    packed = data.get_batch(iterators[0], build_data_fields(args), pad_multiplier=8, pack_device=torch.device("cpu"))
    assert microbatches == [1]
    assert packed["tokens"].tolist() == [[1, 2, 99, 3, 99, 4, 99, 5, 6, 99] + [0] * 6]
    assert packed["dynamic_global_batch_size"] == 2
    assert packed["packed_seq_params"].cu_seqlens_q_cpu == [0, 3, 5, 7, 10, 16]
    assert [value.tolist() for value in packed["unconcat_tokens"]] == [[1, 2, 99], [3, 99], [4, 99], [5, 6, 99]]
