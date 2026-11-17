# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Launch rejection and model layout regressions without GPU/model imports."""

from types import SimpleNamespace

import pytest

from relax.distributed.replay_manager.codecs import replay_codec
from relax.distributed.replay_manager.validation import (
    indexer_capture_layout,
    indexer_topk,
    validate_indexer_replay_args,
)


def valid_args(model_type, monkeypatch):
    import relax.utils.misc as misc

    monkeypatch.setattr(misc, "get_hf_config", lambda _: SimpleNamespace(model_type=model_type))
    keye = model_type == "KeyeVL2"
    return SimpleNamespace(
        hf_checkpoint="unused",
        loss_type="policy_loss",
        train_backend="megatron",
        multimodal_keys=None,
        colocate=True,
        partial_rollout=False,
        fully_async=False,
        hybrid=False,
        use_slime_router=False,
        use_agentic_rollout=False,
        rollout_function_path="relax.engine.rollout.sglang_rollout.generate_rollout",
        custom_generate_function_path=None,
        qkv_format="thd",
        context_parallel_size=2,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        dynamic_context_parallel=False,
        enable_mtp_training=False,
        sglang_speculative_algorithm=None,
        dsa_indexer_loss_coeff=0,
        recompute_granularity="full",
        recompute_method="uniform",
        cp_partition_mode="zigzag" if keye else "contiguous",
        allgather_cp=not keye,
        sglang_enable_dp_attention=not keye,
        pipeline_model_parallel_size=1,
        sglang_disable_overlap_schedule=True,
        sglang_disable_radix_cache=True,
    )


@pytest.mark.parametrize("model_type", ["KeyeVL2", "deepseek_v4"])
def test_indexer_validation_accepts_supported_launches(model_type, monkeypatch):
    validate_indexer_replay_args(valid_args(model_type, monkeypatch), is_offline=False)


@pytest.mark.parametrize(
    "model_type,field,value",
    [
        ("KeyeVL2", "cp_partition_mode", "contiguous"),
        ("deepseek_v4", "cp_partition_mode", "zigzag"),
        ("KeyeVL2", "sglang_enable_dp_attention", True),
        ("deepseek_v4", "sglang_enable_dp_attention", False),
        ("KeyeVL2", "pipeline_model_parallel_size", 2),
        ("KeyeVL2", "sglang_disable_overlap_schedule", False),
        ("KeyeVL2", "sglang_disable_radix_cache", False),
        ("deepseek_v4", "tensor_model_parallel_size", 2),
        ("KeyeVL2", "fully_async", True),
    ],
)
def test_indexer_validation_rejects_unsupported_layouts(model_type, field, value, monkeypatch):
    args = valid_args(model_type, monkeypatch)
    setattr(args, field, value)
    with pytest.raises(ValueError, match="unsupported settings"):
        validate_indexer_replay_args(args, is_offline=False)


def test_unknown_model_cannot_fall_back_to_dsv4(monkeypatch):
    with pytest.raises(ValueError, match="unsupported for model_type"):
        validate_indexer_replay_args(valid_args("unknown", monkeypatch), is_offline=False)
    with pytest.raises(ValueError, match="unsupported for model_type"):
        indexer_capture_layout(
            SimpleNamespace(
                hf_config=SimpleNamespace(model_type="unknown"),
                csa_compress_ratios=[4],
                dsa_indexer_topk=2,
                num_layers=1,
            )
        )


def test_capture_layout_preserves_dsv4_sparse_and_keye_all_layers():
    dsv4 = SimpleNamespace(csa_compress_ratios=[4, 128, 4], dsa_indexer_topk=512, num_layers=3)
    assert indexer_capture_layout(dsv4) == (512, (0, 2))
    keye = SimpleNamespace(hf_config=SimpleNamespace(model_type="KeyeVL2", sa_config={"topk": 2048}), num_layers=3)
    assert indexer_capture_layout(keye) == (2048, (0, 1, 2))
    dsv4.csa_compress_ratios = [4]
    with pytest.raises(ValueError, match="shorter"):
        indexer_capture_layout(dsv4)


def test_codec_contract_rejects_wrong_kind_topk_and_width():
    codec = replay_codec("keye-bitmap-v1")
    codec.validate("indexer", 2048)
    codec.validate_width(1)
    codec.validate_width(1025)
    for kind, topk in [("moe", 2048), ("indexer", 1024)]:
        with pytest.raises(ValueError, match="shape/kind"):
            codec.validate(kind, topk)
    with pytest.raises(ValueError, match="row width"):
        codec.validate_width(1026)
    qsa = replay_codec("qsa-r4-bitmap-v1")
    qsa.validate("indexer", 512)
    qsa.validate_width(2049)
    with pytest.raises(ValueError, match="shape/kind"):
        qsa.validate("indexer", 2048)


def test_dsv4_does_not_inherit_keye_only_restrictions(monkeypatch):
    args = valid_args("deepseek_v4", monkeypatch)
    args.pipeline_model_parallel_size = 2
    args.sglang_disable_overlap_schedule = False
    args.sglang_disable_radix_cache = False
    validate_indexer_replay_args(args, is_offline=False)


def test_capture_topk_uses_registered_hf_layout():
    assert indexer_topk(SimpleNamespace(model_type="deepseek_v4", index_topk=512)) == 512
    text = SimpleNamespace(model_type="KeyeVL2", sa_config={"topk": 2048})
    assert indexer_topk(SimpleNamespace(text_config=text)) == 2048


def test_qsa_reference_layout_uses_blocks_and_sparse_layers():
    text = SimpleNamespace(model_type="qwen4_exp_text", indexer_budget=2048, indexer_compress_ratio=4)
    assert indexer_topk(SimpleNamespace(text_config=text)) == 512
    config = SimpleNamespace(
        num_layers=8,
        qsa_indexer_n_heads=4,
        qsa_indexer_budget=2048,
        qsa_indexer_compress_ratio=4,
        qsa_indexer_loss_coeff=0,
        linear_attention_freq=[1, 1, 1, 0, 1, 1, 1, 0],
    )
    assert indexer_capture_layout(config) == (512, (3, 7))
    config.qsa_indexer_loss_coeff = 1
    with pytest.raises(ValueError, match="zero indexer loss"):
        indexer_capture_layout(config)


def test_qsa_reference_accepts_tp4_cp1_and_rejects_full_delivery(monkeypatch):
    monkeypatch.setenv("RELAX_REPLAY_REFERENCES", "1")
    monkeypatch.setenv("RELAX_QWEN38_QSA_COMMUNITY_BACKEND", "mixed")
    monkeypatch.setenv("RELAX_QWEN38_QSA_FUSED_ATTENTION", "0")
    monkeypatch.setenv("RELAX_QWEN38_QSA_TP_SELECTOR", "1")
    monkeypatch.setenv("RELAX_QWEN38_QSA_FUSED_SELECTOR", "0")
    args = valid_args("qwen4_exp", monkeypatch)
    args.context_parallel_size = 1
    args.tensor_model_parallel_size = 4
    args.sequence_parallel = True
    args.cp_partition_mode = "zigzag"
    args.allgather_cp = False
    args.sglang_enable_dp_attention = False
    args.pipeline_model_parallel_size = 4
    validate_indexer_replay_args(args, is_offline=False)
    monkeypatch.setenv("RELAX_REPLAY_REFERENCES", "0")
    with pytest.raises(ValueError, match="reference delivery"):
        validate_indexer_replay_args(args, is_offline=False)


def test_qsa_reference_rejects_cp2_and_conflicting_attention(monkeypatch):
    monkeypatch.setenv("RELAX_REPLAY_REFERENCES", "1")
    monkeypatch.setenv("RELAX_QWEN38_QSA_COMMUNITY_BACKEND", "mixed")
    monkeypatch.setenv("RELAX_QWEN38_QSA_FUSED_ATTENTION", "1")
    args = valid_args("qwen4_exp", monkeypatch)
    with pytest.raises(ValueError, match="requires context_parallel_size=1.*fused attention disabled"):
        validate_indexer_replay_args(args, is_offline=False)
