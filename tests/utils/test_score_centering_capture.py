# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import ast
import asyncio
import copy
import importlib
import math
import pickle
import sys
from pathlib import Path
from time import monotonic
from types import ModuleType, SimpleNamespace

import numpy as np
import pybase64
import pytest

from relax.utils.score_centering import (
    append_score_centering_candidates,
    configure_score_centering_request,
    parse_score_centering_response,
    resolve_score_centering_vocab_size,
    validate_score_centering_sample,
)
from relax.utils.types import Sample


def _args(**overrides):
    values = dict(
        advantage_estimator="score_centering",
        rollout_temperature=0.7,
        rollout_top_logprobs_num=2,
        sglang_speculative_algorithm=None,
        score_centering_vocab_size=8,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _metadata():
    return {
        "output_token_logprobs": [[math.log(0.2), 4, None], [math.log(0.6), 2, None]],
        "output_top_logprobs": [
            [[math.log(0.5), 1, None], [math.log(0.3), 2, None]],
            [[math.log(0.6), 2, None], [math.log(0.3), 3, None]],
        ],
        "finish_reason": {"type": "abort"},
    }


def _packed_metadata(meta):
    meta = copy.deepcopy(meta)
    for prefix in ("output_token_logprobs", "output_top_logprobs"):
        rows = meta.pop(prefix)
        if prefix == "output_top_logprobs":
            meta[f"{prefix}_k"] = [len(row) for row in rows]
            rows = [entry for row in rows for entry in row]
        for column, field, dtype in ((0, "val", np.float32), (1, "idx", np.int32)):
            values = np.asarray([entry[column] for entry in rows], dtype=dtype)
            meta[f"{prefix}_{field}_b64"] = pybase64.b64encode(values.tobytes()).decode("ascii")
    return meta


def test_score_centering_capture_decodes_base64_head_and_tail_sample():
    expected = parse_score_centering_response(_metadata(), 2, vocab_size=8)
    actual = parse_score_centering_response(_packed_metadata(_metadata()), 2, expected_tokens=[4, 2], vocab_size=8)
    assert actual[0] == expected[0]
    np.testing.assert_allclose(actual[1], expected[1])
    np.testing.assert_array_equal(actual[2], expected[2])
    np.testing.assert_allclose(actual[3], expected[3])


@pytest.mark.parametrize("change", ["missing_ids", "short_values", "row_lengths", "sample_alignment"])
def test_score_centering_capture_rejects_misaligned_base64(change):
    meta = _packed_metadata(_metadata())
    if change == "missing_ids":
        del meta["output_token_logprobs_idx_b64"]
    elif change == "short_values":
        meta["output_top_logprobs_val_b64"] = pybase64.b64encode(np.zeros(3, dtype=np.float32).tobytes()).decode()
    elif change == "row_lengths":
        meta["output_top_logprobs_k"] = [1, 3]
    else:
        meta["output_token_logprobs_idx_b64"] = pybase64.b64encode(np.array([2, 4], dtype=np.int32).tobytes()).decode()
    with pytest.raises(ValueError, match="Score Centering"):
        parse_score_centering_response(meta, 2, expected_tokens=[4, 2], vocab_size=8)


def _sample():
    tokens, sampled, ids, logprobs = parse_score_centering_response(_metadata(), 2, vocab_size=8)
    return Sample(
        index=7,
        tokens=[0] + tokens,
        response_length=2,
        rollout_log_probs=sampled,
        rollout_topk_token_ids=ids,
        rollout_topk_log_probs=logprobs,
    )


def test_score_centering_capture_uses_head_q_without_renormalizing_and_keeps_tail_sample():
    sample = _sample()
    np.testing.assert_allclose(np.exp(sample.rollout_topk_log_probs).sum(-1), [0.8, 0.9])
    assert sample.rollout_log_probs[0] == pytest.approx(math.log(0.2))
    assert 4 not in sample.rollout_topk_token_ids[0]
    validate_score_centering_sample(sample, 2, vocab_size=8)
    restored = Sample.from_dict(pickle.loads(pickle.dumps(sample.to_dict())))
    np.testing.assert_array_equal(restored.rollout_topk_token_ids, sample.rollout_topk_token_ids)
    np.testing.assert_array_equal(restored.rollout_topk_log_probs, sample.rollout_topk_log_probs)


@pytest.mark.parametrize("packed", [False, True])
def test_score_centering_capture_filtered_support_uses_actual_sampled_probability(packed):
    meta = {
        "output_token_logprobs": [[math.log(0.14), 4, None]],
        "output_token_sampling_mask": [[4, 1, 2]],
        "output_token_sampling_logprobs": [math.log(0.2)],
        "output_top_logprobs": [[[math.log(0.35), 1], [math.log(0.21), 2]]],
    }
    response = _packed_metadata(meta) if packed else meta
    tokens, sampled, ids, logprobs = parse_score_centering_response(response, 2, sampling_mode="support", vocab_size=8)
    assert tokens == [4]
    assert sampled == pytest.approx([math.log(0.2)])
    np.testing.assert_array_equal(ids, [[1, 2]])
    np.testing.assert_allclose(np.exp(logprobs), [[0.5, 0.3]])
    meta["output_top_logprobs"][0].extend([[math.log(0.14), 4], [math.log(0.1), 5]])
    response = _packed_metadata(meta) if packed else meta
    _, _, ids, logprobs = parse_score_centering_response(response, 4, sampling_mode="support", vocab_size=8)
    np.testing.assert_array_equal(ids, [[1, 2, 4, -1]])
    assert np.isneginf(logprobs[0, 3])


@pytest.mark.parametrize("change", ["missing", "short", "duplicate", "nan", "mass", "mismatch", "vocab"])
def test_score_centering_capture_rejects_corrupt_probability_metadata(change):
    meta = _metadata()
    if change == "missing":
        del meta["output_top_logprobs"]
    elif change == "short":
        meta["output_top_logprobs"][0].pop()
    elif change == "duplicate":
        meta["output_top_logprobs"][0][1][1] = 1
    elif change == "nan":
        meta["output_top_logprobs"][0][0][0] = float("nan")
    elif change == "mass":
        meta["output_top_logprobs"][0][0][0] = math.log(0.9)
    elif change == "mismatch":
        meta["output_token_logprobs"][1][0] = math.log(0.5)
    elif change == "vocab":
        meta["output_top_logprobs"][0][0][1] = 8
    with pytest.raises(ValueError, match="Score Centering"):
        parse_score_centering_response(meta, 2, vocab_size=8)


def test_score_centering_capture_requires_complete_actual_support_metadata():
    with pytest.raises(ValueError, match="actual behavior q"):
        parse_score_centering_response(_metadata(), 2, sampling_mode="support")
    meta = {
        "output_token_logprobs": [[-1.0, 2]],
        "output_token_sampling_mask": [[1, 2]],
        "output_token_sampling_logprobs": [-0.9],
        "output_top_logprobs": [[[-1.0, 2], [-1.1, 1]]],
    }
    with pytest.raises(ValueError, match="complete sampling support must sum to 1"):
        parse_score_centering_response(meta, 2, sampling_mode="support")


def test_score_centering_capture_singleton_support_handles_argmax_topk_ties():
    meta = {
        "output_token_logprobs": [[math.log(0.5), 1]],
        "output_top_logprobs": [[[math.log(0.5), 2]]],
        "output_token_sampling_mask": [[1]],
        "output_token_sampling_logprobs": [0.0],
    }
    _, sampled, ids, logprobs = parse_score_centering_response(meta, 1, sampling_mode="support")
    assert sampled == [0.0]
    np.testing.assert_array_equal(ids, [[1]])
    np.testing.assert_array_equal(logprobs, [[0.0]])


def test_score_centering_capture_empty_abort_and_token_alignment():
    _, sampled, ids, logprobs = parse_score_centering_response({}, 3, expected_tokens=[])
    assert sampled == [] and ids.shape == logprobs.shape == (0, 3)
    with pytest.raises(ValueError, match="do not match"):
        parse_score_centering_response(_metadata(), 2, expected_tokens=[2, 4])


def test_score_centering_capture_observation_rows_and_partial_append():
    sample = _sample()
    sample.tokens.append(6)
    sample.response_length += 1
    sample.rollout_log_probs.append(0.0)
    sample.loss_mask = [1, 1, 0]
    append_score_centering_candidates(
        sample, np.full((1, 2), -1, dtype=np.int32), np.full((1, 2), -np.inf, dtype=np.float32)
    )
    tokens, sampled, ids, logprobs = parse_score_centering_response(_metadata(), 2)
    sample.tokens.extend(tokens)
    sample.response_length += len(tokens)
    sample.rollout_log_probs.extend(sampled)
    sample.loss_mask.extend([1, 1])
    append_score_centering_candidates(sample, ids, logprobs)
    validate_score_centering_sample(sample, 2, vocab_size=8)
    np.testing.assert_array_equal(sample.rollout_topk_token_ids, [[1, 2], [2, 3], [-1, -1], [1, 2], [2, 3]])
    sample.loss_mask[2] = 1
    with pytest.raises(ValueError, match="sample 7, position 2"):
        validate_score_centering_sample(sample, 2, vocab_size=8)


@pytest.mark.parametrize("overrides", [{"top_p": 0.9, "top_k": 4}, {"top_k": 3}, {"min_p": 0.1, "top_k": 4}])
def test_score_centering_capture_request_selects_behavior_support_for_filters(overrides):
    payload = {"sampling_params": overrides.copy()}
    assert configure_score_centering_request(_args(sglang_sampling_backend="pytorch"), payload) == "support"
    assert payload["return_sampling_mask"] and "sampling_logprobs_mode" not in payload
    assert payload["top_logprobs_num"] == 2
    assert all(payload["sampling_params"][key] == value for key, value in overrides.items())


@pytest.mark.parametrize("backend", [None, "flashinfer"])
@pytest.mark.parametrize("min_p", [0.0, 0.1])
def test_score_centering_capture_rejects_top_p_with_batch_dependent_support(backend, min_p):
    # For [0.4, 0.3, 0.2, 0.1], top_k=2 then renormalized top_p=0.5
    # keeps only token 0. The reported joint support instead keeps 0 and 1.
    # min_p=0 is also unsafe: another request can enable the batch min-p path.
    payload = {"sampling_params": {"top_k": 2, "top_p": 0.5, "min_p": min_p}}
    with pytest.raises(ValueError, match="batch.*min-p"):
        configure_score_centering_request(_args(sglang_sampling_backend=backend), payload)
    assert "return_sampling_mask" not in payload


@pytest.mark.parametrize("backend", [None, "flashinfer", "pytorch"])
@pytest.mark.parametrize("min_p", [0.0, 0.1])
def test_score_centering_capture_top_k_min_p_without_top_p_is_batch_independent(backend, min_p):
    payload = {"sampling_params": {"top_k": 2, "top_p": 1.0, "min_p": min_p}}
    assert configure_score_centering_request(_args(sglang_sampling_backend=backend), payload) == "support"


def test_score_centering_capture_request_preserves_penalties_and_checks_temperature(monkeypatch):
    monkeypatch.delenv("SGLANG_RETURN_ORIGINAL_LOGPROB", raising=False)
    sampling = {"repetition_penalty": 1.1, "regex": "[a-z]+"}
    payload = {"sampling_params": sampling}
    assert configure_score_centering_request(_args(), payload) == "selected"
    assert payload["top_logprobs_num"] == 2 and payload["return_logprob"]
    assert payload["sampling_params"]["repetition_penalty"] == 1.1
    assert payload["sampling_params"]["regex"] == "[a-z]+"
    assert "temperature" not in sampling
    with pytest.raises(ValueError, match="same positive"):
        configure_score_centering_request(_args(), {"sampling_params": {"temperature": 1.0}})
    monkeypatch.setenv("SGLANG_RETURN_ORIGINAL_LOGPROB", "1")
    with pytest.raises(ValueError, match="pre-temperature"):
        configure_score_centering_request(_args(), {"sampling_params": {}})
    disabled = {"sampling_params": {"temperature": 0}}
    assert configure_score_centering_request(_args(advantage_estimator="grpo"), disabled) is None
    assert disabled == {"sampling_params": {"temperature": 0}}


@pytest.mark.parametrize(
    "args,sampling,message",
    [
        (_args(rollout_temperature=1e-7), {}, "greedy"),
        (_args(sglang_speculative_algorithm="EAGLE"), {}, "per-accepted-token"),
        (_args(sglang_rl_on_policy_target="fsdp"), {}, "rounded BF16"),
        (_args(), {"top_p": 0.9}, "finite top_k"),
        (_args(sglang_sampling_backend="ascend"), {"top_k": 2}, "ascend"),
    ],
)
def test_score_centering_capture_reports_specific_backend_probability_gaps(args, sampling, message):
    with pytest.raises(ValueError, match=message):
        configure_score_centering_request(args, {"sampling_params": sampling})


def test_score_centering_capture_rejects_actual_multimodal_sample():
    sample = _sample()
    sample.multimodal_inputs = {"images": [object()]}
    with pytest.raises(ValueError, match="text-only"):
        validate_score_centering_sample(sample, 2, vocab_size=8)
    sample.multimodal_inputs = {"images": []}
    sample.multimodal_train_inputs = {"pixel_values": np.zeros((1, 3, 2, 2))}
    with pytest.raises(ValueError, match="text-only"):
        validate_score_centering_sample(sample, 2, vocab_size=8)


def test_score_centering_capture_resolves_real_model_vocab_once(monkeypatch):
    import relax.utils.misc as misc

    args = _args(hf_checkpoint="model", score_centering_vocab_size=None, vocab_size=128)
    calls = []
    monkeypatch.setattr(
        misc,
        "get_hf_config",
        lambda path: calls.append(path) or SimpleNamespace(text_config=SimpleNamespace(vocab_size=7)),
    )
    assert resolve_score_centering_vocab_size(args) == resolve_score_centering_vocab_size(args) == 7
    assert calls == ["model"]


@pytest.mark.parametrize("packed", [False, True])
def test_score_centering_native_generation_appends_resume_without_rescoring(monkeypatch, packed):
    """Run the real generate function with only its HTTP/processor boundary
    mocked."""
    source = Path(__file__).parents[2] / "relax/engine/rollout/sglang_rollout.py"
    tree = ast.parse(source.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "generate")
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias("annotations")], level=0), function],
        type_ignores=[],
    )
    tokenizer = SimpleNamespace(encode=lambda *args, **kwargs: [0])
    state = SimpleNamespace(tokenizer=tokenizer, processor=None, opd_manager=None)
    calls = []

    async def post_generation(args, state, url, payload, headers, evaluation):
        calls.append(copy.deepcopy(payload))
        meta = _packed_metadata(_metadata()) if packed else _metadata()
        return {"text": "answer", "output_ids": [4, 2], "meta_info": meta}

    namespace = {
        "Sample": Sample,
        "GenerateState": lambda args: state,
        "monotonic": monotonic,
        "_post_generation": post_generation,
        "configure_score_centering_request": configure_score_centering_request,
        "parse_score_centering_response": parse_score_centering_response,
        "append_score_centering_candidates": append_score_centering_candidates,
        "resolve_score_centering_vocab_size": resolve_score_centering_vocab_size,
    }
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    args = _args(
        ci_test=False,
        sglang_router_ip="localhost",
        sglang_router_port=1,
        use_rollout_routing_replay=False,
        use_slime_router=False,
        sglang_router_policy="round_robin",
        partial_rollout=True,
        mask_offpolicy_in_partial_rollout=True,
    )
    sample = Sample(prompt="prompt")
    asyncio.run(namespace["generate"](args, sample, {"max_new_tokens": 8, "temperature": 0.7}))
    sample.loss_mask = [0, 0]
    first_head = sample.rollout_topk_log_probs.copy()
    asyncio.run(namespace["generate"](args, sample, {"max_new_tokens": 8, "temperature": 0.7}))
    assert calls[0]["top_logprobs_num"] == 2
    assert calls[1]["input_ids"] == [0, 4, 2]
    assert sample.tokens == [0, 4, 2, 4, 2] and sample.loss_mask == [0, 0, 1, 1]
    np.testing.assert_array_equal(sample.rollout_topk_log_probs[:2], first_head)
    validate_score_centering_sample(sample, 2, vocab_size=8)


def test_score_centering_converter_flattens_candidates_and_rejects_missing_fields(monkeypatch):
    tensordict = ModuleType("tensordict")
    tensordict.TensorDict = dict
    monkeypatch.setitem(sys.modules, "tensordict", tensordict)
    monkeypatch.setattr(importlib.import_module("relax.utils"), "utils", None, raising=False)
    previous = sys.modules.pop("relax.utils.utils", None)
    try:
        module = importlib.import_module("relax.utils.utils")
        monkeypatch.setattr(module, "post_process_rewards", lambda args, samples: [0.0] * len(samples))
        args = _args(debug_train_only=True, multimodal_keys=None, use_opd=False)
        sample = _sample()
        data = module.convert_samples_to_train_data(args, [sample])
        assert data["rollout_topk_token_ids"] == [[1, 2, 2, 3]]
        np.testing.assert_allclose(np.reshape(data["rollout_topk_log_probs"], (2, 2)), sample.rollout_topk_log_probs)
        monkeypatch.setattr(module, "post_process_rewards", lambda args, samples: [[0.3, -0.2]] * len(samples))
        assert module.convert_samples_to_train_data(args, [sample])["rewards"] == [[0.3, -0.2]]
        args.custom_convert_samples_to_train_data_path = "custom.converter"
        monkeypatch.setattr(module, "load_function", lambda path: lambda args, samples: {"tokens": [[0, 4, 2]]})
        with pytest.raises(ValueError, match="custom converter omitted"):
            module.convert_samples_to_train_data(args, [sample])
        monkeypatch.setattr(module, "load_function", lambda path: lambda args, samples: data)
        # Custom converters own their input layout, including grouped samples.
        assert module.convert_samples_to_train_data(args, [[sample]]) is data
        args.custom_convert_samples_to_train_data_path = None
        sample.rollout_topk_log_probs = None
        with pytest.raises(ValueError, match="rollout_topk_log_probs"):
            module.convert_samples_to_train_data(args, [sample])
    finally:
        sys.modules.pop("relax.utils.utils", None)
        if previous is not None:
            sys.modules["relax.utils.utils"] = previous
