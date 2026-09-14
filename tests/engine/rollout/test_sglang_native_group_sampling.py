# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import argparse
import asyncio
import contextlib
from argparse import Namespace
from pathlib import Path

import pytest

from relax.utils.types import Sample


INFERENCE_IMPORT_ERROR = ""
try:
    from sglang.srt.managers.io_struct import GenerateReqInput

    from relax.engine.rollout import sglang_rollout
    from relax.engine.rollout.request_permit import GenerationAborted
    from relax.engine.rollout.sglang_rollout import (
        _generate_native_group,
        _holding_session_lock,
        _native_group_sampling_eligible,
        _native_group_sampling_media_payload,
        _native_group_sampling_outputs,
        _native_group_sampling_params,
    )
    from relax.utils.arguments import get_slime_extra_args_provider

    HAS_INFERENCE_DEPS = True
except (ImportError, OSError, RuntimeError) as exc:
    HAS_INFERENCE_DEPS = False
    INFERENCE_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

pytestmark = pytest.mark.skipif(
    not HAS_INFERENCE_DEPS,
    reason=f"Ray/SGLang inference stack unavailable: {INFERENCE_IMPORT_ERROR}",
)


class _Tokenizer:
    def encode(self, prompt, add_special_tokens=False):
        assert add_special_tokens is False
        return [11, 12]


class _NativeState:
    def __init__(self, *, processor=None, aborted: bool = False) -> None:
        self.aborted = aborted
        self.dp_entries = 0
        self.opd_manager = None
        self.processor = processor
        self.semaphore = asyncio.Semaphore(1)
        self.tokenizer = _Tokenizer()

    @contextlib.contextmanager
    def dp_rank_context(self):
        self.dp_entries += 1
        yield 0


def _native_args(**overrides) -> Namespace:
    values = {
        "sglang_native_group_sampling": True,
        "group_rm": True,
        "partial_rollout": False,
        "mask_offpolicy_in_partial_rollout": False,
        "use_slime_router": False,
        "use_opd": False,
        "use_rollout_routing_replay": False,
        "lora_rank": 0,
        "lora_adapter_mode": False,
        "sglang_enable_deterministic_inference": False,
        "custom_generate_function_path": None,
        "sglang_router_ip": "router.test",
        "sglang_router_port": 30000,
        "sglang_router_policy": None,
        "slime_router_sticky": False,
        "sglang_speculative_algorithm": None,
    }
    values.update(overrides)
    return Namespace(**values)


def _response(token_id: int, text: str, *, finish_reason: str = "stop") -> dict:
    return {
        "text": text,
        "meta_info": {
            "output_token_logprobs": [(-0.1, token_id, text)],
            "finish_reason": {"type": finish_reason},
            "cached_tokens": 2,
            "prompt_tokens": 2,
        },
    }


def test_native_group_sampling_params_sets_n_without_mutating_input() -> None:
    sampling_params = {"temperature": 1.0, "max_new_tokens": 32}

    result = _native_group_sampling_params(sampling_params, 8)

    assert result == {"temperature": 1.0, "max_new_tokens": 32, "n": 8}
    assert sampling_params == {"temperature": 1.0, "max_new_tokens": 32}


def test_native_group_sampling_params_requires_multiple_samples() -> None:
    with pytest.raises(ValueError, match="at least two"):
        _native_group_sampling_params({}, 1)


def test_native_group_sampling_media_payload_adds_batch_dimension() -> None:
    encoded_mm = {
        "image_data": ["image-a", "image-b"],
        "audio_data": ["audio-a"],
    }

    assert _native_group_sampling_media_payload(encoded_mm) == {
        "image_data": [["image-a", "image-b"]],
        "audio_data": [["audio-a"]],
    }
    assert encoded_mm == {
        "image_data": ["image-a", "image-b"],
        "audio_data": ["audio-a"],
    }


def test_native_group_sampling_media_payload_matches_sglang_batch_contract() -> None:
    payload = _native_group_sampling_media_payload({"image_data": ["image-a", "image-b"]})
    request = GenerateReqInput(
        input_ids=[11, 12],
        image_data=payload["image_data"],
        sampling_params={"n": 2, "max_new_tokens": 8},
    )

    request.normalize_batch_and_arguments()

    assert request.input_ids == [[11, 12], [11, 12]]
    assert request.image_data == [
        ["image-a", "image-b"],
        ["image-a", "image-b"],
    ]
    assert request.modalities == ["multi-images", "multi-images"]


def test_native_group_sampling_outputs_requires_native_list_shape() -> None:
    outputs = [{"text": "a"}, {"text": "b"}]

    assert _native_group_sampling_outputs(outputs, 2) is outputs

    with pytest.raises(TypeError, match="must return a list"):
        _native_group_sampling_outputs(outputs[0], 2)
    with pytest.raises(ValueError, match="unexpected number"):
        _native_group_sampling_outputs(outputs, 8)


def test_native_group_sampling_outputs_rejects_non_dict_items() -> None:
    with pytest.raises(TypeError, match="responses must be dictionaries"):
        _native_group_sampling_outputs([{"text": "a"}, "not-a-response"], 2)


def test_native_group_sampling_cli_is_opt_in() -> None:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    get_slime_extra_args_provider()(parser)

    defaults, _ = parser.parse_known_args([])
    enabled, _ = parser.parse_known_args(["--sglang-native-group-sampling"])

    assert defaults.sglang_native_group_sampling is False
    assert enabled.sglang_native_group_sampling is True


def test_qwen3_vl_launcher_couples_native_sampling_with_group_rm_and_balanced_routing() -> None:
    launcher = Path(__file__).parents[3] / "scripts/training/multimodal/run-qwen3-vl-4B-8xgpu.sh"
    source = launcher.read_text()
    guard = 'if [ "${SGLANG_NATIVE_GROUP_SAMPLING:-0}" = "1" ]; then'
    before_native, separator, after_native = source.partition(guard)
    native_branch, else_separator, explicit_override_branch = after_native.partition("elif")

    assert separator == guard
    assert else_separator == "elif"
    assert "--group-rm" not in before_native
    assert "--group-rm --sglang-native-group-sampling" in native_branch
    assert "${SGLANG_ROUTER_POLICY:-round_robin}" in native_branch
    assert "${SGLANG_ROUTER_POLICY}" in explicit_override_branch


@pytest.mark.parametrize(
    "overrides",
    [
        {"sglang_native_group_sampling": False},
        {"group_rm": False},
        {"partial_rollout": True},
        {"use_slime_router": True},
        {"use_opd": True},
        {"use_rollout_routing_replay": True},
        {"lora_rank": 1, "lora_adapter_mode": True},
        {"sglang_enable_deterministic_inference": True},
        {"sglang_speculative_algorithm": "EAGLE"},
        {"custom_generate_function_path": "custom.generate"},
    ],
)
def test_native_group_sampling_eligibility_rejects_incompatible_modes(overrides) -> None:
    group = [Sample(prompt="same"), Sample(prompt="same")]
    assert not _native_group_sampling_eligible(_native_args(**overrides), group, evaluation=False)


def test_native_group_sampling_eligibility_rejects_evaluation_or_single_sample() -> None:
    group = [Sample(prompt="same"), Sample(prompt="same")]
    assert not _native_group_sampling_eligible(_native_args(), group, evaluation=True)
    assert not _native_group_sampling_eligible(_native_args(), group[:1], evaluation=False)


def test_native_group_sampling_eligibility_requires_homogeneous_fresh_group() -> None:
    shared_media = {"images": [object(), object()], "videos": [], "audio": []}
    group = [
        Sample(prompt="same", multimodal_inputs=shared_media),
        Sample(prompt="same", multimodal_inputs=shared_media),
    ]
    assert _native_group_sampling_eligible(_native_args(), group, evaluation=False)

    group[1].prompt = "different"
    assert not _native_group_sampling_eligible(_native_args(), group, evaluation=False)
    group[1].prompt = "same"

    group[1].multimodal_inputs = {"images": [object()], "videos": [], "audio": []}
    assert not _native_group_sampling_eligible(_native_args(), group, evaluation=False)
    group[1].multimodal_inputs = shared_media

    group[1].tokens = [1]
    assert not _native_group_sampling_eligible(_native_args(), group, evaluation=False)
    group[1].tokens = []

    group[1].loss_mask = []
    assert not _native_group_sampling_eligible(_native_args(), group, evaluation=False)
    group[1].loss_mask = None

    group[1].generate_function_path = "custom.generate"
    assert not _native_group_sampling_eligible(_native_args(), group, evaluation=False)
    group[1].generate_function_path = None

    group[1].status = Sample.Status.FAILED
    assert not _native_group_sampling_eligible(_native_args(), group, evaluation=False)


async def test_generate_native_group_uses_one_n_request_and_maps_outputs(monkeypatch) -> None:
    state = _NativeState()
    args = _native_args()
    group = [Sample(prompt="same") for _ in range(8)]
    sampling_params = {"temperature": 1.0, "max_new_tokens": 32}
    requests = []

    async def fake_post(url, payload, headers=None):
        assert state.semaphore.locked()
        assert _holding_session_lock.get() is True
        requests.append((url, payload, headers))
        return [_response(100 + index, f"r{index}") for index in range(8)]

    monkeypatch.setattr(sglang_rollout, "post", fake_post)

    result = await _generate_native_group(args, state, group, sampling_params)

    assert result is group
    assert len(requests) == 1
    url, payload, headers = requests[0]
    assert url == "http://router.test:30000/generate"
    assert payload["sampling_params"] == {"temperature": 1.0, "max_new_tokens": 32, "n": 8}
    assert payload["return_logprob"] is True
    assert payload["input_ids"] == [11, 12]
    assert headers is None
    assert sampling_params == {"temperature": 1.0, "max_new_tokens": 32}
    assert state.dp_entries == 1
    assert not state.semaphore.locked()
    assert _holding_session_lock.get() is False

    for index, sample in enumerate(group):
        assert sample.tokens == [11, 12, 100 + index]
        assert sample.rollout_tokens == [11, 12, 100 + index]
        assert sample.response == f"r{index}"
        assert sample.response_length == 1
        assert sample.rollout_log_probs == [-0.1]
        assert sample.status == Sample.Status.COMPLETED
        assert set(sample.metadata["_timing"]) == {"generate", "post_generate"}


async def test_generate_native_group_encodes_shared_multimodal_input_once(monkeypatch) -> None:
    state = _NativeState(processor=object())
    args = _native_args()
    shared_media = {"images": [object(), object()], "videos": [], "audio": []}
    group = [Sample(prompt="same", multimodal_inputs=shared_media) for _ in range(2)]
    train_inputs = {"pixel_values": object()}
    calls = {"processor": 0, "encode": 0, "post": 0}

    async def fake_processor(state_arg, args_arg, prompt, multimodal_inputs):
        assert state_arg is state
        assert args_arg is args
        assert prompt == "same"
        assert multimodal_inputs is shared_media
        calls["processor"] += 1
        return [21, 22, 23], train_inputs, 0.25

    async def fake_encode(multimodal_inputs):
        assert multimodal_inputs is shared_media
        calls["encode"] += 1
        return {"image_data": ["encoded-image-a", "encoded-image-b"]}, 0.5

    async def fake_post(url, payload, headers=None):
        calls["post"] += 1
        assert payload["image_data"] == [["encoded-image-a", "encoded-image-b"]]
        return [_response(101, "a"), _response(102, "b")]

    monkeypatch.setattr(sglang_rollout, "_run_image_processor", fake_processor)
    monkeypatch.setattr(sglang_rollout, "_encode_multimodal_inputs", fake_encode)
    monkeypatch.setattr(sglang_rollout, "post", fake_post)

    await _generate_native_group(args, state, group, {"max_new_tokens": 32})

    assert calls == {"processor": 1, "encode": 1, "post": 1}
    for sample in group:
        assert sample.tokens[:3] == [21, 22, 23]
        assert sample.rollout_tokens[:2] == [11, 12]
        assert sample.multimodal_train_inputs is train_inputs
        assert sample.metadata["_timing"]["image_processor"] == 0.25
        assert sample.metadata["_timing"]["mm_encode"] == 0.5


async def test_generate_native_group_uses_consistent_hash_routing_key(monkeypatch) -> None:
    state = _NativeState()
    args = _native_args(sglang_router_policy="consistent_hashing")
    group = [Sample(prompt="same", session_id="group-session") for _ in range(2)]
    observed_headers = []

    async def fake_post(url, payload, headers=None):
        observed_headers.append(headers)
        return [_response(101, "a"), _response(102, "b")]

    monkeypatch.setattr(sglang_rollout, "post", fake_post)

    await _generate_native_group(args, state, group, {"max_new_tokens": 32})

    assert observed_headers == [{"X-SMG-Routing-Key": "group-session"}]


async def test_generate_native_group_marks_entire_group_aborted_before_post(monkeypatch) -> None:
    state = _NativeState(aborted=True)
    group = [Sample(prompt="same") for _ in range(2)]

    async def unexpected_post(url, payload, headers=None):
        raise AssertionError("aborted group must not issue a request")

    monkeypatch.setattr(sglang_rollout, "post", unexpected_post)

    result = await _generate_native_group(_native_args(), state, group, {"max_new_tokens": 32})

    assert all(sample.status == Sample.Status.ABORTED for sample in result)
    assert not state.semaphore.locked()
    assert _holding_session_lock.get() is False


async def test_generate_native_group_rejects_wrong_response_cardinality(monkeypatch) -> None:
    state = _NativeState()
    group = [Sample(prompt="same") for _ in range(2)]

    async def fake_post(url, payload, headers=None):
        return [_response(101, "only-one")]

    monkeypatch.setattr(sglang_rollout, "post", fake_post)

    with pytest.raises(ValueError, match="expected 2, got 1"):
        await _generate_native_group(_native_args(), state, group, {"max_new_tokens": 32})

    assert not state.semaphore.locked()
    assert _holding_session_lock.get() is False


async def test_generate_native_group_abort_before_post_keeps_group_native_eligible(monkeypatch) -> None:
    state = _NativeState(aborted=True)
    args = _native_args()
    group = [Sample(prompt="same") for _ in range(2)]

    async def unexpected_post(url, payload, headers=None):
        raise AssertionError("aborted group must not issue a request")

    monkeypatch.setattr(sglang_rollout, "post", unexpected_post)

    result = await _generate_native_group(args, state, group, {"max_new_tokens": 32})

    for sample in result:
        assert sample.status == Sample.Status.ABORTED
        assert sample.tokens == []
        assert sample.rollout_tokens == []
        assert sample.multimodal_train_inputs is None
    assert _native_group_sampling_eligible(args, result, evaluation=False)


async def test_generate_native_group_reverts_group_when_request_is_aborted(monkeypatch) -> None:
    state = _NativeState()
    args = _native_args()
    group = [Sample(prompt="same") for _ in range(2)]

    async def aborted_post(url, payload, headers=None):
        raise GenerationAborted("router draining")

    monkeypatch.setattr(sglang_rollout, "post", aborted_post)

    result = await _generate_native_group(args, state, group, {"max_new_tokens": 32})

    assert not state.semaphore.locked()
    assert _holding_session_lock.get() is False
    for sample in result:
        assert sample.status == Sample.Status.ABORTED
        assert sample.tokens == []
        assert sample.rollout_tokens == []
    assert _native_group_sampling_eligible(args, result, evaluation=False)


async def test_generate_native_group_maps_mid_flight_abort_outputs(monkeypatch) -> None:
    state = _NativeState()
    args = _native_args()
    group = [Sample(prompt="same") for _ in range(2)]

    async def fake_post(url, payload, headers=None):
        return [
            _response(101, "partial-a", finish_reason="abort"),
            _response(102, "partial-b", finish_reason="abort"),
        ]

    monkeypatch.setattr(sglang_rollout, "post", fake_post)

    result = await _generate_native_group(args, state, group, {"max_new_tokens": 32})

    for index, sample in enumerate(result):
        assert sample.status == Sample.Status.ABORTED
        assert sample.tokens == [11, 12, 101 + index]
        assert sample.response == ("partial-a", "partial-b")[index]
    # Partially generated samples keep their tokens and must retry per-sample.
    assert not _native_group_sampling_eligible(args, result, evaluation=False)


async def test_generate_and_rm_group_strips_pre_encoded_media_on_success(monkeypatch) -> None:
    state = _NativeState(processor=object())
    args = _native_args()
    shared_media = {"images": [object()], "videos": [], "audio": []}
    group = [Sample(prompt="same", multimodal_inputs=shared_media) for _ in range(2)]
    calls = {"encode": 0}

    async def fake_processor(state_arg, args_arg, prompt, multimodal_inputs):
        return [21, 22], {"pixel_values": object()}, 0.25

    async def fake_encode(multimodal_inputs):
        calls["encode"] += 1
        return {"image_data": ["encoded"]}, 0.5

    async def fake_post(url, payload, headers=None):
        assert payload["image_data"] == [["encoded"]]
        return [_response(101, "a"), _response(102, "b")]

    async def fake_rm(args_arg, group_arg):
        return [1.0] * len(group_arg)

    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda _args: state)
    monkeypatch.setattr(sglang_rollout, "_run_image_processor", fake_processor)
    monkeypatch.setattr(sglang_rollout, "_encode_multimodal_inputs", fake_encode)
    monkeypatch.setattr(sglang_rollout, "post", fake_post)
    monkeypatch.setattr(sglang_rollout, "batched_async_rm", fake_rm)

    result = await sglang_rollout.generate_and_rm_group(args, group, {"max_new_tokens": 32})

    assert calls["encode"] == 1
    for sample in result:
        assert sample.reward == 1.0
        assert not hasattr(sample, "_pre_encoded_mm")
        assert not hasattr(sample, "_pre_encoded_mm_elapsed")
        assert "_pre_encoded_mm" not in sample.to_dict()


async def test_generate_and_rm_group_strips_pre_encoded_media_when_fallback_fails(monkeypatch) -> None:
    state = _NativeState()
    args = _native_args(sglang_native_group_sampling=False)
    shared_media = {"images": [object()], "videos": [], "audio": []}
    group = [Sample(prompt="same", multimodal_inputs=shared_media)]

    async def fake_encode(multimodal_inputs):
        return {"image_data": ["encoded"]}, 0.5

    async def failing_generate_and_rm(args_arg, sample, sampling_params, evaluation=False):
        raise RuntimeError("worker exploded")

    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda _args: state)
    monkeypatch.setattr(sglang_rollout, "_encode_multimodal_inputs", fake_encode)
    monkeypatch.setattr(sglang_rollout, "generate_and_rm", failing_generate_and_rm)

    with pytest.raises(RuntimeError, match="worker exploded"):
        await sglang_rollout.generate_and_rm_group(args, group, {"max_new_tokens": 32})

    for sample in group:
        assert not hasattr(sample, "_pre_encoded_mm")
        assert not hasattr(sample, "_pre_encoded_mm_elapsed")


async def test_generate_and_rm_group_skips_pre_encoding_for_empty_media(monkeypatch) -> None:
    state = _NativeState()
    args = _native_args(sglang_native_group_sampling=False)
    group = [Sample(prompt="same", multimodal_inputs={})]

    async def unexpected_encode(multimodal_inputs):
        raise AssertionError("empty multimodal inputs must not be encoded")

    async def fake_generate_and_rm(args_arg, sample, sampling_params, evaluation=False):
        return sample

    async def fake_rm(args_arg, group_arg):
        return [0.0] * len(group_arg)

    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda _args: state)
    monkeypatch.setattr(sglang_rollout, "_encode_multimodal_inputs", unexpected_encode)
    monkeypatch.setattr(sglang_rollout, "generate_and_rm", fake_generate_and_rm)
    monkeypatch.setattr(sglang_rollout, "batched_async_rm", fake_rm)

    result = await sglang_rollout.generate_and_rm_group(args, group, {"max_new_tokens": 32})

    assert result[0] is group[0]
    assert result[0].reward == 0.0


async def test_generate_and_rm_group_warns_once_for_ineligible_groups(monkeypatch) -> None:
    state = _NativeState()
    args = _native_args(partial_rollout=True)
    calls = {"generate": 0}
    warnings = []

    async def fake_generate_and_rm(args_arg, sample, sampling_params, evaluation=False):
        calls["generate"] += 1
        return sample

    async def fake_rm(args_arg, group_arg):
        return [0.0] * len(group_arg)

    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda _args: state)
    monkeypatch.setattr(sglang_rollout, "generate_and_rm", fake_generate_and_rm)
    monkeypatch.setattr(sglang_rollout, "batched_async_rm", fake_rm)
    monkeypatch.setattr(sglang_rollout, "_NATIVE_GROUP_SAMPLING_LOGGED", set())
    monkeypatch.setattr(sglang_rollout.logger, "warning", lambda message, *a, **k: warnings.append(message))

    for _ in range(2):
        group = [Sample(prompt="same"), Sample(prompt="same")]
        await sglang_rollout.generate_and_rm_group(args, group, {"max_new_tokens": 32})

    assert calls["generate"] == 4
    assert len(warnings) == 1
    assert "ineligible" in warnings[0]
