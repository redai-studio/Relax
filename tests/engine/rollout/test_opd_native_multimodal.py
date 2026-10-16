# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import copy
import pickle
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pybase64
import pytest

from relax.engine.rollout.on_policy_distillation import OpdManager
from relax.utils.opd.opd_main_worker import LogprobResponse
from relax.utils.opd.opd_multimodal import (
    encode_image_inputs,
    image_inputs,
    validate_image_input_ids,
    validate_prompt_token_ids,
)
from relax.utils.opd.opd_opsd_worker import OpsdWorker
from relax.utils.types import Sample


@pytest.fixture
def processor_state(monkeypatch):
    state = SimpleNamespace(
        tokenizer=SimpleNamespace(
            image_token_id=99,
            encode=lambda text, **kwargs: [8, 99, 9] if text == "image prompt" else [8, 9],
            apply_chat_template=lambda messages, **kwargs: "image prompt",
        ),
        processor=SimpleNamespace(image_token_id=99),
    )
    module = ModuleType("relax.engine.rollout.sglang_rollout")
    module.GenerateState = lambda args: state

    async def process(state, args, prompt, media):
        assert isinstance(prompt, str)
        assert media["images"]
        return [8, 99, 99, 99, 9], {"image_grid_thw": [[1, 2, 6]]}, 0.0

    module._run_image_processor = process
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return state


def _sample() -> Sample:
    return Sample(
        prompt="image prompt",
        tokens=[1, 99, 99, 99, 2, 10, 11],
        rollout_tokens=[1, 99, 2, 10, 11],
        response_length=2,
        multimodal_inputs={"images": [object()], "_opd_image_b64_cache": ["student-image"]},
    )


def _manager(**overrides) -> OpdManager:
    args = dict(
        opd_token_selection="student_sampled",
        opd_log_prob_top_k=2,
        opd_kl_coef=1.0,
        opd_loss_coef=0.0,
        opd_teacher_url="http://teacher/generate",
        sglang_router_ip="student",
        sglang_router_port=80,
    )
    args.update(overrides)
    return OpdManager(SimpleNamespace(**args))


def _response(expected_ids: list[int], **meta) -> dict:
    return {
        "prompt_token_ids": expected_ids,
        "meta_info": {
            "input_token_logprobs": [[None, expected_ids[-3]], [-0.2, expected_ids[-2]], [-0.3, expected_ids[-1]]],
            **meta,
        },
    }


class _Session:
    def __init__(self, response: dict, status: int = 200):
        self.response = response
        self.status = status
        self.requests = []

    def post(self, url, **kwargs):
        import json

        self.requests.append((url, json.loads(kwargs["data"])))
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def json(self):
        return self.response

    async def text(self):
        return "failed"

    def raise_for_status(self):
        raise RuntimeError("HTTP failure")


@pytest.mark.asyncio
async def test_plain_image_opd_uses_collapsed_ids_raw_images_and_expanded_offset(processor_state):
    sample = _sample()
    session = _Session(_response(sample.tokens))
    assert await _manager()._teacher_prefill(sample, session)
    payload = session.requests[0][1]
    assert payload["input_ids"] == sample.rollout_tokens
    assert payload["image_data"] == ["student-image"]
    assert payload["logprob_start_len"] == 4
    assert payload["return_prompt_token_ids"] is True
    assert sample.teacher_log_probs == pytest.approx([-0.2, -0.3])


@pytest.mark.asyncio
async def test_image_teacher_prefix_uses_collapsed_prompt_boundary(processor_state):
    sample = _sample()
    session = _Session(_response(sample.tokens[:-2] + [20, 21] + sample.tokens[-2:]))
    await _manager(opd_teacher_response_prefix_token_ids=[20, 21])._teacher_prefill(sample, session)
    payload = session.requests[0][1]
    assert payload["input_ids"] == [1, 99, 2, 20, 21, 10, 11]
    assert payload["logprob_start_len"] == 6


@pytest.mark.asyncio
async def test_image_teacher_rejects_same_length_retokenization_before_assigning_logprobs(processor_state):
    sample = _sample()
    session = _Session(_response(sample.tokens[:-2] + [12, 13]))
    with pytest.raises(ValueError, match="expanded input IDs differ"):
        await _manager()._teacher_prefill(sample, session)
    assert sample.teacher_log_probs is None


@pytest.mark.asyncio
async def test_image_teacher_http_failure_cannot_be_masked_by_partial_batch_success(processor_state):
    sample = _sample()
    with pytest.raises(RuntimeError, match="Native image OPD teacher prefill failed"):
        await _manager()._teacher_prefill(sample, _Session({}, status=500))
    assert sample.teacher_log_probs is None


@pytest.mark.asyncio
async def test_image_batch_failure_never_assembles_partial_training_data(processor_state, monkeypatch):
    good, bad = _sample(), _sample()
    bad.tokens[0] = bad.rollout_tokens[0] = 3

    class BatchSession(_Session):
        def post(self, url, **kwargs):
            import json

            ids = json.loads(kwargs["data"])["input_ids"]
            return _Session(_response(good.tokens)) if ids[0] == 1 else _Session({}, status=500)

    monkeypatch.setattr(
        "relax.engine.rollout.on_policy_distillation._create_teacher_client_session",
        lambda args: BatchSession({}),
    )
    manager = _manager()
    assembled = []
    monkeypatch.setattr(manager, "_assemble_transfer", lambda samples: assembled.append(samples))
    with pytest.raises(RuntimeError, match="Native image OPD teacher prefill failed"):
        await manager.prefill([good, bad])
    assert not assembled


@pytest.mark.asyncio
async def test_image_eos_mapping_cannot_rewrite_response(processor_state):
    with pytest.raises(ValueError, match="EOS remapping"):
        await _manager(opd_student_eos_token_id=11, opd_teacher_eos_token_id=12)._teacher_prefill(
            _sample(), _Session({})
        )


@pytest.mark.asyncio
async def test_opsd_prompt_override_reuses_student_images_without_retokenizing_response(processor_state):
    sample = _sample()
    sample.teacher_prompt = [{"role": "user", "content": "image prompt"}]
    await OpsdWorker(is_opsd=True).build_teacher_inputs(SimpleNamespace(), sample)
    assert sample.teacher_rollout_tokens == [8, 99, 9, 10, 11]
    assert sample.teacher_tokens == [8, 99, 99, 99, 9, 10, 11]
    assert sample.teacher_prompt_length == 5


@pytest.mark.asyncio
async def test_opsd_explicit_empty_media_does_not_fall_back_to_student_images(processor_state):
    sample = _sample()
    sample.teacher_prompt = "text prompt"
    sample.teacher_multimodal_inputs = {}
    manager = _manager(opd_teacher_prompt_key="teacher_prompt")
    await manager.opsd_worker.build_teacher_inputs(manager.args, sample)
    assert sample.teacher_tokens == [8, 9, 10, 11]
    session = _Session(_response(sample.teacher_tokens))
    await manager._teacher_prefill(sample, session)
    assert "image_data" not in session.requests[0][1]
    assert session.requests[0][1]["logprob_start_len"] == 1


@pytest.mark.asyncio
async def test_student_cross_topk_uses_student_images_and_checks_query_order(processor_state):
    sample = _sample()
    sample.teacher_multimodal_inputs = {"images": [object()], "_opd_image_b64_cache": ["teacher-image"]}
    sample.teacher_topk_token_ids = np.array([[0, 5], [7, 9]], dtype=np.int32)
    session = _Session(
        _response(
            sample.tokens,
            input_token_ids_logprobs=[None, [[-1.0, 0], [-2.0, 5]], [[-3.0, 7], [-4.0, 9]]],
        )
    )
    await _manager(opd_token_selection="teacher_topk")._student_prefill(sample, session, None)
    payload = session.requests[0][1]
    assert payload["image_data"] == ["student-image"]
    assert payload["input_ids"] == sample.rollout_tokens
    assert payload["token_ids_logprob"] == [0, 5, 7, 9, 0, 0]
    np.testing.assert_array_equal(sample.student_at_teacher_topk_log_probs, [[-1, -2], [-3, -4]])


def test_native_prefill_checks_base_response_ids_and_preserves_nan_sentinel():
    response = _response([1, 2, 10, 11])
    LogprobResponse(response).validate_native_prefill([1, 2, 10, 11], 2)
    response["meta_info"]["input_token_logprobs"][1][1] = 12
    with pytest.raises(ValueError, match="logprob token IDs"):
        LogprobResponse(response).validate_native_prefill([1, 2, 10, 11], 2)


@pytest.mark.parametrize("wire_format", ["legacy", "base64", "flat", "flat_base64"])
def test_strict_topk_decodes_all_supported_wire_formats(wire_format):
    values = np.array([[-1, -2], [-3, -4]], dtype=np.float32)
    ids = np.array([[0, 5], [7, 9]], dtype=np.int32)
    prefix = "input_top_logprobs"
    if wire_format == "legacy":
        meta = {prefix: [None, [[-1, 0], [-2, 5]], [[-3, 7], [-4, 9]]]}
    elif wire_format == "base64":
        meta = {
            f"{prefix}_val_b64": pybase64.b64encode(values.tobytes()).decode(),
            f"{prefix}_idx_b64": pybase64.b64encode(ids.tobytes()).decode(),
            f"{prefix}_k": [0, 2, 2],
        }
    else:
        meta = {f"{prefix}_shape": [2, 2], f"{prefix}_null_prefix": 1}
        for name, array in (("val", values), ("idx", ids)):
            if wire_format == "flat":
                meta[f"{prefix}_{name}_flat"] = array.reshape(-1).tolist()
            else:
                key = f"{prefix}_{name}_flat_b64"
                meta[key] = pybase64.b64encode(array.tobytes()).decode()
                meta[f"{key}_dtype"] = str(array.dtype)
    actual_ids, actual_values = LogprobResponse({"meta_info": meta}).strict_topk(prefix, 2, 2)
    np.testing.assert_array_equal(actual_ids, ids)
    np.testing.assert_array_equal(actual_values, values)


@pytest.mark.parametrize(
    "rows",
    [
        [None, [[-1, 0]], [[-3, 7], [-4, 9]]],
        [None, None, [[-3, 7], [-4, 9]]],
        [None, [[-1, 0], [-2, 5]], [[-3, 7], [-4, 9]], [[-5, 10], [-6, 11]]],
    ],
)
def test_strict_topk_rejects_short_missing_and_extra_rows(rows):
    with pytest.raises(ValueError):
        LogprobResponse({"meta_info": {"input_top_logprobs": rows}}).strict_topk("input_top_logprobs", 2, 2)


def test_strict_cross_topk_rejects_shifted_query_ids():
    response = {"meta_info": {"input_token_ids_logprobs": [None, [[-1, 7], [-2, 9]], [[-3, 0], [-4, 5]]]}}
    with pytest.raises(ValueError, match="query token IDs differ"):
        LogprobResponse(response).strict_topk("input_token_ids_logprobs", 2, 2, np.array([[0, 5], [7, 9]]))


@pytest.mark.asyncio
@pytest.mark.parametrize("side", ["student", "teacher"])
@pytest.mark.parametrize("reorder", ["positions", "tokens"])
async def test_native_cross_prefill_rejects_reordered_query_ids(processor_state, side, reorder):
    sample = _sample()
    query_ids = np.array([[0, 5], [7, 9]], dtype=np.int32)
    returned_ids = query_ids[::-1] if reorder == "positions" else query_ids[:, ::-1]
    response = _response(
        sample.tokens,
        input_token_ids_logprobs=[None] + [[[-1.0, int(token)] for token in row] for row in returned_ids],
    )
    session = _Session(response)
    with pytest.raises(ValueError, match="query token IDs differ"):
        if side == "student":
            sample.teacher_topk_token_ids = query_ids
            await _manager(opd_token_selection="teacher_topk")._student_prefill(sample, session, None)
        else:
            sample.student_topk_token_ids = query_ids
            await _manager(opd_token_selection="student_topk")._teacher_prefill(sample, session)
    assert sample.student_at_teacher_topk_log_probs is None
    assert sample.teacher_at_student_topk_log_probs is None


@pytest.mark.parametrize("packed", [False, True])
def test_native_student_rollout_topk_requires_complete_rows(packed):
    sample = _sample()
    sample.rollout_log_probs = [-0.2, -0.3]
    meta = {"output_top_logprobs": [[[-1, 0]], [[-3, 7], [-4, 9]]]}
    if packed:
        meta = {
            "output_top_logprobs_val_b64": pybase64.b64encode(np.array([-1, -3, -4], dtype=np.float32)).decode(),
            "output_top_logprobs_idx_b64": pybase64.b64encode(np.array([0, 7, 9], dtype=np.int32)).decode(),
            "output_top_logprobs_k": [1, 2],
        }
    with pytest.raises(ValueError):
        _manager(opd_token_selection="student_topk").after_rollout(sample, {"output_ids": [10, 11], "meta_info": meta})
    assert sample.student_topk_token_ids is None


def test_native_student_rollout_topk_appends_exact_rows_across_turns():
    sample = _sample()
    sample.student_topk_token_ids = np.array([[2, 3]], dtype=np.int32)
    sample.student_topk_log_probs = np.array([[-2, -3]], dtype=np.float32)
    sample.rollout_log_probs = [-0.2, -0.3]
    output = {"output_ids": [11], "meta_info": {"output_top_logprobs": [[[-3, 7], [-4, 9]]]}}
    _manager(opd_token_selection="student_topk").after_rollout(sample, output)
    np.testing.assert_array_equal(sample.student_topk_token_ids, [[2, 3], [7, 9]])


def test_sampled_student_rollout_decodes_compact_logprobs():
    meta = {
        "output_token_logprobs_val_b64": pybase64.b64encode(np.array([-0.2, -0.3], dtype=np.float32)).decode(),
        "output_token_logprobs_idx_b64": pybase64.b64encode(np.array([10, 11], dtype=np.int32)).decode(),
    }
    ids, logprobs = _manager().parse_rollout_logprobs(meta, [10, 11], [])
    assert ids == [10, 11]
    assert logprobs == pytest.approx([-0.2, -0.3])


def test_image_contract_rejects_expanded_payload_missing_prompt_ids_and_unsupported_media():
    sample = _sample()
    with pytest.raises(ValueError, match="one collapsed placeholder"):
        validate_image_input_ids(sample.tokens, sample.tokens, 2, 1, 99)
    with pytest.raises(ValueError, match="return_prompt_token_ids"):
        validate_prompt_token_ids({}, sample.tokens)
    with pytest.raises(ValueError, match="audio/video"):
        image_inputs({"images": [object()], "videos": [object()]})


@pytest.fixture
def native_generate(monkeypatch):
    sglang_rollout = pytest.importorskip("relax.engine.rollout.sglang_rollout", exc_type=ModuleNotFoundError)
    args = SimpleNamespace(
        ci_test=False,
        sglang_router_ip="unused",
        sglang_router_port=1,
        use_slime_router=False,
        use_rollout_routing_replay=False,
        sglang_router_policy="round_robin",
        sglang_speculative_algorithm=None,
    )
    state = SimpleNamespace(
        tokenizer=SimpleNamespace(
            image_token_id=99,
            audio_token_id=98,
            video_token_id=97,
            pad_token_id=0,
            encode=lambda text, **kwargs: [1, 99, 2],
        ),
        processor=SimpleNamespace(image_token_id=99),
        opd_manager=_manager(),
    )
    output = {
        "prompt_token_ids": [1, 99, 99, 99, 2],
        "output_ids": [10, 11],
        "text": "response",
        "meta_info": {"output_token_logprobs": [[-0.2, 10], [-0.3, 11]], "finish_reason": {"type": "stop"}},
    }
    requests = []

    async def process(state, args, prompt, media):
        return [1, 99, 99, 99, 2], {"image_grid_thw": [[1, 2, 6]]}, 0.0

    async def post(args, state, url, payload, headers, evaluation):
        requests.append(copy.deepcopy(payload))
        return output

    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda args: state)
    monkeypatch.setattr(sglang_rollout, "_run_image_processor", process)
    monkeypatch.setattr(sglang_rollout, "_post_generation", post)
    sample = Sample(prompt="image prompt", multimodal_inputs={"images": [object()]})
    sample._pre_encoded_mm = {"image_data": ["student-image"]}
    return sglang_rollout, args, sample, state, output, requests


@pytest.mark.asyncio
async def test_native_generate_sends_collapsed_ids_and_preserves_expanded_sample(native_generate):
    rollout, args, sample, _, _, requests = native_generate
    assert await rollout.generate(args, sample, {"max_new_tokens": 2}) is sample
    assert requests[0]["input_ids"] == [1, 99, 2]
    assert requests[0]["return_prompt_token_ids"] is True
    assert requests[0]["image_data"] == ["student-image"]
    assert sample.tokens == [1, 99, 99, 99, 2, 10, 11]
    assert sample.rollout_tokens == [1, 99, 2, 10, 11]
    assert sample.rollout_log_probs == pytest.approx([-0.2, -0.3])
    assert sample.response_length == 2
    assert sample.status == Sample.Status.COMPLETED


@pytest.mark.asyncio
async def test_native_generate_rejects_expanded_request_before_posting(native_generate):
    rollout, args, sample, state, _, requests = native_generate
    state.tokenizer.encode = lambda text, **kwargs: [1, 99, 99, 99, 2]
    with pytest.raises(ValueError, match="one collapsed placeholder"):
        await rollout.generate(args, sample, {"max_new_tokens": 2})
    assert not requests


@pytest.mark.asyncio
@pytest.mark.parametrize("prompt_ids", [None, [1, 99, 99, 99, 3]])
async def test_native_generate_rejects_missing_or_changed_prompt_ids(native_generate, prompt_ids):
    rollout, args, sample, _, output, _ = native_generate
    if prompt_ids is None:
        del output["prompt_token_ids"]
    else:
        output["prompt_token_ids"] = prompt_ids
    with pytest.raises(ValueError, match="return_prompt_token_ids|expanded input IDs differ"):
        await rollout.generate(args, sample, {"max_new_tokens": 2})
    assert sample.response_length == 0
    assert sample.rollout_log_probs is None


@pytest.mark.asyncio
@pytest.mark.parametrize("media_token", [99, 98, 97])
async def test_native_generate_rejects_sampled_media_tokens_before_replacement(native_generate, media_token):
    rollout, args, sample, _, output, _ = native_generate
    output["output_ids"] = [media_token]
    output["meta_info"]["output_token_logprobs"] = [[-0.2, media_token]]
    with pytest.raises(ValueError, match="cannot replace sampled media tokens"):
        await rollout.generate(args, sample, {"max_new_tokens": 2})
    assert sample.tokens == [1, 99, 99, 99, 2]
    assert sample.rollout_tokens == [1, 99, 2]
    assert sample.response_length == 0
    assert sample.rollout_log_probs is None


@pytest.mark.asyncio
@pytest.mark.parametrize("image_side", ["student", "teacher"])
async def test_image_opd_rejects_text_radix_routing_before_generation(monkeypatch, image_side):
    sglang_rollout = pytest.importorskip("relax.engine.rollout.sglang_rollout", exc_type=ModuleNotFoundError)

    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda args: SimpleNamespace(opd_manager=object()))
    args = SimpleNamespace(
        ci_test=False,
        sglang_router_ip="unused",
        sglang_router_port=1,
        use_slime_router=True,
        slime_router_middleware_paths=["RadixTreeMiddleware"],
    )
    sample = Sample(prompt="text")
    if image_side == "student":
        sample.multimodal_inputs = {"images": [object()]}
    else:
        sample.teacher_multimodal_inputs = {"images": [object()]}
    with pytest.raises(ValueError, match="RadixTreeMiddleware uses text lookup"):
        await sglang_rollout.generate(args, sample, {"max_new_tokens": 4})


@pytest.mark.asyncio
async def test_image_encoding_keeps_shared_samples_serializable_while_pending(monkeypatch):
    from relax.utils.data import processing_utils

    started, finish = asyncio.Event(), asyncio.Event()
    calls = []

    async def encode(image):
        calls.append(image)
        started.set()
        await finish.wait()
        return "encoded-image"

    monkeypatch.setattr(processing_utils, "async_encode_image_for_rollout_engine", encode)
    sample = Sample(prompt="image prompt", multimodal_inputs={"images": ["raw-image"]})
    task = asyncio.create_task(encode_image_inputs(sample.multimodal_inputs))
    await started.wait()
    try:
        assert copy.deepcopy(sample).multimodal_inputs == {"images": ["raw-image"]}
        assert pickle.loads(pickle.dumps(sample)).multimodal_inputs == {"images": ["raw-image"]}
    finally:
        finish.set()
        payload = await task
    assert payload == {"image_data": ["encoded-image"]}
    restored = pickle.loads(pickle.dumps(sample))
    assert await encode_image_inputs(restored.multimodal_inputs) == payload
    assert calls == ["raw-image"]


@pytest.mark.asyncio
@pytest.mark.parametrize("use_pool", [False, True])
async def test_image_processor_excludes_transport_cache_from_local_and_worker_inputs(monkeypatch, use_pool):
    sglang_rollout = pytest.importorskip("relax.engine.rollout.sglang_rollout", exc_type=ModuleNotFoundError)

    media = {"images": ["raw-image"], "_opd_image_b64_cache": ["encoded-image"]}
    received = []

    def processor(text, **kwargs):
        assert "_opd_image_b64_cache" not in kwargs
        received.append(kwargs["images"])
        return {"input_ids": [[1, 99, 2]]}

    def worker(text, inputs, kwargs):
        assert "_opd_image_b64_cache" not in inputs
        received.append(inputs["images"])
        return [1, 99, 2], None

    monkeypatch.setattr(sglang_rollout, "process_sample_in_worker", worker)
    state = SimpleNamespace(
        processor=processor,
        processor_pool=SimpleNamespace(executor=None) if use_pool else None,
        encode_executor=None,
    )
    ids, train_inputs, _ = await sglang_rollout._run_image_processor(
        state, SimpleNamespace(use_audio_in_video=False), "image prompt", media
    )
    assert ids == [1, 99, 2]
    assert train_inputs is None
    assert received == [["raw-image"]]
    assert media["_opd_image_b64_cache"] == ["encoded-image"]
