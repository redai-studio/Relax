# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Opt-in HTTP checks against an already running SGLang v0.5.17 engine/router.

Set OPD_GPU_ENDPOINT (base URL) and OPD_GPU_MODEL_PATH (matching HF
checkpoint). The endpoint needs Relax's per-position patch. These tests do not
launch or stop services. Use a prefill chunk smaller than 594 and a logits
chunk smaller than 514 to exercise both chunk boundaries in the long-response
cases.
"""

import base64
import io
import os
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from urllib.parse import urlparse

import numpy as np
import pytest
import requests

from relax.utils.opd.opd_main_worker import LogprobResponse, build_prefill_payload_base
from relax.utils.opd.opd_multimodal import validate_image_input_ids, validate_prompt_token_ids


pytestmark = pytest.mark.skipif(
    not (os.environ.get("OPD_GPU_ENDPOINT") and os.environ.get("OPD_GPU_MODEL_PATH")),
    reason="Requires an explicit GPU SGLang v0.5.17 endpoint and its matching model/processor.",
)


@pytest.fixture(scope="module")
def processor():
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(os.environ["OPD_GPU_MODEL_PATH"], local_files_only=True)


def _post(payload: dict) -> dict:
    response = requests.post(os.environ["OPD_GPU_ENDPOINT"].rstrip("/") + "/generate", json=payload, timeout=300)
    response.raise_for_status()
    return response.json()


def _inputs(processor, image_count: int, repeats: int) -> tuple[dict, list[int], int]:
    from PIL import Image

    images = [Image.new("RGB", (64 + i * 32, 64), color) for i, color in enumerate(("red", "blue")[:image_count])]
    content = [{"type": "image"} for _ in images] + [{"type": "text", "text": "Describe the colors."}]
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
    )
    tokenizer = processor.tokenizer
    response_ids = tokenizer.encode("h", add_special_tokens=False) + tokenizer.encode("ello", add_special_tokens=False)
    assert tokenizer.encode(tokenizer.decode(response_ids), add_special_tokens=False) != response_ids
    response_ids *= repeats
    collapsed = tokenizer.encode(prompt, add_special_tokens=False)
    expanded = list(processor(text=prompt, images=images)["input_ids"][0])
    image_data = []
    for image in images:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_data.append(base64.b64encode(buffer.getvalue()).decode())
    payload = build_prefill_payload_base(collapsed + response_ids, len(expanded) - 1)
    payload.update(image_data=image_data, return_prompt_token_ids=True, top_logprobs_num=16)
    expected_ids = expanded + response_ids
    validate_image_input_ids(
        payload["input_ids"], expected_ids, len(response_ids), image_count, processor.image_token_id
    )
    return payload, expected_ids, len(response_ids)


@pytest.mark.parametrize("image_count,repeats,top_k", [(1, 1, 1), (2, 1, 2), (1, 257, 2), (2, 257, 1)])
def test_native_image_per_position_logprobs_match_same_response_self_topk(processor, image_count, repeats, top_k):
    payload, expected, length = _inputs(processor, image_count, repeats)
    first = LogprobResponse(_post(payload))
    first.validate_native_prefill(expected, length)
    self_ids, _ = first.strict_topk("input_top_logprobs", length, 16)
    query_ids = np.asarray([row[i % 4 : i % 4 + top_k] for i, row in enumerate(self_ids)], dtype=np.int32)
    payload["token_ids_logprob"] = query_ids.reshape(-1).tolist() + [0] * top_k
    payload["sampling_params"]["custom_params"] = {"relax_opd_token_ids_logprob_k": top_k}
    actual = LogprobResponse(_post(payload))
    actual.validate_native_prefill(expected, length)
    ids, values = actual.strict_topk("input_top_logprobs", length, 16)
    _, queried = actual.strict_topk("input_token_ids_logprobs", length, top_k, query_ids)
    # Use the SAME forward's independent self-topk values as an oracle. Prefix
    # and vision caches can change BF16 rounding between separate forwards.
    expected_values = [
        [dict(zip(row_ids, row_values))[token] for token in row_queries]
        for row_ids, row_values, row_queries in zip(ids, values, query_ids, strict=True)
    ]
    np.testing.assert_allclose(queried, expected_values, atol=1e-6, rtol=0)


def test_native_image_rollout_preserves_prompt_and_returns_complete_output_topk(processor):
    payload, expanded, length = _inputs(processor, 1, 1)
    payload["input_ids"] = payload["input_ids"][:-length]
    payload.pop("logprob_start_len")
    payload["sampling_params"]["max_new_tokens"] = 8
    response = _post(payload)
    validate_prompt_token_ids(response, expanded[:-length])
    output_ids = response["output_ids"]
    assert output_ids
    ids, values = LogprobResponse(response).strict_topk("output_top_logprobs", len(output_ids), 16)
    assert ids.shape == values.shape == (len(output_ids), 16)


def test_native_image_mixed_query_widths_and_malformed_request(processor):
    requests_to_send = []
    for top_k in (1, 3):
        payload, expected, length = _inputs(processor, 1, 1)
        queries = np.arange(length * top_k, dtype=np.int32).reshape(length, top_k)
        payload["token_ids_logprob"] = queries.reshape(-1).tolist() + [0] * top_k
        payload["sampling_params"]["custom_params"] = {"relax_opd_token_ids_logprob_k": top_k}
        requests_to_send.append((payload, expected, length, queries))
    with ThreadPoolExecutor(max_workers=2) as pool:
        outputs = list(pool.map(_post, [item[0] for item in requests_to_send]))
    for output, (_, expected, length, queries) in zip(outputs, requests_to_send, strict=True):
        response = LogprobResponse(output)
        response.validate_native_prefill(expected, length)
        response.strict_topk("input_token_ids_logprobs", length, queries.shape[1], queries)
    bad_payload = requests_to_send[0][0]
    bad_payload["token_ids_logprob"].pop()
    response = requests.post(os.environ["OPD_GPU_ENDPOINT"].rstrip("/") + "/generate", json=bad_payload, timeout=30)
    assert response.status_code == 400, response.text[:300]
    # Rejection must not poison the scheduler or the next well-formed request.
    _post(requests_to_send[1][0])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "selection,override_teacher",
    [
        ("student_sampled", False),
        ("student_sampled", True),
        ("student_topk", True),
        ("teacher_topk", True),
        ("union", True),
    ],
)
async def test_native_image_opd_manager_completes_real_rollout_and_teacher_prefills(
    processor, monkeypatch, selection, override_teacher
):
    from PIL import Image

    from relax.engine.rollout import sglang_rollout
    from relax.engine.rollout.on_policy_distillation import OpdManager
    from relax.utils.types import Sample

    endpoint = urlparse(os.environ["OPD_GPU_ENDPOINT"])
    args = SimpleNamespace(
        opd_token_selection=selection,
        opd_log_prob_top_k=4,
        opd_kl_coef=1.0,
        opd_loss_coef=0.0,
        opd_teacher_url=os.environ["OPD_GPU_ENDPOINT"].rstrip("/") + "/generate",
        opd_teacher_timeout_s=300,
        opd_teacher_prompt_key="teacher_prompt" if override_teacher else None,
        opd_teacher_response_prefix_token_ids=[] if override_teacher else processor.tokenizer.encode("Answer:"),
        sglang_router_ip=endpoint.hostname,
        sglang_router_port=endpoint.port or 80,
        use_audio_in_video=False,
    )
    state = SimpleNamespace(
        processor=processor, tokenizer=processor.tokenizer, processor_pool=None, encode_executor=None
    )
    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda args: state)
    manager = OpdManager(args)
    payload, expanded, length = _inputs(processor, 1, 1)
    collapsed = payload["input_ids"][:-length]
    payload["input_ids"] = collapsed
    payload.pop("logprob_start_len")
    payload.pop("top_logprobs_num")
    payload["sampling_params"]["max_new_tokens"] = 8
    manager.before_rollout(payload)
    output = _post(payload)
    validate_prompt_token_ids(output, expanded[:-length])
    response_ids, logprobs = manager.parse_rollout_logprobs(output["meta_info"], output["output_ids"], [])
    if not logprobs:
        logprobs = [row[0] for row in output["meta_info"]["output_token_logprobs"]]
    sample = Sample(
        prompt=processor.tokenizer.decode(collapsed),
        tokens=expanded[:-length] + response_ids,
        rollout_tokens=collapsed + response_ids,
        response=processor.tokenizer.decode(response_ids),
        response_length=len(response_ids),
        rollout_log_probs=logprobs,
        multimodal_inputs={"images": [Image.new("RGB", (64, 64), "red")]},
    )
    if override_teacher:
        sample.teacher_prompt = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Name the dominant color."}]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        sample.teacher_multimodal_inputs = {"images": [Image.new("RGB", (64, 64), "blue")]}
    captured = {}
    post_logprob = manager._post_logprob

    async def record_post(session, url, request, sample, err_tag, expected_input_ids=None):
        captured[err_tag] = request
        return await post_logprob(session, url, request, sample, err_tag, expected_input_ids)

    monkeypatch.setattr(manager, "_post_logprob", record_post)
    manager.after_rollout(sample, output)
    await manager.prefill([sample])
    assert sample.tokens[-sample.response_length :] == output["output_ids"]
    if override_teacher:
        assert sample.teacher_rollout_tokens[-sample.response_length :] == response_ids
        assert sample.teacher_rollout_tokens[: -sample.response_length] != collapsed
    for request_name, request in captured.items():
        image = Image.open(io.BytesIO(base64.b64decode(request["image_data"][0]))).convert("RGB")
        red, _, blue = image.getpixel((0, 0))
        teacher_image = request_name == "teacher prefill" and override_teacher
        assert (blue > 200 and red < 10) if teacher_image else (red > 200 and blue < 10)
    if selection == "student_sampled":
        assert len(sample.teacher_log_probs) == sample.response_length
        assert np.isfinite(sample.teacher_log_probs).all()
    else:
        assert sample.opd_topk_token_ids.shape[0] == sample.response_length
        assert sample.opd_topk_student_log_probs.shape == sample.opd_topk_teacher_log_probs.shape
        assert sample.opd_topk_student_log_probs.shape == sample.opd_topk_token_ids.shape
        assert np.isfinite(sample.opd_topk_student_log_probs).all()
        assert np.isfinite(sample.opd_topk_teacher_log_probs).all()
        if selection in ("teacher_topk", "union"):
            assert "student-at-teacher-topk" in captured
