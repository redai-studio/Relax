# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""OpdManager / TopkWorker payload construction + logprob parsing.

Refactored: the old module-level helpers (``_extract_teacher_topk_pair``,
``_post_teacher_request_with_diagnostics``, ``fetch_teacher_log_probs``,
``_compute_image_id``) were folded into ``LogprobResponse`` /
``TopkWorker`` / ``OpdManager``. Teacher/student top-k logprobs are now
carried as sglang base64 fields, decoded into numpy arrays.
"""

from types import SimpleNamespace

import numpy as np
import pybase64

from relax.engine.rollout.on_policy_distillation import OpdManager
from relax.utils.opd.opd_main_worker import LogprobResponse, TopkWorker


def _b64(arr: np.ndarray) -> str:
    return pybase64.b64encode(arr.tobytes()).decode("utf-8")


def test_teacher_prefill_self_topk_keeps_token_id_zero_and_takes_tail() -> None:
    # Total n=2 rows; response_length=1 -> keep the last row. token_id 0 kept.
    vals = np.array([-1.0, -2.0, -0.1, -0.2], dtype=np.float32)
    ids = np.array([100, 200, 0, 5], dtype=np.int32)
    resp = {
        "meta_info": {
            "input_top_logprobs_val_b64": _b64(vals),
            "input_top_logprobs_idx_b64": _b64(ids),
        }
    }

    pair = LogprobResponse(resp).self_topk("prefill", top_k=2, response_length=1)

    assert pair is not None
    out_ids, out_lps = pair
    np.testing.assert_array_equal(out_ids, np.array([[0, 5]], dtype=np.int32))
    np.testing.assert_allclose(out_lps, np.array([[-0.1, -0.2]], dtype=np.float32))


def test_base_logprobs_1d_from_b64_and_legacy() -> None:
    val = np.array([-0.5, -0.7, -0.9], dtype=np.float32)
    r1 = {"meta_info": {"input_token_logprobs_val_b64": _b64(val)}}
    np.testing.assert_allclose(LogprobResponse(r1).base_logprobs_1d(), val)

    # Legacy plain-list fallback: list of [logprob, token_id] pairs.
    r2 = {"meta_info": {"input_token_logprobs": [[-0.5, 1], [-0.7, 2]]}}
    np.testing.assert_allclose(LogprobResponse(r2).base_logprobs_1d(), np.array([-0.5, -0.7], dtype=np.float32))


def test_build_teacher_payload_adds_student_topk_query_ids() -> None:
    # student_topk + kl_coef != 0 => teacher_at_student=True => token_ids_logprob present.
    w = TopkWorker("student_topk", top_k=2, opd_kl_coef=1.0, opd_loss_coef=0.0)
    student_topk_ids = np.array([[3, 5], [7, 9]], dtype=np.int64)  # R=2, K=2

    payload = w.build_teacher_payload(
        input_ids=[1, 2, 3, 4],
        logprob_start_len=1,
        student_topk_ids=student_topk_ids,
        response_length=2,
    )

    assert payload["input_ids"] == [1, 2, 3, 4]
    assert payload["logprob_start_len"] == 1
    assert payload["return_logprob"] is True
    # _flatten_other_topk_ids: [[3,5],[7,9]] flattened + trailing [0]*top_k
    assert payload["token_ids_logprob"] == [3, 5, 7, 9, 0, 0]


def test_build_transfer_channels_union_merges_and_pads() -> None:
    # union + kl_coef != 0 (is_advantage=True): merge student/teacher self-topk
    # ids per position, keep first-seen logprobs, pad ragged rows.
    w = TopkWorker("union", top_k=2, opd_kl_coef=1.0, opd_loss_coef=0.0)

    s_ids = np.array([[3, 5], [1, 2]], dtype=np.int32)
    s_student_lp = np.array([[-0.1, -0.2], [-0.5, -0.6]], dtype=np.float32)
    t_ids = np.array([[5, 7], [1, 2]], dtype=np.int32)
    t_teacher_lp = np.array([[-0.3, -0.4], [-0.7, -0.8]], dtype=np.float32)
    teacher_at_student_lp = np.array([[-1.0, -1.1], [-1.5, -1.6]], dtype=np.float32)
    student_at_teacher_lp = np.array([[-2.0, -2.1], [-2.5, -2.6]], dtype=np.float32)

    channels = w.build_transfer_channels(
        student_self_topk=(s_ids, s_student_lp),
        teacher_self_topk=(t_ids, t_teacher_lp),
        teacher_at_student_lp=teacher_at_student_lp,
        student_at_teacher_lp=student_at_teacher_lp,
    )

    # row0: union({3,5},{5,7}) = [3,5,7] (kp=3); row1: union({1,2},{1,2}) = [1,2] (kp=2, padded to 3 with -1)
    np.testing.assert_array_equal(
        channels[TopkWorker.TRANSFER_TOKEN_IDS],
        np.array([[3, 5, 7], [1, 2, -1]], dtype=np.int32),
    )
    # teacher logprob per unique id (first-seen): row0 keeps s_teacher for 3,5 and t_teacher for 7
    np.testing.assert_allclose(
        channels[TopkWorker.TRANSFER_TEACHER_LOG_PROBS],
        np.array([[-1.0, -1.1, -0.4], [-1.5, -1.6, 0.0]], dtype=np.float32),
    )
    # student logprob per unique id (first-seen)
    np.testing.assert_allclose(
        channels[TopkWorker.TRANSFER_STUDENT_LOG_PROBS],
        np.array([[-0.1, -0.2, -2.1], [-0.5, -0.6, 0.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(channels[TopkWorker.TRANSFER_K_LENGTHS], np.array([3, 2], dtype=np.int32))


def test_deferred_union_transfer_is_strict_and_serializable() -> None:
    worker = TopkWorker("union", top_k=2, opd_kl_coef=1.0, opd_loss_coef=0.0)
    manager = OpdManager.__new__(OpdManager)
    manager.topk_worker = worker
    manager.sampled_worker = None

    samples = []
    for offset in (0, 10):
        student_ids = np.array([[offset + 1, offset + 2], [offset + 3, offset + 4]], dtype=np.int32)
        student_lps = np.full((2, 2), -0.1 - offset, dtype=np.float32)
        teacher_ids = np.array([[offset + 2, offset + 5], [offset + 3, offset + 6]], dtype=np.int32)
        teacher_lps = np.full((2, 2), -0.3 - offset, dtype=np.float32)
        sample = SimpleNamespace(
            index=offset,
            response_length=2,
            student_topk_token_ids=student_ids,
            student_topk_log_probs=student_lps,
            teacher_topk_token_ids=teacher_ids,
            teacher_topk_log_probs=teacher_lps,
            teacher_at_student_topk_log_probs=np.full((2, 2), -0.5 - offset, dtype=np.float32),
            student_at_teacher_topk_log_probs=np.full((2, 2), -0.7 - offset, dtype=np.float32),
        )
        samples.append(sample)

    manager.finish_prefill(samples, strict=True)
    train_data = {}
    manager.produce_opd_transfer_data(samples, train_data)

    assert all(len(row) == 6 for row in train_data[TopkWorker.TRANSFER_TOKEN_IDS])
    assert all(len(row) == 6 for row in train_data[TopkWorker.TRANSFER_TEACHER_LOG_PROBS])
    assert all(len(row) == 6 for row in train_data[TopkWorker.TRANSFER_STUDENT_LOG_PROBS])
    assert train_data[TopkWorker.TRANSFER_K_LENGTHS] == [[3, 3], [3, 3]]
