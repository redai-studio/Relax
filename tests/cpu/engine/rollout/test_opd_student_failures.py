# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from contextlib import asynccontextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from relax.engine.inference import deferred
from relax.engine.rollout import on_policy_distillation as opd
from relax.utils import utils as relax_utils
from relax.utils.types import Sample
from tests.cpu.engine.inference.test_deferred_scoring import _DataSystem, _teacher_args, _teacher_coordinator


@pytest.mark.parametrize("selection", ["teacher_topk", "union"])
@pytest.mark.parametrize("failure", ["all", "partial", "malformed"])
async def test_deferred_student_failure_matches_inline_and_gates_publication(monkeypatch, selection, failure):
    args = _teacher_args(
        opd_token_selection=selection,
        opd_log_prob_top_k=2,
        opd_kl_coef=1.0,
        opd_loss_coef=0.0,
        sglang_router_ip="student",
        sglang_router_port=1,
    )
    manager = opd.OpdManager(args)
    events, published = [], _DataSystem()

    @asynccontextmanager
    async def session(args):
        yield object()

    async def teacher(sample, session):
        sample.teacher_topk_token_ids = np.array([[2, 3]], dtype=np.int32)
        sample.teacher_topk_log_probs = np.array([[-0.2, -0.3]], dtype=np.float32)
        sample.student_topk_token_ids = sample.teacher_topk_token_ids.copy()
        sample.student_topk_log_probs = sample.teacher_topk_log_probs.copy()
        sample.teacher_at_student_topk_log_probs = sample.teacher_topk_log_probs.copy()
        return True

    async def post(session, url, payload, sample, *args, **kwargs):
        if failure == "all" or (failure == "partial" and sample.index == 0):
            return None
        values = None if failure == "malformed" else np.array([[-0.4, -0.5]], dtype=np.float32)
        return SimpleNamespace(other_topk=lambda *args: values)

    monkeypatch.setattr(opd, "_create_teacher_client_session", session)
    monkeypatch.setattr(manager, "_teacher_prefill", teacher)
    monkeypatch.setattr(manager, "_post_logprob", post)
    monkeypatch.setattr(opd, "OpdManager", lambda args: manager)
    monkeypatch.setattr(deferred, "build_local_coordinator", lambda args: _teacher_coordinator(args, events))
    monkeypatch.setattr(
        relax_utils,
        "convert_samples_to_train_data",
        lambda args, samples: SimpleNamespace(samples=list(samples), numel=lambda: len(samples)),
    )
    monkeypatch.setattr(relax_utils, "build_rollout_custom_meta", lambda batch: {})
    monkeypatch.setattr(relax_utils, "CURRENT_ROLLOUT_BATCH", [])

    def samples():
        return [Sample(index=i, tokens=[1, 2], response_length=1, reward=1.0) for i in range(2)]

    inline, batch = samples(), samples()
    if failure != "partial":
        with pytest.raises(RuntimeError, match="All OPD student fetches failed"):
            await relax_utils.transfer_batch_to_data_system(args, [batch], 1, 0, published)
        with pytest.raises(RuntimeError, match="All OPD student fetches failed"):
            await manager.prefill(inline)
        assert published.published == [] and relax_utils.CURRENT_ROLLOUT_BATCH == []
        assert events[-2:] == ["teacher offload", "rollout onload"]
    else:
        await manager.prefill(inline)
        await relax_utils.transfer_batch_to_data_system(args, [batch], 1, 0, published)
        inline_data, deferred_data = {}, {}
        manager.produce_opd_transfer_data(inline, inline_data)
        manager.produce_opd_transfer_data(batch, deferred_data)
        assert inline_data == deferred_data
        assert deferred_data["opd_topk_student_log_probs"][0] == []
        assert deferred_data["opd_topk_student_log_probs"][1]
        assert published.published == [batch]
