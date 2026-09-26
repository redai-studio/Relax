# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""GPU smoke test for production capture (PR B).

Runs the real hook path — capture_hooks.capture_policy_loss with CUDA
tensors through the async CaptureManager — and verifies the produced bundle
replays cleanly on CPU. Skipped on CPU-only hosts.

This is a semi-integration smoke: it exercises the actual hook functions and
the deferred GPU→CPU copy on the writer thread, but not build_identity
(which needs a Megatron mpu world). Full DP/TP parity is validated by a
real training run on the target cluster.
"""

from __future__ import annotations

import pytest
import torch

from relax.utils.replay import capture_hooks
from relax.utils.replay.capture import CaptureConfig, begin_step, disable, enable, end_step
from relax.utils.replay.runner import replay
from relax.utils.replay.schema import StageId
from tests.utils.replay.helpers import make_capture_record


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def test_capture_policy_loss_gpu_roundtrip(tmp_path):
    device = torch.device("cuda")
    reference = make_capture_record("b-gpu")

    enable(CaptureConfig(enabled=True, output_dir=str(tmp_path)))
    begin_step((120, 0), identity=reference.identity, config=reference.config, bundle_id="b-gpu")

    reported_loss = {
        field: torch.tensor(value, device=device)
        for field, value in reference.expected[StageId.LOSS_POLICY.value].items()
    }
    capture_hooks.capture_policy_loss(
        old_log_probs=reference.tensors["old_log_probs"].to(device),
        log_probs=reference.tensors["log_probs"].to(device),
        entropy=reference.tensors["entropy"].to(device),
        advantages=reference.tensors["advantages"].to(device),
        loss_masks=[torch.tensor(sample.loss_mask, device=device) for sample in reference.samples],
        response_lengths=[sample.response_length for sample in reference.samples],
        total_lengths=[sample.total_length for sample in reference.samples],
        reported_loss=reported_loss,
    )
    end_step()
    disable()

    assert (tmp_path / "b-gpu" / "COMPLETE").exists()
    assert replay(tmp_path / "b-gpu").passed
