# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace

import pytest
import torch


try:
    from relax.backends.megatron import actor as actor_module
except (ImportError, AssertionError) as exc:
    pytest.skip(f"relax.backends.megatron.actor unavailable: {exc}", allow_module_level=True)


def test_sft_prepacked_iterator_close_releases_device_batch_and_event():
    iterator = actor_module._SFTPrepackedDeviceIterator(
        packed_cpu=[],
        first_device_micro_batch=(actor_module.PrepackedBatch(), None),
        first_ready_event=object(),
        copy_stream=object(),
        device=torch.device("cpu"),
    )

    iterator.close()

    assert iterator._packed_cpu == []
    assert iterator._next_device_micro_batch is None
    assert iterator._next_ready_event is None


def test_sft_peer_error_agreement_propagates_peer_failure(monkeypatch):
    groups = []

    def mark_peer_error(error_flag, *, op, group):
        groups.append(group)
        error_flag.fill_(1)

    monkeypatch.setattr(actor_module.dist, "all_reduce", mark_peer_error)

    with pytest.raises(RuntimeError, match="failed on a peer rank"):
        actor_module._raise_if_sft_peer_error(
            None,
            phase="validation",
            device=torch.device("cpu"),
            tp_group="tp",
            dp_group="dp",
        )

    assert groups == ["tp", "dp"]


def test_sft_lookahead_pauses_on_checkpoint_boundary(monkeypatch):
    monkeypatch.setattr(actor_module, "should_run_sft_eval", lambda *_args: False)
    monkeypatch.setattr(actor_module, "should_run_sft_predict", lambda *_args: False)
    args = Namespace(num_rollout=100, save="/checkpoint", rotate_ckpt=False, save_interval=20)

    assert actor_module._should_pause_sft_prepack_lookahead(args, rollout_id=19) is True
    assert actor_module._should_pause_sft_prepack_lookahead(args, rollout_id=18) is False
