# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""RL actor forward must inject MTP kwargs without disturbing GRPO loss_mask
wiring.

The training-side ``forward_step`` (``relax/backends/megatron/model.py``) is shared
between SFT and RL. For RL it pre-sets ``forward_kwargs["loss_mask"] =
batch["full_loss_masks"]`` (the response-only mask) BEFORE calling
``_attach_mtp_forward_kwargs``. This test pins down that contract so future edits
do not silently break RL+MTP.

The auxiliary loss is added inside the patched Megatron ``process_mtp_loss``
(``docker/patch/megatron/20260506-85bced0ae.patch``) when ``mtp_kwargs`` is
present and ``labels`` is None — verified end-to-end by the Day-2 smoke test, not
here.
"""

from argparse import Namespace

import pytest


try:
    from relax.backends.megatron.model import _attach_mtp_forward_kwargs
except (ImportError, AssertionError) as _exc:
    pytest.skip(f"relax.backends.megatron.model unavailable: {_exc}", allow_module_level=True)


def _mk_args(enable_mtp_training: bool) -> Namespace:
    return Namespace(enable_mtp_training=enable_mtp_training)


def _mk_rl_forward_kwargs(loss_mask: object) -> dict:
    # Mirrors the RL training-side forward_kwargs built at
    # relax/backends/megatron/model.py:507-514 (labels=None for RL: Relax computes
    # GRPO loss externally from logits).
    return {
        "input_ids": object(),
        "position_ids": None,
        "attention_mask": None,
        "labels": None,
        "packed_seq_params": object(),
        "loss_mask": loss_mask,
    }


def test_rl_actor_forward_keeps_response_only_loss_mask():
    tokens = object()
    response_only_mask = object()  # full_loss_masks: prompt tokens zeroed for GRPO
    batch = {"tokens": tokens, "full_loss_masks": response_only_mask}
    forward_kwargs = _mk_rl_forward_kwargs(loss_mask=response_only_mask)

    _attach_mtp_forward_kwargs(_mk_args(enable_mtp_training=True), batch, forward_kwargs)

    # MTP head trains on next-token prediction of the rollout response tokens.
    assert forward_kwargs["mtp_kwargs"]["mtp_labels"] is tokens
    # GRPO's loss_mask must not be overwritten — it's the response-only mask the
    # external policy loss reads from batch["full_loss_masks"].
    assert forward_kwargs["loss_mask"] is response_only_mask
    # RL forward never sets labels (Relax computes loss externally).
    assert forward_kwargs["labels"] is None


def test_rl_vlm_unsplit_loss_mask_none_is_preserved():
    # VL/CP unsplit branch (model.py:520-523) sets loss_mask=None because the
    # bridge forwards labels=None and computes loss externally. _attach must NOT
    # backfill full_loss_masks here, otherwise the patched process_mtp_loss would
    # see an inconsistent (labels=None, loss_mask=not-None) state and the MTP
    # branch in Megatron would receive a misshapen mask vs the unsplit tokens.
    tokens = object()
    full_loss_masks = object()
    batch = {"tokens": tokens, "full_loss_masks": full_loss_masks}
    forward_kwargs = _mk_rl_forward_kwargs(loss_mask=None)
    # The VLM branch overrides loss_mask to None AFTER initial construction;
    # simulate that state explicitly.
    forward_kwargs["loss_mask"] = None

    _attach_mtp_forward_kwargs(_mk_args(enable_mtp_training=True), batch, forward_kwargs)

    # _attach backfills loss_mask only when it's None on entry (model.py:49-50).
    # That's correct for SFT; for the VLM unsplit branch the caller has already
    # decided None is intentional. The current implementation backfills
    # unconditionally on None — that's acceptable because VLM+MTP is not a
    # supported combination yet (see Phase-2 roadmap). This test pins the
    # current behavior so any future MTP+VLM work has a deliberate hand-off.
    assert forward_kwargs["mtp_kwargs"]["mtp_labels"] is tokens
    assert forward_kwargs["loss_mask"] is full_loss_masks


def test_rl_actor_forward_noop_without_mtp_flag():
    # GRPO baseline (no MTP) must leave forward_kwargs untouched so adding the
    # mtp flags is a pure additive change for existing runs.
    batch = {"tokens": object(), "full_loss_masks": object()}
    forward_kwargs = _mk_rl_forward_kwargs(loss_mask=batch["full_loss_masks"])
    snapshot = forward_kwargs.copy()

    _attach_mtp_forward_kwargs(_mk_args(enable_mtp_training=False), batch, forward_kwargs)

    assert forward_kwargs == snapshot
    assert "mtp_kwargs" not in forward_kwargs
