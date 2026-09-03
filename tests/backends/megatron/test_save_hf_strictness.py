# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Strictness decision in ``save_hf_model``.

Bridge under ``strict=True`` refuses every safetensors shard containing a key
the training model never emits, which loses the real tensors sharing those
shards rather than just the absent ones. So strictness is relaxed exactly when
the reference declares a group the model structurally cannot produce -- MTP
layers a model trained without MTP, or the vision tower of a VL base trained
text-only -- and stays on for everything else.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
pytest.importorskip("megatron.bridge")

import relax.utils.hf_export as hf_export  # noqa: E402
from relax.backends.megatron import model as model_mod  # noqa: E402


class _NullCtx:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def _install_fakes(monkeypatch, recorded):
    class _FakeBridge:
        def save_hf_pretrained(self, model, path, strict):
            recorded["strict"] = strict
            (path / "config.json").write_text("{}")

    monkeypatch.setattr(
        "megatron.bridge.AutoBridge.from_hf_pretrained",
        classmethod(lambda cls, *a, **k: _FakeBridge()),
    )
    monkeypatch.setattr(model_mod, "patch_megatron_model", _NullCtx)
    monkeypatch.setattr(model_mod, "is_lora_enabled", lambda args: False)
    monkeypatch.setattr(model_mod.torch.distributed, "is_initialized", lambda: False)


def _args(tmp_path):
    return SimpleNamespace(
        save_hf=str(tmp_path / "hf_out/iter_{rollout_id}"),
        save_hf_dtype="bf16",
        hf_checkpoint="/tmp/fake-hf",
        mtp_num_layers=None,
    )


def _model(*, vision: bool):
    """A model whose real ``get_model_config`` walk finds a provider-like
    config.

    Only the VL providers declare ``vision_config``; the plain text providers
    have no such field, which is what distinguishes them here.
    """
    config = SimpleNamespace(vision_config=object()) if vision else SimpleNamespace()
    return [SimpleNamespace(config=config)]


def test_vision_reference_with_text_only_model_relaxes_strict(monkeypatch, tmp_path):
    recorded = {}
    _install_fakes(monkeypatch, recorded)
    monkeypatch.setattr(hf_export, "reference_expects_mtp", lambda path: False)
    monkeypatch.setattr(hf_export, "reference_expects_vision", lambda path: True)

    model_mod.save_hf_model(_args(tmp_path), rollout_id=1, model=_model(vision=False))

    assert recorded["strict"] is False


def test_vision_reference_with_vl_model_keeps_strict(monkeypatch, tmp_path):
    recorded = {}
    _install_fakes(monkeypatch, recorded)
    monkeypatch.setattr(hf_export, "reference_expects_mtp", lambda path: False)
    monkeypatch.setattr(hf_export, "reference_expects_vision", lambda path: True)

    # The model does build a vision tower, so nothing is structurally absent.
    model_mod.save_hf_model(_args(tmp_path), rollout_id=2, model=_model(vision=True))

    assert recorded["strict"] is True


def test_plain_reference_keeps_strict_without_touching_the_model(monkeypatch, tmp_path):
    """A non-VL reference must short-circuit before the model config is read.

    ``model=[]`` makes ``get_model_config(model[0])`` raise, and save_hf_model
    swallows exceptions, so an eager check here silently skipped the whole
    export.
    """
    recorded = {}
    _install_fakes(monkeypatch, recorded)
    monkeypatch.setattr(hf_export, "reference_expects_mtp", lambda path: False)
    monkeypatch.setattr(hf_export, "reference_expects_vision", lambda path: False)

    model_mod.save_hf_model(_args(tmp_path), rollout_id=3, model=[])

    assert recorded["strict"] is True


def test_mtp_reference_without_mtp_model_relaxes_strict(monkeypatch, tmp_path):
    recorded = {}
    _install_fakes(monkeypatch, recorded)
    monkeypatch.setattr(hf_export, "reference_expects_mtp", lambda path: True)
    monkeypatch.setattr(hf_export, "reference_expects_vision", lambda path: False)

    model_mod.save_hf_model(_args(tmp_path), rollout_id=4, model=[])

    assert recorded["strict"] is False


def _record_reconcile(monkeypatch):
    """Capture how save_hf_model calls the reconciler instead of running it."""
    calls = []

    def _fake(path, reference_hf_dir=None, supplement_mtp=True, **kwargs):
        calls.append({"path": path, "reference_hf_dir": reference_hf_dir, "supplement_mtp": supplement_mtp})

    monkeypatch.setattr(hf_export, "reconcile_hf_export_index", _fake)
    return calls


def test_reconcile_runs_for_a_vision_only_relaxation_and_leaves_mtp_alone(monkeypatch, tmp_path):
    """Ghost entries can come from either relaxation, MTP supplementation
    cannot.

    Reconcile has to run whenever the save was non-strict, but pulling mtp.*
    weights out of the base is only ever right when MTP was the relaxed group
    -- otherwise a genuinely missing MTP tensor gets papered over with base
    weights.
    """
    recorded = {}
    _install_fakes(monkeypatch, recorded)
    calls = _record_reconcile(monkeypatch)
    monkeypatch.setattr(hf_export, "reference_expects_mtp", lambda path: False)
    monkeypatch.setattr(hf_export, "reference_expects_vision", lambda path: True)

    model_mod.save_hf_model(_args(tmp_path), rollout_id=6, model=_model(vision=False))

    assert recorded["strict"] is False
    assert len(calls) == 1
    assert calls[0]["supplement_mtp"] is False


def test_reconcile_supplements_mtp_for_the_mtp_relaxation(monkeypatch, tmp_path):
    recorded = {}
    _install_fakes(monkeypatch, recorded)
    calls = _record_reconcile(monkeypatch)
    monkeypatch.setattr(hf_export, "reference_expects_mtp", lambda path: True)
    monkeypatch.setattr(hf_export, "reference_expects_vision", lambda path: False)

    model_mod.save_hf_model(_args(tmp_path), rollout_id=7, model=[])

    assert recorded["strict"] is False
    assert len(calls) == 1
    assert calls[0]["supplement_mtp"] is True


def test_reconcile_skipped_when_the_save_was_strict(monkeypatch, tmp_path):
    """Nothing was relaxed, so there can be no ghost entries to reconcile."""
    recorded = {}
    _install_fakes(monkeypatch, recorded)
    calls = _record_reconcile(monkeypatch)
    monkeypatch.setattr(hf_export, "reference_expects_mtp", lambda path: False)
    monkeypatch.setattr(hf_export, "reference_expects_vision", lambda path: False)

    model_mod.save_hf_model(_args(tmp_path), rollout_id=8, model=[])

    assert recorded["strict"] is True
    assert calls == []
