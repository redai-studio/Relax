# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Strictness decision in ``save_hf_model``.

Bridge under ``strict=True`` refuses every safetensors shard containing a key
the training model never emits, which loses the real tensors sharing those
shards rather than just the absent ones. So strictness is relaxed exactly when
a group the reference declares really will be absent from the export: MTP
layers a model trained without MTP, or a vision tower that could not be copied
in from the reference (FP8, or no safetensors source to copy from). Supplying
the tower leaves nothing missing, so those exports keep strict on.
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
        def __init__(self):
            # A safetensors-backed source, as the vision supplement expects to find.
            self.hf_pretrained = SimpleNamespace(
                state=SimpleNamespace(
                    source=SimpleNamespace(key_to_filename_map={}, save_generator=lambda *a, **k: None)
                )
            )

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


def test_vision_reference_with_text_only_model_keeps_strict(monkeypatch, tmp_path):
    """The tower is copied in from the reference, so nothing ends up absent."""
    recorded = {}
    _install_fakes(monkeypatch, recorded)
    monkeypatch.setattr(hf_export, "reference_expects_mtp", lambda path: False)
    monkeypatch.setattr(hf_export, "reference_expects_vision", lambda path: True)

    model_mod.save_hf_model(_args(tmp_path), rollout_id=1, model=_model(vision=False))

    assert recorded["strict"] is True


def test_vision_relaxation_returns_when_the_tower_cannot_be_copied(monkeypatch, tmp_path):
    """No safetensors source to read the tower from -- fall back to
    relaxing."""
    recorded = {}
    _install_fakes(monkeypatch, recorded)
    monkeypatch.setattr(hf_export, "reference_expects_mtp", lambda path: False)
    monkeypatch.setattr(hf_export, "reference_expects_vision", lambda path: True)
    monkeypatch.setattr(model_mod, "_install_vision_supplement", lambda bridge, reference: None)

    model_mod.save_hf_model(_args(tmp_path), rollout_id=1, model=_model(vision=False))

    assert recorded["strict"] is False


def test_fp8_still_relaxes_for_a_text_only_model(monkeypatch, tmp_path):
    """FP8 cannot take a BF16 tower, so the group really is absent there."""
    recorded = {}
    _install_fakes(monkeypatch, recorded)
    monkeypatch.setattr(hf_export, "reference_expects_mtp", lambda path: False)
    monkeypatch.setattr(hf_export, "reference_expects_vision", lambda path: True)
    monkeypatch.setattr(model_mod, "_install_streaming_fp8_writer", lambda *a: (None, None))
    monkeypatch.setattr(model_mod, "_apply_fp8_quantization_config", lambda *a: None)
    args = _args(tmp_path)
    args.save_hf_dtype = "fp8"
    args.save_hf_fp8_quant_mode = "block"
    args.save_hf_fp8_block_size = [128, 128]

    model_mod.save_hf_model(args, rollout_id=1, model=_model(vision=False))

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

    ``model=[]`` makes ``get_model_config(model[0])`` raise, so the short-
    circuit also verifies that a plain reference does not abort the export.
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
    # Only an export that could not take the tower is still short of it.
    monkeypatch.setattr(model_mod, "_install_vision_supplement", lambda bridge, reference: None)

    model_mod.save_hf_model(_args(tmp_path), rollout_id=6, model=_model(vision=False))

    assert recorded["strict"] is False
    assert len(calls) == 1
    assert calls[0]["supplement_mtp"] is False


def test_reconcile_skipped_once_the_tower_was_supplied(monkeypatch, tmp_path):
    """Nothing was relaxed, so there are no ghost entries to reconcile."""
    recorded = {}
    _install_fakes(monkeypatch, recorded)
    calls = _record_reconcile(monkeypatch)
    monkeypatch.setattr(hf_export, "reference_expects_mtp", lambda path: False)
    monkeypatch.setattr(hf_export, "reference_expects_vision", lambda path: True)

    model_mod.save_hf_model(_args(tmp_path), rollout_id=6, model=_model(vision=False))

    assert recorded["strict"] is True
    assert calls == []


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
