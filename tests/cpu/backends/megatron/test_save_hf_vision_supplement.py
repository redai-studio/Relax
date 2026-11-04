# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Feeding a text-only export the reference's vision tower.

A VL base trained text-only emits no vision tensors, so the shard holding them
never completes and Bridge writes it short.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
pytest.importorskip("megatron.bridge")

import safetensors  # noqa: E402
import safetensors.torch  # noqa: E402

from relax.backends.megatron import model as model_mod  # noqa: E402


_SHARD = "model-00001-of-00001.safetensors"
_LANG = ["model.language_model.layers.0.self_attn.q_proj.weight"]
_VISION = ["model.vision_tower.encoder.layers.0.mlp.down_proj.weight", "model.embed_vision.projection.weight"]


@pytest.fixture
def reference(tmp_path):
    """A reference checkpoint holding both language and vision weights."""
    directory = tmp_path / "base"
    directory.mkdir()
    safetensors.torch.save_file(
        {k: torch.ones(4, dtype=torch.bfloat16) for k in _LANG + _VISION},
        str(directory / _SHARD),
        metadata={"format": "pt"},
    )
    return str(directory)


class _Source:
    """Stands in for Bridge's SafeTensorsStateSource."""

    def __init__(self):
        self.key_to_filename_map = dict.fromkeys(_LANG + _VISION, _SHARD)
        self.consumed: dict = {}
        self.received_args = None

    def save_generator(self, generator, *args, **kwargs):
        self.received_args = (args, kwargs)
        self.consumed = dict(generator)


def _bridge(source=None):
    source = source if source is not None else _Source()
    return SimpleNamespace(hf_pretrained=SimpleNamespace(state=SimpleNamespace(source=source))), source


class TestInstallVisionSupplement:
    def test_model_tensors_are_kept_and_vision_appended(self, reference):
        bridge, source = _bridge()
        restore = model_mod._install_vision_supplement(bridge, reference)

        source.save_generator(iter([(k, torch.zeros(4)) for k in _LANG]))

        assert sorted(source.consumed) == sorted(_LANG + _VISION)
        restore()

    def test_extra_arguments_are_passed_through(self, reference):
        bridge, source = _bridge()
        model_mod._install_vision_supplement(bridge, reference)

        source.save_generator(iter([]), "positional", strict=True, ignored_source_key_prefixes=("mtp.",))

        assert source.received_args == (("positional",), {"strict": True, "ignored_source_key_prefixes": ("mtp.",)})

    def test_restore_stops_the_supplementing(self, reference):
        """Later exports on the same bridge must not be hijacked."""
        bridge, source = _bridge()
        restore = model_mod._install_vision_supplement(bridge, reference)
        source.save_generator(iter([]))
        assert sorted(source.consumed) == sorted(_VISION)

        restore()
        source.save_generator(iter([(_LANG[0], torch.zeros(4))]))

        assert sorted(source.consumed) == _LANG

    def test_returns_none_when_the_source_is_not_safetensors_backed(self, reference):
        for bridge in (
            SimpleNamespace(),
            SimpleNamespace(hf_pretrained=SimpleNamespace(state=None)),
            SimpleNamespace(hf_pretrained=SimpleNamespace(state=SimpleNamespace(source=SimpleNamespace()))),
        ):
            assert model_mod._install_vision_supplement(bridge, reference) is None


class TestReferenceVisionTensors:
    def test_yields_every_vision_key_the_reference_declares(self, reference):
        pairs = list(model_mod._reference_vision_tensors(reference, dict.fromkeys(_LANG + _VISION, _SHARD)))

        assert sorted(name for name, _t in pairs) == sorted(_VISION)
        assert all(torch.equal(t, torch.ones(4, dtype=torch.bfloat16)) for _n, t in pairs)

    def test_only_rank_zero_reads(self, reference, monkeypatch):
        monkeypatch.setattr(model_mod.torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(model_mod.torch.distributed, "get_rank", lambda group=None: 3)

        assert list(model_mod._reference_vision_tensors(reference, dict.fromkeys(_VISION, _SHARD))) == []

        monkeypatch.setattr(model_mod.torch.distributed, "get_rank", lambda group=None: 0)
        assert len(list(model_mod._reference_vision_tensors(reference, dict.fromkeys(_VISION, _SHARD)))) == 2
