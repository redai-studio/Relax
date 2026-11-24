# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for the INT4 (W4A16) HF-checkpoint converter's fused-expert handling.

Mirrors the FP8 converter: Qwen3.6 fused routed experts
(``...mlp.experts.gate_up_proj`` / ``.down_proj``, 3-D) are sliced per expert,
``gate_up`` is split into ``gate`` + ``up``, and each 2-D projection is packed via
``pack_layer`` into ``weight_packed`` / ``weight_scale`` / ``weight_shape``
(+ ``weight_zero_point`` when asymmetric). The fused path is only taken when the
tensor is *not* matched by ``--ignore-rules``.

``pack_layer`` needs a compiled CUDA kernel, so we stub it (recording shapes) and
fake the safetensors I/O to test the pure slicing / naming logic.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
import safetensors  # noqa: E402
import safetensors.torch  # noqa: E402


_SOURCE = pathlib.Path(__file__).resolve().parents[3] / "scripts/tools/convert_hf_to_int4.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("convert_hf_to_int4_under_test", _SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = _load_module()


class _FakeReader:
    def __init__(self, tensors):
        self._tensors = tensors

    def keys(self):
        return list(self._tensors.keys())

    def get_tensor(self, key):
        return self._tensors[key]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _make_pack_layer(shapes):
    def _fake_pack_layer(weight, group_size, sym):
        shapes.append(tuple(weight.shape))
        rows = weight.shape[0]
        qw = torch.zeros((rows, 1), dtype=torch.int32)
        scale = torch.ones((rows, 1), dtype=torch.float32)
        zp = None if sym else torch.zeros((rows, 1), dtype=torch.int32)
        return qw, scale, zp

    return _fake_pack_layer


@pytest.fixture
def harness(monkeypatch):
    shapes = []
    saved = {}

    def _fake_save_file(tensors, path, metadata=None):
        saved["tensors"] = dict(tensors)

    monkeypatch.setattr(mod, "pack_layer", _make_pack_layer(shapes))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a, **k: 0)
    monkeypatch.setattr(safetensors.torch, "save_file", _fake_save_file)

    def _run(inputs, ignore_rules, is_symmetric=True, group_size=128):
        monkeypatch.setattr(safetensors, "safe_open", lambda *a, **k: _FakeReader(inputs))
        collector = mod.ConversionResult()
        mod._process_file(
            input_path="/in",
            output_path="/out",
            filename="model-00001-of-00001.safetensors",
            group_size=group_size,
            is_symmetric=is_symmetric,
            ignore_rules=ignore_rules,
            result_collector=collector,
        )
        return saved["tensors"], shapes

    return _run


# Representative subset of the recipe's default ignore rules.
_IGNORE = [
    "re:.*self_attn.*",
    "re:.*visual.*",
    "re:.*mlp\\.gate\\.weight",
    "re:.*norm.*",
]


class TestStoreQuantized:
    def test_symmetric_omits_zero_point(self, monkeypatch):
        monkeypatch.setattr(mod, "pack_layer", _make_pack_layer([]))
        out = {}
        weight = torch.zeros(4, 8)
        mod._store_quantized(out, "a.b", weight, group_size=128, is_symmetric=True)
        assert set(out) == {"a.b.weight_packed", "a.b.weight_scale", "a.b.weight_shape"}
        # weight_shape records the *original* 2-D shape as int32
        assert out["a.b.weight_shape"].dtype == torch.int32
        assert out["a.b.weight_shape"].tolist() == [4, 8]

    def test_asymmetric_includes_zero_point(self, monkeypatch):
        monkeypatch.setattr(mod, "pack_layer", _make_pack_layer([]))
        out = {}
        mod._store_quantized(out, "a.b", torch.zeros(4, 8), group_size=128, is_symmetric=False)
        assert "a.b.weight_zero_point" in out


class TestProcessFileFusedExperts:
    def test_fused_gate_up_and_down_split_per_expert(self, harness):
        inputs = {
            "L.mlp.experts.gate_up_proj": torch.randn(2, 8, 3),
            "L.mlp.experts.down_proj": torch.randn(2, 3, 4),
        }
        saved, shapes = harness(inputs, ignore_rules=_IGNORE)

        for i in range(2):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                base = f"L.mlp.experts.{i}.{proj}"
                assert f"{base}.weight_packed" in saved
                assert f"{base}.weight_scale" in saved
                assert f"{base}.weight_shape" in saved
        # gate_up rows (8) chunked into two (4, 3); down kept (3, 4)
        assert shapes.count((4, 3)) == 4
        assert shapes.count((3, 4)) == 2

    def test_ignored_and_regular_weights(self, harness):
        q = torch.randn(4, 4)
        vis = torch.randn(2, 2)
        gate = torch.randn(2, 2)
        regular = torch.randn(4, 8)
        inputs = {
            "L.self_attn.q_proj.weight": q,  # ignored (self_attn)
            "L.visual.x.weight": vis,  # ignored (visual)
            "L.mlp.gate.weight": gate,  # ignored (router gate)
            "L.regular.weight": regular,  # packed
        }
        saved, _shapes = harness(inputs, ignore_rules=_IGNORE)

        # ignored weights pass through unchanged, no packed triplet
        assert saved["L.self_attn.q_proj.weight"] is q
        assert saved["L.visual.x.weight"] is vis
        assert saved["L.mlp.gate.weight"] is gate
        assert "L.self_attn.q_proj.weight_packed" not in saved
        # regular weight is packed
        assert "L.regular.weight_packed" in saved
        assert "L.regular.weight_shape" in saved

    def test_ignore_rule_prevents_fused_split(self, harness):
        # A fused expert tensor whose name matches an ignore rule must NOT be
        # split; it is stored verbatim (guards the ``not is_ignored`` condition).
        fused = torch.randn(2, 8, 3)
        inputs = {"L.self_attn.experts.gate_up_proj": fused}
        saved, shapes = harness(inputs, ignore_rules=_IGNORE)

        assert saved["L.self_attn.experts.gate_up_proj"] is fused
        assert not any(k.startswith("L.self_attn.experts.0.") for k in saved)
        assert shapes == []  # pack_layer never called

    @pytest.mark.parametrize("is_symmetric", [True, False])
    def test_zero_point_only_when_asymmetric(self, harness, is_symmetric):
        inputs = {"L.regular.weight": torch.randn(4, 8)}
        saved, _shapes = harness(inputs, ignore_rules=_IGNORE, is_symmetric=is_symmetric)
        assert ("L.regular.weight_zero_point" in saved) is (not is_symmetric)
