# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for training-time save-hf FP8 export helpers.

`save_hf_model` (relax.backends.megatron.model) grows two small helpers to
support `--save-hf-dtype fp8`:

* `_install_streaming_fp8_writer(bridge, ...)` monkey-patches the bridge
  source's `save_generator` with a `StreamingFP8Writer` so that Bridge's
  `save_hf_pretrained` streams quantized shards instead of BF16 shards.
* `_apply_fp8_quantization_config(path, ...)` merges `quantization_config`
  into the exported `config.json`.

Plus argument validation: `--save-hf-dtype fp8` requires `--save-hf`.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

try:
    from relax.backends.megatron import model as model_mod  # noqa: E402
except (ImportError, AssertionError) as _exc:
    pytest.skip(f"relax.backends.megatron.model unavailable: {_exc}", allow_module_level=True)

from relax.utils.quant_cast import fp8_checkpoint as checkpoint_mod  # noqa: E402


class _FakeSource:
    def __init__(self, key_to_filename_map):
        self.key_to_filename_map = dict(key_to_filename_map)
        self.save_generator = self._original_save_generator

    def _original_save_generator(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("original save_generator should not be invoked in tests")


class _FakeBridge:
    def __init__(self, source):
        self.hf_pretrained = SimpleNamespace(state=SimpleNamespace(source=source))


class TestInstallStreamingFp8Writer:
    def test_installs_writer_and_restores_original_save_generator(self):
        source = _FakeSource({"a.weight": "model.safetensors"})
        original = source.save_generator
        bridge = _FakeBridge(source)

        writer, restore = model_mod._install_streaming_fp8_writer(
            bridge,
            strategy="block",
            block_size=[128, 128],
        )

        assert isinstance(writer, checkpoint_mod.StreamingFP8Writer)
        assert source.save_generator == writer.save_generator
        assert source.save_generator != original
        restore()
        assert source.save_generator == original

    def test_rejects_bridge_without_safetensors_source(self):
        bridge = SimpleNamespace(hf_pretrained=SimpleNamespace(state=None))
        with pytest.raises(ValueError, match="safetensors"):
            model_mod._install_streaming_fp8_writer(
                bridge,
                strategy="tensor",
                block_size=None,
            )

    def test_rejects_source_without_key_map(self):
        source = SimpleNamespace(save_generator=lambda *a, **k: None)
        bridge = _FakeBridge(source)
        with pytest.raises(ValueError, match="safetensors"):
            model_mod._install_streaming_fp8_writer(
                bridge,
                strategy="tensor",
                block_size=None,
            )


class TestApplyFp8QuantizationConfig:
    def test_merges_quantization_config_into_existing_config(self, tmp_path):
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(json.dumps({"model_type": "qwen3", "hidden_size": 1024}))

        model_mod._apply_fp8_quantization_config(
            str(cfg_path),
            strategy="block",
            block_size=[128, 128],
            modules_to_not_convert=["lm_head", "norm"],
        )

        cfg = json.loads(cfg_path.read_text())
        assert cfg["model_type"] == "qwen3"
        assert cfg["hidden_size"] == 1024
        qc = cfg["quantization_config"]
        assert qc["quant_method"] == "fp8"
        assert qc["fmt"] == "e4m3"
        assert qc["weight_block_size"] == [128, 128]
        assert qc["modules_to_not_convert"] == ["lm_head", "norm"]

    def test_noop_when_config_missing(self, tmp_path):
        # Should not raise if config.json isn't there.
        model_mod._apply_fp8_quantization_config(
            str(tmp_path / "config.json"),
            strategy="tensor",
            block_size=None,
            modules_to_not_convert=[],
        )

    def test_tensor_strategy_writes_no_block_size(self, tmp_path):
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(json.dumps({"model_type": "qwen3"}))

        model_mod._apply_fp8_quantization_config(
            str(cfg_path),
            strategy="tensor",
            block_size=None,
            modules_to_not_convert=[],
        )

        cfg = json.loads(cfg_path.read_text())
        assert "weight_block_size" not in cfg["quantization_config"]


class TestSaveHfDtypeValidation:
    def test_fp8_without_save_hf_is_rejected(self):
        from relax.utils.arguments import validate_save_hf_fp8_args

        args = SimpleNamespace(
            save_hf=None,
            save_hf_dtype="fp8",
            save_hf_fp8_quant_mode="block",
            save_hf_fp8_block_size=[128, 128],
        )
        with pytest.raises(ValueError, match="--save-hf"):
            validate_save_hf_fp8_args(args)

    def test_fp8_with_save_hf_and_block_defaults_ok(self):
        from relax.utils.arguments import validate_save_hf_fp8_args

        args = SimpleNamespace(
            save_hf="/tmp/hf_out/iter_{rollout_id}",
            save_hf_dtype="fp8",
            save_hf_fp8_quant_mode="block",
            save_hf_fp8_block_size=None,
        )
        validate_save_hf_fp8_args(args)
        assert args.save_hf_fp8_block_size == [128, 128]

    def test_bf16_default_is_noop(self):
        from relax.utils.arguments import validate_save_hf_fp8_args

        args = SimpleNamespace(
            save_hf=None,
            save_hf_dtype="bf16",
            save_hf_fp8_quant_mode="block",
            save_hf_fp8_block_size=None,
        )
        validate_save_hf_fp8_args(args)
        assert args.save_hf_fp8_block_size is None

    def test_tensor_mode_with_block_size_is_rejected(self):
        from relax.utils.arguments import validate_save_hf_fp8_args

        args = SimpleNamespace(
            save_hf="/tmp/out",
            save_hf_dtype="fp8",
            save_hf_fp8_quant_mode="tensor",
            save_hf_fp8_block_size=[128, 128],
        )
        with pytest.raises(ValueError):
            validate_save_hf_fp8_args(args)


class TestResolveSaveHfPath:
    def test_appends_iteration_to_root_path(self):
        assert model_mod._resolve_save_hf_path("/tmp/hf_output", 7) == Path("/tmp/hf_output/iter_0000007")

    def test_preserves_rollout_id_template(self):
        assert model_mod._resolve_save_hf_path("/tmp/hf_output/iter_{rollout_id}", 7) == Path("/tmp/hf_output/iter_7")

    def test_supports_rollout_id_format_spec(self):
        assert model_mod._resolve_save_hf_path("/tmp/hf_output/iter_{rollout_id:07d}", 7) == Path(
            "/tmp/hf_output/iter_0000007"
        )


class TestSaveHfModelFp8Branch:
    """Integration-ish: save_hf_model routes to the FP8 helpers, skips MTP reconcile."""

    def test_fp8_dtype_installs_writer_and_writes_quantization_config(self, monkeypatch, tmp_path):
        pytest.importorskip("megatron.bridge")
        # Fake bridge: exposes an in-memory source with a save_generator hook
        # that mimics Bridge yielding one tensor and Bridge writing config.json.
        source = _FakeSource({"a.weight": "model.safetensors"})
        recorded = {}

        class _FakeBridgeWithSave:
            hf_pretrained = SimpleNamespace(state=SimpleNamespace(source=source))

            def save_hf_pretrained(self, model, path, strict):
                # Trigger the FP8 writer by invoking the (now-patched) generator.
                source.save_generator(
                    iter([("a.weight", torch.ones(2, 2))]),
                    path,
                    strict=strict,
                )
                (path / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
                recorded["strict"] = strict

        monkeypatch.setattr(
            "megatron.bridge.AutoBridge.from_hf_pretrained",
            classmethod(lambda cls, *a, **k: _FakeBridgeWithSave()),
        )
        monkeypatch.setattr(model_mod, "patch_megatron_model", _NullCtx)
        monkeypatch.setattr(model_mod, "is_lora_enabled", lambda args: False)
        # Force the non-distributed WORLD-rank-0 branch.
        monkeypatch.setattr(model_mod.torch.distributed, "is_initialized", lambda: False)

        import relax.utils.hf_export as hf_export

        reconcile_calls = []
        monkeypatch.setattr(
            hf_export,
            "reconcile_hf_export_index",
            lambda *a, **k: reconcile_calls.append((a, k)),
        )
        monkeypatch.setattr(hf_export, "reference_expects_mtp", lambda path: False)

        args = SimpleNamespace(
            save_hf=str(tmp_path / "hf_out/iter_{rollout_id}"),
            save_hf_dtype="fp8",
            save_hf_fp8_quant_mode="tensor",
            save_hf_fp8_block_size=None,
            hf_checkpoint="/tmp/fake-hf",
            mtp_num_layers=None,
        )

        model_mod.save_hf_model(args, rollout_id=7, model=[])

        out_dir = tmp_path / "hf_out" / "iter_7"
        cfg = json.loads((out_dir / "config.json").read_text())
        assert cfg["quantization_config"]["quant_method"] == "fp8"
        assert cfg["quantization_config"]["fmt"] == "e4m3"
        # Reconcile must not run in the FP8 branch.
        assert reconcile_calls == []
        # save_generator must be restored on the source after the call returns.
        assert source.save_generator == source._original_save_generator


class _NullCtx:
    """Minimal context manager replacement for `patch_megatron_model`."""

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
