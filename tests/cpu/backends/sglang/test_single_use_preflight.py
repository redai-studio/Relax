"""Metadata reuse is bounded to one load and cannot hide file replacement."""

import json

import pytest
import torch
from safetensors.torch import save_file

from relax.distributed.weight_sync.model_config import configuration_signature
from relax.distributed.weight_sync.sglang_loader import CheckpointTensors, EmbeddedSGLangGroup


def fixture(tmp_path):
    config = dict(model_type="qwen3", num_attention_heads=1, num_key_value_heads=1)
    (tmp_path / "config.json").write_text(json.dumps(config))
    save_file({"model.norm.weight": torch.ones(4, dtype=torch.bfloat16)}, tmp_path / "model.safetensors")
    group = object.__new__(EmbeddedSGLangGroup)
    group.config_hash = configuration_signature(config)
    group.startup_schema = CheckpointTensors(tmp_path).schema
    group.tp_size = group.pp_size = 1
    group.worker_receipts = [
        dict(
            rank=0,
            layout=dict(moe_ep_rank=0, moe_ep_size=1, moe_tp_rank=0, moe_tp_size=1),
            physical_schema={"model.norm.weight": [[4], "torch.bfloat16"]},
        )
    ]
    calls = []
    group._rpc = lambda phase, **kwargs: calls.append(phase)
    return group, calls


def test_preflight_then_load_reuses_only_same_unchanged_inventory(tmp_path, monkeypatch):
    import safetensors

    group, calls = fixture(tmp_path)
    reader, opened = safetensors.safe_open, []

    def tracked(*args, **kwargs):
        opened.append(str(args[0]))
        return reader(*args, **kwargs)

    monkeypatch.setattr(safetensors, "safe_open", tracked)
    group.preflight(tmp_path)
    group.load(tmp_path, "version1")
    assert len(opened) == 1 and calls == ["identify", "load"]
    group.load(tmp_path, "version2")
    assert len(opened) == 2 and calls[-2:] == ["identify", "load"]


@pytest.mark.parametrize("change", ["shape", "dtype", "extra_shard", "config"])
def test_changed_files_cannot_reuse_the_previous_preflight(tmp_path, change):
    group, calls = fixture(tmp_path)
    group.preflight(tmp_path)
    if change in ("shape", "dtype"):
        save_file(
            {
                "model.norm.weight": torch.ones(
                    8 if change == "shape" else 4, dtype=torch.float32 if change == "dtype" else torch.bfloat16
                )
            },
            tmp_path / "model.safetensors",
        )
    elif change == "extra_shard":
        save_file({"unexpected.weight": torch.ones(1)}, tmp_path / "extra.safetensors")
    else:
        config = json.loads((tmp_path / "config.json").read_bytes())
        config["num_attention_heads"] = 2
        (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError):
        group.load(tmp_path, "version1")
    assert "load" not in calls


def test_lost_load_reply_consumes_metadata_permission(tmp_path):
    group, calls = fixture(tmp_path)
    group.preflight(tmp_path)

    def lost(phase, **kwargs):
        calls.append(phase)
        if phase == "load":
            raise OSError("reply lost")

    group._rpc = lost
    with pytest.raises(OSError):
        group.load(tmp_path, "version1")
    assert group._pending_preflight is None and group._loaded_inspection is None
    group._rpc = lambda phase, **kwargs: calls.append(phase)
    group.load(tmp_path, "version1")
    assert calls[-2:] == ["identify", "load"]
