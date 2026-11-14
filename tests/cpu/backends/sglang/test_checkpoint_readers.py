# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Bounded checkpoint mappings amortize large MoE headers within one target."""

import pytest
import torch
from safetensors.torch import save_file

from relax.distributed.weight_sync.sglang_loader import CheckpointTensors


def observed_readers(monkeypatch):
    import safetensors

    original = safetensors.safe_open
    counts = {"opened": 0, "active": 0, "peak": 0}

    class Reader:
        def __init__(self, *args, **kwargs):
            self.reader = original(*args, **kwargs)

        def __enter__(self):
            self.reader.__enter__()
            counts["opened"] += 1
            counts["active"] += 1
            counts["peak"] = max(counts["peak"], counts["active"])
            return self

        def __exit__(self, *error):
            counts["active"] -= 1
            self.reader.__exit__(*error)

        def __getattr__(self, name):
            return getattr(self.reader, name)

    monkeypatch.setattr(safetensors, "safe_open", Reader)
    return counts


def test_one_header_parse_per_shard_within_verification_and_fresh_next_target(tmp_path, monkeypatch):
    values = {f"expert.{n}": torch.full((2, 3), n, dtype=torch.bfloat16) for n in range(256)}
    path = tmp_path / "model.safetensors"
    save_file(values, str(path))
    counts = observed_readers(monkeypatch)
    with CheckpointTensors(tmp_path) as tensors:
        for name in tensors:
            assert torch.equal(tensors[name], values[name])
    assert counts == {"opened": 2, "active": 0, "peak": 1}
    replacement = {key: value + 1 for key, value in values.items()}
    save_file(replacement, str(path))
    with CheckpointTensors(tmp_path) as tensors:
        assert torch.equal(tensors["expert.0"], replacement["expert.0"])
    assert counts["active"] == 0


def test_shard_mapping_limit_eviction_and_exception_cleanup(tmp_path, monkeypatch):
    for index in range(7):
        save_file({f"w{index}": torch.full((3,), index, dtype=torch.float32)}, str(tmp_path / f"{index}.safetensors"))
    counts = observed_readers(monkeypatch)
    with pytest.raises(ValueError, match="injected"):
        with CheckpointTensors(tmp_path) as tensors:
            retained = tensors["w0"]
            for key in tensors:
                assert tensors[key].numel() == 3
            # Evicting a reader does not invalidate a tensor already returned.
            assert torch.equal(retained, torch.zeros(3))
            assert torch.equal(tensors["w0"], retained)
            assert counts["active"] == 4
            raise ValueError("injected verification failure")
    assert counts["active"] == 0
    assert counts["peak"] == 4


def test_metadata_only_scope_never_opens_tensor_readers(tmp_path, monkeypatch):
    save_file({"w": torch.ones((3, 2))}, str(tmp_path / "model.safetensors"))
    counts = observed_readers(monkeypatch)
    with CheckpointTensors(tmp_path, metadata_only=True) as tensors:
        assert tensors["w"].device.type == "meta"
        assert tuple(tensors["w"].shape) == (3, 2)
    assert counts == {"opened": 1, "active": 0, "peak": 1}


def test_membership_checks_only_the_index(tmp_path, monkeypatch):
    save_file({"w": torch.ones((3, 2))}, str(tmp_path / "model.safetensors"))
    tensors = CheckpointTensors(tmp_path)

    def forbidden(*args):
        raise AssertionError("Membership must not materialize tensor data")

    monkeypatch.setattr(CheckpointTensors, "__getitem__", forbidden)
    assert "w" in tensors
    assert "missing" not in tensors
