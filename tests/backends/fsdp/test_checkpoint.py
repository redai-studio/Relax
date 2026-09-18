# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""DCP checkpoint round-trip + COMMITTED gating (single-rank gloo)."""

from __future__ import annotations

import multiprocessing as mp
import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

from relax.backends.fsdp import checkpoint as ckpt


@pytest.fixture(scope="module")
def _dist():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29613")
    if not dist.is_initialized():
        dist.init_process_group("gloo", rank=0, world_size=1)
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


def _trained_model():
    model = torch.nn.Linear(4, 4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.randn(2, 4)).sum().backward()
    opt.step()
    return model, opt


def _checkpoint_failure_worker(rank: int, save_dir: str, failure_phase: str, queue) -> None:
    rendezvous = os.path.join(save_dir, f"{failure_phase}-rendezvous")
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=15),
    )
    from relax.utils import distributed_utils

    distributed_utils.GLOO_GROUP = dist.new_group(backend="gloo", timeout=timedelta(seconds=15))
    model, opt = _trained_model()
    if failure_phase == "metadata" and rank == 0:

        def _fail(*_args, **_kwargs):
            raise OSError("metadata disk full")

        ckpt._write_json = _fail
    if failure_phase == "state" and rank == 1:

        def _fail_state(*_args, **_kwargs):
            raise OSError("state preparation OOM")

        ckpt.get_state_dict = _fail_state
    try:
        ckpt.save_checkpoint(
            save_dir,
            "t2i",
            11,
            model,
            opt,
            trainer_state={"rollout_id": 11},
            weight_sync_manifest={},
            adapter_contract={},
            is_rank0=rank == 0,
        )
        queue.put((rank, "unexpected-success"))
    except Exception as exc:
        queue.put((rank, f"{type(exc).__name__}: {exc}"))
    finally:
        dist.destroy_process_group()


def test_dcp_save_load_restores_model_and_state(_dist, tmp_path):
    model, opt = _trained_model()
    save_dir = str(tmp_path)
    ckpt.save_checkpoint(
        save_dir,
        "t2i",
        7,
        model,
        opt,
        trainer_state={"rollout_id": 7, "policy_version": 3, "dataset_cursor": 42},
        weight_sync_manifest={"tensor_count": 2},
        adapter_contract={"family": "test"},
    )
    assert ckpt.find_latest_committed(save_dir, "t2i") == 7
    ckpt_dir = os.path.join(save_dir, "t2i", ckpt.iter_dir_name(7))
    for name in ("COMMITTED", "trainer_state.json", "weight_sync_manifest.json", "adapter_contract.json"):
        assert os.path.isfile(os.path.join(ckpt_dir, name))
    assert os.path.isfile(os.path.join(ckpt_dir, "fsdp", ".metadata"))

    before = model.weight.detach().clone()
    with torch.no_grad():
        model.weight.add_(5.0)
    state = ckpt.load_checkpoint(save_dir, "t2i", 7, model, opt)
    assert torch.allclose(model.weight, before)
    assert state["policy_version"] == 3
    assert state["dataset_cursor"] == 42


def test_checkpoint_metadata_failure_is_not_committed(_dist, tmp_path, monkeypatch):
    model, opt = _trained_model()

    def _fail(*_args, **_kwargs):
        raise OSError("metadata disk full")

    monkeypatch.setattr(ckpt, "_write_json", _fail)
    with pytest.raises(OSError, match="metadata disk full"):
        ckpt.save_checkpoint(
            str(tmp_path),
            "t2i",
            9,
            model,
            opt,
            trainer_state={"rollout_id": 9},
            weight_sync_manifest={},
            adapter_contract={},
        )
    ckpt_dir = os.path.join(str(tmp_path), "t2i", ckpt.iter_dir_name(9))
    assert not os.path.exists(os.path.join(ckpt_dir, ckpt.COMMITTED_MARKER))


def test_checkpoint_waits_for_all_rng_writes_before_publish(_dist, tmp_path, monkeypatch):
    model, opt = _trained_model()
    events = []
    original_all_ranks_ok = ckpt._all_ranks_ok
    original_write_json = ckpt._write_json

    def _record_agreement(ok):
        events.append("agreement")
        return original_all_ranks_ok(ok)

    def _record_rng(_staging_dir):
        events.append("rng")

    def _record_json(path, payload):
        events.append("publish")
        return original_write_json(path, payload)

    monkeypatch.setattr(ckpt, "_all_ranks_ok", _record_agreement)
    monkeypatch.setattr(ckpt, "_save_rng", _record_rng)
    monkeypatch.setattr(ckpt, "_write_json", _record_json)

    ckpt.save_checkpoint(
        str(tmp_path),
        "t2i",
        10,
        model,
        opt,
        trainer_state={"rollout_id": 10},
        weight_sync_manifest={},
        adapter_contract={},
    )

    rng_index = events.index("rng")
    publish_index = events.index("publish")
    assert "agreement" in events[rng_index + 1 : publish_index]


def test_checkpoint_metadata_failure_exits_all_ranks_without_hang(tmp_path):
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    processes = [
        ctx.Process(target=_checkpoint_failure_worker, args=(rank, str(tmp_path), "metadata", queue))
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            pytest.fail("checkpoint rank hung after rank-0 metadata publication failure")
        assert process.exitcode == 0
    results = sorted(queue.get(timeout=2) for _ in range(2))
    assert "OSError: metadata disk full" in results[0][1]
    assert "RuntimeError: Checkpoint metadata publication failed on rank 0" in results[1][1]
    ckpt_dir = os.path.join(str(tmp_path), "t2i", ckpt.iter_dir_name(11))
    assert not os.path.exists(os.path.join(ckpt_dir, ckpt.COMMITTED_MARKER))


def test_checkpoint_state_preparation_failure_exits_all_ranks_without_hang(tmp_path):
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    processes = [
        ctx.Process(target=_checkpoint_failure_worker, args=(rank, str(tmp_path), "state", queue)) for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            pytest.fail("checkpoint rank hung after non-rank-0 state preparation failure")
        assert process.exitcode == 0
    results = sorted(queue.get(timeout=2) for _ in range(2))
    assert "RuntimeError: Checkpoint state-dict preparation failed on another rank" in results[0][1]
    assert "OSError: state preparation OOM" in results[1][1]


def test_refuses_uncommitted_checkpoint(_dist, tmp_path):
    model, opt = _trained_model()
    save_dir = str(tmp_path)
    ckpt_dir = os.path.join(save_dir, "t2i", ckpt.iter_dir_name(3), "fsdp")
    os.makedirs(ckpt_dir, exist_ok=True)  # exists but no COMMITTED marker
    with pytest.raises(FileNotFoundError):
        ckpt.load_checkpoint(save_dir, "t2i", 3, model, opt)


def test_find_latest_committed_picks_highest(_dist, tmp_path):
    model, opt = _trained_model()
    save_dir = str(tmp_path)
    for it in (2, 5, 3):
        ckpt.save_checkpoint(
            save_dir,
            "t2v",
            it,
            model,
            opt,
            trainer_state={"rollout_id": it},
            weight_sync_manifest={},
            adapter_contract={},
        )
    assert ckpt.find_latest_committed(save_dir, "t2v") == 5


def test_same_iteration_retry_cannot_overwrite_committed_checkpoint(_dist, tmp_path, monkeypatch):
    model, opt = _trained_model()
    save_dir = str(tmp_path)
    metadata = dict(
        trainer_state={"rollout_id": 6, "policy_version": 2},
        weight_sync_manifest={"tensor_count": 2},
        adapter_contract={"family": "test"},
    )
    ckpt_dir = ckpt.save_checkpoint(save_dir, "t2i", 6, model, opt, **metadata)
    before = {
        name: (tmp_path / "t2i" / ckpt.iter_dir_name(6) / name).read_bytes()
        for name in ("COMMITTED", "trainer_state.json", "weight_sync_manifest.json", "adapter_contract.json")
    }

    def _must_not_rewrite(*_args, **_kwargs):
        raise AssertionError("an identical committed retry must not rewrite DCP shards")

    monkeypatch.setattr(ckpt.dcp, "save", _must_not_rewrite)
    assert ckpt.save_checkpoint(save_dir, "t2i", 6, model, opt, **metadata) == ckpt_dir
    after = {name: (tmp_path / "t2i" / ckpt.iter_dir_name(6) / name).read_bytes() for name in before}
    assert after == before

    changed = {**metadata, "trainer_state": {"rollout_id": 6, "policy_version": 3}}
    with pytest.raises(RuntimeError, match="Refusing to overwrite committed checkpoint"):
        ckpt.save_checkpoint(save_dir, "t2i", 6, model, opt, **changed)
    assert {name: (tmp_path / "t2i" / ckpt.iter_dir_name(6) / name).read_bytes() for name in before} == before


# ---------------------------------------------------------------------------
# LoRA: adapter-only save/load and the two export forms
# ---------------------------------------------------------------------------


class _FakeLoraLinear(torch.nn.Module):
    """PEFT's post-injection layout: frozen base + trainable adapter."""

    def __init__(self, dim=4, rank=2):
        super().__init__()
        self.base_layer = torch.nn.Linear(dim, dim, bias=False)
        self.base_layer.weight.requires_grad_(False)
        self.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(dim, rank, bias=False)})
        self.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(rank, dim, bias=False)})


def _lora_model():
    torch.manual_seed(0)
    model = torch.nn.Module()
    model.to_q = _FakeLoraLinear()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    return model, opt


_LORA_CONTRACT = {
    "family": "test",
    "trainable_mode": "lora",
    "save_mode": "adapter",
    "base_model_sha256": "deadbeef",
    "lora": {
        "rank": 2,
        "alpha": 4,
        "dropout": 0.0,
        "target_modules": ["to_q"],
        "bias": "none",
        "task_type": "FEATURE_EXTRACTION",
        "adapter_name": "default",
        "scaling": 2.0,
    },
}


def _save_lora(save_dir, iteration=1):
    model, opt = _lora_model()
    ckpt.save_checkpoint(
        save_dir,
        "t2i",
        iteration,
        model,
        opt,
        trainer_state={"rollout_id": iteration, "policy_version": 1},
        weight_sync_manifest={},
        adapter_contract=_LORA_CONTRACT,
        adapter_only=True,
    )
    return model, opt


def test_adapter_only_save_omits_the_frozen_base(_dist, tmp_path):
    from torch.distributed.checkpoint import FileSystemReader

    save_dir = str(tmp_path)
    _save_lora(save_dir)

    metadata = FileSystemReader(os.path.join(save_dir, "t2i", ckpt.iter_dir_name(1), "fsdp")).read_metadata()
    model_keys = [k for k in metadata.state_dict_metadata if k.startswith("model.")]
    assert model_keys, "adapter-only save wrote nothing"
    assert all(".lora_" in k for k in model_keys), sorted(model_keys)


def test_adapter_only_round_trip_restores_adapter_and_spares_base(_dist, tmp_path):
    """The symmetric load is mandatory: asking DCP for base keys an adapter-
    only save never wrote fails, and only at the FIRST resume."""
    save_dir = str(tmp_path)
    model, opt = _save_lora(save_dir)

    adapter_before = model.to_q.lora_B["default"].weight.detach().clone()
    base_before = model.to_q.base_layer.weight.detach().clone()
    with torch.no_grad():
        model.to_q.lora_B["default"].weight.add_(5.0)
        model.to_q.base_layer.weight.add_(7.0)

    ckpt.load_checkpoint(save_dir, "t2i", 1, model, opt, adapter_only=True)
    assert torch.allclose(model.to_q.lora_B["default"].weight, adapter_before)
    # The base is NOT in the checkpoint; the load must leave it alone rather
    # than raise on a missing key.
    assert torch.allclose(model.to_q.base_layer.weight, base_before + 7.0)


def test_adapter_contract_round_trips(_dist, tmp_path):
    save_dir = str(tmp_path)
    _save_lora(save_dir)
    contract = ckpt.read_adapter_contract(save_dir, "t2i", 1)
    assert contract["save_mode"] == "adapter"
    assert contract["base_model_sha256"] == "deadbeef"
    # alpha cannot be recovered from the weights, so it must be recorded.
    assert contract["lora"]["alpha"] == 4


def test_export_hf_refuses_a_lora_checkpoint_without_a_base(_dist, tmp_path):
    """The worst footgun: the export would otherwise write a "transformer" made
    entirely of lora_A/lora_B, which is non-empty and finite."""
    save_dir = str(tmp_path / "ckpt")
    _save_lora(save_dir)
    with pytest.raises(ValueError, match="base_model_path is required"):
        ckpt.export_hf(save_dir, "t2i", 1, str(tmp_path / "out"))


def test_export_hf_folds_the_adapter_into_the_base(_dist, tmp_path):
    from safetensors.torch import save_file

    from relax.backends.fsdp.lora import fold_lora_delta

    save_dir = str(tmp_path / "ckpt")
    model, _opt = _save_lora(save_dir)

    base_dir = tmp_path / "base" / "transformer"
    base_dir.mkdir(parents=True)
    base_weight = torch.randn(4, 4)
    save_file({"to_q.weight": base_weight}, str(base_dir / "diffusion_pytorch_model.safetensors"))

    out = ckpt.export_hf(save_dir, "t2i", 1, str(tmp_path / "out"), base_model_path=str(tmp_path / "base"))

    from safetensors.torch import load_file

    exported = load_file(os.path.join(out, "transformer", "diffusion_pytorch_model.safetensors"))
    assert sorted(exported) == ["to_q.weight"], "adapter tensors must not leak into the merged tree"
    expected = fold_lora_delta(
        base_weight,
        model.to_q.lora_A["default"].weight.detach(),
        model.to_q.lora_B["default"].weight.detach(),
        _LORA_CONTRACT["lora"]["scaling"],
    )
    assert torch.allclose(exported["to_q.weight"], expected)


def test_export_peft_adapter_writes_a_loadable_directory(_dist, tmp_path):
    save_dir = str(tmp_path / "ckpt")
    _save_lora(save_dir)
    out = ckpt.export_peft_adapter(save_dir, "t2i", 1, str(tmp_path / "adapter"))

    import json

    with open(os.path.join(out, "adapter_config.json"), encoding="utf-8") as f:
        config = json.load(f)
    assert config["r"] == 2 and config["lora_alpha"] == 4
    # Diffusion DiTs must not be tagged CAUSAL_LM or a PEFT loader adds an LM head.
    assert config["task_type"] == "FEATURE_EXTRACTION"

    from safetensors.torch import load_file

    weights = load_file(os.path.join(out, "adapter_model.safetensors"))
    assert sorted(weights) == [
        "base_model.model.to_q.lora_A.weight",
        "base_model.model.to_q.lora_B.weight",
    ]


def test_export_peft_adapter_refuses_a_full_ft_checkpoint(_dist, tmp_path):
    save_dir = str(tmp_path / "ckpt")
    model, opt = _trained_model()
    ckpt.save_checkpoint(
        save_dir,
        "t2i",
        1,
        model,
        opt,
        trainer_state={"rollout_id": 1},
        weight_sync_manifest={},
        adapter_contract={"family": "test", "save_mode": "full"},
    )
    with pytest.raises(ValueError, match="not a LoRA checkpoint"):
        ckpt.export_peft_adapter(save_dir, "t2i", 1, str(tmp_path / "adapter"))


def test_export_hf_reads_the_full_ft_checkpoint(_dist, tmp_path):
    """Regression: ``dcp.load({"model": {}})`` is a silent no-op.

    DCP fills the tensors a destination dict already declares, so loading into
    an empty one succeeded and returned nothing — every export raised "refusing
    to write an empty export". The destination must be pre-allocated from the
    checkpoint metadata.
    """
    from safetensors.torch import load_file

    save_dir = str(tmp_path / "ckpt")
    model, opt = _trained_model()
    ckpt.save_checkpoint(
        save_dir,
        "t2i",
        1,
        model,
        opt,
        trainer_state={"rollout_id": 1},
        weight_sync_manifest={},
        adapter_contract={"family": "test", "save_mode": "full"},
    )
    out = ckpt.export_hf(save_dir, "t2i", 1, str(tmp_path / "out"))
    exported = load_file(os.path.join(out, "diffusion_pytorch_model.safetensors"))
    assert sorted(exported) == ["bias", "weight"]
    assert torch.allclose(exported["weight"], model.weight.detach())
