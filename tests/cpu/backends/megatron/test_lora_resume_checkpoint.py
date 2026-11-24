# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import hashlib
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("megatron.training.checkpointing")

from relax.backends.megatron import checkpoint  # noqa: E402


class _AdapterModel(torch.nn.Module):
    def __init__(self, *, train_base: bool = False):
        super().__init__()
        self.layer = torch.nn.Module()
        self.layer.adapter = torch.nn.Module()
        self.layer.adapter.linear_in = torch.nn.Linear(2, 2, bias=False)
        self.base = torch.nn.Linear(2, 2, bias=False)
        self.base.weight.requires_grad_(train_base)


def _runtime_args():
    return SimpleNamespace(
        hf_checkpoint="/models/base",
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_target_modules=["linear_in"],
        lora_scope="all",
        lora_merge_mode=True,
        lora_adapter_mode=False,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        world_size=4,
    )


def test_filter_lora_checkpoint_keeps_resume_state_and_marks_args():
    marker = {"format_version": 1}
    optimizer = object()
    scheduler = object()
    rng = object()
    state = {
        "args": SimpleNamespace(),
        "model": {
            "decoder.base.weight": object(),
            "decoder.adapter.linear_in.weight": object(),
            "decoder.adapter.linear_out.weight": object(),
        },
        "model1": {
            "vision.base.weight": object(),
            "vision.adapter.linear_in.weight": object(),
        },
        "optimizer": optimizer,
        "opt_param_scheduler": scheduler,
        "rng_state": rng,
        "iteration": 7,
    }

    filtered = checkpoint._filter_lora_checkpoint_state_dict(state, metadata=marker)

    assert tuple(filtered["model"]) == (
        "decoder.adapter.linear_in.weight",
        "decoder.adapter.linear_out.weight",
    )
    assert tuple(filtered["model1"]) == ("vision.adapter.linear_in.weight",)
    assert filtered["optimizer"] is optimizer
    assert filtered["opt_param_scheduler"] is scheduler
    assert filtered["rng_state"] is rng
    assert filtered["iteration"] == 7
    assert getattr(filtered["args"], checkpoint._LORA_CHECKPOINT_METADATA_ATTR) == marker


def test_lora_checkpoint_rejects_non_lora_trainable_parameters():
    with pytest.raises(RuntimeError, match="non-LoRA trainable parameter"):
        checkpoint._global_trainable_lora_parameter_names([_AdapterModel(train_base=True)])


def test_lora_checkpoint_accepts_only_trainable_adapters():
    names = checkpoint._global_trainable_lora_parameter_names([_AdapterModel()])

    assert names == ("layer.adapter.linear_in.weight",)


def test_save_lora_only_filters_model_after_optimizer_state_is_built(monkeypatch):
    import megatron.training.checkpointing as mcore_checkpointing

    seen = {}
    args = _runtime_args()
    args.relax_lora_frozen_hf_base = {"shard_content_sha256": "frozen-base"}

    def fake_generate_state_dict(*_args, **_kwargs):
        seen["optimizer_saw_base"] = True
        return {
            "args": SimpleNamespace(),
            "model": {
                "layer.to_wrap.weight": object(),
                "layer.adapter.linear_in.weight": object(),
            },
            "optimizer": {"state": "kept"},
            "rng_state": "kept",
        }

    def fake_save(*_args, **_kwargs):
        seen["state"] = mcore_checkpointing.generate_state_dict()

    monkeypatch.setattr(mcore_checkpointing, "generate_state_dict", fake_generate_state_dict)
    monkeypatch.setattr(checkpoint, "_save_checkpoint_megatron", fake_save)
    monkeypatch.setattr(checkpoint, "get_args", lambda: args)
    monkeypatch.setattr(
        checkpoint,
        "_global_trainable_lora_parameter_names",
        lambda _model: ("layer.adapter.linear_in.weight",),
    )

    checkpoint.save_checkpoint(3, [object()], None, None, lora_only=True)

    assert seen["optimizer_saw_base"] is True
    assert tuple(seen["state"]["model"]) == ("layer.adapter.linear_in.weight",)
    assert seen["state"]["optimizer"] == {"state": "kept"}
    assert seen["state"]["rng_state"] == "kept"
    metadata = getattr(seen["state"]["args"], checkpoint._LORA_CHECKPOINT_METADATA_ATTR)
    assert metadata["base_hf_checkpoint"] == args.relax_lora_frozen_hf_base
    assert metadata["format_version"] == 2


def test_lora_model_load_allows_missing_base_but_requires_adapter():
    model = _AdapterModel()
    adapter_state = {"layer.adapter.linear_in.weight": torch.ones_like(model.layer.adapter.linear_in.weight)}

    with checkpoint._validate_lora_model_state_load([model]):
        model.load_state_dict(adapter_state, strict=False)

    with checkpoint._validate_lora_model_state_load([model]):
        with pytest.raises(RuntimeError, match="missing_adapters"):
            model.load_state_dict({}, strict=False)


def test_lora_only_save_keeps_full_model_for_hf_export(monkeypatch, tmp_path):
    import megatron.training.checkpointing as mcore_checkpointing
    from megatron.bridge import AutoBridge

    from relax.backends.megatron import model as model_mod

    args = _runtime_args()
    args.relax_lora_frozen_hf_base = {"shard_content_sha256": "frozen-base"}
    args.save_lora_only = True
    args.save_hf = str(tmp_path / "hf")
    args.save_hf_dtype = "bf16"
    model = [_AdapterModel()]
    original = {name: tensor.clone() for name, tensor in model[0].state_dict().items()}
    captured = {}

    def generate_state_dict(*_args, **_kwargs):
        return {"args": args, "model": model[0].state_dict(), "optimizer": {"step": 7}}

    def save_native(*_args, **_kwargs):
        captured["native"] = mcore_checkpointing.generate_state_dict()

    class ExportBridge:
        def save_hf_pretrained(self, chunks, path, strict):
            captured["hf"] = {name: tensor.clone() for name, tensor in chunks[0].state_dict().items()}

    def save_adapter(chunks, path, runtime_args, bridge):
        captured["adapter"] = chunks[0].layer.adapter.linear_in.weight.detach().clone()

    monkeypatch.setattr(checkpoint, "get_args", lambda: args)
    monkeypatch.setattr(mcore_checkpointing, "generate_state_dict", generate_state_dict)
    monkeypatch.setattr(checkpoint, "_save_checkpoint_megatron", save_native)
    monkeypatch.setattr(AutoBridge, "from_hf_pretrained", lambda *_args, **_kwargs: ExportBridge())
    monkeypatch.setattr(model_mod, "patch_megatron_model", lambda _model: nullcontext())
    monkeypatch.setattr(model_mod, "_save_lora_to_checkpoint", save_adapter)

    checkpoint.save_checkpoint(7, model, None, None, lora_only=args.save_lora_only)
    model_mod.save_hf_model(args, 7, model)

    assert set(captured["native"]["model"]) == {"layer.adapter.linear_in.weight"}
    assert captured["native"]["optimizer"] == {"step": 7}
    assert set(captured["hf"]) == set(original)
    for name, tensor in original.items():
        torch.testing.assert_close(captured["hf"][name], tensor)
        torch.testing.assert_close(model[0].state_dict()[name], tensor)
    torch.testing.assert_close(captured["adapter"], original["layer.adapter.linear_in.weight"])
    assert set(mcore_checkpointing.generate_state_dict()["model"]) == set(original)
    assert not hasattr(args, checkpoint._LORA_CHECKPOINT_METADATA_ATTR)


def _check_local_hf_copies(rank: int, directory: str, expected: dict) -> None:
    from datetime import timedelta

    from relax.utils.distributed_utils import init_gloo_group

    root = Path(directory)
    torch.distributed.init_process_group(
        "gloo", rank=rank, world_size=2, init_method=(root / "store").as_uri(), timeout=timedelta(seconds=60)
    )
    try:
        init_gloo_group(1)
        local = root / str(rank)
        assert checkpoint._actor_hf_identity(str(local), 2) == expected
        if rank == 1:
            with (local / "model.safetensors").open("r+b") as shard:
                shard.seek(-1, 2)
                shard.write(b"B")
        assert checkpoint._actor_hf_identity(str(local), 2) is None
        assert checkpoint._actor_hf_identity(str(root / "0") if rank == 0 else "", 2) is None
        if rank == 1:
            (local / "model.safetensors.index.json").write_text("[]")
        assert checkpoint._actor_hf_identity(str(local), 2) is None
    finally:
        torch.distributed.destroy_process_group()


def test_distributed_lora_identity_agrees_on_localized_weights(tmp_path):
    weights = bytes(4 * 1024**2) + b"A"
    # Hash the known bytes in one shot and spell out the single-shard v2 manifest.
    shard_digest = hashlib.sha256(weights).hexdigest()
    manifest = f'[["model.safetensors","{shard_digest}"]]'.encode("ascii")
    expected = {
        "config_sha256": hashlib.sha256(b"{}").hexdigest(),
        "index_sha256": None,
        "shard_content_sha256": hashlib.sha256(manifest).hexdigest(),
        "shard_count": "1",
    }
    for rank in range(2):
        local = tmp_path / str(rank)
        local.mkdir()
        (local / "config.json").write_text("{}")
        (local / "model.safetensors").write_bytes(weights)
    torch.multiprocessing.spawn(_check_local_hf_copies, args=(str(tmp_path), expected), nprocs=2, join=True)


@pytest.mark.parametrize("use_ref", [False, True])
def test_lora_marker_tracks_actor_hf_load_and_survives_full_resume(monkeypatch, tmp_path, use_ref):
    from megatron.bridge import AutoBridge
    from megatron.core import dist_checkpointing
    from megatron.training import checkpointing as native
    from megatron.training.checkpointing import generate_state_dict
    from safetensors.torch import load_file, save_file

    from tests.cpu.backends.megatron.test_checkpoint_chained_optimizer import _CPUSaveStrategy

    args = _runtime_args()
    a, b = tmp_path / "a", tmp_path / "b"
    for directory, value in ((a, 1.0), (b, 2.0)):
        directory.mkdir()
        (directory / "config.json").write_text(json.dumps({"name": directory.name}))
        save_file({"weight": torch.full((2, 2), value)}, directory / "model.safetensors")
    args.hf_checkpoint, args.ref_load = str(a), str(b) if use_ref else None
    args.megatron_to_hf_mode, args.fp16, args.bf16 = "bridge", False, False
    model = [_AdapterModel()]

    def from_hf(path, **kwargs):
        weights = load_file(Path(path) / "model.safetensors")["weight"]
        return SimpleNamespace(load_hf_weights=lambda chunks: chunks[0].base.load_state_dict({"weight": weights}))

    monkeypatch.setattr(AutoBridge, "from_hf_pretrained", from_hf)
    monkeypatch.setattr(checkpoint.megatron_bridge_utils, "patch_megatron_model", lambda _: nullcontext())
    for name in ("use_critic_lm_head_for_hf_load", "use_sequence_classification_lm_head_for_hf_load"):
        monkeypatch.setattr(checkpoint, name, lambda _: nullcontext())
    with monkeypatch.context() as patch:

        def unexpected_hash(*_args):
            pytest.fail("A run without a native save directory hashed the HF weights")

        patch.setattr(checkpoint, "_actor_hf_identity", unexpected_hash)
        checkpoint._load_checkpoint_hf(model, object(), args, None if use_ref else str(b))
    root = tmp_path / "checkpoint"
    args.save = str(root)
    checkpoint._load_checkpoint_hf(model, object(), args, None if use_ref else str(b))
    torch.testing.assert_close(model[0].base.weight, torch.full((2, 2), 2.0))
    expected_identity = checkpoint.hf_checkpoint_identity(b, version=2)
    assert args.relax_lora_frozen_hf_base == expected_identity
    actor_state = {key: value.clone() for key, value in model[0].state_dict().items()}
    checkpoint._load_checkpoint_hf(model, None, args, str(a))
    assert args.relax_lora_frozen_hf_base == expected_identity
    model[0].load_state_dict(actor_state)

    args.no_save_optim = args.no_save_rng = args.no_load_rng = True
    args.load, args.no_load_optim, args.loss_type = str(root), True, "sft"

    def save_common(iteration, *_args, **_kwargs):
        directory = Path(args.save) / f"iter_{iteration:07d}"
        directory.mkdir(parents=True, exist_ok=True)
        # Exercise Megatron's training args serialization and torch_dist common
        # object, replacing only the distributed model and CUDA save finalizer.
        state = generate_state_dict(args, [], None, None, None, iteration=iteration)
        torch.distributed.init_process_group("gloo", init_method=(directory / "store").as_uri(), rank=0, world_size=1)
        try:
            dist_checkpointing.save(state, str(directory), sharded_strategy=_CPUSaveStrategy())
        finally:
            torch.distributed.destroy_process_group()
        (directory.parent / "latest_checkpointed_iteration.txt").write_text(str(iteration))

    monkeypatch.setattr(checkpoint, "get_args", lambda: args)
    monkeypatch.setattr(checkpoint, "_save_checkpoint_megatron", save_common)
    checkpoint.save_checkpoint(1, model, None, None)
    args.relax_lora_frozen_hf_base = None
    native_load = native.load_checkpoint
    monkeypatch.setattr(native, "load_checkpoint", lambda **kw: (1, 0))
    checkpoint.load_checkpoint(model, object(), None, {}, False)
    assert args.relax_lora_frozen_hf_base == expected_identity
    names = checkpoint._global_trainable_lora_parameter_names(model)
    marker = checkpoint._lora_checkpoint_metadata(args, names, args.relax_lora_frozen_hf_base)
    args.ref_load = str(b)
    assert checkpoint._validate_lora_checkpoint_metadata(args, model, marker) == str(b)
    args.ref_load = None
    with pytest.raises(RuntimeError, match="base HF identity mismatch"):
        checkpoint._validate_lora_checkpoint_metadata(args, model, marker)

    # A second DCP at the same iteration wins the native selection. The
    # (iteration, FLOPs) returned to Relax cannot distinguish these sources.
    alternate = tmp_path / "non_persistent"
    args.save = str(alternate)
    checkpoint._load_checkpoint_hf(model, object(), args, str(a))
    checkpoint.save_checkpoint(1, model, None, None)
    args.save = str(root)
    args.non_persistent_ckpt_type, args.non_persistent_global_ckpt_dir = "global", str(alternate)

    loaded = []

    def load_native_common(**_kwargs):
        state, path, release, _ = native._load_base_checkpoint(args.load, args, rank0=True)
        loaded.append((Path(path), state["args"].relax_lora_frozen_hf_base, release))
        return state["iteration"], 0

    with monkeypatch.context() as patch:
        patch.setattr(native, "load_checkpoint", load_native_common)
        checkpoint.load_checkpoint(model, object(), None, {}, False)
        with pytest.raises(RuntimeError, match="verified frozen HF base"):
            checkpoint.save_checkpoint(2, model, None, None, lora_only=True)

        # Both candidates exist. An explicit step of 42 ranks release above step 1.
        (root / "release").symlink_to(root / "iter_0000001", target_is_directory=True)
        (root / "latest_checkpointed_iteration.txt").write_text("release")
        args.ckpt_step = 42
        checkpoint.load_checkpoint(model, object(), None, {}, False)
    assert loaded == [
        (alternate / "iter_0000001", checkpoint.hf_checkpoint_identity(a, version=2), False),
        (root / "release", expected_identity, True),
    ]
    args.non_persistent_ckpt_type = None

    # The pretrained fallback happens before Megatron applies ckpt_step.
    # Both roots have iteration 1, but only the alternate has a tracker.
    (root / "latest_checkpointed_iteration.txt").unlink()
    args.pretrained_checkpoint, args.ckpt_step = str(alternate), 1
    args.no_load_optim = args.no_load_rng = use_ref
    args.finetune = False
    with monkeypatch.context() as patch:
        # Simulate the original tracker appearing after preflight. The loader
        # must use the same root as the common state it already validated.
        read_metadata = checkpoint._load_checkpoint_metadata

        def read_selected_metadata(*positional):
            common = read_metadata(*positional)
            (root / "latest_checkpointed_iteration.txt").write_text("1")
            return common

        patch.setattr(checkpoint, "_load_checkpoint_metadata", read_selected_metadata)
        patch.setattr(native, "get_args", lambda: args)
        args.ckpt_format, args.auto_detect_ckpt_format = "torch_dist", False
        read_base = native._load_base_checkpoint

        class ReadCommon(Exception):
            pass

        def read_common(*positional, **kwargs):
            state, path, _, _ = read_base(*positional, **kwargs)
            assert Path(path) == alternate / "iter_0000001"
            assert state["args"].relax_lora_frozen_hf_base == checkpoint.hf_checkpoint_identity(a, version=2)
            raise ReadCommon

        def load_through_native_selector(**kwargs):
            with pytest.raises(ReadCommon):
                native_load(**kwargs)
            return 0, 0

        patch.setattr(native, "_load_base_checkpoint", read_common)
        patch.setattr(native, "load_checkpoint", load_through_native_selector)
        checkpoint.load_checkpoint(model, object(), None, {}, False)
    assert args.finetune and args.load == str(root)
    assert args.relax_lora_frozen_hf_base == checkpoint.hf_checkpoint_identity(a, version=2)
    args.pretrained_checkpoint = args.ckpt_step = None
    args.no_load_optim = args.no_load_rng = True
    args.finetune = False

    args.moe_router_enable_expert_bias = True
    args.moe_router_bias_update_rate = 1e-3
    checkpoint.save_checkpoint(2, model, None, None)
    assert dist_checkpointing.load_common_state_dict(root / "iter_0000002")["args"].relax_lora_frozen_hf_base is None
    with pytest.raises(RuntimeError, match="MoE expert bias"):
        checkpoint.save_checkpoint(2, model, None, None, lora_only=True)
    args.moe_router_bias_update_rate = 0

    # An old full DCP still loads and can be saved in full. Discarding its
    # base would lose the only authoritative copy of those weights.
    del args.relax_lora_frozen_hf_base
    checkpoint.save_checkpoint(3, model, None, None)
    checkpoint.load_checkpoint(model, object(), None, {}, False)
    checkpoint.save_checkpoint(4, model, None, None)
    with pytest.raises(RuntimeError, match="verified frozen HF base"):
        checkpoint.save_checkpoint(2, model, None, None, lora_only=True)


@pytest.mark.parametrize(
    "load_direct,selected_dir,next_tracker,expected_weight",
    [
        (False, "iter_0000007", "release", 23.0),
        (True, "iter_0000000", "7", 11.0),
        (True, "release", "7", 47.0),
    ],
)
def test_resume_loads_weights_from_preflight_iteration(
    monkeypatch, tmp_path, load_direct, selected_dir, next_tracker, expected_weight
):
    from megatron.training import checkpointing as native

    root = tmp_path / "run"
    model = torch.nn.Linear(1, 1, bias=False)
    for dirname, weight in [("iter_0000000", 11), ("iter_0000007", 23), ("release", 47)]:
        rank = root / dirname / "mp_rank_00"
        rank.mkdir(parents=True)
        torch.save(
            {
                "checkpoint_version": 3.0,
                "iteration": 7 if dirname == "iter_0000007" else 0,
                "model": {"weight": torch.tensor([[float(weight)]])},
            },
            rank / "model_optim_rng.pt",
        )
    tracker = root / "latest_checkpointed_iteration.txt"
    tracker.write_text("7")
    load_path = str(root / selected_dir if load_direct else root)
    args = SimpleNamespace(
        load=load_path,
        pretrained_checkpoint=str(root) if load_direct else None,
        ckpt_step=None,
        ckpt_format="torch",
        auto_detect_ckpt_format=True,
        non_persistent_global_ckpt_dir=None,
        non_persistent_ckpt_type=None,
        no_load_optim=True,
        no_load_rng=True,
        finetune=False,
        bf16=False,
        fp16=False,
    )
    monkeypatch.setattr(checkpoint, "get_args", lambda: args)
    monkeypatch.setattr(native, "get_args", lambda: args)
    monkeypatch.setattr(native, "_CHECKPOINT_VERSION", None)
    for axis in ("tensor", "pipeline", "expert"):
        monkeypatch.setattr(native.mpu, f"get_{axis}_model_parallel_world_size", lambda: 1)
        monkeypatch.setattr(native.mpu, f"get_{axis}_model_parallel_rank", lambda: 0)
    # Change the shared tracker after MCore has opened the first checkpoint,
    # before it opens the rank's file again to restore model parameters.
    read_base = native._load_base_checkpoint

    def move_tracker(*positional, **kwargs):
        result = read_base(*positional, **kwargs)
        if kwargs["rank0"]:
            tracker.write_text(next_tracker)
        return result

    monkeypatch.setattr(native, "_load_base_checkpoint", move_tracker)
    checkpoint.load_checkpoint([model], None, None, {}, False)
    torch.testing.assert_close(model.weight, torch.tensor([[expected_weight]]))
    assert args.load == load_path and not args.finetune


def test_v1_first_run_fallback_requires_confirmed_original_base(monkeypatch, tmp_path):
    from megatron.core.dist_checkpointing.core import CheckpointingConfig, save_config
    from safetensors.torch import save_file

    from scripts.tools.convert_checkpoint_lora import validate_lora_base

    base, other, native = (tmp_path / name for name in ("original", "other", "native"))
    for directory, size in ((base, 2), (other, 4)):
        directory.mkdir()
        (directory / "config.json").write_text("{}")
        save_file({"weight": torch.zeros(size)}, directory / "model.safetensors")
    iteration = native / "iter_0000001"
    iteration.mkdir(parents=True)
    (native / "latest_checkpointed_iteration.txt").write_text("1")
    save_config(CheckpointingConfig("torch_dist"), str(iteration))
    args = _runtime_args()
    args.hf_checkpoint = args.ref_load = str(base)
    args.load = args.save = str(native)
    args.no_load_optim, args.loss_type, args.dist_ckpt_strictness = True, "sft", "raise_all"
    model = [_AdapterModel()]
    names = checkpoint._global_trainable_lora_parameter_names(model)
    marker = checkpoint._lora_checkpoint_metadata(args, names, checkpoint.hf_checkpoint_identity(base), version=1)
    saved = dict(vars(args), relax_lora_checkpoint_metadata=marker)
    torch.save({"args": saved}, iteration / "common.pt")
    monkeypatch.setattr(checkpoint, "get_args", lambda: args)
    monkeypatch.delenv("RELAX_LORA_V1_RESUME_BASE", raising=False)
    with pytest.raises(RuntimeError, match="legacy v1"):
        checkpoint.load_checkpoint(model, object(), None, {}, False)
    monkeypatch.setenv("RELAX_LORA_V1_RESUME_BASE", str(other))
    with pytest.raises(RuntimeError, match="base HF identity mismatch"):
        checkpoint.load_checkpoint(model, object(), None, {}, False)
    monkeypatch.setenv("RELAX_LORA_V1_RESUME_BASE", str(base))
    with pytest.raises(ValueError, match="legacy v1"):
        validate_lora_base(saved, base)
    loaded = []
    monkeypatch.setattr(checkpoint, "_load_checkpoint_hf", lambda **kw: loaded.append(kw["load_path"]))
    monkeypatch.setattr(checkpoint, "_load_checkpoint_megatron", lambda **kw: (1, 0))
    assert checkpoint.load_checkpoint(model, object(), None, {}, False) == (1, 0)
    assert loaded == [str(base)]

    # A v1 BIN-only manifest still matches its BIN shard after a different
    # model.safetensors is placed beside it. Even explicit confirmation must
    # refuse weights that the historical fingerprint never covered.
    (base / "model.safetensors").rename(base / "pytorch_model.bin")
    (base / "pytorch_model.bin.index.json").write_text(json.dumps({"weight_map": {"weight": "pytorch_model.bin"}}))
    marker["base_hf_checkpoint"] = checkpoint.hf_checkpoint_identity(base)
    torch.save({"args": saved}, iteration / "common.pt")
    save_file({"weight": torch.ones(2)}, base / "model.safetensors")
    with pytest.raises(RuntimeError, match="base HF identity mismatch"):
        checkpoint.load_checkpoint(model, object(), None, {}, False)
    assert loaded == [str(base)]


def test_hf_checkpoint_identity_detects_shard_replacement(tmp_path):
    shard = tmp_path / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"first")
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": shard.name}}))
    before = checkpoint.hf_checkpoint_identity(str(tmp_path))

    shard.write_bytes(b"replacement")
    after = checkpoint.hf_checkpoint_identity(str(tmp_path))

    assert before["shard_stat_sha256"] != after["shard_stat_sha256"]


def test_hf_checkpoint_identity_covers_pytorch_bin(tmp_path):
    weights = tmp_path / "pytorch_model.bin"
    weights.write_bytes(b"weights")
    (tmp_path / "config.json").write_text("{}")

    identity = checkpoint.hf_checkpoint_identity(str(tmp_path))

    assert identity["shard_count"] == "1"
    assert identity["shard_stat_sha256"]


@pytest.mark.parametrize(
    ("tracker", "ckpt_step", "direct_path", "loss_type", "directory"),
    [
        ("7", None, False, "sft", "iter_0000007"),
        ("17", 42, False, "rm", "iter_0000042"),
        # Megatron treats ckpt_step=0 as unset; metadata must follow the same tracker.
        ("17", 0, False, "rm", "iter_0000017"),
        ("release", None, False, "sft", "release"),
        (None, None, True, "sft", "iter_0000007"),
    ],
)
def test_lightweight_resume_loads_hf_base_before_lora_dcp(
    monkeypatch, tmp_path, tracker, ckpt_step, direct_path, loss_type, directory
):
    import megatron.core

    resume_path = tmp_path / "checkpoint"
    checkpoint_dir = resume_path / directory
    checkpoint_dir.mkdir(parents=True)
    if tracker is not None:
        (resume_path / "latest_checkpointed_iteration.txt").write_text(tracker)
    (checkpoint_dir / "common.pt").touch()
    args = SimpleNamespace(
        load=str(checkpoint_dir if direct_path else resume_path),
        hf_checkpoint="/models/base",
        dist_ckpt_strictness="assume_ok_unexpected",
        ckpt_step=ckpt_step,
        loss_type=loss_type,
        hidden_size=1024,
    )
    marker = {
        "format_version": 2,
        "base_hf_checkpoint": {
            "config_sha256": "c",
            "index_sha256": None,
            "shard_count": "1",
            "shard_content_sha256": "s",
        },
    }
    common = {
        "args": SimpleNamespace(
            loss_type=loss_type,
            head_type=checkpoint.REWARD_MODEL_HEAD_TYPE if loss_type == "rm" else None,
            checkpoint_role="actor",
            **{checkpoint._LORA_CHECKPOINT_METADATA_ATTR: marker},
        )
    }
    calls = []
    common_loads = []
    tensor_metadata_loads = []
    monkeypatch.setattr(
        megatron.core,
        "dist_checkpointing",
        SimpleNamespace(
            check_is_distributed_checkpoint=lambda path: Path(path) == checkpoint_dir,
            load_common_state_dict=lambda path: common_loads.append(Path(path)) or common,
            load_tensors_metadata=lambda path: (
                tensor_metadata_loads.append(Path(path))
                or {"model.output_layer.weight": SimpleNamespace(global_shape=(1, 1024))}
            ),
        ),
    )
    monkeypatch.setattr(checkpoint, "get_args", lambda: args)
    monkeypatch.setattr(checkpoint, "_alias_renamed_transfer_queue_enum", lambda: None)
    monkeypatch.setattr(checkpoint, "_validate_lora_model_state_load", lambda _model: nullcontext())
    monkeypatch.setattr(
        checkpoint,
        "_validate_lora_checkpoint_metadata",
        lambda current_args, model, metadata: (
            calls.append(("validate", current_args, model, metadata)) or args.hf_checkpoint
        ),
    )
    monkeypatch.setattr(checkpoint, "_load_checkpoint_hf", lambda **_kwargs: calls.append(("hf",)))
    monkeypatch.setattr(
        checkpoint,
        "_load_checkpoint_megatron",
        lambda **kwargs: calls.append(("dcp", kwargs["strict"])) or (7, 0),
    )

    assert checkpoint.load_checkpoint("model", "optimizer", "scheduler", {}, False) == (7, 0)
    assert [call[0] for call in calls] == ["validate", "hf", "dcp"]
    assert calls[-1] == ("dcp", True)
    assert args.dist_ckpt_strictness == "assume_ok_unexpected"
    assert common_loads == [checkpoint_dir]
    assert tensor_metadata_loads == ([checkpoint_dir] if loss_type == "rm" else [])
    assert calls[0][3] is marker


def test_lightweight_resume_preserves_hybrid_optimizer_step(monkeypatch):
    from megatron.core.optimizer.cpu_offloading import HybridDeviceOptimizer

    captured = {}
    parameter = torch.nn.Parameter(torch.zeros(1))
    sub_optimizer = SimpleNamespace(
        state={parameter: {"step": torch.tensor(1.0)}},
        param_groups=[{"params": [parameter], "step": 1}],
    )
    optimizer = SimpleNamespace(
        state={parameter: {"step": torch.tensor(1.0)}},
        param_groups=[{"params": [parameter], "step": 1}],
        sub_optimizers=[sub_optimizer],
    )

    def fake_load_state_dict(_optimizer, state_dict):
        captured.update(state_dict)

    monkeypatch.setattr(HybridDeviceOptimizer, "load_state_dict", fake_load_state_dict)
    state_dict = {
        "param_groups": [{"params": [0], "step": 7}],
        "state": {
            0: {
                "step": torch.tensor(1.0),
                "exp_avg": torch.zeros(1),
                "exp_avg_sq": torch.zeros(1),
            }
        },
    }

    with checkpoint._preserve_hybrid_optimizer_steps_on_load():
        HybridDeviceOptimizer.load_state_dict(optimizer, state_dict)

    assert captured["state"][0]["step"].item() == 7
    assert optimizer.state[parameter]["step"].item() == 7
    assert optimizer.param_groups[0]["step"] == 7
    assert sub_optimizer.state[parameter]["step"].item() == 7
    assert sub_optimizer.param_groups[0]["step"] == 7
    assert HybridDeviceOptimizer.load_state_dict is fake_load_state_dict


def test_lightweight_save_syncs_hybrid_optimizer_checkpoint_step():
    from megatron.core.optimizer.cpu_offloading import HybridDeviceOptimizer

    optimizer = object.__new__(HybridDeviceOptimizer)
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer.state = {parameter: {"step": torch.tensor(7.0)}}
    optimizer.param_groups = [{"params": [parameter], "step": 1}]
    optimizer.cpu_optimizers = [SimpleNamespace(param_groups=[{"params": [parameter], "step": 1}])]
    optimizer.gpu_optimizer = None
    outer = SimpleNamespace(optimizer=optimizer)

    checkpoint._sync_hybrid_optimizer_checkpoint_steps(outer)

    assert optimizer.param_groups[0]["step"] == 7
    assert optimizer.cpu_optimizers[0].param_groups[0]["step"] == 7


def test_lightweight_save_syncs_all_hybrid_steps_in_chained_optimizer() -> None:
    from megatron.core.optimizer.cpu_offloading import HybridDeviceOptimizer
    from megatron.core.optimizer.optimizer import ChainedOptimizer

    children = []
    for step in (7, 11):
        parameter = torch.nn.Parameter(torch.zeros(1))
        optimizer = object.__new__(HybridDeviceOptimizer)
        optimizer.state = {parameter: {"step": torch.tensor(float(step))}}
        optimizer.param_groups = [{"params": [parameter], "step": 1}]
        optimizer.cpu_optimizers = [SimpleNamespace(param_groups=[{"params": [parameter], "step": 1}])]
        optimizer.gpu_optimizer = SimpleNamespace(param_groups=[{"params": [parameter], "step": 1}])
        children.append(SimpleNamespace(optimizer=optimizer))

    # Use MCore's real property contract: .optimizer asserts for multiple children.
    # Bypass initialization because this only tests traversal, not GPU optimizer setup.
    chained = object.__new__(ChainedOptimizer)
    chained.chained_optimizers = children

    checkpoint._sync_hybrid_optimizer_checkpoint_steps(SimpleNamespace(optimizer=chained))

    for child, step in zip(children, (7, 11), strict=True):
        assert child.optimizer.param_groups[0]["step"] == step
        assert child.optimizer.cpu_optimizers[0].param_groups[0]["step"] == step
        assert child.optimizer.gpu_optimizer.param_groups[0]["step"] == step


def test_lightweight_resume_restores_step_after_distributed_parameter_load(monkeypatch):
    from megatron.core.optimizer.cpu_offloading import HybridDeviceOptimizer
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    parameter = torch.nn.Parameter(torch.zeros(1))
    sub_optimizer = SimpleNamespace(
        state={parameter: {"step": torch.tensor(1.0)}},
        param_groups=[{"params": [parameter], "step": 1}],
    )
    hybrid_optimizer = object.__new__(HybridDeviceOptimizer)
    hybrid_optimizer.state = {parameter: {"step": torch.tensor(1.0)}}
    hybrid_optimizer.param_groups = [{"params": [parameter], "step": 1}]
    hybrid_optimizer.cpu_optimizers = [sub_optimizer]
    hybrid_optimizer.gpu_optimizer = None
    distributed_optimizer = object.__new__(DistributedOptimizer)
    distributed_optimizer.optimizer = hybrid_optimizer

    def fake_distributed_load(_optimizer, _state_dict):
        # Model the later parameter-shard phase leaving the bootstrap step.
        hybrid_optimizer.state[parameter]["step"] = torch.tensor(1.0)

    monkeypatch.setattr(DistributedOptimizer, "load_state_dict", fake_distributed_load)
    with checkpoint._preserve_hybrid_optimizer_steps_on_load():
        DistributedOptimizer.load_state_dict(
            distributed_optimizer,
            {"optimizer": {"param_groups": [{"step": 7}]}},
        )

    assert hybrid_optimizer.state[parameter]["step"].item() == 7
    assert hybrid_optimizer.param_groups[0]["step"] == 7
    assert sub_optimizer.state[parameter]["step"].item() == 7
    assert sub_optimizer.param_groups[0]["step"] == 7
    assert DistributedOptimizer.load_state_dict is fake_distributed_load
