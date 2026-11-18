# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""ART CPU actor orchestration, real DCP resume and pre-forward agreement."""

import json
import multiprocessing as mp
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from relax.backends.fsdp import checkpoint as ckpt
from relax.backends.fsdp.actor import FSDPTrainRayActor
from relax.backends.fsdp.arguments import validate_generative_config
from relax.engine.rollout.data_source import RolloutDataSource
from tests.cpu.backends.fsdp.test_save_and_validation import _valid_args
from tests.cpu.backends.fsdp.test_weight_transaction import _manifest
from tests.cpu.backends.fsdp.test_weight_transaction import _shell as _weight_shell
from tests.cpu.engine.rollout.test_art_routing import _acquisition, _Adapter


def _shell(args):
    actor = object.__new__(FSDPTrainRayActor)
    actor.args, actor.device, actor._rank = args, torch.device("cpu"), 0
    actor.model = torch.nn.Linear(2, 1, bias=False)
    torch.nn.init.zeros_(actor.model.weight)
    actor.adapter = _Adapter()
    actor.adapter.family = "fixture"
    actor.adapter.supported_tasks = ("i2i",)
    actor.adapter.full_trainable_targets = ("weight",)
    actor._ema_snapshot = actor._lora_plan = None
    actor._trainable_mode, actor._base_model_sha256 = "full", "b" * 64
    actor.policy_version, actor._lr_steps, actor._base_lr = 7, 0, 0.05
    actor._active_weight_manifest_sha256 = "a" * 64
    actor.optimizer = torch.optim.AdamW(actor.model.parameters(), lr=0.05, foreach=False)
    actor.events = []

    def sync(rollout_id):
        actor.events.append(("publish", rollout_id, actor._lr_steps))
        actor.policy_version += int(not actor._art_unchanged_policy)
        actor.events.append(("ready", actor.policy_version))

    actor._sync_save_and_release = sync
    actor._run_step_evaluation = lambda _rollout: None
    actor._log_train_metrics = lambda *_args: None
    return actor


def _profile():
    return _valid_args(
        generation_task="i2i",
        model_adapter_path="relax.models.sensenova.adapter.SenseNovaU1ImageEditAdapter",
        micro_batch_size=1,
        num_updates_per_batch=2,
        fsdp_param_dtype="bf16",
        fsdp_reduce_dtype="fp32",
        fsdp_master_dtype="fp32",
        artifact_retention_rollouts=1,
        n_samples_per_prompt=8,
        global_batch_size=16,
        rollout_batch_size=2,
        actor_num_nodes=1,
        actor_num_gpus_per_node=8,
        reward_component_weights={"ocr": 1.0},
        sampling_config={
            "algorithm": "art",
            "sde_type": "ode",
            "eta": 0.0,
            "num_inference_steps": 4,
            "storage_dtype": "float16",
            "wave_groups": 1,
            "art": {"train_step_indices": [0, 2], "max_cache_bytes": 10**8, "max_replay_bytes": 10**8},
        },
    )


def test_art_shell_launch_uses_registered_defaults_and_preserves_resume_and_extra_args(tmp_path):
    import argparse
    import os
    import subprocess
    from pathlib import Path

    from relax.backends.fsdp.arguments import add_generative_arguments

    capture = tmp_path / "argv.json"
    ray = tmp_path / "ray"
    ray.write_text(
        "#!/usr/bin/env python3\nimport json, os, sys\n"
        "from pathlib import Path\nPath(os.environ['ART_TEST_ARGV']).write_text(json.dumps(sys.argv[1:]))\n"
    )
    ray.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "ART_TEST_ARGV": str(capture),
        "MODEL_PATH": str(tmp_path / "model with spaces"),
        "PROMPT_SET": str(tmp_path / "prompts.jsonl"),
        "SAVE_DIR": str(tmp_path / "checkpoints"),
        "ARTIFACT_ROOT": str(tmp_path / "artifacts"),
        "REWARD_ENDPOINT": "http://reward.invalid/v1",
        "LOAD_DIR": str(tmp_path / "prior checkpoint"),
        "ART_TRAIN_STEP_INDICES": "[0,10,19]",
        "ART_MAX_CACHE_BYTES": str(1 << 35),
        "ART_MAX_REPLAY_BYTES": str(1 << 36),
        "ART_GROUP_OVERLAP": "0",
        "ART_GENERATION_PREFETCH": "0",
        "N_PROMPTS": "8",
        "N_SAMPLES": "8",
        "STEPS": "20",
        "RAY_NO_WAIT": "1",
    }
    script = Path(__file__).resolve().parents[4] / "scripts/training/diffusion/run-sensenova-i2i-art-full-8xgpu.sh"
    subprocess.run(["bash", str(script), "--use-clearml"], env=env, check=True, capture_output=True)
    submitted = json.loads(capture.read_text())
    assert submitted[:2] == ["job", "submit"] and "--no-wait" in submitted
    command = submitted[submitted.index("--") + 1 :]
    assert command[:3] == ["python3", "-m", "relax.entrypoints.train"]
    argv = command[3:]
    assert argv[argv.index("--model-path") + 1] == env["MODEL_PATH"]
    assert argv[argv.index("--load") + 1] == env["LOAD_DIR"] and argv[-1] == "--use-clearml"
    registered, _ = add_generative_arguments(argparse.ArgumentParser()).parse_known_args(argv)
    assert not hasattr(registered, "generative_reference_policy")
    args = argparse.Namespace(**{**vars(_profile()), **vars(registered)})
    validate_generative_config(args)
    args.generative_reference_policy = "ema"
    with pytest.raises(ValueError, match="fixed"):
        validate_generative_config(args)
    capture.unlink()
    legacy = subprocess.run(["bash", str(script), "--mode", "preflight"], env=env, capture_output=True)
    assert legacy.returncode != 0 and not capture.exists()


@pytest.mark.parametrize(
    "invalid", [None, "budget", "odd_k", "ema", "steps", "options", "precision", "streaming", "staleness"]
)
def test_art_profile_is_explicit_and_rejects_unsupported_semantics(invalid):
    args = _profile()
    if invalid == "budget":
        del args.sampling_config["art"]["max_cache_bytes"]
    elif invalid == "odd_k":
        args.n_samples_per_prompt = 7
    elif invalid == "ema":
        args.generative_reference_policy = "ema"
    elif invalid == "steps":
        args.sampling_config["art"]["train_step_indices"] = [2, 0]
    elif invalid == "options":
        args.sampling_config["art"]["corrective_mask_enabled"] = False
    elif invalid == "precision":
        args.fsdp_master_dtype = "bf16"
    elif invalid == "streaming":
        args.use_streaming_dataset = True
    elif invalid == "staleness":
        args.max_staleness = 1
    if invalid is None:
        validate_generative_config(args)
    else:
        with pytest.raises(ValueError, match="ART"):
            validate_generative_config(args)


@pytest.mark.parametrize("constant", [False, True])
def test_actor_has_two_updates_one_publish_or_global_zero_skip(monkeypatch, constant):
    args, rows, _caches, _ = _acquisition(monkeypatch, constant=constant)
    actor = _shell(args)
    actor.train(0, rows)
    assert actor._lr_steps == (0 if constant else 2)
    assert actor.policy_version == (7 if constant else 8)
    assert actor.events == [("publish", 0, actor._lr_steps), ("ready", actor.policy_version)]
    assert actor.adapter.prefix_calls == (0 if constant else 16)
    if constant:
        assert not actor.optimizer.state
    else:
        assert actor.optimizer.state[actor.model.weight]["step"] == 2
        # The first update moves both dimensions; the entirely masked second
        # window still moves them through AdamW's historical moments.
        assert actor.model.weight[0, 0] < -0.05
        assert actor.model.weight[0, 1] > 0.05


@pytest.mark.parametrize("changed", [None, "seed", "steps", "k", "lr", "budget"])
def test_resume_accepts_legacy_scheduling_keys_but_keeps_math_and_optimizer_strict(tmp_path, changed):
    import json

    actor = _shell(_profile())
    actor.args.seed, actor.args.lr = 7, 1e-5
    recorded = deepcopy(actor._adapter_contract())
    recorded["art"]["sampling"].update(overlap_group_processing=False, overlap_pending_groups=2)
    before = deepcopy(recorded)
    path = tmp_path / "adapter_contract.json"
    path.write_text(json.dumps(recorded))
    if changed == "seed":
        actor.args.seed += 1
    elif changed == "steps":
        actor.args.sampling_config["num_inference_steps"] += 1
    elif changed == "k":
        actor.args.n_samples_per_prompt += 2
    elif changed == "lr":
        actor.args.lr *= 2
    elif changed == "budget":
        actor.args.sampling_config["art"]["max_cache_bytes"] += 1
    if changed is None:
        actor._assert_resumable(str(tmp_path))
    else:
        with pytest.raises(ValueError, match="ART algorithm"):
            actor._assert_resumable(str(tmp_path))
    assert recorded == before and json.loads(path.read_text()) == before


def test_invalid_judge_skip_preserves_existing_adam_moments_and_allows_next_round(monkeypatch):
    args, rows, caches, _ = _acquisition(monkeypatch)
    actor = _shell(args)
    actor.train(0, rows)
    before_weight = actor.model.weight.detach().clone()
    before_state = deepcopy(actor.optimizer.state_dict())
    before_lr_steps, before_version = actor._lr_steps, actor.policy_version
    _, skipped, caches, _ = _acquisition(monkeypatch, invalid=(0,))
    for cache in caches.values():
        cache["policy_versions"].fill_(before_version)
    actor.train(1, skipped)
    assert actor._lr_steps == before_lr_steps and actor.policy_version == before_version
    torch.testing.assert_close(actor.model.weight, before_weight, rtol=0, atol=0)
    after_state = actor.optimizer.state_dict()
    assert before_state["param_groups"] == after_state["param_groups"]
    for key, value in before_state["state"][0].items():
        torch.testing.assert_close(value, after_state["state"][0][key], rtol=0, atol=0)
    _, next_rows, caches, _ = _acquisition(monkeypatch)
    for cache in caches.values():
        cache["policy_versions"].fill_(before_version)
    actor.train(2, next_rows)
    assert actor._lr_steps == before_lr_steps + 2 and actor.policy_version == before_version + 1


def test_second_window_failure_cannot_publish_save_or_repeat_first_update(monkeypatch):
    args, rows, _caches, _ = _acquisition(monkeypatch)
    actor = _shell(args)
    original = actor._prepare_micro_batch

    def prepare(mb):
        if mb["art_update_index"] == 1:
            raise OSError("second-window cache failure")
        return original(mb)

    actor._prepare_micro_batch = prepare
    with pytest.raises(RuntimeError, match="before forward"):
        actor.train(0, rows)
    assert actor._lr_steps == 1 and actor.policy_version == 7
    assert actor.events == []
    with pytest.raises(RuntimeError, match="restore the last committed"):
        actor.train(0, rows)
    assert actor._lr_steps == 1


def test_unchanged_publication_uses_same_version_and_live_tensor_source(monkeypatch):
    actor = _publication_shell(monkeypatch, version=7, resume_pending=False)
    actor._art_unchanged_policy, actor._lora_plan = True, None
    actor.args.sampling_config = {"algorithm": "art"}
    actor._run_weight_transaction = lambda *_args: None
    assert actor._sync_tensor_source() is None  # Live parameters, never NFT EMA.
    actor.update_weights(offload_after_sync=False)
    assert actor.policy_version == 7 and not actor._art_unchanged_policy
    actor.update_weights(offload_after_sync=False)
    assert actor.policy_version == 8


@pytest.mark.parametrize("constant", [False, True])
def test_complete_window_dcp_resume_matches_uninterrupted_next_acquisition(monkeypatch, tmp_path, constant):
    args, rows, _caches, _ = _acquisition(monkeypatch, constant=constant)
    args.generation_task, args.load = "i2i", str(tmp_path)
    actor = _shell(args)
    actor.train(0, rows)
    source = object.__new__(RolloutDataSource)
    source.args = args
    args.rollout_global_dataset, args.save = True, str(tmp_path)
    args.rollout_shuffle = False
    source.sample_offset = source.sample_group_index = 2
    source.sample_index, source.epoch_id = 2 * args.n_samples_per_prompt, 0
    source.metadata, source._use_streaming = {}, False
    source.save(0)
    ckpt.save_checkpoint(
        str(tmp_path),
        "i2i",
        0,
        actor.model,
        actor.optimizer,
        trainer_state={"rollout_id": 0, "policy_version": actor.policy_version, "lr_scheduler_steps": actor._lr_steps},
        weight_sync_manifest={},
        adapter_contract=actor._adapter_contract(),
        require_rng_state=True,
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(args, "_start_rollout_id_explicit", True, raising=False)
        patch.setattr(args, "start_rollout_id", 7, raising=False)
        with pytest.raises(RuntimeError, match="Explicit start_rollout_id"):
            _shell(args)._maybe_resume()
    from relax.utils.utils import recovery_load_path

    # Health recovery replaces the initial explicit cursor with the newest
    # committed step, without weakening validation for a fresh explicit resume.
    args.start_rollout_id, args._start_rollout_id_explicit = 0, True
    args.no_save_optim = args.no_save_rng = False
    recovery_load_path(args)
    restored = _shell(args)
    assert restored._maybe_resume() == 1
    assert restored._resume_initial_sync_pending
    assert restored._lr_steps == actor._lr_steps
    source.sample_offset = source.sample_group_index = source.sample_index = 0
    source.load(0)
    assert (source.sample_offset, source.sample_group_index, source.sample_index) == (
        2,
        2,
        2 * args.n_samples_per_prompt,
    )
    with pytest.raises(RuntimeError, match="publish the restored"):
        restored.train(1, rows)
    restored._resume_initial_sync_pending = False  # CPU stand-in for successful initial publication.
    # Next acquisition must carry the committed live behavior-policy version.
    _caches[3]["policy_versions"].fill_(actor.policy_version)
    _caches[9]["policy_versions"].fill_(actor.policy_version)
    for cache in _caches.values():
        cache["art_old_velocities"].copy_(actor.model.weight.detach().expand_as(cache["art_old_velocities"]))
    actor.train(1, rows)
    restored.train(1, rows)
    assert restored.policy_version == actor.policy_version and restored._lr_steps == actor._lr_steps
    torch.testing.assert_close(restored.model.weight, actor.model.weight, rtol=0, atol=0)
    for key, value in actor.optimizer.state.get(actor.model.weight, {}).items():
        torch.testing.assert_close(restored.optimizer.state[restored.model.weight][key], value, rtol=0, atol=0)
    # Different window/optimizer settings cannot silently resume this checkpoint.
    restored.args.seed += 1
    with pytest.raises(ValueError, match="ART algorithm"):
        restored._assert_resumable(str(tmp_path / "i2i" / ckpt.iter_dir_name(0)))


@pytest.mark.parametrize("failure", ["write", "missing", "reuse", "seed"])
def test_art_save_keeps_previous_recovery_point_until_cursor_and_dcp_are_committed(monkeypatch, tmp_path, failure):
    import ray

    args = _profile()
    args.save = args.load = str(tmp_path)
    args.generation_seed = 42
    args.rollout_global_dataset, args.rollout_shuffle = True, False
    args.max_actor_ckpt_to_keep, args.rotate_ckpt, args.save_interval = 1, False, 1
    actor = _shell(args)
    actor._build_manifest = lambda: SimpleNamespace(to_dict=lambda: {})
    source = object.__new__(RolloutDataSource)
    source.args, source.metadata, source._use_streaming = args, {}, False
    source.sample_offset = source.sample_group_index = 2
    source.sample_index, source.epoch_id = 2 * args.n_samples_per_prompt, 0
    fail = False

    def save_cursor(rollout_id):
        if fail:
            if failure == "write":
                raise OSError("cursor disk full")
            if failure == "missing":
                return  # Simulate a lost cursor before DCP's final publication check.
        source.save(rollout_id)

    # Only the Ray RPC boundary is replaced; real cursor IO, DCP, rotation and
    # actor resume execute, including a failed publication after shard writes.
    actor.rollout_manager = SimpleNamespace(save=SimpleNamespace(remote=save_cursor))
    monkeypatch.setattr(ray, "get", lambda result: result)
    actor.save_model(0)
    cursor = tmp_path / "dataset" / "global_dataset_state_dict_0.pt"
    original_cursor = cursor.read_bytes()
    actor.save_model(0)  # Idempotent retries must reuse, not rewrite, the cursor.
    assert cursor.read_bytes() == original_cursor
    first = tmp_path / "i2i" / ckpt.iter_dir_name(0)
    first_weights = actor.model.weight.detach().clone()
    with torch.no_grad():
        actor.model.weight.add_(1)
    source.sample_offset = source.sample_group_index = 4
    source.sample_index = 4 * args.n_samples_per_prompt
    if failure == "reuse":
        source._dataset_resume_contract = {"replacement": "different manifest"}
    elif failure == "seed":
        source.sample_offset = source.sample_group_index = 2
        source.sample_index = 2 * args.n_samples_per_prompt
        args.generation_seed += 1
    fail = True
    with pytest.raises((RuntimeError, ValueError), match="cursor|metadata"):
        actor.save_model(0 if failure in ("reuse", "seed") else 1)
    assert cursor.read_bytes() == original_cursor
    assert first.is_dir() and ckpt.find_latest_committed(str(tmp_path), "i2i") == 0
    assert not (tmp_path / "i2i" / ckpt.iter_dir_name(1) / "COMMITTED").exists()
    for root in (tmp_path, tmp_path / "i2i"):
        assert (root / "latest_checkpointed_iteration.txt").read_text() == "0"
    args.generation_seed = 42
    restored = _shell(args)
    assert restored._maybe_resume() == 1
    torch.testing.assert_close(restored.model.weight, first_weights, rtol=0, atol=0)
    fail = False
    actor.save_model(1)
    assert not first.exists() and ckpt.find_latest_committed(str(tmp_path), "i2i") == 1
    assert _shell(args)._maybe_resume() == 2


@pytest.mark.parametrize("failure", ["missing", "corrupt", "schema", "k", "acquisition"])
def test_art_resume_rejects_bad_data_cursor_before_loading_training_state(monkeypatch, tmp_path, failure):
    from unittest.mock import Mock

    args = _profile()
    args.load, args.rollout_global_dataset = str(tmp_path), True
    actor = _shell(args)
    root = tmp_path / "i2i" / ckpt.iter_dir_name(0)
    root.mkdir(parents=True)
    (root / "COMMITTED").touch()
    (root / "adapter_contract.json").write_text(json.dumps(actor._adapter_contract()))
    (root / "trainer_state.json").write_text(json.dumps({"rollout_id": 0}))
    cursor = tmp_path / "dataset" / "global_dataset_state_dict_0.pt"
    cursor.parent.mkdir()
    state = dict(sample_offset=2, epoch_id=0, sample_group_index=2, sample_index=16, metadata={}, use_streaming=False)
    if failure == "corrupt":
        cursor.write_bytes(b"incomplete snapshot")
    elif failure != "missing":
        if failure == "schema":
            del state["sample_offset"]
        elif failure == "k":
            state["sample_index"] = 15
        else:
            state["rollout_id"] = 1
        torch.save(state, cursor)
    load = Mock(return_value={"rollout_id": 0, "policy_version": 7, "lr_scheduler_steps": 0})
    monkeypatch.setattr(ckpt, "load_checkpoint", load)
    before = actor.model.weight.detach().clone()
    with pytest.raises(RuntimeError, match="ART resume contract validation failed"):
        actor._maybe_resume()
    load.assert_not_called()
    torch.testing.assert_close(actor.model.weight, before, rtol=0, atol=0)
    assert (actor.policy_version, actor._lr_steps) == (7, 0)
    source = object.__new__(RolloutDataSource)
    source.args = args
    source.sample_index = 99
    with pytest.raises(ValueError, match="ART"):
        source.load(0)
    assert source.sample_index == 99


@pytest.mark.parametrize("failure", ["missing", "incomplete", "corrupt"])
def test_strict_rng_load_cannot_silently_continue(tmp_path, failure):
    path = tmp_path / "rng_rank0.pt"
    if failure == "incomplete":
        torch.save({"torch": torch.get_rng_state()}, path)
    elif failure == "corrupt":
        path.write_bytes(b"not a torch checkpoint")
    with pytest.raises(Exception):
        ckpt._load_rng(str(tmp_path), strict=True)


def _prepare_failure_worker(rank, path, queue):
    dist.init_process_group(
        "gloo", init_method=f"file://{path}", rank=rank, world_size=2, timeout=timedelta(seconds=20)
    )
    from relax.utils import distributed_utils

    distributed_utils.GLOO_GROUP = dist.new_group(backend="gloo", timeout=timedelta(seconds=20))
    try:
        with pytest.MonkeyPatch.context() as patch:
            args, rows, _caches, _ = _acquisition(patch)
            actor = _shell(args)
            actor._rank = rank
            if rank == 1:
                actor._prepare_micro_batch = lambda _mb: (_ for _ in ()).throw(OSError("rank-one load failure"))
            try:
                actor.train(0, rows)
                queue.put((rank, "unexpected success", actor.adapter.prefix_calls))
            except RuntimeError as exc:
                queue.put((rank, str(exc), actor.adapter.prefix_calls))
    finally:
        dist.destroy_process_group()


def test_two_gloo_ranks_reject_a_peer_load_failure_before_any_model_forward(tmp_path):
    context = mp.get_context("spawn")
    queue = context.Queue()
    workers = [
        context.Process(target=_prepare_failure_worker, args=(rank, str(tmp_path / "rendezvous"), queue))
        for rank in range(2)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=45)
        assert not worker.is_alive() and worker.exitcode == 0
    results = [queue.get(timeout=5) for _ in workers]
    assert {value[0] for value in results} == {0, 1}
    assert all("before forward" in value[1] and value[2] == 0 for value in results)


def _publication_shell(monkeypatch, *, version, resume_pending):
    shell = _weight_shell()
    shell.device = torch.device("cpu")
    shell._rank = 1
    shell.policy_version = version
    shell._resume_initial_sync_pending = resume_pending
    shell._weight_sync_failures = 0
    shell._build_manifest = lambda: replace(_manifest(), policy_version=shell.policy_version)
    shell._build_weight_iterator = lambda: iter(())
    shell._invalidate_ipc_topology = lambda: None
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *_: 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *_: 0)
    return shell
