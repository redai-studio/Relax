# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Run the real actor lifecycle with CPU tensors and no Megatron workers."""

import ast
import sys
import weakref
from collections import deque
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from relax.utils.training.indexer_replay import IndexerReplay


_ROOT = Path(__file__).resolve().parents[4]


def _load_definitions(path, names, namespace, *, class_name=None):
    nodes = ast.parse((_ROOT / path).read_text()).body
    if class_name is not None:
        nodes = next(node for node in nodes if isinstance(node, ast.ClassDef) and node.name == class_name).body
    definitions = [node for node in nodes if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert {node.name for node in definitions} == set(names)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *definitions],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(_ROOT / path), "exec"), namespace)


def _make_run(monkeypatch, *, offload=True, offline=False):
    events, tensor_refs, retained_iterators = [], [], []
    batches = deque()
    for index in range(2):
        pixels = torch.full((4, 8), float(index))
        tensor_refs.append(weakref.ref(pixels))
        batches.append(
            {
                "total_lengths": [8 + index],
                "loss_masks": [torch.tensor([0, 1, 1]) if index == 0 else [0, 1, 1, 1]],
                "rollout_indexer_topk": [torch.zeros(1, 1, dtype=torch.int32)],
                "multimodal_train_inputs": [
                    {
                        "pixel_values": pixels,
                        "image_grid_thw": torch.tensor([[index + 1, 2, 3]]),
                        "feature_attention_mask": torch.ones(1, index + 2, dtype=torch.int64),
                    }
                ],
            }
        )
    del pixels
    timer_state = SimpleNamespace()
    should_release = offload and not offline

    def check_live(event):
        assert all(ref() is not None for ref in tensor_refs), event
        events.append(event)

    def check_released(event):
        assert all((ref() is None) == should_release for ref in tensor_refs), event
        events.append(event)

    def all_gather_object(destination, value, **_kwargs):
        destination[0] = value

    def log_perf(*_args, **_kwargs):
        assert timer_state.seq_lens == [8, 9]
        assert timer_state.response_lens == [2, 3]
        assert timer_state.image_grid_thw == [(1, 2, 3), (2, 2, 3)]
        assert timer_state.images_seqlens == [6, 6, 6]
        assert timer_state.audio_seqlens == [2, 3]
        events.append("perf")

    @contextmanager
    def timed(_name):
        yield

    # The wholepack sampler is unrelated to this RL lifecycle. Keep this CPU
    # harness independent of the optional TransferQueue runtime it imports.
    wholepack = ModuleType("relax.utils.data.wholepack")
    wholepack.enabled = lambda _args: False
    wholepack.all_ready = lambda ready: ready
    monkeypatch.setitem(sys.modules, wholepack.__name__, wholepack)

    namespace = {
        "torch": torch,
        "timer": timed,
        "manual_gc_training": lambda fn: fn,
        "can_reuse_colocate_log_probs": lambda *_args, **_kwargs: False,
        "inverse_timer": lambda _name: nullcontext(),
        "Timer": lambda: timer_state,
        "dist": SimpleNamespace(all_gather_object=all_gather_object, barrier=lambda **_kwargs: None),
        "mpu": SimpleNamespace(
            get_data_parallel_world_size=lambda **_kwargs: 1,
            get_data_parallel_group=lambda **_kwargs: None,
        ),
        "is_offline_mode": lambda args: args.offline,
        "is_sft_mode": lambda args: args.offline,
        "is_preference_mode": lambda _args: False,
        "algorithm_needs_critic": lambda _args: False,
        "is_megatron_main_rank": lambda: True,
        "sft_task_name": lambda *_args, **_kwargs: "train",
        "build_rollout_minibatch_plan": lambda *_args: SimpleNamespace(
            mini_local_sample_request=1, num_rollout_minis=2
        ),
        "build_data_fields": lambda *_args, **_kwargs: [],
        "Envs": SimpleNamespace(RELAX_EMPTY_POLL_SLEEP_MS=0),
        "logger": SimpleNamespace(info=lambda *_args: None),
        "log_rollout_data": lambda *_args: check_live("batch_metrics"),
        "train_dump_utils": SimpleNamespace(save_debug_train_data=lambda *_args, **_kwargs: check_live("dump")),
        "log_perf_data": log_perf,
        "tracking_utils": SimpleNamespace(flush_metrics=lambda *_args, **_kwargs: events.append("flush")),
        "compute_rollout_step": lambda _args, step: step,
        "get_gloo_group": lambda: None,
        "evaluation_step_for_rollout": lambda _args, step: step,
        "ROLLOUT_MINI_LOCAL_SAMPLE_COUNTS_KEY": "rollout_mini_local_sample_counts",
        "ROLLOUT_MINI_BATCH_METAS_KEY": "rollout_mini_batch_metas",
    }
    _load_definitions(
        "relax/backends/megatron/data.py",
        {"DataIterator", "concat_rollout_batches", "_same_scalar_value"},
        namespace,
    )
    _load_definitions("relax/utils/utils.py", {"_extract_image_grids", "_extract_audio_seqlens"}, namespace)
    actor_methods = {"train", "train_actor", "_backup_actor_weights_after_train"}
    _load_definitions("relax/backends/megatron/actor.py", actor_methods, namespace, class_name="MegatronTrainRayActor")
    actor_type = type("ActorHarness", (), {name: namespace[name] for name in actor_methods})

    def make_iterators(_args, _model, data, **_kwargs):
        # Keep both virtual stages, including independent log-prob iterators,
        # alive beyond backup: clearing only a local iterator variable is insufficient.
        iterators = [namespace["DataIterator"](data, micro_batch_size=1) for _ in range(2)]
        retained_iterators.extend(iterators)
        return iterators, [2]

    def train(_step, _model, _optimizer, _scheduler, iterators, _microbatches, *, profiler=None):
        check_live("compute")
        for iterator in iterators:
            for index in range(2):
                batch = iterator.get_next(["multimodal_train_inputs"])
                pixels = batch["multimodal_train_inputs"][0]["pixel_values"]
                assert torch.equal(pixels, torch.full((4, 8), float(index)))

    def fill_indexer(_iterators, _microbatches, data):
        assert len(data.pop("rollout_indexer_topk")) == 2

    namespace.update(get_data_iterator=make_iterators, train=train)
    monkeypatch.setattr(IndexerReplay, "set_stage", lambda _stage: None)
    monkeypatch.setattr(IndexerReplay, "finish_step", lambda **_kwargs: check_live("replay"))
    actor = actor_type()
    actor.args = SimpleNamespace(
        offload_rollout=False,
        offload_train=offload,
        debug_train_only=False,
        offline=offline,
        partial_rollout=False,
        use_rollout_indexer_replay=True,
        use_dynamic_batch_size=True,
        log_probs_max_tokens_per_gpu=16,
        max_tokens_per_gpu=8,
        use_rollout_routing_replay=False,
        compute_advantages_and_returns=False,
        use_routing_replay=False,
        ref_update_interval=1,
        num_rollout=1,
        save="unused-checkpoint",
        rotate_ckpt=False,
        save_interval=1,
        recompute_granularity="full",
    )
    actor.role = "actor"
    actor._per_step_rollout = not offline
    actor._sft_train_prefetch = actor._sft_window_prefetcher = None
    actor._is_standard_dpo = lambda: False
    actor._resident_sft_overlap_enabled = lambda: False
    actor.all_consumed = lambda *_args: False
    actor._take_sft_train_prefetch = lambda *_args: None
    actor._get_data_from_transfer_queue = lambda *_args, **_kwargs: (batches.popleft(), None)
    actor.wake_up = lambda: events.append("wake")
    actor.model = actor.optimizer = actor.opt_param_scheduler = actor.tokenizer = None
    actor.rollout_data_postprocess = actor.flops_counter = None
    actor.prof = SimpleNamespace(step=lambda **_kwargs: None)
    actor.fill_indexer_replay = fill_indexer
    actor.weights_backuper = SimpleNamespace(
        backup_tags={"actor", "ref"}, backup=lambda tag: check_released(f"backup_{tag}")
    )
    actor.save_model = lambda *_args, **_kwargs: check_released("checkpoint")
    actor.sleep = lambda: check_released("offload")
    actor.update_weights = lambda: events.append("update_weights")
    actor.rollout_manager = object()
    actor._run_step_evaluation = lambda *_args: events.append("eval")
    actor._wait_for_previous_eval = lambda: events.append("eval_wait")
    return actor, batches, tensor_refs, events, retained_iterators, namespace


def test_train_releases_minibatches_and_iterator_aliases_before_backup(monkeypatch):
    actor, batches, tensor_refs, events, iterators, _namespace = _make_run(monkeypatch)

    actor.train(0)

    assert not batches
    assert all(ref() is None for ref in tensor_refs)
    assert len(iterators) == 4
    assert all(iterator.rollout_data == {} for iterator in iterators)
    assert {"dump", "replay", "backup_actor", "backup_ref", "perf", "checkpoint", "offload"} <= set(events)


@pytest.mark.parametrize("offline,offload", [(False, False), (True, True)])
def test_train_actor_preserves_batch_without_per_step_offload(monkeypatch, offline, offload):
    actor, batches, tensor_refs, events, _iterators, namespace = _make_run(
        monkeypatch, offline=offline, offload=offload
    )
    batch = namespace["concat_rollout_batches"](list(batches))
    batches.clear()

    actor.train_actor(0, batch)

    assert all(ref() is not None for ref in tensor_refs)
    assert len(batch["multimodal_train_inputs"]) == 2
    assert "offload" not in events
