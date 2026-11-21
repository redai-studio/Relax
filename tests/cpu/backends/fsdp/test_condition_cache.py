# Copyright (c) 2026 Relax Authors. All Rights Reserved.
import gc
import multiprocessing as mp
import weakref
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from relax.backends.fsdp.condition_cache import ConditionCacheBinding
from relax.backends.fsdp.sft import FSDPSFTTrainer
from relax.engine.sft.condition_cache import ConditionFieldSpec, ConditionRequest


class Adapter:
    def condition_field_specs(self, model: torch.nn.Module) -> tuple[ConditionFieldSpec, ...]:
        return (ConditionFieldSpec("features", ("output",), ("encoder",), ("input",), ("preprocess",), 1),)

    def bind_condition_cache(self, model: torch.nn.Module, cache: object) -> None:
        model._condition_cache = cache

    def prepare_condition_request(self, model: torch.nn.Module, data: dict) -> dict[str, ConditionRequest]:
        return {
            "features": ConditionRequest(
                {
                    "input": data["input"],
                    "weights": "local",
                    "text": "v1",
                    "precision": "float32",
                    "preprocess": 1,
                },
                data.get("eligible", True),
            )
        }


def binding() -> ConditionCacheBinding:
    model = torch.nn.Module()
    model.encoder = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout(0.5)).eval().requires_grad_(False)
    return ConditionCacheBinding(model, Adapter(), torch.device("cpu"), {})


def micro(cache: ConditionCacheBinding, value: str = "sample") -> bool:
    state = cache.prepare({"input": value})
    cache.decide([state])
    hit = "features" in cache.values
    if not hit:
        dtype = next(cache.model.encoder.parameters()).dtype
        cache.put("features", {"output": cache.model.encoder(torch.ones(4, dtype=dtype))})
    cache.end_micro()
    return hit


@pytest.mark.parametrize("change", ["slice", "parameter", "module", "load", "explicit", "dtype"])
def test_encoder_changes_invalidate_and_rebind(change: str) -> None:
    cache = binding()
    assert not micro(cache)
    assert micro(cache)
    generation = cache.generations["features"]
    model = cache.model
    old = model.encoder
    if change == "slice":
        with torch.no_grad():
            model.encoder[0].weight[2:3, 1:3].add_(1)
    elif change == "parameter":
        model.encoder[0].weight = torch.nn.Parameter(torch.zeros(4, 4), requires_grad=False)
    elif change == "module":
        model.encoder = torch.nn.Sequential(torch.nn.Linear(4, 4)).eval().requires_grad_(False)
    elif change == "load":
        model.encoder.load_state_dict(model.encoder.state_dict())
    elif change == "dtype":
        model.encoder.double()
    else:
        cache.bump_encoder_version(version_tag="v1")
        cache.bump_encoder_version(version_tag="v1")
    assert not micro(cache)
    assert cache.generations["features"] > generation
    assert micro(cache)
    if change == "module":
        assert not old._load_state_dict_post_hooks
    cache.close()
    assert model._condition_cache is None
    assert not model.encoder._load_state_dict_post_hooks


def test_recursive_training_and_request_eligibility() -> None:
    cache = binding()
    micro(cache)
    cache.model.encoder[1].train()
    assert not micro(cache)
    assert cache.cache.stats()["entries"] == 0
    cache.model.encoder.eval()
    assert not micro(cache)
    assert micro(cache)
    states = cache.prepare({"input": "sample", "eligible": False})
    cache.decide([states])
    assert not cache.values
    assert not cache.put("features", {"output": torch.ones(4, requires_grad=True)})
    cache.end_micro()
    assert not micro(cache)
    assert micro(cache)
    cache.close()


def test_consumers_hold_budget_until_backward_and_close_releases() -> None:
    cache = binding()
    micro(cache)
    model_reference = weakref.ref(cache.model)
    states = cache.prepare({"input": "sample"})
    cache.decide([states])
    value = cache.values["features"]["output"]
    weight = torch.nn.Parameter(torch.ones(4))
    loss = (value * weight).sum()
    charged = cache.stats()["charged_bytes"]
    with pytest.raises(RuntimeError, match="idle"):
        cache.prepare({"input": "next"})
    with pytest.raises(RuntimeError, match="idle"):
        cache.bump_encoder_version()
    loss.backward()
    assert torch.equal(weight.grad, value)
    assert cache.stats()["charged_bytes"] == charged
    del loss, value
    cache.end_micro()
    cache.close()
    gc.collect()
    assert model_reference() is None
    assert cache.stats()["charged_bytes"] == 0


def _distributed_worker(rank: int, rendezvous: str, queue: object) -> None:
    from relax.utils.distributed_utils import init_gloo_group

    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=20)
    )
    cache = binding()
    trainer = FSDPSFTTrainer.__new__(FSDPSFTTrainer)
    trainer.condition_cache = cache
    trainer.control_group = init_gloo_group()
    trainer.world = 2
    try:
        results = []
        for index in range(5):
            if index == 2 and rank == 0:
                cache.cache.clear()
            if index == 4 and rank == 1:
                cache.bump_encoder_version()
            trainer._prepare_conditions({"input": f"rank{rank}"}, index, 0, 1)
            results.append(bool(cache.values))
            if not cache.values:
                cache.put("features", {"output": torch.ones(4)})
            cache.end_micro()
        if rank == 0:
            cache.generations["features"] += 1
        with pytest.raises(ValueError, match="generations"):
            trainer._prepare_conditions({"input": f"rank{rank}"}, 5, 0, 1)
        cache.end_micro()
        with pytest.raises(ValueError, match="sequence"):
            trainer._prepare_conditions({"input": f"rank{rank}"}, 6, 0, rank + 1)
        cache.end_micro()
        with pytest.raises(ValueError, match="preparation failed"):
            trainer._prepare_conditions({} if rank == 0 else {"input": "sample"}, 7, 0, 1)
        queue.put((rank, results, cache.stats()["group_forced_online"]))
    finally:
        cache.close()
        dist.destroy_process_group()


def test_two_rank_hits_misses_and_version_agreement(tmp_path) -> None:
    context = mp.get_context("spawn")
    queue = context.Queue()
    processes = [
        context.Process(target=_distributed_worker, args=(rank, str(tmp_path / "rendezvous"), queue))
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    try:
        for process in processes:
            process.join(timeout=45)
            assert not process.is_alive()
            assert process.exitcode == 0
        results = sorted(queue.get(timeout=5) for _ in processes)
        assert all(result[1] == [False, True, False, True, False] for result in results)
        assert results[1][2] >= 1
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        queue.close()


@pytest.mark.parametrize("failure", ["prefetch", "tracker"])
def test_cleanup_failure_releases_cache(failure: str) -> None:
    cache = binding()
    micro(cache)
    trainer = FSDPSFTTrainer.__new__(FSDPSFTTrainer)
    trainer.rank = 0
    trainer.condition_cache = cache

    def fail() -> None:
        raise RuntimeError("cleanup failed")

    trainer.prefetch = SimpleNamespace(close=fail) if failure == "prefetch" else None
    trainer.tracker = SimpleNamespace(save_summary=fail) if failure == "tracker" else None
    with pytest.raises(RuntimeError, match="cleanup failed"):
        trainer.close()
    assert cache.model is None
    assert cache.stats()["charged_bytes"] == 0


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, RuntimeError])
def test_training_failure_releases_features_with_retained_exception(error_type) -> None:
    cache = binding()
    micro(cache)
    references = []
    trainer = FSDPSFTTrainer.__new__(FSDPSFTTrainer)
    trainer.rank, trainer.prefetch, trainer.tracker = 0, None, None
    trainer.condition_cache = cache

    def fail(step: int) -> None:
        states = cache.prepare({"input": "sample"})
        cache.decide([states])
        features = cache.values["features"]["output"]
        references.append(weakref.ref(features))
        raise error_type("training interrupted")

    trainer._train_step = fail
    with pytest.raises(error_type) as raised:
        trainer.train(0)
    gc.collect()
    assert raised.value.__traceback__ is not None
    assert references[0]() is None
    assert cache.stats()["charged_bytes"] == 0


def test_training_failure_keeps_frames_without_cache() -> None:
    trainer = FSDPSFTTrainer.__new__(FSDPSFTTrainer)
    trainer.rank, trainer.prefetch, trainer.tracker, trainer.condition_cache = 0, None, None, None

    def fail(step: int) -> None:
        sample = {"step": step}
        raise RuntimeError(f"training interrupted {sample}")

    trainer._train_step = fail
    with pytest.raises(RuntimeError) as raised:
        trainer.train(3)
    frames = []
    traceback = raised.value.__traceback__
    while traceback is not None:
        frames.append(traceback.tb_frame)
        traceback = traceback.tb_next
    assert [frame.f_locals["sample"] for frame in frames if frame.f_code is fail.__code__] == [{"step": 3}]


def test_dcp_restore_invalidates_and_failure_closes(tmp_path, monkeypatch) -> None:
    from relax.backends.fsdp import checkpoint

    cache = binding()
    model = cache.model
    dist.init_process_group(
        "gloo",
        init_method=f"file://{tmp_path / 'rendezvous'}",
        rank=0,
        world_size=1,
        timeout=timedelta(seconds=20),
    )
    try:
        checkpoint.save_checkpoint(
            str(tmp_path),
            "i2i_sft",
            1,
            model,
            None,
            trainer_state={"step": 1},
            weight_sync_manifest={},
            adapter_contract={},
        )
        micro(cache)
        generation = cache.generations["features"]
        state = checkpoint.load_checkpoint(str(tmp_path), "i2i_sft", 1, model, None)
        assert state == {"step": 1}
        assert cache.generations["features"] > generation
        assert not micro(cache)
        assert micro(cache)

        def fail(*args, **kwargs) -> None:
            raise RuntimeError("DCP read failed")

        monkeypatch.setattr(checkpoint.dcp, "load", fail)
        with pytest.raises(RuntimeError, match="DCP read failed"):
            checkpoint.load_checkpoint(str(tmp_path), "i2i_sft", 1, model, None)
        assert model._condition_cache is None
        assert not model.encoder._load_state_dict_post_hooks
        assert cache.stats()["charged_bytes"] == 0
    finally:
        cache.close()
        dist.destroy_process_group()


def test_managed_dependency_replacement_restores_tensor_checks(tmp_path) -> None:
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Replicate, distribute_tensor

    dist.init_process_group(
        "gloo",
        init_method=f"file://{tmp_path / 'rendezvous'}",
        rank=0,
        world_size=1,
        timeout=timedelta(seconds=20),
    )
    cache = binding()
    try:
        model = cache.model
        mesh = init_device_mesh("cpu", (1,))
        cache.bump_encoder_version()
        model.encoder[0].weight = torch.nn.Parameter(
            distribute_tensor(model.encoder[0].weight.detach(), mesh, [Replicate()]),
            requires_grad=False,
        )
        states = cache.prepare({"input": "sample"})
        cache.decide([states])
        cache.put("features", {"output": torch.ones(4)})
        cache.end_micro()
        generation = cache.generations["features"]
        with torch.no_grad():
            model.encoder[0].weight.add_(1)
        states = cache.prepare({"input": "sample"})
        assert cache.generations["features"] == generation
        cache.decide([states])
        assert cache.values
        cache.end_micro()
        cache.bump_encoder_version()
        model.encoder = torch.nn.Sequential(torch.nn.Linear(4, 4)).eval().requires_grad_(False)
        assert not micro(cache)
        assert micro(cache)
        with torch.no_grad():
            model.encoder[0].weight[1:3, 1:3].add_(1)
        assert not micro(cache)
    finally:
        cache.close()
        dist.destroy_process_group()
