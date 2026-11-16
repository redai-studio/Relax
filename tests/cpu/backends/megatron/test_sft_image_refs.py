# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Rank-side image-ref rebuild wiring on the Megatron train actor.

Covers the ``--sft-image-preprocess-on-rank`` consumer contract without GPU,
Megatron process groups or a real processor: background prefetch stashing,
foreground re-attachment before get_batch, and failure surfacing.
"""

import gc
import multiprocessing
import time
import weakref
from argparse import Namespace
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist

from relax.engine.sft import image_prefetch as image_prefetch_module
from relax.engine.sft.image_prefetch import LazySFTImageInput, SFTImagePrefetch, resolve_sft_image_micro_batch
from relax.utils.data.image_refs import SFT_IMAGE_REFS_FIELD


def _descriptor(**overrides):
    descriptor = {
        "version": 1,
        "kind": "kimi_k3_sft_image_v1",
        "image_refs": ["/data/a.jpg"],
        "pixel_shape": [4, 3],
        "image_grid_thw": [[1, 2, 4]],
    }
    descriptor.update(overrides)
    return descriptor


class _Chunk:
    def __init__(self, pre_process: bool):
        self.pre_process = pre_process


def _actor(*, vision: bool) -> SFTImagePrefetch:
    helper = SFTImagePrefetch(Namespace(is_vl_model=True, hf_checkpoint="/tmp/model"), MagicMock())
    helper.model = [_Chunk(vision)]
    return helper


def _fake_builder(features):
    """Replace the pool-based rebuild with a canned in-process one."""

    def build(pool, descriptors):
        return [None if descriptor is None else features for descriptor in descriptors], {}

    return build


def _resolve_failure_worker(rank: int, rendezvous: str, output_dir: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    image_prefetch_module.get_gloo_group = lambda: dist.group.WORLD
    actor = _actor(vision=rank == 0)
    actor._ensure_pool = lambda: MagicMock()
    rollout_data = {
        "tokens": [[1, 2, 3, 4]],
        SFT_IMAGE_REFS_FIELD: [_descriptor()],
        "multimodal_train_inputs": [{"image_grid_thw": torch.tensor([[1, 2, 4]])}],
    }
    if rank == 0:
        # Rollout preparation still agrees on metadata errors before training;
        # pixel decoding itself is deliberately deferred to microbatches.
        rollout_data["multimodal_train_inputs"][0] = None
    else:
        # Model a DP replica that polls an empty partition before receiving its
        # cached slice and joining the actor-world failure agreement.
        assert actor.resolve_across_ranks(None, 5, model=actor.model) is False
        time.sleep(0.5)
    try:
        actor.resolve_across_ranks(rollout_data, 5, model=actor.model)
        outcome = "returned"
    except Exception as exc:  # noqa: BLE001
        outcome = f"{type(exc).__name__}: {exc}"
    finally:
        dist.destroy_process_group()
    Path(output_dir, f"rank-{rank}.txt").write_text(outcome)


def test_background_prefetch_keeps_only_metadata_for_vision_rank(monkeypatch):
    actor = _actor(vision=True)
    descriptor = _descriptor()
    monkeypatch.setattr(
        actor,
        "fetch_data",
        lambda **_kwargs: [[{SFT_IMAGE_REFS_FIELD: [descriptor]}, "meta"], 0.5],
    )
    builder = MagicMock(side_effect=AssertionError("whole-rollout prefetch must not decode image pixels"))
    monkeypatch.setattr("relax.utils.data.image_rebuild.build_batch_image_features", builder)
    actor._ensure_pool = lambda: MagicMock()

    prefetched = actor.fetch(
        7,
        model=actor.model,
        data_fields=["tokens", SFT_IMAGE_REFS_FIELD],
        batch_size=1,
        partition_id="sft_7",
        task_name="sft_train",
        sampling_config={"dp_rank": 0},
    )

    assert prefetched[1] == 0.5  # the raw TQ payload is returned untouched
    assert actor.features == {}
    builder.assert_not_called()


def test_background_prefetch_skips_rebuild_for_non_vision_rank(monkeypatch):
    actor = _actor(vision=False)
    monkeypatch.setattr(
        actor,
        "fetch_data",
        lambda **_kwargs: [[{SFT_IMAGE_REFS_FIELD: [_descriptor()]}, "meta"], 0.1],
    )

    def _fail(*_args):
        raise AssertionError("non-vision ranks must not rebuild pixels")

    monkeypatch.setattr("relax.utils.data.image_rebuild.build_batch_image_features", _fail)

    actor.fetch(
        7,
        model=actor.model,
        data_fields=["tokens"],
        batch_size=1,
        partition_id="sft_7",
        task_name="sft_train",
        sampling_config={"dp_rank": 0},
    )

    assert actor.features == {}


def test_resolve_attaches_lazy_inputs_and_drops_descriptor_field(monkeypatch):
    actor = _actor(vision=True)
    descriptor = _descriptor()
    features = {"pixel_values": torch.zeros(4, 3), "image_grid_thw": torch.tensor([[1, 2, 4]])}
    builder = MagicMock(side_effect=_fake_builder(features))
    monkeypatch.setattr("relax.utils.data.image_rebuild.build_batch_image_features", builder)
    actor._ensure_pool = lambda: MagicMock()
    rollout_data = {
        "tokens": [[1, 2, 3, 4]],
        SFT_IMAGE_REFS_FIELD: [descriptor],
        "multimodal_train_inputs": [{"image_grid_thw": torch.tensor([[1, 2, 4]])}],
    }

    actor.resolve(rollout_data, 5, model=actor.model)

    assert SFT_IMAGE_REFS_FIELD not in rollout_data
    mm = rollout_data["multimodal_train_inputs"][0]
    assert isinstance(mm, LazySFTImageInput)
    assert "pixel_values" not in mm
    assert mm["image_grid_thw"].tolist() == [[1, 2, 4]]
    assert actor.features == {}
    builder.assert_not_called()
    microbatch = {"multimodal_train_inputs": [mm]}
    resolve_sft_image_micro_batch(microbatch)
    assert microbatch["multimodal_train_inputs"][0]["pixel_values"] is features["pixel_values"]
    assert "pixel_values" not in rollout_data["multimodal_train_inputs"][0]


def test_resolve_keeps_only_grid_metadata_on_non_vision_rank():
    actor = _actor(vision=False)
    rollout_data = {
        "tokens": [[1, 2, 3, 4]],
        SFT_IMAGE_REFS_FIELD: [_descriptor()],
        "multimodal_train_inputs": [{"image_grid_thw": torch.tensor([[1, 2, 4]])}],
    }

    actor.resolve(rollout_data, 5, model=actor.model)

    assert SFT_IMAGE_REFS_FIELD not in rollout_data
    mm = rollout_data["multimodal_train_inputs"][0]
    assert list(mm) == ["image_grid_thw"]
    assert mm["image_grid_thw"].tolist() == [[1, 2, 4]]


def test_microbatch_resolution_rejects_grid_mismatch(monkeypatch):
    actor = _actor(vision=True)
    descriptor = _descriptor()
    features = {"pixel_values": torch.zeros(4, 3), "image_grid_thw": torch.tensor([[1, 8, 8]])}
    monkeypatch.setattr("relax.utils.data.image_rebuild.build_batch_image_features", _fake_builder(features))
    actor._ensure_pool = lambda: MagicMock()
    rollout_data = {
        "tokens": [[1, 2, 3, 4]],
        SFT_IMAGE_REFS_FIELD: [descriptor],
        "multimodal_train_inputs": [{"image_grid_thw": torch.tensor([[1, 2, 4]])}],
    }

    actor.resolve(rollout_data, 5, model=actor.model)
    with pytest.raises(RuntimeError, match="rollout_id=5 sample 0.*does not match the"):
        resolve_sft_image_micro_batch(rollout_data)


def test_microbatch_resolution_rebuilds_without_rollout_stash(monkeypatch):
    actor = _actor(vision=True)
    descriptor = _descriptor()
    features = {"pixel_values": torch.zeros(4, 3), "image_grid_thw": torch.tensor([[1, 2, 4]])}
    monkeypatch.setattr("relax.utils.data.image_rebuild.build_batch_image_features", _fake_builder(features))
    actor._ensure_pool = lambda: MagicMock()
    rollout_data = {
        "tokens": [[1, 2, 3, 4]],
        SFT_IMAGE_REFS_FIELD: [descriptor],
        "multimodal_train_inputs": [{"image_grid_thw": torch.tensor([[1, 2, 4]])}],
    }

    actor.resolve(rollout_data, 5, model=actor.model)
    resolve_sft_image_micro_batch(rollout_data)

    assert rollout_data["multimodal_train_inputs"][0]["pixel_values"] is features["pixel_values"]


def test_resolve_without_descriptors_is_a_noop():
    actor = _actor(vision=True)
    rollout_data = {"tokens": [[1, 2]], "multimodal_train_inputs": [None]}

    actor.resolve(rollout_data, 5, model=actor.model)

    assert rollout_data == {"tokens": [[1, 2]], "multimodal_train_inputs": [None]}


def test_resolve_across_ranks_without_rollout_skips_agreement(monkeypatch):
    actor = _actor(vision=True)
    all_reduce = MagicMock(side_effect=AssertionError("an empty DP poll must not enter actor-world agreement"))
    monkeypatch.setattr(image_prefetch_module.dist, "all_reduce", all_reduce)

    assert actor.resolve_across_ranks(None, 5, model=actor.model) is False
    all_reduce.assert_not_called()


def test_resolve_across_ranks_success_proceeds_after_world_agreement(monkeypatch):
    actor = _actor(vision=False)
    all_reduce = MagicMock()
    monkeypatch.setattr(image_prefetch_module.dist, "all_reduce", all_reduce)
    monkeypatch.setattr(image_prefetch_module, "get_gloo_group", lambda: "actor-world")
    rollout_data = {
        "tokens": [[1, 2, 3, 4]],
        SFT_IMAGE_REFS_FIELD: [_descriptor()],
        "multimodal_train_inputs": [{"image_grid_thw": torch.tensor([[1, 2, 4]])}],
    }

    assert actor.resolve_across_ranks(rollout_data, 5, model=actor.model) is True
    all_reduce.assert_called_once()
    error_flag = all_reduce.call_args.args[0]
    assert error_flag.tolist() == [0]
    assert all_reduce.call_args.kwargs == {"op": dist.ReduceOp.MAX, "group": "actor-world"}


def test_resolve_across_ranks_propagates_vision_failure_to_delayed_peer(tmp_path):
    context = multiprocessing.get_context("spawn")
    rendezvous = str(tmp_path / "gloo-rendezvous")
    processes = [
        context.Process(target=_resolve_failure_worker, args=(rank, rendezvous, str(tmp_path))) for rank in range(2)
    ]

    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=60)
            assert not process.is_alive(), "rank hung during actor-world image resolution error agreement"
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)

    assert "image references without grid metadata" in (tmp_path / "rank-0.txt").read_text()
    assert (
        "RuntimeError: SFT rank-side image resolution failed on a peer rank" in (tmp_path / "rank-1.txt").read_text()
    )


def test_large_rollout_materializes_only_selected_microbatch_and_releases_pixels(monkeypatch):
    actor = _actor(vision=True)
    actor._ensure_pool = lambda: MagicMock()
    decoded = []
    weak_pixels = []

    def build(_pool, descriptors):
        assert len(descriptors) == 1
        decoded.append(descriptors[0]["image_refs"][0])
        pixels = torch.zeros(4, 3)
        weak_pixels.append(weakref.ref(pixels))
        return [{"pixel_values": pixels, "image_grid_thw": torch.tensor([[1, 2, 4]])}], {}

    monkeypatch.setattr("relax.utils.data.image_rebuild.build_batch_image_features", build)
    rollout = {
        SFT_IMAGE_REFS_FIELD: [_descriptor(image_refs=[f"/data/{i}.jpg"]) for i in range(1024)],
        "multimodal_train_inputs": [{"image_grid_thw": torch.tensor([[1, 2, 4]])} for _ in range(1024)],
    }
    actor.resolve(rollout, 0, model=actor.model)
    assert decoded == []
    for index in [0, 1, 0]:  # Iterator reset/replay reconstructs from metadata.
        microbatch = {"multimodal_train_inputs": [rollout["multimodal_train_inputs"][index]]}
        resolve_sft_image_micro_batch(microbatch)
        assert weak_pixels[-1]() is not None
        assert all("pixel_values" not in value for value in rollout["multimodal_train_inputs"])
        del microbatch
        gc.collect()
        assert all(reference() is None for reference in weak_pixels)
    assert decoded == ["/data/0.jpg", "/data/1.jpg", "/data/0.jpg"]
