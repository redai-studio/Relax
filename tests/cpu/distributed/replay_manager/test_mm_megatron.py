# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU contracts for frozen visual injection and native-path preservation."""

from types import SimpleNamespace

import pytest
import torch

from relax.distributed.replay_manager.backend.mm_megatron import (
    VisionReplayInput,
    bind_mm_replay_batch,
    close_mm_replay,
    install_vision_replay,
    native_vision_forward,
    replay_layout,
)


def test_visual_replay_layout_keeps_bf16_and_media_order_without_pixels():
    image = {"grid": [1, 4, 4], "modality": "image"}
    video = {"grid": [2, 2, 4], "modality": "video"}
    values = torch.arange(8 * 2560, dtype=torch.float32).reshape(8, 2560).bfloat16()
    payload = VisionReplayInput(values, (image, video))
    tokens = torch.tensor([[7, 31, 31, 31, 31, 8, 32, 32, 32, 32]])
    carrier, grid, mask = replay_layout(tokens, payload, 31, 32)
    assert carrier.shape == (32, 0) and carrier.numel() == 0
    assert grid.tolist() == [[1, 4, 4], [2, 2, 4]]
    assert mask.tolist() == [[False, True, True, True, True, False, True, True, True, True]]
    assert torch.equal(payload.embeddings.view(torch.int16), values.view(torch.int16))


def test_visual_replay_rejects_grid_shape_mismatch():
    payload = VisionReplayInput(torch.zeros(3, 2560, dtype=torch.bfloat16), ({"grid": [1, 4, 4]},))
    with pytest.raises(ValueError, match="shape/dtype"):
        replay_layout(torch.ones(1, 4, dtype=torch.long), payload, 1, 2)


def test_visual_replay_refuses_trainable_merger_before_installation():
    vision = torch.nn.Linear(4, 4)
    model = SimpleNamespace(vision_model=vision)
    with pytest.raises(ValueError, match="entire VE and merger"):
        install_vision_replay(model)


def test_visual_replay_batch_fails_closed_outside_reference_region():
    close_mm_replay()
    with pytest.raises(RuntimeError, match="no active reference"):
        bind_mm_replay_batch({}, None)


def test_mm_replay_reads_embeddings_only_for_the_input_virtual_pipeline_chunk():
    from unittest.mock import Mock

    from relax.distributed.replay_manager.backend import mm_megatron
    from relax.distributed.replay_manager.ref_data import ReplayRef

    ref = ReplayRef(2, "abcd", "abcd", "f" * 32, "mm_embedding", 4, 1, 2560, 4, "a" * 64, "bf16-raw-v1")
    media = dict(hash=ref.tokens, grid=[1, 4, 4], modality="image", ref=ref.to_dict(), lease="b" * 32)
    manifests = [{"mm_embedding": {"version": 1, "media": [media]}}]
    grids = torch.tensor([[1, 4, 4]])
    embeddings = torch.ones(4, 2560, dtype=torch.bfloat16)
    reader = SimpleNamespace(enabled=True, load=Mock(return_value=embeddings))
    token = mm_megatron._active_reader.set(reader)
    try:
        for input_chunk in (True, False):
            inputs = {"image_grid_thw": grids}
            batch = {"rollout_replay_refs": manifests, "multimodal_train_inputs": inputs}
            bind_mm_replay_batch(batch, SimpleNamespace(module=SimpleNamespace(pre_process=input_chunk)))
            assert inputs["image_grid_thw"] is grids
            if input_chunk:
                assert inputs["_relax_mm_embedding_replay"].embeddings is embeddings
            else:
                assert "_relax_mm_embedding_replay" not in inputs
        reader.load.assert_called_once_with([media])
    finally:
        mm_megatron._active_reader.reset(token)


class _FrozenVision(torch.nn.Module):
    def __init__(self, hidden_size=2560):
        super().__init__()
        self.hidden_size = hidden_size
        self.weight = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.calls = 0

    def forward(self, hidden_states, grid_thw):
        self.calls += 1
        rows = sum(t * h * w // 4 for t, h, w in grid_thw.tolist())
        return (2 * self.weight).to(hidden_states).expand(rows, self.hidden_size), []


def reorganize_inputs(input_ids, image_grid_thw=None, pixel_values=None):
    return pixel_values, image_grid_thw, input_ids == 31


class _VisualModel(torch.nn.Module):
    def __init__(self, hidden_size=2560):
        super().__init__()
        self.vision_model = _FrozenVision(hidden_size)
        self.gain = torch.nn.Parameter(torch.tensor(3.0))
        self.config = SimpleNamespace(hidden_size=hidden_size, context_parallel_size=1)
        self.vision_transformer_config = SimpleNamespace(deepstack_visual_indexes=[])
        self.image_token_id, self.video_token_id = 31, 32

    def forward(self, input_ids, image_grid_thw=None, pixel_values=None):
        pixels, grids, mask = reorganize_inputs(
            input_ids=input_ids, image_grid_thw=image_grid_thw, pixel_values=pixel_values
        )
        visual, _ = self.vision_model(hidden_states=pixels, grid_thw=grids)
        combined = torch.zeros((*input_ids.shape, self.config.hidden_size), dtype=visual.dtype)
        combined[mask] = visual
        return combined * self.gain


@pytest.mark.parametrize("hidden_size", [2048, 2560])
def test_frozen_vision_injection_skips_encoder_and_preserves_language_gradient(monkeypatch, hidden_size):
    import sys

    monkeypatch.setattr(sys.modules[__name__], "reorganize_inputs", reorganize_inputs)
    model = _VisualModel(hidden_size)
    tokens = torch.tensor([[7, 31, 31, 31, 31]])
    grids = torch.tensor([[1, 4, 4]])
    original = model(tokens, image_grid_thw=grids, pixel_values=torch.ones(16, 4))
    assert model.vision_model.calls == 1
    install_vision_replay(model)
    embedding = torch.full((4, hidden_size), 2, dtype=torch.bfloat16)
    payload = VisionReplayInput(embedding, ({"grid": [1, 4, 4], "modality": "image"},))
    actual = model(tokens, image_grid_thw=grids, _relax_mm_embedding_replay=payload)
    assert model.vision_model.calls == 1
    torch.testing.assert_close(actual, original.bfloat16().float(), rtol=0, atol=0, check_dtype=False)
    actual.sum().backward()
    assert model.gain.grad == 4 * hidden_size * 2
    assert model.vision_model.weight.grad is None
    with pytest.raises(RuntimeError, match="has no replay embeddings"):
        model(tokens, image_grid_thw=grids)
    assert model.vision_model.calls == 1


def test_reference_forward_uses_its_own_visual_weights(monkeypatch):
    import sys

    monkeypatch.setattr(sys.modules[__name__], "reorganize_inputs", reorganize_inputs)
    model = _VisualModel()
    tokens = torch.tensor([[7, 31, 31, 31, 31]])
    grids = torch.tensor([[1, 4, 4]])
    pixels = torch.ones(16, 4, dtype=torch.bfloat16)
    actor_embedding, _ = model.vision_model(pixels, grids)
    actor_payload = VisionReplayInput(actor_embedding, ({"grid": [1, 4, 4], "modality": "image"},))
    model.vision_model.weight.fill_(2)
    expected = model(tokens, image_grid_thw=grids, pixel_values=pixels)

    install_vision_replay(model)
    with native_vision_forward():
        actual = model(tokens, image_grid_thw=grids, pixel_values=pixels)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actor = model(tokens, image_grid_thw=grids, _relax_mm_embedding_replay=actor_payload)
    assert not torch.equal(actor, actual)
    with pytest.raises(RuntimeError, match="has no replay embeddings"):
        model(tokens, image_grid_thw=grids)


@pytest.mark.parametrize(
    "field, value, reason",
    [
        ("freeze_vision_model", False, "frozen vision"),
        ("freeze_vision_projection", False, "frozen vision"),
        ("context_parallel_size", 2, "CP1"),
    ],
)
def test_mm_validation_rejects_unsupported_training_and_capture(monkeypatch, field, value, reason):
    from relax.distributed.replay_manager.validation import validate_mm_embedding_replay_args

    vision = SimpleNamespace(out_hidden_size=2560, spatial_merge_size=2, deepstack_visual_indexes=[])
    config = SimpleNamespace(
        model_type="qwen4_exp", vision_config=vision, text_config=SimpleNamespace(hidden_size=2560)
    )
    monkeypatch.setattr("relax.utils.misc.get_hf_config", lambda _: config)
    args = SimpleNamespace(
        hf_checkpoint="test-model",
        train_backend="megatron",
        colocate=True,
        context_parallel_size=1,
        freeze_vision_model=True,
        freeze_vision_projection=True,
        multimodal_keys={"image": "images"},
    )
    validate_mm_embedding_replay_args(args, is_offline=False)
    setattr(args, field, value)
    with pytest.raises(ValueError, match=reason):
        validate_mm_embedding_replay_args(args, is_offline=False)


def test_mm_input_stage_metrics_initialize_service_once_and_report_each_step(monkeypatch):
    import sys
    from types import ModuleType

    from relax.distributed.replay_manager.backend import mm_megatron
    from relax.utils import utils
    from relax.utils.metrics import metrics_service_adapter
    from relax.utils.metrics.client import MetricsClient

    core = ModuleType("megatron.core")
    core.mpu = SimpleNamespace(get_data_parallel_rank=lambda **kwargs: 0)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setattr(
        utils.ray,
        "nodes",
        lambda: [{"Alive": True, "Resources": {"node:__internal_head__": 1}, "NodeManagerAddress": "192.0.2.8"}],
    )
    urls, batches, reports = [], [], []
    client = MetricsClient()
    monkeypatch.setattr(client, "health_check", lambda: {"status": "healthy"})

    def get_client(url):
        urls.append(url)
        return client

    def send_batch(step, metrics, tags=None):
        batches.append((step, metrics))
        return True

    def report(step):
        reports.append(step)
        return {"status": "success"}

    monkeypatch.setattr(metrics_service_adapter, "get_metrics_client", get_client)
    monkeypatch.setattr(metrics_service_adapter, "_global_adapter", None)
    monkeypatch.setattr(client, "_send_metrics_batch", send_batch)
    monkeypatch.setattr(client, "_send_report_step", report)
    args = SimpleNamespace(use_metrics_service=True, wandb_always_use_train_step=False, timeline_dump_dir=None)
    reader = SimpleNamespace(enabled=True, tp_rank=0, metrics={"read_bytes": 5120})
    token = mm_megatron._active_reader.set(reader)
    try:
        mm_megatron.log_mm_replay_metrics(args, 2)
        mm_megatron.log_mm_replay_metrics(args, 3)
        assert urls == ["http://192.0.2.8:8000/metrics"]
        assert batches == [(2, {"replay/mm_read_bytes": 5120}), (3, {"replay/mm_read_bytes": 5120})]
        assert reports == [2, 3]
        assert client.get_buffered_metrics_count() == 0
        args.use_metrics_service = False
        mm_megatron.log_mm_replay_metrics(args, 4)
        assert reports == [2, 3]
    finally:
        mm_megatron._active_reader.reset(token)
        adapter = metrics_service_adapter.get_metrics_service_adapter()
        if adapter is not None:
            adapter.finish_async_reporting()


def _mm_tp_worker(rank, rendezvous):
    from dataclasses import replace
    from datetime import timedelta
    from unittest.mock import patch

    import torch.distributed as dist

    from relax.distributed.replay_manager import mm_embedding
    from relax.distributed.replay_manager.backend import mm_megatron
    from relax.distributed.replay_manager.ref_data import ReplayRef

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=15))
    try:
        group = dist.group.WORLD
        ref = ReplayRef(2, "abcd", "abcd", "f" * 32, "mm_embedding", 4, 1, 2560, 4, "a" * 64, "bf16-raw-v1")
        media = dict(hash="a" * 64, grid=[1, 4, 4], modality="image", ref=ref.to_dict(), lease="b" * 32)
        second_ref = replace(ref, record="e" * 32, tokens="d" * 64)
        second_media = dict(media, hash=second_ref.tokens, ref=second_ref.to_dict())
        entries = [media, second_media, media]
        batch = {"rollout_replay_refs": [{"mm_embedding": {"version": 1, "media": entries}}]}
        # Replace only CUDA device selection and external TQ I/O. The real
        # reader constructor, cache, close and every collective execute here.
        cpu_torch = SimpleNamespace(
            Tensor=torch.Tensor,
            device=lambda *args: torch.device("cpu"),
            cuda=SimpleNamespace(current_device=lambda: 0),
            empty=torch.empty,
            cat=torch.cat,
            tensor=torch.tensor,
            uint8=torch.uint8,
            bfloat16=torch.bfloat16,
        )
        for failure in ("acquire", "read", None):
            acquired, released, reads = [], [], []
            fail_read = failure == "read"

            def acquire(refs, *, partition):
                assert rank == 0
                if failure == "acquire":
                    raise RuntimeError("injected acquire failure")
                acquired.extend(refs)
                return ["c" * 32] * len(refs)

            def read(_store, entry, **kwargs):
                assert rank == 0
                reads.append(entry["hash"])
                if fail_read and entry is second_media:
                    raise RuntimeError("injected read failure")
                return torch.full((4, 2560), 2 if entry is media else 3, dtype=torch.bfloat16)

            store = SimpleNamespace(acquire_many=acquire, release_many=lambda refs, leases: released.extend(leases))
            manager = SimpleNamespace(store=lambda: store, config=SimpleNamespace(materialization_bytes=1 << 20))
            model = _VisualModel()
            model.pg_collection = SimpleNamespace(tp=group)
            with patch.object(mm_megatron, "torch", cpu_torch), patch.object(mm_embedding, "read_mm_embedding", read):
                if failure == "acquire":
                    with pytest.raises(RuntimeError, match="injected acquire failure"):
                        mm_megatron.MMReplayReader(manager, batch, SimpleNamespace(partition="train_0"), model)
                    continue
                reader = mm_megatron.MMReplayReader(manager, batch, SimpleNamespace(partition="train_0"), model)
                try:
                    if failure == "read":
                        with pytest.raises(RuntimeError, match="injected read failure"):
                            reader.load(entries)
                        assert reader.cache_bytes == 0
                        fail_read = False
                        reads.clear()
                    hits_before = reader.metrics["cache_hits"]
                    expected = torch.tensor([2] * 4 + [3] * 4 + [2] * 4, dtype=torch.bfloat16)
                    for _ in range(2):
                        actual = reader.load(entries)
                        torch.testing.assert_close(actual, expected[:, None].expand(-1, 2560))
                    assert reads == ([media["hash"], second_media["hash"]] if rank == 0 else [])
                    assert reader.metrics["cache_hits"] - hits_before == 4
                finally:
                    reader.close()
                assert len(acquired) == len(released) == (2 if rank == 0 else 0)
    finally:
        dist.destroy_process_group()


def test_mm_tp_io_failures_reach_all_ranks_and_success_keeps_cache(tmp_path):
    import torch.multiprocessing as mp

    # Mocks cannot expose a peer waiting on an unmatched collective. One
    # two-process Gloo run covers both I/O failures and the successful path;
    # every collective has a 15-second timeout and no CUDA is required.
    mp.spawn(_mm_tp_worker, args=((tmp_path / "mm_tp_gloo").as_uri(),), nprocs=2, join=True)


@pytest.mark.parametrize("hybrid", [False, True])
def test_qwen35_media_validation_accepts_agentic_pp1_and_rejects_unwired_async(monkeypatch, hybrid):
    from relax.distributed.replay_manager.validation import validate_mm_embedding_replay_args

    config = SimpleNamespace(
        model_type="qwen3_5_moe",
        vision_config=SimpleNamespace(out_hidden_size=2048, spatial_merge_size=2, deepstack_visual_indexes=[]),
        text_config=SimpleNamespace(hidden_size=2048),
    )
    monkeypatch.setattr("relax.utils.misc.get_hf_config", lambda _: config)
    args = SimpleNamespace(
        hf_checkpoint="qwen35",
        train_backend="megatron",
        colocate=False,
        fully_async=not hybrid,
        hybrid=hybrid,
        use_agentic_rollout=True,
        rollout_function_path="relax.agentic.rollout.generate_rollout",
        context_parallel_size=1,
        pipeline_model_parallel_size=1,
        use_dynamic_batch_size=True,
        rollout_batch_size=16,
        n_samples_per_prompt=8,
        global_batch_size=128,
        freeze_vision_model=True,
        freeze_vision_projection=True,
        multimodal_keys={"image": "images"},
    )
    validate_mm_embedding_replay_args(args, is_offline=False)
    if not hybrid:
        args.global_batch_size = 256
        with pytest.raises(ValueError, match="inline actor log-probabilities"):
            validate_mm_embedding_replay_args(args, is_offline=False)
        args.global_batch_size = 128
    args.pipeline_model_parallel_size = 2
    with pytest.raises(ValueError, match="PP1"):
        validate_mm_embedding_replay_args(args, is_offline=False)


def _mm_control_bootstrap_worker(rank, rendezvous):
    from datetime import timedelta

    import torch.distributed as dist

    from relax.distributed.replay_manager.backend.mm_megatron import initialize_mm_replay_control_groups

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4, timeout=timedelta(seconds=15))
    world = dist.group.WORLD
    try:
        first = dist.new_group(ranks=[0, 1], backend="gloo")
        second = dist.new_group(ranks=[2, 3], backend="gloo")
        model = SimpleNamespace(pre_process=rank < 2, pg_collection=SimpleNamespace(tp=first if rank < 2 else second))
        initialize_mm_replay_control_groups([model], world)
        # DCS registers an extra group only on the weight-update source.
        if rank == 0:
            dist.new_group(ranks=[0], backend="gloo", use_local_synchronization=True)
        dist.barrier(group=world)
        if rank < 2:
            control = model._relax_mm_replay_control_group
            assert dist.get_process_group_ranks(control) == [0, 1]
            value = torch.tensor([71 if rank == 0 else 0], dtype=torch.int64)
            dist.broadcast(value, src=0, group=control)
            assert value.tolist() == [71]
        else:
            assert not hasattr(model, "_relax_mm_replay_control_group")
        dist.barrier(group=world)
    finally:
        dist.destroy_process_group(world)


def test_mm_control_bootstrap_survives_asymmetric_dcs_group_and_non_input_stages(tmp_path):
    import time

    import torch.multiprocessing as mp

    context = mp.spawn(_mm_control_bootstrap_worker, args=(f"file://{tmp_path / 'control'}",), nprocs=4, join=False)
    deadline = time.monotonic() + 45
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail("MM control bootstrap did not finish within 45 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
