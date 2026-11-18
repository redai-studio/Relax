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
        bind_mm_replay_batch({})


class _FrozenVision(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.calls = 0

    def forward(self, hidden_states, grid_thw):
        self.calls += 1
        rows = sum(t * h * w // 4 for t, h, w in grid_thw.tolist())
        return hidden_states.new_full((rows, 2560), 2), []


def reorganize_inputs(input_ids, image_grid_thw=None, pixel_values=None):
    return pixel_values, image_grid_thw, input_ids == 31


class _VisualModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model = _FrozenVision()
        self.gain = torch.nn.Parameter(torch.tensor(3.0))
        self.config = SimpleNamespace(hidden_size=2560, context_parallel_size=1)
        self.vision_transformer_config = SimpleNamespace(deepstack_visual_indexes=[])
        self.image_token_id, self.video_token_id = 31, 32

    def forward(self, input_ids, image_grid_thw=None, pixel_values=None):
        pixels, grids, mask = reorganize_inputs(
            input_ids=input_ids, image_grid_thw=image_grid_thw, pixel_values=pixel_values
        )
        visual, _ = self.vision_model(hidden_states=pixels, grid_thw=grids)
        combined = torch.zeros((*input_ids.shape, 2560), dtype=visual.dtype)
        combined[mask] = visual
        return combined * self.gain


def test_frozen_vision_injection_skips_encoder_and_preserves_language_gradient(monkeypatch):
    import sys

    monkeypatch.setattr(sys.modules[__name__], "reorganize_inputs", reorganize_inputs)
    model = _VisualModel()
    tokens = torch.tensor([[7, 31, 31, 31, 31]])
    grids = torch.tensor([[1, 4, 4]])
    original = model(tokens, image_grid_thw=grids, pixel_values=torch.ones(16, 4))
    assert model.vision_model.calls == 1
    install_vision_replay(model)
    embedding = torch.full((4, 2560), 2, dtype=torch.bfloat16)
    payload = VisionReplayInput(embedding, ({"grid": [1, 4, 4], "modality": "image"},))
    actual = model(tokens, image_grid_thw=grids, _relax_mm_embedding_replay=payload)
    assert model.vision_model.calls == 1
    torch.testing.assert_close(actual, original.bfloat16().float(), rtol=0, atol=0, check_dtype=False)
    actual.sum().backward()
    assert model.gain.grad == 4 * 2560 * 2
    assert model.vision_model.weight.grad is None
    with pytest.raises(RuntimeError, match="has no replay embeddings"):
        model(tokens, image_grid_thw=grids)
    assert model.vision_model.calls == 1


@pytest.mark.parametrize("field", ["freeze_vision_model", "freeze_vision_projection", "context_parallel_size"])
def test_mm_validation_rejects_trainable_vision_or_cp(monkeypatch, field):
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
    setattr(args, field, 2 if field == "context_parallel_size" else False)
    with pytest.raises(ValueError, match="use-rollout-mm-embedding-replay"):
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
