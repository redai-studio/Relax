# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Real MoonViT/NCCL regression; opt in with KIMI_K3_TEST_CHECKPOINT."""

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _vision_tp_worker(rank: int, world_size: int, rendezvous: str, checkpoint_path: str) -> None:
    from transformers import AutoConfig

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=world_size)
    group = dist.new_group(ranks=list(range(world_size)))
    try:
        spec = importlib.util.spec_from_file_location(
            "_k3_vision_gpu_model", Path(__file__).parents[2] / "relax/models/kimi_k3/model.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        hf_config = AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=True)
        # Exercise real attention, spatial merge and projector without loading
        # the language model or allocating all 27 vision blocks for a unit test.
        hf_config.vision_config.vt_num_hidden_layers = 2
        config = SimpleNamespace(
            hf_model_path=checkpoint_path,
            trust_remote_code=True,
            vision_config=hf_config.vision_config,
            params_dtype=torch.bfloat16,
        )
        torch.manual_seed(42)
        model = module.KimiK3VLModel.__new__(module.KimiK3VLModel)
        torch.nn.Module.__init__(model)
        model.config = config
        model.tp_group = group
        model.vision_tower, model.mm_projector = module._build_vision_modules(config)
        model.cuda(rank).train()
        model.vision_tower.requires_grad_(False)
        for grids in ([[1, 4, 4], [1, 6, 4], [1, 2, 2]], [[1, 2, 2]] * 5):
            grid = torch.tensor(grids, dtype=torch.long)
            patch_count = sum(t * h * w for t, h, w in grids)
            pixels = torch.randn(patch_count, 3, 14, 14, device=rank, dtype=torch.bfloat16)
            model.vision_dp_when_tp = False
            reference = model._image_features(pixels, grid)
            reference.float().square().mean().backward()
            reference_grads = [p.grad.clone() for p in model.mm_projector.parameters()]
            model.zero_grad(set_to_none=True)
            model.vision_dp_when_tp = True
            # Cover the device grid that caused the first real training failure.
            output = model._image_features(pixels, grid.cuda(rank))
            torch.testing.assert_close(output, reference, atol=0.02, rtol=0.02)
            output.float().square().mean().backward()
            for parameter, expected in zip(model.mm_projector.parameters(), reference_grads, strict=True):
                torch.testing.assert_close(parameter.grad, expected, atol=2e-4, rtol=0.02)
            assert all(p.grad is None for p in model.vision_tower.parameters())
            model.zero_grad(set_to_none=True)
    finally:
        dist.destroy_process_group(group)
        dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() < 4 or not os.environ.get("KIMI_K3_TEST_CHECKPOINT"),
    reason="Requires four GPUs and KIMI_K3_TEST_CHECKPOINT with the K3 HF custom modules.",
)
def test_kimi_k3_real_vision_tp_matches_unsplit_features_and_projector_gradients(tmp_path: Path) -> None:
    mp.spawn(
        _vision_tp_worker,
        args=(4, f"file://{tmp_path / 'nccl_init'}", os.environ["KIMI_K3_TEST_CHECKPOINT"]),
        nprocs=4,
        join=True,
    )
