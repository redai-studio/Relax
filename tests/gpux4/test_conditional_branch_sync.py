# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Two-GPU regression for vision hooks across changing SFT batch modalities.

CPU autograd tests cannot exercise MCore's first-batch readiness bookkeeping or
collective completion. This uses one tiny mixed text/vision bucket, two
microbatches per step and a bounded two-process timeout.
"""

import importlib.util
import time
from argparse import Namespace
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist


def _worker(rank: int, root: str) -> None:
    from megatron.core import parallel_state
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.tensor_parallel.layers import set_defaults_if_not_set_tensor_model_parallel_attributes
    from megatron.core.transformer.module import MegatronModule
    from megatron.core.transformer.transformer_config import TransformerConfig

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=(Path(root) / "rendezvous").as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    parallel_state.initialize_model_parallel()
    group = parallel_state.get_data_parallel_group()
    config = TransformerConfig(num_layers=1, hidden_size=4, num_attention_heads=1, gradient_accumulation_fusion=False)
    source = Path(__file__).resolve().parents[2] / "relax/backends/megatron/conditional_branch_sync.py"
    spec = importlib.util.spec_from_file_location("conditional_sync_gpu_test", source)
    sync = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sync)

    class Vision(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = torch.nn.Module()
            self.patch_embed.proj = torch.nn.Linear(2, 2, bias=False, device="cuda")

        def forward(self, hidden_states, grid_thw=None):
            return self.patch_embed.proj(hidden_states)

    class Model(MegatronModule):
        def __init__(self):
            super().__init__(config)
            self.text = torch.nn.Linear(2, 2, bias=False, device="cuda")
            self.vision_model = Vision()
            self.vision_transformer_config = SimpleNamespace(
                spatial_merge_size=1,
                in_channels=2,
                temporal_patch_size=1,
                patch_size=1,
            )
            self.pg_collection = SimpleNamespace(dp=group)

        def forward(self, input_ids, pixel_values=None, image_grid_thw=None):
            output = self.text(input_ids)
            if pixel_values is not None:
                output = output + self.vision_model(pixel_values, image_grid_thw).sum()
            return output

    torch.manual_seed(1234)
    model, reference = Model(), Model()
    reference.load_state_dict(model.state_dict())
    # Only model recognition is isolated; dummy vision, autograd and DDP are real.
    sync._is_qwen3vl_with_conditional_vision = lambda module: isinstance(module, Model)
    sync.install_conditional_branch_sync(Namespace(overlap_grad_reduce=True), model)
    for parameter in model.parameters():
        set_defaults_if_not_set_tensor_model_parallel_attributes(parameter)
    ddp = DistributedDataParallel(
        config,
        DistributedDataParallelConfig(overlap_grad_reduce=True, grad_reduce_in_fp32=True),
        model,
        disable_bucketing=True,
    )
    ddp.zero_grad_buffer()
    for modality in ("text", "image", "text", "mixed"):
        for microbatch in range(2):
            values = torch.full((2, 2), float(rank + microbatch + 1), device="cuda")
            has_image = modality == "image" or (modality == "mixed" and rank == 0)
            kwargs = {"pixel_values": values, "image_grid_thw": torch.ones(1, 3, device="cuda")} if has_image else {}
            with ddp.no_sync() if microbatch == 0 else nullcontext():
                actual, expected = ddp(values, **kwargs), reference(values, **kwargs)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                actual.sum().backward()
                expected.sum().backward()
        ddp.finish_grad_sync()
        for (name, parameter), (_, baseline) in zip(model.named_parameters(), reference.named_parameters()):
            expected = baseline.grad.clone() if baseline.grad is not None else torch.zeros_like(baseline)
            dist.all_reduce(expected, group=group)
            expected /= dist.get_world_size(group)
            torch.testing.assert_close(parameter.main_grad, expected, rtol=0, atol=0, msg=f"{modality}: {name}")
        vision_grad = model.vision_model.patch_embed.proj.weight.main_grad
        assert bool(torch.any(vision_grad != 0)) == (modality != "text")
        ddp.zero_grad_buffer()  # First all-text batch used to fail here.
        reference.zero_grad(set_to_none=True)
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two GPUs for real MCore gradient synchronization")
def test_conditional_vision_sync_handles_text_image_transitions(tmp_path, monkeypatch):
    import torch.multiprocessing as mp

    monkeypatch.setenv("RELAX_DISABLE_VISION_BRANCH_SYNC", "0")
    context = mp.spawn(_worker, args=(str(tmp_path),), nprocs=2, join=False)
    deadline = time.monotonic() + 120
    try:
        while not context.join(timeout=5):
            if time.monotonic() > deadline:
                raise TimeoutError("Conditional vision DDP regression exceeded 120 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=10)
