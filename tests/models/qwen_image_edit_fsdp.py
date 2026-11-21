# Copyright (c) 2026 Relax Authors. All Rights Reserved.
from __future__ import annotations

import gc
import multiprocessing as mp
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.attention import SDPBackend, sdpa_kernel

from relax.backends.fsdp.checkpoint import load_checkpoint, save_checkpoint
from relax.backends.fsdp.runtime import fsdp2_wrap
from relax.utils.distributed_utils import init_gloo_group
from tests.cpu.models.qwen_image_edit_helpers import build_adapter, build_model, small_sample


def fsdp_worker(rank: int, world: int, directory: str, mode: str) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{directory}/rendezvous",
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=90),
    )
    init_gloo_group()
    try:
        with sdpa_kernel(SDPBackend.MATH):
            adapter, model = build_adapter(), build_model(mode, torch.bfloat16)
            data = small_sample(adapter)
            model.transformer.enable_gradient_checkpointing()
            model.to(device)
            adapter.on_device_ready(model, device)
            model = fsdp2_wrap(model, master_dtype="fp32")
            counters = {}
            handles = []
            watched = (
                "text_encoder.model.language_model.layers.0",
                "text_encoder.model.visual.blocks.0",
                "vae.encoder",
                "text_encoder.lm_head",
                "vae.decoder",
            )
            for name in watched:
                counters[name] = 0

                def count(module, args, name=name):
                    counters[name] += 1

                handles.append(model.get_submodule(name).register_forward_pre_hook(count))
            parameters = [p for p in model.parameters() if p.requires_grad]
            assert all(p.dtype == torch.float32 for p in parameters)
            optimizer = torch.optim.AdamW(parameters, lr=1e-3)
            torch.manual_seed(1991 + rank)
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                counters.update(dict.fromkeys(counters, 0))
                loss = model(data)
                assert torch.isfinite(loss)
                loss.backward()
                optimizer.step()
                for name in watched[:2]:
                    assert counters[name] == 1
                assert counters["vae.encoder"] == 2
                assert counters["text_encoder.lm_head"] == counters["vae.decoder"] == 0
                assert data["target_pixels"].device.type == "cpu"
                assert data["text_inputs"]["pixel_values"].device.type == "cpu"
            if mode == "full":
                output = str(Path(directory) / "checkpoint")
                expected = [p.to_local().detach().clone() for p in parameters]
                expected_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state(device).clone()
                save_checkpoint(
                    output,
                    "qwen",
                    2,
                    model,
                    optimizer,
                    trainer_state={"completed_steps": 2},
                    weight_sync_manifest={},
                    adapter_contract={"trainable_mode": mode},
                    is_rank0=rank == 0,
                    require_rng_state=True,
                )
                with torch.no_grad():
                    for p in parameters:
                        p.add_(1)
                restored = load_checkpoint(output, "qwen", 2, model, optimizer)
                assert restored["completed_steps"] == 2
                for actual, original in zip(parameters, expected, strict=True):
                    assert torch.equal(actual.to_local(), original)
                assert torch.equal(torch.get_rng_state(), expected_rng[0])
                assert torch.equal(torch.cuda.get_rng_state(device), expected_rng[1])
            for handle in handles:
                handle.remove()
            del model, parameters, optimizer
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        dist.destroy_process_group()


def run_fsdp_processes(directory: Path, world: int, mode: str) -> None:
    context = mp.get_context("spawn")
    processes = [
        context.Process(target=fsdp_worker, args=(rank, world, str(directory), mode)) for rank in range(world)
    ]
    for process in processes:
        process.start()
    try:
        for process in processes:
            process.join(timeout=180)
            assert not process.is_alive(), "Qwen FSDP worker exceeded its timeout"
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
