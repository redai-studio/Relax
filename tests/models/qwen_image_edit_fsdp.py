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
            model = fsdp2_wrap(model, master_dtype="fp32")
            adapter.on_device_ready(model, device)
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
            optimizer = torch.optim.AdamW(parameters, lr=1e-3, foreach=mode != "full")
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
            adapter_only = mode == "lora"
            output = str(Path(directory) / "checkpoint")
            expected_step2 = [p.to_local().detach().clone() for p in parameters]
            expected_step2_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state(device).clone()
            expected_moments = [
                (
                    optimizer.state[p]["exp_avg"].to_local().detach().clone(),
                    optimizer.state[p]["exp_avg_sq"].to_local().detach().clone(),
                )
                for p in parameters
                if p in optimizer.state and "exp_avg" in optimizer.state[p]
            ]
            assert expected_moments
            assert any(bool((m != 0).any()) for m, _ in expected_moments)
            save_checkpoint(
                output,
                "qwen",
                2,
                model,
                optimizer,
                trainer_state={"completed_steps": 2},
                weight_sync_manifest={},
                adapter_contract={"trainable_mode": mode},
                adapter_only=adapter_only,
                is_rank0=rank == 0,
                require_rng_state=True,
            )
            # 连续执行第 3 步，作为从第 2 步 checkpoint 恢复后首步更新的基准。
            optimizer.zero_grad(set_to_none=True)
            continuous_loss = model(data)
            assert torch.isfinite(continuous_loss)
            continuous_loss.backward()
            optimizer.step()
            expected_step3_loss = continuous_loss.detach().clone()
            expected_step3 = [p.to_local().detach().clone() for p in parameters]
            expected_step3_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state(device).clone()

            for handle in handles:
                handle.remove()
            del model, parameters, optimizer
            gc.collect()

            # 构建全新模型和未初始化的 AdamW 优化器，验证分片权重与优化器状态恢复后的续训一致性。
            resumed_model = build_model(mode, torch.bfloat16)
            resumed_model.transformer.enable_gradient_checkpointing()
            resumed_model = fsdp2_wrap(resumed_model, master_dtype="fp32")
            adapter.on_device_ready(resumed_model, device)
            resumed_parameters = [p for p in resumed_model.parameters() if p.requires_grad]
            with torch.no_grad():
                for p in resumed_parameters:
                    p.add_(1)
            resumed_optimizer = torch.optim.AdamW(resumed_parameters, lr=1e-3, foreach=mode != "full")
            restored = load_checkpoint(
                output,
                "qwen",
                2,
                resumed_model,
                resumed_optimizer,
                adapter_only=adapter_only,
                require_rng_state=True,
            )
            assert restored["completed_steps"] == 2
            for actual, original in zip(resumed_parameters, expected_step2, strict=True):
                assert torch.equal(actual.to_local(), original)
            assert torch.equal(torch.get_rng_state(), expected_step2_rng[0])
            assert torch.equal(torch.cuda.get_rng_state(device), expected_step2_rng[1])
            restored_moments = [
                (
                    resumed_optimizer.state[p]["exp_avg"].to_local(),
                    resumed_optimizer.state[p]["exp_avg_sq"].to_local(),
                )
                for p in resumed_parameters
                if p in resumed_optimizer.state and "exp_avg" in resumed_optimizer.state[p]
            ]
            assert len(restored_moments) == len(expected_moments)
            for (act_m, act_v), (exp_m, exp_v) in zip(restored_moments, expected_moments, strict=True):
                assert torch.equal(act_m, exp_m)
                assert torch.equal(act_v, exp_v)

            resumed_optimizer.zero_grad(set_to_none=True)
            resumed_loss = resumed_model(data)
            assert torch.equal(resumed_loss.detach(), expected_step3_loss)
            resumed_loss.backward()
            resumed_optimizer.step()
            for actual, expected in zip(resumed_parameters, expected_step3, strict=True):
                assert torch.equal(actual.to_local(), expected)
            assert torch.equal(torch.get_rng_state(), expected_step3_rng[0])
            assert torch.equal(torch.cuda.get_rng_state(device), expected_step3_rng[1])

            del resumed_model, resumed_parameters, resumed_optimizer
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
