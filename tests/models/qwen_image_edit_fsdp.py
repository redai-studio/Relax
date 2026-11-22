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
from relax.backends.fsdp.condition_cache import ConditionCacheBinding
from relax.backends.fsdp.runtime import fsdp2_wrap
from relax.backends.fsdp.sft import FSDPSFTTrainer
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
    control = init_gloo_group()
    histories = []
    try:
        with sdpa_kernel(SDPBackend.MATH):
            for enabled in (False, True):
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
                cache = (
                    ConditionCacheBinding(model, adapter, device, {"condition_cache_max_entries": 2 - rank})
                    if enabled
                    else None
                )
                trainer = FSDPSFTTrainer.__new__(FSDPSFTTrainer)
                trainer.world, trainer.control_group, trainer.condition_cache = world, control, cache
                parameters = [p for p in model.parameters() if p.requires_grad]
                assert all(p.dtype == torch.float32 for p in parameters)
                optimizer = torch.optim.AdamW(parameters, lr=1e-3, foreach=mode != "full")
                torch.manual_seed(1991 + rank)
                history = []
                for step in range(6):
                    if enabled and step == 2 and rank == 0:
                        cache.cache.clear()
                    if step == 4:
                        with torch.no_grad():
                            next(model.text_encoder.parameters()).add_(0.001)
                        if enabled and rank == 0:
                            cache.bump_encoder_version(version_tag="updated")
                    optimizer.zero_grad(set_to_none=True)
                    before = [p.to_local().detach().clone() for p in parameters]
                    counters.update(dict.fromkeys(counters, 0))
                    if enabled:
                        trainer._prepare_conditions(data, step, 0, 1)
                    reused = enabled and bool(cache.values)
                    loss = model(data)
                    loss.backward()
                    gradients = [
                        p.grad.to_local().detach().cpu().clone() if p.grad is not None else None for p in parameters
                    ]
                    optimizer.step()
                    history.append(
                        [
                            loss.detach().cpu(),
                            *gradients,
                            *((p.to_local().detach() - old).cpu() for p, old in zip(parameters, before, strict=True)),
                            torch.get_rng_state(),
                            torch.cuda.get_rng_state(device),
                        ]
                    )
                    for name in watched[:2]:
                        assert counters[name] == (0 if reused else 1)
                    assert counters["vae.encoder"] == (0 if reused else 2)
                    assert counters["text_encoder.lm_head"] == counters["vae.decoder"] == 0
                    assert data["target_pixels"].device.type == "cpu"
                    assert data["text_inputs"]["pixel_values"].device.type == "cpu"
                    if enabled:
                        cache.end_micro()
                if enabled:
                    stats = cache.stats()
                    assert stats["effective_reuses"] == 3
                    assert stats["peak_entries"] <= 2 - rank
                    assert stats["peak_charged_bytes"] <= cache.cache.max_bytes
                adapter_only = mode == "lora"
                output = str(Path(directory) / f"checkpoint-{enabled}")
                expected_step6 = [p.to_local().detach().clone() for p in parameters]
                expected_step6_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state(device).clone()
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
                    6,
                    model,
                    optimizer,
                    trainer_state={"completed_steps": 6},
                    weight_sync_manifest={},
                    adapter_contract={"trainable_mode": mode},
                    adapter_only=adapter_only,
                    is_rank0=rank == 0,
                    require_rng_state=True,
                )
                optimizer.zero_grad(set_to_none=True)
                if enabled:
                    trainer._prepare_conditions(data, 6, 0, 1)
                continuous_loss = model(data)
                assert torch.isfinite(continuous_loss)
                continuous_loss.backward()
                optimizer.step()
                if enabled:
                    cache.end_micro()
                expected_step7_loss = continuous_loss.detach().clone()
                expected_step7 = [p.to_local().detach().clone() for p in parameters]
                expected_step7_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state(device).clone()

                histories.append(history)
                for handle in handles:
                    handle.remove()
                if cache is not None:
                    cache.close()
                del model, parameters, optimizer, cache, trainer
                gc.collect()

                # 构建全新模型和未初始化的 AdamW 优化器，验证恢复后优化器状态与第 7 步更新一致。
                resumed_model = build_model(mode, torch.bfloat16)
                resumed_model.transformer.enable_gradient_checkpointing()
                resumed_model = fsdp2_wrap(resumed_model, master_dtype="fp32")
                adapter.on_device_ready(resumed_model, device)
                if mode == "lora":
                    # LoRA 仅保存适配器权重，需要重放第 4 步对冻结 text_encoder 权重的修改。
                    with torch.no_grad():
                        next(resumed_model.text_encoder.parameters()).add_(0.001)
                resumed_cache = (
                    ConditionCacheBinding(resumed_model, adapter, device, {"condition_cache_max_entries": 2 - rank})
                    if enabled
                    else None
                )
                resumed_trainer = FSDPSFTTrainer.__new__(FSDPSFTTrainer)
                resumed_trainer.world, resumed_trainer.control_group, resumed_trainer.condition_cache = (
                    world,
                    control,
                    resumed_cache,
                )
                if enabled:
                    # 加载 checkpoint 前先经根模块前向写入一条缓存，验证 DCP 恢复会真实清空已有缓存并推进失效计数。
                    resumed_trainer._prepare_conditions(data, 5, 0, 1)
                    with torch.no_grad():
                        resumed_model(data)
                    resumed_cache.end_micro()
                    assert resumed_cache.stats()["entries"] == 1
                    assert resumed_cache.stats()["version_invalidations"] == 0
                resumed_parameters = [p for p in resumed_model.parameters() if p.requires_grad]
                with torch.no_grad():
                    for p in resumed_parameters:
                        p.add_(1)
                resumed_optimizer = torch.optim.AdamW(resumed_parameters, lr=1e-3, foreach=mode != "full")
                restored = load_checkpoint(
                    output,
                    "qwen",
                    6,
                    resumed_model,
                    resumed_optimizer,
                    adapter_only=adapter_only,
                    require_rng_state=True,
                )
                assert restored["completed_steps"] == 6
                for actual, original in zip(resumed_parameters, expected_step6, strict=True):
                    assert torch.equal(actual.to_local(), original)
                assert torch.equal(torch.get_rng_state(), expected_step6_rng[0])
                assert torch.equal(torch.cuda.get_rng_state(device), expected_step6_rng[1])
                if enabled:
                    assert resumed_cache.stats()["entries"] == 0
                    assert resumed_cache.stats()["version_invalidations"] >= 1
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
                if enabled:
                    resumed_trainer._prepare_conditions(data, 6, 0, 1)
                resumed_loss = resumed_model(data)
                assert torch.equal(resumed_loss.detach(), expected_step7_loss)
                resumed_loss.backward()
                resumed_optimizer.step()
                if enabled:
                    resumed_cache.end_micro()
                    resumed_cache.close()
                for actual, expected in zip(resumed_parameters, expected_step7, strict=True):
                    assert torch.equal(actual.to_local(), expected)
                assert torch.equal(torch.get_rng_state(), expected_step7_rng[0])
                assert torch.equal(torch.cuda.get_rng_state(device), expected_step7_rng[1])
                del resumed_model, resumed_parameters, resumed_optimizer, resumed_cache, resumed_trainer
                gc.collect()
                torch.cuda.empty_cache()
            for online, cached in zip(*histories, strict=True):
                for expected, actual in zip(online, cached, strict=True):
                    if expected is None:
                        assert actual is None
                    else:
                        assert torch.isfinite(actual).all()
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
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
