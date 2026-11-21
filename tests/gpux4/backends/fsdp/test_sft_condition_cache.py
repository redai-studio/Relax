# Copyright (c) 2026 Relax Authors. All Rights Reserved.
import gc
import multiprocessing as mp
import os
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from relax.backends.fsdp.condition_cache import ConditionCacheBinding
from relax.backends.fsdp.runtime import fsdp2_wrap
from relax.backends.fsdp.sft import FSDPSFTTrainer
from relax.models.sensenova.adapter import SenseNovaSFTAdapter
from tests.cpu.models.test_sensenova_condition_cache import build_model, sample


def _worker(rank: int, directory: str) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{directory}/rendezvous",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    control = dist.new_group(backend="gloo", timeout=timedelta(seconds=90))
    device = torch.device("cuda", rank)
    root = Path(directory) / str(rank)
    root.mkdir()
    histories = []
    try:
        for enabled in (False, True):
            torch.manual_seed(191)
            model = build_model(root)
            model.pipe.to(device=device)
            model = fsdp2_wrap(model, param_dtype="fp32", reduce_dtype="fp32", master_dtype="fp32")
            cache = (
                ConditionCacheBinding(
                    model,
                    SenseNovaSFTAdapter(),
                    device,
                    {"condition_cache_max_entries": 2 - rank},
                )
                if enabled
                else None
            )
            trainer = FSDPSFTTrainer.__new__(FSDPSFTTrainer)
            trainer.world, trainer.control_group, trainer.condition_cache = 2, control, cache
            parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
            optimizer = torch.optim.SGD(parameters, lr=0.01)
            torch.manual_seed(192 + rank)
            history = []
            try:
                for step in range(5):
                    optimizer.zero_grad(set_to_none=True)
                    before = [parameter.to_local().detach().clone() for parameter in parameters]
                    losses = []
                    for micro in range(2):
                        index = 2 * step + micro
                        data = sample()
                        data["prompt"] += str(rank) + (" change" if index == 4 else "")
                        if index == 2 and enabled and rank == 0:
                            cache.cache.clear()
                        if index == 8:
                            if enabled and rank == 0:
                                cache.bump_encoder_version()
                            with torch.no_grad():
                                model.pipe.dit.language_model.model.embed_tokens.weight.add_(0.01)
                        if enabled:
                            trainer._prepare_conditions(data, step, micro, 2)
                        loss = model(data)
                        (loss / 2).backward()
                        losses.append(loss.detach().cpu())
                        del data, loss
                        if enabled:
                            cache.end_micro()
                    gradients = [parameter.grad.to_local().detach().cpu().clone() for parameter in parameters]
                    optimizer.step()
                    updates = [
                        (parameter.to_local().detach() - previous).cpu()
                        for parameter, previous in zip(parameters, before, strict=True)
                    ]
                    history.append(
                        (losses, gradients, updates, torch.get_rng_state(), torch.cuda.get_rng_state(device))
                    )
                if enabled:
                    assert cache.stats()["effective_reuses"] == 5
                    assert cache.generations["prefix_kv"] == 2
                    assert cache.stats()["peak_entries"] <= 2 - rank
                    assert cache.stats()["peak_charged_bytes"] <= cache.cache.max_bytes
                histories.append(history)
            finally:
                if cache is not None:
                    cache.close()
            del model, optimizer, parameters, trainer, cache
            gc.collect()
            torch.cuda.empty_cache()
        for online, cached in zip(*histories, strict=True):
            for reference, actual in zip(
                online[0] + online[1] + online[2], cached[0] + cached[1] + cached[2], strict=True
            ):
                assert torch.isfinite(actual).all()
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            assert torch.equal(online[3], cached[3])
            assert torch.equal(online[4], cached[4])
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two CUDA devices for real FSDP2 execution")
def test_sensenova_fsdp_cache_switching_gradients_and_rng(tmp_path) -> None:
    context = mp.get_context("spawn")
    processes = [context.Process(target=_worker, args=(rank, str(tmp_path))) for rank in range(2)]
    for process in processes:
        process.start()
    try:
        for process in processes:
            process.join(timeout=180)
            assert not process.is_alive()
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
