# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""HC/PLE LoRA parity with real CUDA lookup, halo exchange and TP SUM.

Each rank first computes a complete TP1 reference, then TP2 partitions the
tokens and the PLE vocabulary. Each step starts from the same reference weights
and optimizer state, separating TP errors from accumulated BF16 optimizer
rounding. No communication or model kernels are replaced.
"""

import copy
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from tests.cpu.models.test_qwen38_lora import _adapters, _inject, _loss, _model, _runtime


def _forward(model: torch.nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    tokens = (torch.arange(32, device=hidden.device).remainder(31) + 1).unsqueeze(0)
    # The second document crosses the TP boundary, exercising the real PLE halo.
    model.ple.prepare(tokens, torch.tensor([0, 12, 32], device=hidden.device, dtype=torch.int32))
    hidden = hidden + model.ple(hidden)
    block_input, h_res, h_post, residual = model.hc(hidden)
    return model.hc.fused_h_res_h_post_bda(h_res, residual, h_post, (block_input, None), 0.0, True, False)


def _parameters(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters()}


def _check_frozen(model: torch.nn.Module, initial: dict[str, torch.Tensor]) -> None:
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            assert parameter.grad is None, name
            torch.testing.assert_close(parameter, initial[name], rtol=0, atol=0, msg=name)


def _check_error(actual: torch.Tensor, expected: torch.Tensor, name: str, tolerance: float = 0.04) -> float:
    actual, expected = actual.float(), expected.float()
    assert torch.isfinite(actual).all(), name
    difference = torch.linalg.vector_norm(actual - expected).item()
    reference = torch.linalg.vector_norm(expected).item()
    # A relative norm bound catches missing/SUM-vs-AVG gradients even when every
    # element is tiny; an elementwise BF16 absolute tolerance alone would not.
    assert difference <= tolerance * reference + 1e-7, (
        f"{name}: relative L2 error={difference / max(reference, 1e-12):.6g}, "
        f"absolute L2 error={difference:.6g}, reference L2={reference:.6g}"
    )
    return difference / max(reference, 1e-12)


def _reference(hidden: torch.Tensor, weights: torch.Tensor) -> tuple[dict, list[dict]]:
    model = _model(1, with_ple=True).to(device=hidden.device, dtype=hidden.dtype)
    model.config.params_dtype = hidden.dtype
    _inject(model)
    initial = _parameters(model)
    optimizer = torch.optim.Adam(_adapters(model).values(), lr=0.01)
    steps = []
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        inputs = hidden.clone().requires_grad_()
        output = _forward(model, inputs)
        _loss(output.float(), weights).backward()
        gradients = {}
        for name, parameter in _adapters(model).items():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            if step or "linear_out" in name:
                assert torch.count_nonzero(parameter.grad) > 0, name
            gradients[name] = parameter.grad.detach().clone()
        optimizer.step()
        _check_frozen(model, initial)
        steps.append(
            {
                "output": output.detach().clone(),
                "input_grad": inputs.grad.detach().clone(),
                "grads": gradients,
                "parameters": {name: value.detach().clone() for name, value in _adapters(model).items()},
                "optimizer": copy.deepcopy(optimizer.state_dict()),
            }
        )
    for name, parameter in _adapters(model).items():
        assert not torch.equal(parameter, initial[name]), f"{name}: no update after two steps"
    return initial, steps


def _worker(rank: int, dtype_name: str, rendezvous: str) -> None:
    from megatron.core import parallel_state
    from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig

    from relax.models.common.chunked_grad_coalesce import _allreduce_non_tensor_model_parallel_grads
    from relax.models.qwen3_8.lora import validate_qwen38_lora
    from relax.utils.distributed_utils import init_gloo_group
    from relax.utils.logging_utils import get_logger

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    dtype = getattr(torch, dtype_name)
    tolerance = 1e-4 if dtype == torch.float32 else 0.04
    logger = get_logger(__name__)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=120))
    try:
        generator = torch.Generator().manual_seed(514)
        hidden = torch.randn(32, 1, 16, generator=generator).to(device="cuda", dtype=dtype)
        weights = torch.arange(1, 33, device="cuda", dtype=torch.float32).view(32, 1, 1) / 32
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=1)
        initial_reference, reference_steps = _reference(hidden, weights)
        parallel_state.destroy_model_parallel()
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=2)

        model = _model(2, with_ple=True).to(device="cuda", dtype=dtype)
        model.config.params_dtype = dtype
        args = _inject(model)
        model.ddp_config = DistributedDataParallelConfig()
        table = model.ple.ple_embedding.ngram_embedding
        # Compare the same model, independently of RNG consumption during TP
        # sharding. The production validator must then repair a divergent A.
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                expected = initial_reference[name]
                if name == "ple.ple_embedding.ngram_embedding.weight":
                    expected = expected[table.vocab_start_index : table.vocab_end_index]
                parameter.copy_(expected)
                if parameter.requires_grad and "linear_in" in name:
                    parameter.add_(rank)
        init_gloo_group(distributed_timeout_minutes=2)
        validate_qwen38_lora([model], args)
        initial = _parameters(model)
        for name, parameter in initial.items():
            expected = initial_reference[name]
            if name == "ple.ple_embedding.ngram_embedding.weight":
                expected = expected[table.vocab_start_index : table.vocab_end_index]
            torch.testing.assert_close(parameter, expected, rtol=0, atol=0, msg=f"initial {name}")

        optimizer = torch.optim.Adam(_adapters(model).values(), lr=0.01)
        for step, reference in enumerate(reference_steps):
            start = initial if step == 0 else reference_steps[step - 1]["parameters"]
            if step:
                with torch.no_grad():
                    for name, parameter in _adapters(model).items():
                        parameter.copy_(start[name])
                optimizer.load_state_dict(copy.deepcopy(reference_steps[step - 1]["optimizer"]))
            errors = dict(output=0.0, input_grad=0.0, adapter_grad=0.0, adapter_update=0.0)
            optimizer.zero_grad(set_to_none=True)
            inputs = hidden.chunk(2)[rank].clone().requires_grad_()
            output = _forward(model, inputs)
            errors["output"] = max(
                errors["output"],
                _check_error(output, reference["output"].chunk(2)[rank], f"step {step} output", tolerance),
            )
            _loss(output.float(), weights.chunk(2)[rank]).backward()
            _allreduce_non_tensor_model_parallel_grads([model], model.config, dist.group.WORLD)
            errors["input_grad"] = max(
                errors["input_grad"],
                _check_error(
                    inputs.grad, reference["input_grad"].chunk(2)[rank], f"step {step} input gradient", tolerance
                ),
            )
            for name, parameter in _adapters(model).items():
                assert parameter.grad is not None, name
                if step or "linear_out" in name:
                    assert torch.count_nonzero(parameter.grad) > 0, name
                errors["adapter_grad"] = max(
                    errors["adapter_grad"],
                    _check_error(parameter.grad, reference["grads"][name], f"step {step} {name} gradient", tolerance),
                )
            optimizer.step()
            _check_frozen(model, initial)
            for name, parameter in _adapters(model).items():
                errors["adapter_update"] = max(
                    errors["adapter_update"],
                    _check_error(
                        parameter.float() - start[name].float(),
                        reference["parameters"][name].float() - start[name].float(),
                        f"step {step} {name} update",
                        tolerance,
                    ),
                )
                replicas = [torch.empty_like(parameter) for _ in range(2)]
                dist.all_gather(replicas, parameter, group=dist.group.WORLD)
                torch.testing.assert_close(replicas[0], replicas[1], rtol=0, atol=0, msg=name)
            logger.info(
                "HC/PLE %s TP2 LoRA parity rank=%d step=%d max_relative_errors=%s", dtype_name, rank, step, errors
            )
        for name, parameter in _adapters(model).items():
            assert not torch.equal(parameter, initial[name]), f"{name}: no update after two steps"
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="HC/PLE TP2 LoRA requires two CUDA GPUs and NCCL")
@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
def test_qwen38_lora_tp2_sp_parity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dtype_name: str) -> None:
    _runtime()
    monkeypatch.setenv("QWEN38_PLE_CPU_OFFLOAD", "0")
    monkeypatch.setenv("RELAX_QWEN38_PLE_STREAMED", "0")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    torch.multiprocessing.spawn(_worker, args=(dtype_name, f"file://{tmp_path / 'rendezvous'}"), nprocs=2, join=True)
