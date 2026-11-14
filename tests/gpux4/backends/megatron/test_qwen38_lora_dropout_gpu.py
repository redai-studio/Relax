# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Nonzero PLE LoRA dropout on real TP/CP replicas and native RNG checkpoints.

Four ranks exercise TP2/CP2, then TP1/CP2/DP2 with different token lengths. The
latter catches checkpoint layouts that accidentally restore DP0's RNG on every
replica. Model kernels and collectives are not replaced.
"""

import copy
import random
import time
from argparse import Namespace
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist

from tests.cpu.models.test_qwen38_lora import _adapters, _inject, _loss, _model, _runtime
from tests.gpux4.backends.megatron.test_qwen38_lora_gpu import _check_error, _check_frozen, _parameters


def _rng() -> tuple:
    from megatron.core.tensor_parallel.random import _get_all_rng_states

    return copy.deepcopy(_get_all_rng_states())


def _assert_rng(actual: tuple, expected: tuple) -> None:
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    assert actual[3] == expected[3]
    assert actual[2].keys() == expected[2].keys()
    for name in actual[2]:
        torch.testing.assert_close(actual[2][name], expected[2][name], rtol=0, atol=0, msg=name)


def _set_ple_rng(state: torch.Tensor) -> None:
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

    from relax.models.qwen3_8.lora import _PLE_LORA_RNG_NAME

    tracker = get_cuda_rng_tracker()
    states = tracker.get_states()
    states[_PLE_LORA_RNG_NAME] = state.clone()
    tracker.set_states(states)


def _data(length: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(514)
    hidden = torch.randn(length, 1, 16, generator=generator).to(device="cuda", dtype=dtype)
    weights = torch.arange(1, length + 1, device="cuda", dtype=torch.float32).view(length, 1, 1) / length
    tokens = (torch.arange(length, device="cuda").remainder(31) + 1).unsqueeze(0)
    boundaries = torch.tensor([0, 12 if length == 32 else 8, length], device="cuda", dtype=torch.int32)
    return hidden, weights, tokens, boundaries


def _split(tensor: torch.Tensor, boundaries: torch.Tensor, *, dim: int = 0, sp: bool = True) -> torch.Tensor:
    from megatron.core import parallel_state
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.ssm.mamba_context_parallel import split_tensor_cp

    packed = PackedSeqParams(cu_seqlens_q=boundaries, cu_seqlens_kv=boundaries, qkv_format="thd")
    result = split_tensor_cp(tensor, packed, dim=dim)
    if sp:
        result = result.chunk(parallel_state.get_tensor_model_parallel_world_size(), dim=dim)[
            parallel_state.get_tensor_model_parallel_rank()
        ]
    return result.contiguous()


def _build(tp: int, cp: int, dtype: torch.dtype, initial: dict | None = None) -> torch.nn.Module:
    from megatron.core import parallel_state
    from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    from relax.models.qwen3_8.lora import PLE_TARGETS, validate_qwen38_lora

    model = _model(tp, with_ple=True, cp_size=cp).to(device="cuda", dtype=dtype)
    model.config.params_dtype = dtype
    args = _inject(model, dropout=0.25, targets=list(PLE_TARGETS))
    args.data_parallel_random_init = True
    model.ddp_config = DistributedDataParallelConfig()
    table = model.ple.ple_embedding.ngram_embedding
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if initial is not None:
                value = initial[name]
                if name == "ple.ple_embedding.ngram_embedding.weight":
                    value = value[table.vocab_start_index : table.vocab_end_index]
                parameter.copy_(value)
            elif parameter.requires_grad and "linear_out" in name:
                # B=0 hides dropout in the initial forward and A gradient.
                values = torch.arange(parameter.numel(), device="cuda", dtype=torch.float32)
                parameter.copy_((0.03 * (values + 1).sin()).reshape(parameter.shape))
    model_parallel_cuda_manual_seed(8421 + 10 * parallel_state.get_data_parallel_rank())
    validate_qwen38_lora([model], args)
    return model


def _step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data: tuple,
    *,
    mode: str = "direct",
    reference: bool = False,
) -> dict:
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel import random as tp_random

    from relax.models.common.chunked_grad_coalesce import _allreduce_non_tensor_model_parallel_grads
    from relax.models.qwen3_8.lora import _PLE_LORA_RNG_NAME

    hidden, weights, tokens, boundaries = data
    model.ple.prepare(_split(tokens, boundaries, dim=1, sp=False), boundaries)
    inputs = _split(hidden, boundaries).clone().requires_grad_()
    optimizer.zero_grad(set_to_none=True)
    masks: dict[str, list[torch.Tensor]] = {name: [] for name in ("key_proj", "value_proj")}
    handles = [
        getattr(model.ple, name).adapter.dropout.register_forward_hook(
            lambda _module, _inputs, output, name=name: masks[name].append(output.detach().ne(0))
        )
        for name in masks
    ]
    try:
        # Match the outer SP RNG context. The CP1 reference explicitly uses the
        # full-stream RNG; production CP2 must select it inside PLE itself.
        name = _PLE_LORA_RNG_NAME if reference else tp_random._MODEL_PARALLEL_RNG_TRACKER_NAME
        with tp_random.get_cuda_rng_tracker().fork(name):
            if mode == "direct":
                output = model.ple(inputs)
            else:
                with pytest.MonkeyPatch.context() as patch:
                    patch.setenv("RELAX_QWEN38_QSA_RECOMPUTE_REPLAY", "1" if mode == "replay" else "0")
                    output = tp_random.checkpoint(model.ple, False, inputs)
        _loss(output.float(), _split(weights, boundaries)).backward()
    finally:
        for handle in handles:
            handle.remove()
    for records in masks.values():
        assert len(records) == (1 if mode == "direct" else 2)
        assert records[0].any() and (~records[0]).any(), "Dropout did not affect the projection"
        if mode != "direct":
            torch.testing.assert_close(records[0], records[1], rtol=0, atol=0)
    assert not torch.equal(masks["key_proj"][0], masks["value_proj"][0]), "Key/value reused a dropout mask"
    if not reference:
        tp_group = parallel_state.get_tensor_model_parallel_group()
        _allreduce_non_tensor_model_parallel_grads([model], model.config, tp_group)
        cp_group = parallel_state.get_context_parallel_group()
        dp_group = parallel_state.get_data_parallel_group()
        for parameter in _adapters(model).values():
            dist.all_reduce(parameter.grad, group=cp_group)
            dist.all_reduce(parameter.grad, group=dp_group)
            parameter.grad.div_(dist.get_world_size(group=dp_group))
    gradients = {}
    for name, parameter in _adapters(model).items():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad) > 0, name
        gradients[name] = parameter.grad.detach().clone()
    optimizer.step()
    return {
        "output": output.detach().clone(),
        "input_grad": inputs.grad.detach().clone(),
        "grads": gradients,
        "parameters": {name: parameter.detach().clone() for name, parameter in _adapters(model).items()},
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "rng": _rng(),
        "masks": {name: records[0] for name, records in masks.items()},
    }


def _assert_step(actual: dict, expected: dict) -> None:
    for key in ("output", "input_grad"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0, msg=key)
    for key in ("grads", "parameters", "masks"):
        assert actual[key].keys() == expected[key].keys()
        for name in actual[key]:
            torch.testing.assert_close(actual[key][name], expected[key][name], rtol=0, atol=0, msg=f"{key} {name}")
    assert actual["optimizer"]["param_groups"] == expected["optimizer"]["param_groups"]
    for parameter_id, states in actual["optimizer"]["state"].items():
        for name, value in states.items():
            torch.testing.assert_close(value, expected["optimizer"]["state"][parameter_id][name], rtol=0, atol=0)
    _assert_rng(actual["rng"], expected["rng"])


def _native_state(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> dict:
    from megatron.core import parallel_state
    from megatron.core.dist_checkpointing.mapping import ShardedObject
    from megatron.training.checkpointing import get_rng_state

    training_state = {
        "adapter": {name: parameter.detach().cpu() for name, parameter in _adapters(model).items()},
        "optimizer": optimizer.state_dict(),
    }
    return {
        "rng_state": get_rng_state(
            "torch_dist",
            parallel_state.get_tensor_model_parallel_group(),
            parallel_state.get_pipeline_model_parallel_group(),
            dp_cp_group=parallel_state.get_data_parallel_group(with_context_parallel=True),
            dp_group=parallel_state.get_data_parallel_group(),
        ),
        "training": ShardedObject(
            "training",
            training_state,
            (dist.get_world_size(group=dist.group.WORLD),),
            (dist.get_rank(group=dist.group.WORLD),),
        ),
    }


def _resume(model: torch.nn.Module, optimizer: torch.optim.Optimizer, data: tuple, directory: Path) -> None:
    from megatron.core import dist_checkpointing, parallel_state
    from megatron.core.tensor_parallel import random as tp_random

    from relax.models.qwen3_8.lora import _PLE_LORA_RNG_NAME

    directory.mkdir(parents=True, exist_ok=True)
    frozen = _parameters(model)
    dist_checkpointing.save(_native_state(model, optimizer), str(directory))
    expected = _step(model, optimizer, data, mode="replay")
    restored = _build(model.config.tensor_model_parallel_size, model.config.context_parallel_size, data[0].dtype)
    # A lightweight checkpoint restores the same frozen base separately.
    with torch.no_grad():
        for name, parameter in restored.named_parameters():
            if not parameter.requires_grad:
                parameter.copy_(frozen[name])
    restored_optimizer = torch.optim.Adam(_adapters(restored).values(), lr=0.001)
    loaded = dist_checkpointing.load(_native_state(restored, restored_optimizer), str(directory), strict="raise_all")
    with torch.no_grad():
        for name, parameter in _adapters(restored).items():
            parameter.copy_(loaded["training"]["adapter"][name])
    restored_optimizer.load_state_dict(loaded["training"]["optimizer"])
    rng = loaded["rng_state"][parallel_state.get_data_parallel_rank()]
    assert _PLE_LORA_RNG_NAME in rng["rng_tracker_states"]
    random.setstate(rng["random_rng_state"])
    np.random.set_state(rng["np_rng_state"])
    torch.set_rng_state(rng["torch_rng_state"])
    torch.cuda.set_rng_state(rng["cuda_rng_state"])
    tracker = tp_random.get_cuda_rng_tracker()
    tracker.set_states(
        {
            name: tp_random.convert_cuda_rng_state(
                state, to_graphable=tp_random.is_graph_safe_cuda_rng_tracker(tracker)
            )
            for name, state in rng["rng_tracker_states"].items()
        }
    )
    _assert_step(_step(restored, restored_optimizer, data, mode="replay"), expected)


def _worker(rank: int, dtype_name: str, rendezvous: str, directory: str) -> None:
    from megatron.core import parallel_state
    from megatron.core.ssm.mamba_context_parallel import split_tensor_cp
    from megatron.core.tensor_parallel import random as tp_random
    from megatron.training.global_vars import set_args

    from relax.models.qwen3_8.lora import _PLE_LORA_RNG_NAME, PLE_TARGETS, validate_qwen38_lora
    from relax.utils.distributed_utils import init_gloo_group
    from relax.utils.logging_utils import get_logger

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=4, timeout=timedelta(seconds=180))
    try:
        set_args(Namespace(data_parallel_random_init=True))
        init_gloo_group(distributed_timeout_minutes=3)
        dtype = getattr(torch, dtype_name)
        tolerance = 1e-4 if dtype == torch.float32 else 0.04
        data = _data(32, dtype)
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=1)
        reference = _build(1, 1, dtype)
        tp_random.get_cuda_rng_tracker().add(_PLE_LORA_RNG_NAME, 99017)
        initial = _parameters(reference)
        reference_optimizer = torch.optim.Adam(_adapters(reference).values(), lr=0.001)
        references = []
        for _ in range(2):
            start_rng = _rng()[2][_PLE_LORA_RNG_NAME]
            result = _step(reference, reference_optimizer, data, reference=True)
            result["start_rng"] = start_rng
            references.append(result)
        parallel_state.destroy_model_parallel()
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=2, context_parallel_size=2)
        model = _build(2, 2, dtype, initial)
        frozen = _parameters(model)
        tracker = tp_random.get_cuda_rng_tracker()
        assert _PLE_LORA_RNG_NAME in tracker.get_states(), "PLE RNG was not registered before checkpoint"
        replicas = [None] * 4
        dist.all_gather_object(replicas, tracker.get_states()[_PLE_LORA_RNG_NAME], group=dist.group.WORLD)
        for state in replicas[1:]:
            torch.testing.assert_close(state, replicas[0], rtol=0, atol=0)
        model.ple.prepare(split_tensor_cp(data[2], None, dim=1))
        with pytest.raises(ValueError, match="requires packed THD inputs"):
            model.ple(split_tensor_cp(data[0], None).chunk(2)[parallel_state.get_tensor_model_parallel_rank()])
        model.ple.prepare(_split(data[2], data[3], dim=1, sp=False), data[3])
        states = tracker.get_states()
        tracker.set_states({name: state for name, state in states.items() if name != _PLE_LORA_RNG_NAME})
        with pytest.raises(RuntimeError, match="Checkpoint lacks the PLE LoRA dropout RNG stream"):
            model.ple(_split(data[0], data[3]))
        tracker.set_states(states)
        optimizer = torch.optim.Adam(_adapters(model).values(), lr=0.001)
        for index, expected in enumerate(references):
            start = initial if index == 0 else references[index - 1]["parameters"]
            if index:
                with torch.no_grad():
                    for name, parameter in _adapters(model).items():
                        parameter.copy_(start[name])
                optimizer.load_state_dict(copy.deepcopy(references[index - 1]["optimizer"]))
            _set_ple_rng(expected["start_rng"])
            # Outer SP RNG consumption differs by TP rank. PLE must select its
            # own stream; CP copies retain the native checkpoint RNG contract.
            with tracker.fork():
                torch.rand(11 + parallel_state.get_tensor_model_parallel_rank(), device="cuda")
            start_rng, start_optimizer = _rng(), copy.deepcopy(optimizer.state_dict())
            actual = _step(model, optimizer, data)
            _check_error(actual["output"], _split(expected["output"], data[3]), "output", tolerance)
            _check_error(actual["input_grad"], _split(expected["input_grad"], data[3]), "input gradient", tolerance)
            for name in actual["grads"]:
                _check_error(actual["grads"][name], expected["grads"][name], f"{name} gradient", tolerance)
                _check_error(
                    actual["parameters"][name].float() - start[name].float(),
                    expected["parameters"][name].float() - start[name].float(),
                    f"{name} update",
                    tolerance,
                )
            for name in actual["masks"]:
                torch.testing.assert_close(actual["masks"][name], expected["masks"][name], rtol=0, atol=0)
            if index:
                assert not torch.equal(actual["masks"]["key_proj"], references[0]["masks"]["key_proj"])
            for mode in ("checkpoint", "replay"):
                with torch.no_grad():
                    for name, parameter in _adapters(model).items():
                        parameter.copy_(start[name])
                optimizer.load_state_dict(copy.deepcopy(start_optimizer))
                tp_random._set_all_rng_states(*copy.deepcopy(start_rng))
                _assert_step(_step(model, optimizer, data, mode=mode), actual)
            _check_frozen(model, frozen)
        _resume(model, optimizer, data, Path(directory) / "tp2-cp2")

        # The native RNG checkpoint stores a list indexed by logical DP rank.
        # Different sequence lengths consume different RNG offsets on each DP.
        parallel_state.destroy_model_parallel()
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=2)
        dp_rank = parallel_state.get_data_parallel_rank()
        data = _data(16 if dp_rank == 0 else 32, dtype)
        model = _build(1, 2, dtype)
        args = Namespace(
            lora_rank=2,
            lora_dropout=0.25,
            lora_target_modules=list(PLE_TARGETS),
            data_parallel_random_init=False,
        )
        with pytest.raises(ValueError, match="requires --data-parallel-random-init"):
            validate_qwen38_lora([model], args)
        args.data_parallel_random_init = True
        validate_qwen38_lora([model], args)
        optimizer = torch.optim.Adam(_adapters(model).values(), lr=0.001)
        for _ in range(2):
            _step(model, optimizer, data)
        progressed_rng = _rng()
        validate_qwen38_lora([model], args)
        _assert_rng(_rng(), progressed_rng)
        dp_states = [None] * 2
        dist.all_gather_object(
            dp_states,
            tp_random.get_cuda_rng_tracker().get_states()[_PLE_LORA_RNG_NAME],
            group=parallel_state.get_data_parallel_group(),
        )
        assert not torch.equal(dp_states[0], dp_states[1]), "Logical DP replicas reused the PLE random stream"
        _resume(model, optimizer, data, Path(directory) / "dp2-cp2")
        get_logger(__name__).info(
            "PLE LoRA dropout %s: TP2/CP2 parity, recompute and native DP2 RNG resume passed", dtype_name
        )
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="PLE dropout TP2/CP2 requires four CUDA GPUs and NCCL")
@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
def test_qwen38_ple_dropout_parity_recompute_and_native_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dtype_name: str
) -> None:
    _runtime()
    monkeypatch.setenv("QWEN38_PLE_CPU_OFFLOAD", "0")
    monkeypatch.setenv("RELAX_QWEN38_PLE_STREAMED", "0")
    monkeypatch.setenv("RELAX_QWEN38_PLE_FUSED_CONV", "0")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    processes = torch.multiprocessing.start_processes(
        _worker,
        args=(dtype_name, (tmp_path / "rendezvous").as_uri(), str(tmp_path)),
        nprocs=4,
        join=False,
        start_method="spawn",
    )
    try:
        deadline = time.monotonic() + 300
        while not processes.join(timeout=1):
            assert time.monotonic() < deadline, "PLE dropout worker timed out"
    finally:
        for process in processes.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
