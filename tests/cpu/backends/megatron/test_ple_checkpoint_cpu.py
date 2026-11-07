# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Small, trainable production PLE round trips using CPU AdamW and DCP.

The actual PLE source and checkpoint serializer come from the shipped patches.
CPU FP32 master parameters replace CUDA optimizer construction and transfers;
this does not exercise HDO, NCCL, TP/SP communication, or full-size memory use.
"""

from __future__ import annotations

import ast
import copy
import importlib.util
import subprocess
import sys
import types
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


pytest.importorskip("megatron.core", exc_type=ImportError)

from megatron.core import dist_checkpointing, parallel_state
from megatron.core.dist_checkpointing.strategies.torch import (
    MCoreSavePlanner,
    TorchDistSaveShardedStrategy,
    _replace_state_dict_keys_with_sharded_keys,
    mcore_to_pyt_state_dict,
)
from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.optimizer.optimizer import MegatronOptimizer
from megatron.core.transformer.transformer_config import TransformerConfig

from tests.cpu.backends.megatron.test_dp_checkpoint_patch import patched_source as patched_source


ROOT = Path(__file__).resolve().parents[4]
MODEL_PATCH = ROOT / "docker/patch/megatron/20261016-qwen38-models.patch"


def _section(patch: Path, relative: str) -> str:
    return patch.read_text().split(f"diff --git a/{relative} b/{relative}\n", 1)[1].split("\ndiff --git ", 1)[0]


def _additions(relative: str) -> str:
    return (
        "\n".join(
            line[1:]
            for line in _section(MODEL_PATCH, relative).splitlines()
            if line.startswith("+") and not line.startswith("+++")
        )
        + "\n"
    )


def _load_file(name: str, path: str | Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ple_source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    relative = "megatron/core/transformer/per_layer_embedding.py"
    path = tmp_path_factory.mktemp("ple-checkpoint-source") / "per_layer_embedding.py"
    path.write_text(_additions(relative))
    # Include the same operator-dispatch additions as the image build.
    subprocess.run(
        ["patch", "--batch", "--forward", "--fuzz=0", str(path)],
        input=_section(ROOT / "docker/patch/megatron/20261020-model-operators.patch", relative),
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )
    return path


def _load_ple(source: str) -> type[torch.nn.Module]:
    # PyPI MCore lacks these Qwen38 additions. Load the real shipped helpers,
    # preserving the real Megatron embedding, module, and checkpoint classes.
    hyper = types.ModuleType("megatron.core.transformer.hyper_connection")
    hyper.__dict__.update(torch=torch, Tensor=torch.Tensor)
    tree = ast.parse(_additions("megatron/core/transformer/hyper_connection.py"))
    norm = next(node for node in tree.body if getattr(node, "name", None) == "gated_residual_group_rmsnorm")
    exec(compile(ast.Module(body=[norm], type_ignores=[]), str(MODEL_PATCH), "exec"), hyper.__dict__)
    sys.modules[hyper.__name__] = hyper

    from megatron.core.ssm import mamba_context_parallel

    helpers = _additions("megatron/core/ssm/mamba_context_parallel.py")
    helpers = helpers[helpers.index("def reconstruct_tensor_cp(") :]
    scope = dict(mamba_context_parallel.__dict__)
    exec(compile(helpers, str(MODEL_PATCH), "exec"), scope)
    for name in ("reconstruct_tensor_cp", "split_tensor_cp"):
        setattr(mamba_context_parallel, name, scope[name])
    return _load_file("cpu_checkpoint_ple", source).PerLayerEmbedding


class _CPUSaveStrategy(TorchDistSaveShardedStrategy):
    def save(self, sharded_state_dict: dict, checkpoint_dir: str | Path) -> None:
        shards, _, _ = _replace_state_dict_keys_with_sharded_keys(sharded_state_dict, True)
        torch.distributed.checkpoint.save(
            mcore_to_pyt_state_dict(shards, False),
            storage_writer=torch.distributed.checkpoint.FileSystemWriter(checkpoint_dir),
            planner=MCoreSavePlanner(flatten_state_dict=False, flatten_sharded_tensors=False),
        )


def _create(
    ple_class: type[torch.nn.Module], dtype: torch.dtype, seed: int
) -> tuple[torch.nn.Module, DistributedOptimizer]:
    torch.manual_seed(seed)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=1,
        use_cpu_initialization=True,
        params_dtype=dtype,
        bf16=dtype == torch.bfloat16,
    )
    for name, value in {
        "num_residual_streams": 4,
        "ple_layer_ids": [1],
        "ple_embed_dim": 16,
        "ple_ngram_size": 3,
        "ple_heads_per_ngram": 2,
        "ple_ngram_vocab_size_base": 17,
        "ple_ngram_vocab_divisible_by": 8,
        "ple_unigram_vocab_size": 32,
        "ple_eos_token_id": 0,
        "ple_seed": 11,
        "ple_conv_kernel_size": 3,
    }.items():
        setattr(config, name, value)
    model = ple_class(config, layer_number=1)
    parameters = list(model.parameters())
    assert len(parameters) == 7 and all(p.device.type == "cpu" and p.requires_grad for p in parameters)
    masters = [torch.nn.Parameter(p.detach().float().flatten().clone()) for p in parameters]
    optimizer = torch.optim.AdamW(
        [{"params": masters, "wd_mult": 1.0, "lr_mult": 1.0, "is_expert_parallel": False, "is_decoupled_lr": False}],
        lr=0.02,
        betas=(0.8, 0.95),
        weight_decay=0.01,
        foreach=False,
    )
    # Initialize Adam state on CPU without changing the initial master weights.
    optimizer.param_groups[0]["lr"] = 0.0
    for master in masters:
        master.grad = torch.zeros_like(master)
    optimizer.step()
    optimizer.param_groups[0]["lr"] = 0.02
    for state in optimizer.state.values():
        state["step"].zero_()

    distributed = object.__new__(DistributedOptimizer)
    MegatronOptimizer.__init__(distributed, optimizer, OptimizerConfig(lr=0.02))
    distributed.grad_scaler = None
    distributed.ddp_config = SimpleNamespace(use_megatron_fsdp=False)
    distributed.data_parallel_group = torch.distributed.group.WORLD
    distributed.data_parallel_group_idx = 0
    distributed.distributed_optimizer_instance_id = 0
    distributed.model_param_group_index_map = {p: (0, i) for i, p in enumerate(parameters)}
    ranges, offset = {}, 0
    for parameter in parameters:
        # Real parameter tensors and Adam states, with alignment gaps between
        # parameters and a padded bucket tail, exercise the patched serializer.
        start = (offset + 63) // 64 * 64
        offset = start + parameter.numel()
        ranges[parameter] = {"gbuf_local": SimpleNamespace(start=start, end=offset)}
    padded = (offset + 127) // 128 * 128
    assert padded > offset
    buffer_dtype = (dtype, torch.float32)
    distributed.gbuf_ranges = [{buffer_dtype: [{"param_map": ranges}]}]
    distributed.per_bucket_numel = [[padded]]
    distributed.per_bucket_numel_unpadded = [[offset]]
    distributed.buffers = [
        SimpleNamespace(buckets=[SimpleNamespace(grad_data=torch.empty(padded), numel_unpadded=offset)])
    ]
    return model, distributed


def _batch(step: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(70 + step)
    tokens = (torch.arange(13).unsqueeze(0) + step + 1) % 17
    tokens[0, 2] = tokens[0, 8] = 0
    boundaries = torch.tensor([0, 7, 13], dtype=torch.int32)
    hidden = torch.randn(13, 1, 32, generator=generator).to(dtype)
    target = torch.randn(13, 1, 32, generator=generator)
    return tokens, boundaries, hidden, target


def _step(model: torch.nn.Module, distributed: DistributedOptimizer, number: int) -> torch.Tensor:
    optimizer = distributed.optimizer
    optimizer.zero_grad(set_to_none=True)
    model.zero_grad(set_to_none=True)
    tokens, boundaries, hidden, target = _batch(number, next(model.parameters()).dtype)
    model.prepare(tokens, boundaries)
    output = hidden + model(hidden)
    loss = (output.float() - target).square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    table = model.ple_embedding.ngram_embedding.weight
    assert table.grad is not None and torch.count_nonzero(table.grad) > 0
    for parameter, master in zip(model.parameters(), optimizer.param_groups[0]["params"], strict=True):
        assert parameter.grad is not None
        master.grad = parameter.grad.detach().float().flatten().clone()
    optimizer.step()
    with torch.no_grad():
        for parameter, master in zip(model.parameters(), optimizer.param_groups[0]["params"], strict=True):
            parameter.copy_(master.view_as(parameter))
    return output.detach().clone()


def _capture(model: torch.nn.Module, distributed: DistributedOptimizer) -> tuple[dict[str, torch.Tensor], list[dict]]:
    values = {f"model/{name}": tensor.detach().clone() for name, tensor in model.state_dict().items()}
    for index, master in enumerate(distributed.optimizer.param_groups[0]["params"]):
        values[f"master/{index}"] = master.detach().clone()
        for name, tensor in distributed.optimizer.state[master].items():
            values[f"optimizer/{index}/{name}"] = tensor.detach().clone()
    return values, copy.deepcopy(distributed.optimizer.state_dict()["param_groups"])


def _common_state(optimizer: torch.optim.Optimizer) -> dict:
    state = optimizer.state_dict()
    # Persist real per-parameter Adam counters outside the bucket tensors.
    # The CPU adapter uses AdamW's state_dict API rather than the GPU/HDO
    # distributed-optimizer common-state loader.
    return {
        "param_groups": state["param_groups"],
        "steps": {index: item["step"] for index, item in state["state"].items()},
    }


def _load_common_state(optimizer: torch.optim.Optimizer, common: dict) -> None:
    state = optimizer.state_dict()
    state["param_groups"] = common["param_groups"]
    for index, step in common["steps"].items():
        state["state"][index]["step"] = step
    optimizer.load_state_dict(state)


def _assert_equal(
    actual: tuple[dict[str, torch.Tensor], list[dict]], expected: tuple[dict[str, torch.Tensor], list[dict]]
) -> None:
    assert actual[0].keys() == expected[0].keys()
    for name in actual[0]:
        torch.testing.assert_close(actual[0][name], expected[0][name], rtol=0, atol=0, msg=name)
    assert actual[1] == expected[1]


def _roundtrip_worker(ple_source: str, checkpoint_source: str, directory: str, dtype_name: str) -> None:
    torch.set_num_threads(1)
    dtype = getattr(torch, dtype_name)
    directory = Path(directory)
    torch.distributed.init_process_group(
        "gloo", init_method=(directory / "rendezvous").as_uri(), rank=0, world_size=1, timeout=timedelta(seconds=30)
    )
    try:
        parallel_state.initialize_model_parallel()
        ple_class = _load_ple(ple_source)
        serializer = _load_file("cpu_ple_checkpoint_serializer", checkpoint_source).sharded_param_state_dp_reshardable
        model, distributed = _create(ple_class, dtype, seed=17)
        initial_table = model.ple_embedding.ngram_embedding.weight.detach().clone()
        for step in range(3):
            _step(model, distributed, step)
        assert not torch.equal(initial_table, model.ple_embedding.ngram_embedding.weight)
        expected_loaded = _capture(model, distributed)
        path = directory / "checkpoint"
        path.mkdir()

        def payload(model, distributed):
            return {
                "model": model.sharded_state_dict(),
                "common": _common_state(distributed.optimizer),
                "parameters": serializer(distributed, {}),
            }

        dist_checkpointing.save(
            payload(model, distributed), str(path), sharded_strategy=_CPUSaveStrategy("torch_dist", 1)
        )
        expected_output = _step(model, distributed, 3)
        expected_next = _capture(model, distributed)

        restored, restored_distributed = _create(ple_class, dtype, seed=193)
        with torch.no_grad():
            for tensor in restored.state_dict().values():
                tensor.add_(3)
            for master in restored_distributed.optimizer.param_groups[0]["params"]:
                master.fill_(-3)
                for tensor in restored_distributed.optimizer.state[master].values():
                    tensor.fill_(73)
        restored_distributed.optimizer.param_groups[0]["lr"] = 0.9
        # Step counters are common checkpoint metadata, not sharded parameter
        # data. Read them before constructing local parameter-load templates.
        common = dist_checkpointing.load_common_state_dict(str(path))["common"]
        _load_common_state(restored_distributed.optimizer, common)
        loaded = dist_checkpointing.load(payload(restored, restored_distributed), str(path))
        restored.load_state_dict(loaded["model"])
        with torch.no_grad():
            restored_distributed.load_parameter_state_from_dp_reshardable(loaded["parameters"])
        _assert_equal(_capture(restored, restored_distributed), expected_loaded)
        # prepare() must recompute nonpersistent n-gram/position caches.
        actual_output = _step(restored, restored_distributed, 3)
        torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=0)
        _assert_equal(_capture(restored, restored_distributed), expected_next)
    finally:
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
def test_trainable_ple_cpu_checkpoint_load_and_next_update(
    tmp_path: Path, ple_source: Path, patched_source: Path, monkeypatch: pytest.MonkeyPatch, dtype_name: str
) -> None:
    from multiprocessing import get_context

    # Set before the fresh worker imports torch, including on GPU-equipped CI.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("QWEN38_PLE_CPU_OFFLOAD", "0")
    monkeypatch.setenv("RELAX_QWEN38_PLE_STREAMED", "0")
    monkeypatch.setenv("RELAX_QWEN38_PLE_FUSED_CONV", "0")
    process = get_context("spawn").Process(
        target=_roundtrip_worker, args=(str(ple_source), str(patched_source), str(tmp_path), dtype_name)
    )
    process.start()
    process.join(timeout=90)
    assert not process.is_alive(), "CPU PLE checkpoint worker exceeded its timeout"
    assert process.exitcode == 0
