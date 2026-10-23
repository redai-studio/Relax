# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""GPU numerical regressions; distributed cases require two torchrun workers.

Run on the training image, with TF32 disabled and synthetic frozen FP8 tables.
The references below use independent equations, never replacement kernels.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("megatron.core")

from megatron.core import parallel_state  # noqa: E402
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed  # noqa: E402
from megatron.core.transformer.transformer_config import TransformerConfig  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from tokenizers.models import WordLevel  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402

from relax.models.deepseekv41.engram import DeepSeekV41Engram, engram_gate  # noqa: E402
from relax.models.deepseekv41.model import V41HyperConnection  # noqa: E402
from relax.models.deepseekv41.runtime import V41Runtime  # noqa: E402


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires the remote training GPU image")


def _close(actual: torch.Tensor, expected: torch.Tensor, name: str, *, atol: float, rtol: float) -> None:
    delta = (actual.float() - expected.float()).detach()
    rel_l2 = delta.norm() / expected.detach().float().norm().clamp_min(1e-12)
    print(
        f"METRIC {name} shape={tuple(actual.shape)} rel_l2={rel_l2.item():.8g} max_abs={delta.abs().max().item():.8g}"
    )
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.fixture(autouse=True)
def gpu_seed() -> None:
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    torch.manual_seed(41017)
    torch.cuda.manual_seed(41017)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _sinkhorn(logits: torch.Tensor, steps: int, epsilon: float) -> torch.Tensor:
    probabilities = torch.exp(logits - logits.max(dim=-1, keepdim=True).values)
    probabilities = probabilities / probabilities.sum(-1, keepdim=True) + epsilon
    for step in range(steps):
        if step:
            probabilities = probabilities / (probabilities.sum(-1, keepdim=True) + epsilon)
        probabilities = probabilities / (probabilities.sum(-2, keepdim=True) + epsilon)
    return probabilities


def _mhc_reference(
    module: V41HyperConnection, hidden: torch.Tensor, pre: torch.Tensor | None
) -> tuple[torch.Tensor, ...]:
    streams = hidden.float().unflatten(-1, (module.n, module.hidden_size))
    # The mapping is one FP32 branch: cast its combined gradient only once.
    mapping_input = hidden.float()
    rms = mapping_input.square().mean(-1, keepdim=True).sqrt() + module.norm_eps
    projection = torch.nn.functional.linear(mapping_input, module.mapping_proj.weight) / rms
    pre_logits = projection[..., : module.n] * module.alpha_pre + module.bias[: module.n]
    post_logits = projection[..., module.n : 2 * module.n] * module.alpha_post + module.bias[module.n : 2 * module.n]
    residual_logits = projection[..., 2 * module.n :] * module.alpha_res + module.bias[2 * module.n :]
    next_pre = pre_logits.sigmoid() + module.compute_h_eps
    post = 2 * post_logits.sigmoid()
    mixing = _sinkhorn(
        residual_logits.unflatten(-1, (module.n, module.n)), module.sinkhorn_iterations, module.sinkhorn_eps
    )
    aggregate = streams[..., 0, :] if pre is None else (streams * pre.unsqueeze(-1)).sum(-2)
    return aggregate.to(hidden.dtype), mixing, post, hidden, next_pre


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("scale", [1.0, 1e-5])
def test_mhc_gpu_previous_pre_sinkhorn_and_gradients(dtype: torch.dtype, scale: float) -> None:
    config = SimpleNamespace(
        num_residual_streams=4,
        hidden_size=32,
        mhc_sinkhorn_iterations=5,
        mhc_init_gating_factor=0.3,
        sequence_parallel=False,
        use_fused_mhc=False,
        layernorm_epsilon=1e-6,
        v41_hc_eps=1e-4,
    )
    runtime = V41Runtime()
    module = V41HyperConnection(config, 1, runtime).cuda()
    with torch.no_grad():
        module.bias.copy_(torch.linspace(-0.9, 0.8, module.bias.numel(), device="cuda"))
    hidden = (torch.randn(7, 2, 128, device="cuda", dtype=dtype) * scale).requires_grad_()
    previous_pre = torch.rand(7, 2, 4, device="cuda", requires_grad=True)
    runtime.pre = previous_pre
    actual = (*module(hidden), runtime.pre)
    expected = _mhc_reference(module, hidden, previous_pre)
    for index, (result, reference) in enumerate(zip(actual, expected)):
        _close(result, reference, f"mhc/{dtype}/{scale}/out{index}", atol=2e-6, rtol=2e-5)
    cotangents = [torch.randn_like(value) for value in actual]
    inputs = (hidden, previous_pre, *module.parameters())
    actual_grad = torch.autograd.grad(sum((x.float() * g.float()).sum() for x, g in zip(actual, cotangents)), inputs)
    expected_grad = torch.autograd.grad(
        sum((x.float() * g.float()).sum() for x, g in zip(expected, cotangents)), inputs
    )
    for index, (result, reference) in enumerate(zip(actual_grad, expected_grad)):
        tolerance = 0.02 if dtype == torch.bfloat16 and index == 0 else 5e-4
        _close(
            result,
            reference,
            f"mhc/{dtype}/{scale}/grad{index}",
            atol=tolerance,
            rtol=0.015 if dtype == torch.bfloat16 else 3e-4,
        )
    runtime.pre = None
    _close(module(hidden.detach())[0], hidden.detach()[..., :32], "mhc/initial_stream", atol=0, rtol=0)
    output = torch.randn(7, 2, 32, device="cuda", dtype=dtype, requires_grad=True)
    mixing, post = actual[1].detach().requires_grad_(), actual[2].detach().requires_grad_()
    mixed = module.fused_h_res_h_post_bda(mixing, hidden, post, (output, None), 0, True, False)
    streams = hidden.float().view(7, 2, 4, 32)
    reference = (
        (torch.matmul(mixing.transpose(-1, -2), streams) + post.unsqueeze(-1) * output.float().unsqueeze(-2))
        .flatten(-2)
        .to(dtype)
    )
    _close(mixed, reference, "mhc/residual_orientation", atol=0, rtol=0)
    mix_inputs = (hidden, mixing, post, output)
    cotangent = torch.randn_like(mixed)
    for result, reference in zip(
        torch.autograd.grad(mixed, mix_inputs, cotangent), torch.autograd.grad(reference, mix_inputs, cotangent)
    ):
        _close(result, reference, "mhc/residual_gradient", atol=2e-6, rtol=2e-5)


def _gate_reference(x: torch.Tensor, kv: torch.Tensor, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    streams, width = x.shape[-2:]
    key = kv[..., : streams * width].float().reshape(*x.shape)
    value = kv[..., streams * width :].float()
    query = x.float() / (x.float().square().mean(-1, keepdim=True) + 1e-6).sqrt()
    key = key / (key.square().mean(-1, keepdim=True) + 1e-6).sqrt()
    score = ((query * q) * (key * k)).sum(-1) / width**0.5
    transformed = torch.where(score < 0, -score.abs().clamp_min(1e-6).sqrt(), score.abs().clamp_min(1e-6).sqrt())
    return (x.float() + transformed.sigmoid().unsqueeze(-1) * value.unsqueeze(-2)).to(x.dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_engram_gpu_gate_formula_and_checkpoint(dtype: torch.dtype) -> None:
    x = torch.randn(17, 2, 4, 32, dtype=dtype, device="cuda", requires_grad=True)
    kv = torch.randn(17, 2, 160, dtype=dtype, device="cuda", requires_grad=True)
    q = torch.randn(4, 32, device="cuda", requires_grad=True)
    k = torch.randn(4, 32, device="cuda", requires_grad=True)
    inputs = (x, kv, q, k)
    reference = _gate_reference(*inputs)
    cotangent = torch.randn_like(reference)
    expected_grad = torch.autograd.grad(reference, inputs, cotangent)
    for chunk in (0, 5):
        result = engram_gate(*inputs, 1e-6, 1e-6, chunk=chunk)
        _close(result, reference, f"engram/gate/{dtype}/chunk{chunk}", atol=2e-6, rtol=2e-5)
        for index, (actual, expected) in enumerate(zip(torch.autograd.grad(result, inputs, cotangent), expected_grad)):
            _close(
                actual,
                expected,
                f"engram/gate/{dtype}/chunk{chunk}/grad{index}",
                atol=0.004 if dtype == torch.bfloat16 else 2e-5,
                rtol=0.02 if dtype == torch.bfloat16 else 1e-4,
            )


@pytest.fixture(scope="module")
def process_group(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        torch.distributed.init_process_group("nccl", timeout=timedelta(seconds=90))
    else:
        rendezvous = tmp_path_factory.mktemp("engram_dist") / "rendezvous"
        torch.distributed.init_process_group(
            "nccl", init_method=f"file://{rendezvous}", rank=0, world_size=1, timeout=timedelta(seconds=90)
        )
    try:
        yield
    finally:
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()


def _checkpoint(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    vocabulary = {"<unk>": 0, "<pad>": 1, "a": 2, "b": 3, "c": 4, "d": 5, "e": 6, "f": 7}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocabulary, unk_token="<unk>")), unk_token="<unk>", pad_token="<pad>"
    )
    tokenizer.save_pretrained(path)
    values = (torch.arange(36 * 32).reshape(36, 32) % 29 - 14).float().div(8).to(torch.float8_e4m3fn).view(torch.uint8)
    scales = (torch.arange(36).remainder(5) + 125).to(torch.uint8).view(36, 1)
    tensors = {"layers.0.engram.embed.weight": values, "layers.0.engram.embed.scale": scales}
    save_file(tensors, path / "table.safetensors")
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "table.safetensors" for name in tensors}})
    )
    return values, scales


def _decoded_table(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    bits = values.int()
    exponent, mantissa = (bits >> 3) & 15, bits & 7
    magnitude = torch.where(
        exponent == 0, mantissa.float() * 2**-9, (1 + mantissa.float() / 8) * torch.pow(2.0, exponent - 7)
    )
    scale = torch.pow(2.0, scales.int() - 127).repeat_interleave(32, dim=-1)
    return (torch.where(bits & 128 != 0, -magnitude, magnitude) * scale).bfloat16()


def _engram_config(path: Path, tp: int, cp: int) -> TransformerConfig:
    config = TransformerConfig(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=2,
        bf16=True,
        params_dtype=torch.bfloat16,
        tensor_model_parallel_size=tp,
        context_parallel_size=cp,
        sequence_parallel=tp > 1,
        gradient_accumulation_fusion=False,
        hidden_dropout=0,
        attention_dropout=0,
    )
    for name, value in dict(
        num_residual_streams=4,
        v41_hf_checkpoint=str(path),
        v41_engram_layer_ids=(0,),
        v41_engram_num_embeddings=(36,),
        v41_engram_max_ngram_size=3,
        v41_engram_vocab_size=5,
        v41_engram_compressed_vocab_size=8,
        v41_engram_n_heads=2,
        v41_engram_head_dim=32,
        v41_engram_pad_token_id=1,
    ).items():
        setattr(config, name, value)
    return config


@pytest.mark.parametrize("mode", ["local", "cp", "tp_sp"])
def test_engram_gpu_mmap_distributed_forward_and_gradients(process_group: None, tmp_path: Path, mode: str) -> None:
    world = torch.distributed.get_world_size()
    if (mode == "local" and world != 1) or (mode != "local" and world != 2):
        pytest.skip("local requires one worker; CP/TP-SP require two torchrun workers")
    tp, cp = (2 if mode == "tp_sp" else 1), (2 if mode == "cp" else 1)
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=tp, context_parallel_size=cp)
    try:
        model_parallel_cuda_manual_seed(41017)
        values, scales = _checkpoint(tmp_path)
        config = _engram_config(tmp_path, tp, cp)
        module = DeepSeekV41Engram(
            config, 0, parallel_state.get_tensor_model_parallel_group(), parallel_state.get_context_parallel_group()
        ).cuda()
        # Identical replicated parameters, deliberately nontrivial gates.
        with torch.no_grad():
            module.linear_wkv.weight.copy_(
                torch.linspace(-0.08, 0.11, module.linear_wkv.weight.numel(), device="cuda").reshape_as(
                    module.linear_wkv.weight
                )
            )
            module.q_weight.copy_(torch.linspace(-0.9, 1.2, 128, device="cuda").view(4, 32))
            module.k_weight.copy_(torch.linspace(0.2, 1.1, 128, device="cuda").view(4, 32))
        token_ids = torch.tensor([[2, 3, 6, 4, 5, 7, 3, 2], [7, 5, 4, 2, 3, 6, 7, 4]], device="cuda")
        full_hidden = torch.randn(8, 2, 128, device="cuda", dtype=torch.bfloat16)
        full_cotangent = torch.randn_like(full_hidden)
        rank = torch.distributed.get_rank()
        chunk = slice(rank * 4, (rank + 1) * 4) if world == 2 else slice(None)
        hidden = full_hidden[chunk].clone().requires_grad_() if world == 2 else full_hidden.clone().requires_grad_()
        input_ids = token_ids[:, chunk].contiguous() if mode == "cp" else token_ids
        actual = module(hidden, input_ids)
        expected_hidden = full_hidden.clone().requires_grad_()
        hashes = module.hash_ids(token_ids).transpose(0, 1)
        decoded = _decoded_table(values, scales).cuda()[hashes].flatten(-2)
        kv = torch.nn.functional.linear(decoded, module.linear_wkv.weight)
        expected = _gate_reference(expected_hidden.view(8, 2, 4, 32), kv, module.q_weight, module.k_weight).flatten(-2)
        _close(actual, expected[chunk], f"engram/{mode}/forward", atol=0.016, rtol=0.01)
        actual_grad = torch.autograd.grad(actual, (hidden, *module.parameters()), full_cotangent[chunk])
        expected_grad = torch.autograd.grad(expected, (expected_hidden, *module.parameters()), full_cotangent)
        _close(actual_grad[0], expected_grad[0][chunk], f"engram/{mode}/hidden_grad", atol=0.016, rtol=0.015)
        for index, (result, reference) in enumerate(zip(actual_grad[1:], expected_grad[1:])):
            if mode == "cp":
                torch.distributed.all_reduce(result, group=parallel_state.get_context_parallel_group())
            _close(
                result,
                reference,
                f"engram/{mode}/param_grad{index}",
                atol=0.15 if result.dtype == torch.bfloat16 else 0.02,
                rtol=0.02,
            )
        assert module.table.device.type == "cpu" and module.table_scale.device.type == "cpu"
        assert module.table.shape[0] == (18 if tp == 2 else 36)
        assert set(module.state_dict()) == {"q_weight", "k_weight", "linear_wkv.weight", "linear_wkv._extra_state"}
    finally:
        parallel_state.destroy_model_parallel()
