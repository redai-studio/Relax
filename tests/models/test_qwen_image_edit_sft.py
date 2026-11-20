# Copyright (c) 2026 Relax Authors. All Rights Reserved.
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest
import torch
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformerBlock

from relax.models.qwen_image.attention import QwenImageFlashAttnProcessor
from tests.models.qwen_image_edit_fsdp import run_fsdp_processes


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires one CUDA device for real FSDP2 execution")
@pytest.mark.parametrize("mode", ["full", "lora"])
def test_qwen_single_gpu_fsdp_and_checkpoint(tmp_path: Path, mode: str) -> None:
    run_fsdp_processes(tmp_path, 1, mode)


@pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("flash_attn") is None,
    reason="Requires CUDA and FlashAttention 2",
)
def test_qwen_flash_attention_outputs_gradients_determinism(record_property) -> None:
    torch.manual_seed(157)
    reference = QwenImageTransformerBlock(128, 2, 64).attn.to(device="cuda", dtype=torch.bfloat16)
    reference.processor._attention_backend = "native"
    actual = copy.deepcopy(reference)
    actual.set_processor(QwenImageFlashAttnProcessor(deterministic=True))
    image = torch.randn(1, 16, 128, device="cuda", dtype=torch.bfloat16)
    text = torch.randn(1, 20, 128, device="cuda", dtype=torch.bfloat16)

    def run(module):
        x, y = image.detach().clone().requires_grad_(), text.detach().clone().requires_grad_()
        outputs = module(x, encoder_hidden_states=y)
        loss = sum(value.float().square().mean() for value in outputs)
        gradients = torch.autograd.grad(loss, (x, y))
        return [*outputs, *gradients]

    expected, outputs, repeated = run(reference), run(actual), run(actual)
    for index, (target, result, repeat) in enumerate(zip(expected, outputs, repeated, strict=True)):
        error = (result.float() - target.float()).norm() / target.float().norm()
        record_property(f"tensor_{index}_relative_l2", error.item())
        record_property(f"tensor_{index}_max_absolute_error", (result.float() - target.float()).abs().max().item())
        assert error <= 0.01
        assert torch.equal(result, repeat)
