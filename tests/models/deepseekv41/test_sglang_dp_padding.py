# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise the real SGLang V4.1 source entry without allocating model
weights."""

import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from relax.backends.sglang.deepseekv41.patch import apply_dp_prefill_padding_patch, apply_engram_idle_patch


def test_sglang_v41_hook_does_not_import_training_extensions():
    # Import order matters: the training and bundled inference FlashMLA
    # extensions export conflicting native symbols in the current image.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib, sys; "
            "importlib.import_module('relax.backends.sglang.deepseekv41.model'); "
            "assert 'relax.models' not in sys.modules; "
            "assert 'megatron.core' not in sys.modules; "
            "assert 'flash_mla' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("num_tokens", [0, 8])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sglang_v41_idle_engram_preserves_padding_and_history(num_tokens, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required for the GPU padded-idle regression")
    from sglang.srt.layers.engram import EngramHasher
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    apply_engram_idle_patch()
    history = torch.arange(20, dtype=torch.int32, device=device).reshape(5, 4)
    hasher = SimpleNamespace(
        primes=torch.empty(2, 4, 8, device=device),
        offsets=torch.empty(2, 32, device=device),
        history=history,
    )
    batch = SimpleNamespace(forward_mode=ForwardMode.IDLE)
    result = EngramHasher.forward(hasher, torch.zeros(num_tokens, dtype=torch.int64, device=device), batch)
    assert result.shape == (num_tokens, 2, 32)
    assert result.dtype == torch.int64
    assert result.device == history.device
    torch.testing.assert_close(result, torch.zeros_like(result))
    torch.testing.assert_close(history, torch.arange(20, dtype=torch.int32, device=device).reshape(5, 4))


@pytest.mark.parametrize("lengths,padded", [([1], 8), ([3], 8), ([2, 3], 8), ([1, 8], 16), ([8], 8)])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sglang_v41_dp_prefill_sources_ignore_padding(monkeypatch, lengths, padded, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required for the GPU repeat_interleave regression")
    from sglang.srt.layers.attention import deepseek_v4_backend as backend
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    apply_dp_prefill_padding_patch()
    monkeypatch.setattr(backend, "get_parallel", lambda: SimpleNamespace(attn_dp_size=4))
    monkeypatch.setattr(backend, "dsa_use_prefill_cp", lambda batch: False)
    req = torch.tensor([11, 17][: len(lengths)], device=device)
    batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        req_pool_indices=req,
        extend_seq_lens=torch.tensor(lengths, device=device),
        extend_seq_lens_cpu=lengths,
        extend_num_tokens=padded,
        encoder_swa_replay=False,
    )
    calls = []

    def compress(layer, x, req, pos, forward_batch):
        calls.append(("compress", x.shape[0], pos.shape[0], req))

    def index(layer, x, q_lora, req, pos, forward_batch):
        calls.append(("index", x.shape[0], q_lora.shape[0], req))

    instance = SimpleNamespace(
        low_ratio_prefill_graph=False,
        forward_metadata=SimpleNamespace(),
        _low_ratio_in_prefill_graph=lambda: False,
        _low_ratio_compress=compress,
        _low_ratio_index_topk=index,
    )
    backend.DeepseekV4AttnBackend.forward_low_ratio_sources(
        instance,
        layer=SimpleNamespace(compressor=object(), indexer=object()),
        x=torch.zeros(padded, 4, device=device),
        q_lora=torch.zeros(padded, 2, device=device),
        positions=torch.arange(padded, device=device),
        forward_batch=batch,
    )
    expected = torch.repeat_interleave(req, torch.tensor(lengths, device=device))
    assert [call[0] for call in calls] == ["compress", "index"]
    for _, x_rows, other_rows, actual in calls:
        assert x_rows == other_rows == sum(lengths)
        torch.testing.assert_close(actual, expected)
    assert batch.extend_num_tokens == padded
