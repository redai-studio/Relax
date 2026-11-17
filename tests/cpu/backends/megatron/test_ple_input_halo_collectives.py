# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Two Gloo ranks exercise the shipped PLE and its differentiable input gather.

Selects MCore's tensor-based THD indexing fallback instead of the TE CUDA
kernel. Uses small frozen tables on CPU; no GPU kernels, pinned host lookup, or
optimizer. The CP1 native forward is an independent reference for local outputs
and gradients.
"""

import os
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.utils.checkpoint import checkpoint


class Lookup(nn.Module):
    def __init__(self):
        super().__init__()
        self.table = nn.Embedding(128, 8)

    def forward(self, ids):
        return self.table(ids.squeeze(-1)).transpose(0, 1)


def _model(lengths):
    from megatron.core.transformer.per_layer_embedding import PerLayerEmbedding

    model = PerLayerEmbedding.__new__(PerLayerEmbedding)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(sequence_parallel=False)
    model.n, model.hidden_size, model.norm_eps = 2, 4, 1e-6
    model.conv_kernel_size, model.conv_dilation, model.halo = 3, 2, 4
    model.ple_embedding = Lookup()
    model.key_proj = nn.Linear(8, 8, bias=False)
    model.value_proj = nn.Linear(8, 4, bias=False)
    model.conv1d = nn.Conv1d(8, 8, 3, groups=8, bias=False)
    for name in ("norm_key", "norm_query", "norm_conv"):
        norm = nn.Module()
        norm.weight = nn.Parameter(torch.randn(8) / 10)
        setattr(model, name, norm)
    model.tp_group = SimpleNamespace(size=lambda: 1, rank=lambda: 0)
    model._ngram_ids = (torch.arange(sum(lengths)) % 128).reshape(1, -1, 1)
    model._position_in_sequence = torch.cat([torch.arange(n) for n in lengths]).unsqueeze(0)
    model._cp_packed_seq_params = None
    model.requires_grad_(False)
    return model


def _worker(rank, rendezvous):
    from megatron.core import parallel_state
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.transformer.ple_input_halo_cp import _gather_input_halo, prepare_input_halo_cp

    from relax.backends.megatron.qsa_layout_metadata import bind_physical_layout

    torch.set_num_threads(1)
    for flag in ("RELAX_QWEN38_PLE_STREAMED", "RELAX_QWEN38_PLE_FUSED_CONV", "RELAX_QWEN38_PLE_CP_FUSED_CONV"):
        os.environ[flag] = "0"
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=45))
    try:
        # Short chunks, multiple documents, long chunks, and zero halo are distinct risks.
        for lengths, halo in [((4, 12), 4), ((32, 16), 4), ((128,), 4), ((16,), 0)]:
            owned, needed = [], []
            for peer in range(2):
                rows, required, base = [], set(), 0
                for length in lengths:
                    chunks = torch.arange(base, base + length).reshape(4, -1)
                    local = chunks[[peer, 3 - peer]].flatten().tolist()
                    rows.extend(local)
                    for row in local:
                        required.update(range(max(base, row - halo), row + 1))
                    base += length
                owned.append(rows)
                needed.append(sorted(required))
            source = (torch.arange(sum(lengths) * 8).reshape(-1, 1, 8) % 19 - 9).double()
            x = source[owned[rank]].clone().requires_grad_()
            with patch("torch.cuda.current_device", return_value="cpu"):
                y = _gather_input_halo(x, lengths, needed[rank], halo, dist.group.WORLD)
                (y.square().sum() * (rank + 1)).backward()
            counts = torch.zeros(sum(lengths), dtype=torch.float64)
            for peer in range(2):
                counts[needed[peer]] += peer + 1
            assert torch.equal(y, source[needed[rank]])
            assert torch.equal(x.grad, (2 * source * counts[:, None, None])[owned[rank]])
            if halo == 0:
                continue
            torch.manual_seed(43)
            model = _model(lengths)
            hidden = torch.randn(sum(lengths), 1, 8)
            derivative = torch.randn_like(hidden)
            reference_input = hidden.clone().requires_grad_()
            os.environ["QWEN38_PLE_INPUT_HALO_CP"] = "0"
            with patch.object(parallel_state, "get_context_parallel_world_size", return_value=1):
                reference = model(reference_input)
                (reference * derivative).sum().backward()
            cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
            packed = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu)
            bind_physical_layout(packed, tuple(cu.tolist()))
            with (
                patch.object(parallel_state, "get_context_parallel_world_size", return_value=2),
                patch.object(parallel_state, "get_context_parallel_rank", return_value=rank),
                patch.object(parallel_state, "get_context_parallel_group", return_value=dist.group.WORLD),
                patch("torch.cuda.current_device", return_value="cpu"),
                patch("megatron.core.ssm.mamba_context_parallel._THD_TE_DEFAULT_SHARED_BYTES", 0),
            ):
                model._cp_packed_seq_params = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu)
                prepare_input_halo_cp(model, packed)
                os.environ["QWEN38_PLE_INPUT_HALO_CP"] = "1"
                for mode in ("full", "halo", "adaptive"):
                    os.environ["QWEN38_PLE_INPUT_HALO_TRANSPORT"] = mode
                    for recompute in (False, True):
                        local = hidden[owned[rank]].clone().requires_grad_()
                        output = checkpoint(model, local, use_reentrant=True) if recompute else model(local)
                        (output * derivative[owned[rank]]).sum().backward()
                        torch.testing.assert_close(output, reference[owned[rank]], rtol=1e-5, atol=1e-6)
                        torch.testing.assert_close(local.grad, reference_input.grad[owned[rank]], rtol=1e-5, atol=1e-6)
    finally:
        dist.destroy_process_group()


def test_two_rank_ple_outputs_gradients_and_checkpoint(tmp_path):
    pytest.importorskip("megatron.core.transformer.ple_input_halo_cp", exc_type=ImportError)
    if not dist.is_gloo_available():
        pytest.skip("Requires PyTorch Gloo")
    context = mp.spawn(_worker, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2, join=False)
    try:
        import time

        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if context.join(timeout=1):
                return
        pytest.fail("PLE Gloo test exceeded 120 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
