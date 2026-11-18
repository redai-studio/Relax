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
        for lengths, halo in [((4, 12), 4), ((4, 32), 4), ((16,), 4), ((32, 16), 4), ((128,), 4), ((16,), 0)]:
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
                # Compare against the original implementation at the SAME CP size,
                # in addition to the independent complete-sequence CP1 reference.
                os.environ["QWEN38_PLE_INPUT_HALO_CP"] = "0"
                native_input = hidden[owned[rank]].clone().requires_grad_()
                native_output = model(native_input)
                (native_output * derivative[owned[rank]]).sum().backward()
                torch.testing.assert_close(native_output, reference[owned[rank]], rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(native_input.grad, reference_input.grad[owned[rank]], rtol=1e-5, atol=1e-6)
                os.environ["QWEN38_PLE_INPUT_HALO_CP"] = "1"
                for mode in ("full", "halo", "adaptive"):
                    os.environ["QWEN38_PLE_INPUT_HALO_TRANSPORT"] = mode
                    for recompute in (False, True):
                        local = hidden[owned[rank]].clone().requires_grad_()
                        output = checkpoint(model, local, use_reentrant=True) if recompute else model(local)
                        (output * derivative[owned[rank]]).sum().backward()
                        torch.testing.assert_close(output, reference[owned[rank]], rtol=1e-5, atol=1e-6)
                        torch.testing.assert_close(local.grad, reference_input.grad[owned[rank]], rtol=1e-5, atol=1e-6)
                        torch.testing.assert_close(output, native_output, rtol=1e-5, atol=1e-6)
                        torch.testing.assert_close(local.grad, native_input.grad, rtol=1e-5, atol=1e-6)
                    zero_input = hidden[owned[rank]].clone().requires_grad_()
                    (model(zero_input) * 0).sum().backward()
                    assert torch.equal(zero_input.grad, torch.zeros_like(zero_input))
        _check_block_interleaving(rank, native=False)
        _check_block_interleaving(rank, native=True)
    finally:
        dist.destroy_process_group()


class PLELayer(nn.Module):
    def __init__(self, ple):
        super().__init__()
        self.per_layer_embedding = ple
        self.layer_number = 1

    def forward(self, hidden_states, context=None, **kwargs):
        if getattr(self, "fail", False):
            raise RuntimeError("injected layer failure")
        return self.per_layer_embedding(hidden_states), context


def _check_block_interleaving(rank, *, native):
    """Execute native GPT and Bridge closures with CPU checkpointing.

    Replaces only the CUDA RNG checkpoint wrapper with PyTorch reentrant
    checkpoint. Microbatches share one PLE instance and backpropagate in
    reverse order, including equal layouts with different n-gram IDs.
    """
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_block import Qwen3VLTransformerBlock
    from megatron.core import parallel_state
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.transformer.ple_input_halo_cp import prepare_input_halo_cp
    from megatron.core.transformer.transformer_block import TransformerBlock

    from relax.backends.megatron.qsa_layout_metadata import bind_physical_layout

    torch.manual_seed(71)
    model = _model((16,))
    layer = PLELayer(model)
    block = SimpleNamespace(
        config=SimpleNamespace(
            fp8=False,
            fp4=False,
            recompute_method="uniform",
            recompute_num_layers=1,
            distribute_saved_activations=False,
        ),
        pre_process=False,
        num_layers_per_pipeline_rank=1,
        _get_layer=lambda index: layer,
    )
    pending = []
    with (
        patch.object(parallel_state, "get_context_parallel_world_size", return_value=2),
        patch.object(parallel_state, "get_context_parallel_rank", return_value=rank),
        patch.object(parallel_state, "get_context_parallel_group", return_value=dist.group.WORLD),
        patch("torch.cuda.current_device", return_value="cpu"),
        patch("megatron.core.ssm.mamba_context_parallel._THD_TE_DEFAULT_SHARED_BYTES", 0),
        patch(
            "megatron.core.tensor_parallel.checkpoint",
            side_effect=lambda function, distribute, *args: checkpoint(function, *args, use_reentrant=True),
        ),
    ):
        for batch, lengths in enumerate(((4, 12), (32, 16), (32, 16), (128,))):
            model._ngram_ids = ((torch.arange(sum(lengths)) + batch * 17) % 128).reshape(1, -1, 1)
            model._position_in_sequence = torch.cat([torch.arange(n) for n in lengths]).unsqueeze(0)
            model._cp_packed_seq_params = None
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
            model._cp_packed_seq_params = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu)
            prepare_input_halo_cp(model, packed)
            indices, base = [], 0
            for length in lengths:
                indices.extend(torch.arange(base, base + length).reshape(4, -1)[[rank, 3 - rank]].flatten().tolist())
                base += length
            local = hidden[indices].clone().requires_grad_()
            os.environ["QWEN38_PLE_INPUT_HALO_CP"] = "1"
            os.environ["QWEN38_PLE_INPUT_HALO_TRANSPORT"] = "adaptive"
            if native:
                output = TransformerBlock._checkpointed_forward(
                    block, local, None, None, None, None, None, packed, False
                )
            else:
                output = Qwen3VLTransformerBlock._relax_native_checkpointed_forward(
                    block, local, None, None, None, None, None, packed, None, False
                )
            torch.testing.assert_close(output, reference[indices], rtol=1e-5, atol=1e-6)
            pending.append((local, output, derivative[indices], reference_input.grad[indices]))
        latest = (model._ngram_ids, model._position_in_sequence, model._cp_packed_seq_params)
        for local, output, derivative, expected_gradient in reversed(pending):
            (output * derivative).sum().backward()
            torch.testing.assert_close(local.grad, expected_gradient, rtol=1e-5, atol=1e-6)
            assert all(
                current is saved
                for current, saved in zip(
                    (model._ngram_ids, model._position_in_sequence, model._cp_packed_seq_params), latest, strict=True
                )
            )

        if native:
            # A failure during replay must still restore the newer microbatch.
            local = hidden[indices].clone().requires_grad_()
            output = TransformerBlock._checkpointed_forward(block, local, None, None, None, None, None, packed, False)
            model._ngram_ids = model._ngram_ids.clone()
            model._position_in_sequence = model._position_in_sequence.clone()
            newer = (model._ngram_ids, model._position_in_sequence, model._cp_packed_seq_params)
            layer.fail = True
            with pytest.raises(RuntimeError, match="injected layer failure"):
                output.sum().backward()
            assert all(
                current is saved
                for current, saved in zip(
                    (model._ngram_ids, model._position_in_sequence, model._cp_packed_seq_params), newer, strict=True
                )
            )


def test_native_checkpoint_without_ple():
    from megatron.core.transformer.transformer_block import TransformerBlock

    class Layer(nn.Module):
        def forward(self, hidden_states, context=None, **kwargs):
            return hidden_states.square(), context

    layer = Layer()
    block = SimpleNamespace(
        config=SimpleNamespace(
            fp8=False,
            fp4=False,
            recompute_method="uniform",
            recompute_num_layers=1,
            distribute_saved_activations=False,
        ),
        num_layers_per_pipeline_rank=1,
        _get_layer=lambda index: layer,
    )
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    with patch(
        "megatron.core.tensor_parallel.checkpoint",
        side_effect=lambda function, distribute, *args: checkpoint(function, *args, use_reentrant=True),
    ):
        output = TransformerBlock._checkpointed_forward(block, x, None, None, None, None, None, None, False)
        output.sum().backward()
    torch.testing.assert_close(output, x.detach().square())
    torch.testing.assert_close(x.grad, 2 * x.detach())


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
