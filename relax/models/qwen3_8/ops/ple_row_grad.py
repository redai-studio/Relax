# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Accumulate BF16 n-gram embedding gradients without a full BF16 table
temporary."""

from argparse import Namespace

import torch
import torch.nn.functional as F
from torch.autograd.function import FunctionCtx

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


class _PLEEmbedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx: FunctionCtx, indices: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(indices)
        ctx.table_shape = weight.shape
        return F.embedding(indices, weight)

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: torch.Tensor) -> tuple[None, torch.Tensor]:
        (indices,) = ctx.saved_tensors
        ids = indices.reshape(-1)
        rows = ids.sort().values
        # The first position of each row is a local slot. A fixed number of slots
        # avoids a host synchronization to discover the number of unique IDs.
        slots = torch.searchsorted(rows, ids)
        values = torch.ops.aten.embedding_dense_backward(
            grad_output.reshape(ids.numel(), ctx.table_shape[1]), slots, ids.numel(), -1, False
        )
        # Every repeated position has a zero gradient. Spread those zeros over
        # the table so masked input IDs do not contend on row zero in index_add_.
        positions = torch.arange(ids.numel(), device=ids.device)
        first = (positions == 0) | (rows != rows.roll(1))
        destinations = torch.where(first, rows, positions % ctx.table_shape[0])
        grad = torch.sparse_coo_tensor(destinations.unsqueeze(0), values, ctx.table_shape)
        return None, grad


def _lookup(indices: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if not (torch.is_grad_enabled() and weight.requires_grad and weight.is_cuda and weight.dtype == torch.bfloat16):
        return F.embedding(indices, weight)

    # Megatron FSDP leases this parameter's reduce-scatter bucket on demand in
    # the backward hook. Requesting it during the lookup increases peak memory.
    if not callable(getattr(weight, "get_main_grad", None)):
        main = getattr(weight, "main_grad", None)
        if main is None or main.layout != torch.strided or main.shape != weight.shape or main.dtype != torch.float32:
            return F.embedding(indices, weight)
    return _PLEEmbedding.apply(indices, weight)


def configure(model: torch.nn.Module, args: Namespace) -> None:
    from megatron.core.tensor_parallel.layers import VocabParallelEmbedding
    from megatron.core.transformer.per_layer_embedding import NGramEmbedding

    if not args.use_distributed_optimizer or getattr(args, "use_torch_fsdp2", False):
        raise ValueError("PLE compact gradients require the Megatron distributed optimizer")
    if getattr(args, "use_megatron_fsdp", False) and (
        args.data_parallel_sharding_strategy not in ("optim_grads", "optim_grads_params")
        or args.megatron_fsdp_main_grads_dtype not in (torch.float32, "fp32")
    ):
        raise ValueError("PLE compact gradients with Megatron FSDP require sharded gradients and FP32 accumulation")
    for owner in model.modules():
        if not isinstance(owner, NGramEmbedding):
            continue
        table = owner.ngram_embedding
        if not isinstance(table, VocabParallelEmbedding) or table.deterministic_mode or not table.weight.requires_grad:
            continue
        if not hasattr(table, "_embedding_lookup"):
            raise RuntimeError("PLE compact gradients require the embedding lookup hook in the Relax Megatron patch")
        table._embedding_lookup = _lookup
        logger.info("Enabled compact gradients for PLE n-gram table shard %s", tuple(table.weight.shape))
