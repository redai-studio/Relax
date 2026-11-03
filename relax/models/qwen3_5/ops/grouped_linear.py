# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Native TE grouped GEMM with the recipe's no-fallback contract."""

from transformer_engine.pytorch.ops.basic.grouped_linear import GroupedLinear


class QwenGroupedLinear(GroupedLinear):
    def _is_graph_safe_path_supported(self, *args, **kwargs):
        result = super()._is_graph_safe_path_supported(*args, **kwargs)
        assert result, "native ops fallback forbidden"
        return result

    def _fuser_forward_grouped_tensor(self, *args, **kwargs):
        import torch

        splits = kwargs["split_sizes"]
        assert splits.is_cuda and splits.dtype == torch.int64
        return super()._fuser_forward_grouped_tensor(*args, **kwargs)
