# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest
import torch
import torch.nn.functional as F

from relax.utils.training import fused_ce


@pytest.mark.parametrize("batch,rows", [(1, 7), (2, 7), (1, 1)])
def test_te_adapter_normalizes_offset_transposed_labels(monkeypatch, batch, rows):
    # Actual MTP layout: contiguous [rows,1] can retain stride (1,128).
    parent = torch.arange(batch * 128).reshape(batch, 128) % 17
    labels = parent[:, 5 : 5 + rows].transpose(0, 1)
    logits = torch.randn(batch, rows, 17).transpose(0, 1).requires_grad_()

    def checked_kernel(x, target, **kwargs):
        assert x.shape == (1, rows * batch, 17)
        assert target.shape == (1, rows * batch)
        if target.numel() > 1:
            assert target.stride(-1) == 1
            assert x.stride(-2) == 17
        assert kwargs["is_cg_capturable"] is True
        assert kwargs["reduce_loss"] is False
        torch.testing.assert_close(target.flatten(), labels.reshape(-1))
        return F.cross_entropy(x[0], target[0], reduction="none").unsqueeze(0)

    monkeypatch.setattr(fused_ce, "_te_ce", lambda: checked_kernel)
    actual = fused_ce.te_cross_entropy(logits, labels)
    expected = F.cross_entropy(logits.reshape(-1, 17), labels.reshape(-1), reduction="none").reshape_as(labels)
    torch.testing.assert_close(actual, expected)
    upstream = torch.randn_like(actual)
    (actual_grad,) = torch.autograd.grad(actual, logits, upstream)
    (expected_grad,) = torch.autograd.grad(expected, logits, upstream)
    torch.testing.assert_close(actual_grad, expected_grad)
