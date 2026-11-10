# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Native wire operation accounting without allocating full model replicas."""

import pytest
import torch

from task_delta_weight_sync.prepared_nccl import plan_capacity, wire_bucket


def test_prepared_plan_keeps_native_order_dtype_and_bytes_with_bounded_reuse():
    first = [
        ("a", torch.empty(3, dtype=torch.bfloat16)),
        ("b", torch.empty(3, dtype=torch.bfloat16)),
        ("c", torch.empty(2, dtype=torch.float32)),
        ("d", torch.empty(3, dtype=torch.bfloat16)),
    ]
    bucket = wire_bucket(first)
    assert bucket == [["bfloat16", 3, 2], ["float32", 2, 1], ["bfloat16", 3, 1]]
    capacities, totals = plan_capacity([bucket, bucket])
    assert capacities == {"bfloat16": 9, "float32": 2}
    assert totals == {"bfloat16": 36, "float32": 16}


@pytest.mark.parametrize(
    "plan", [[], [[["int8", 1, 1]]], [[["bfloat16", -1, 1]]], [[["bfloat16", 1, 0]]], [[["float32", 1 << 30, 1]]]]
)
def test_invalid_or_unbounded_transfer_plan_is_rejected_before_allocation(plan):
    with pytest.raises(ValueError):
        plan_capacity(plan)
