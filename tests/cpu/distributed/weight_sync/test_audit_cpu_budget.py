# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Every TP worker shares the node budget, including not-yet-resident
memory."""

import json

import pytest

from task_delta_weight_sync.audit_cpu_budget import AuditCpuBudget


def test_node_budget_counts_all_workers_and_releases_only_its_own_lease(tmp_path):
    first = AuditCpuBudget(tmp_path, "engine0-tp0", limit_bytes=100, margin_bytes=0, available=lambda: 200)
    second = AuditCpuBudget(tmp_path, "engine1-tp0", limit_bytes=100, margin_bytes=0, available=lambda: 200)
    first.reserve(60)
    first.allocated()
    with pytest.raises(ValueError, match="shared node CPU budget"):
        second.reserve(50)
    second.reserve(40)
    second.release()
    assert set(json.loads(first.path.read_text())["leases"]) == {"engine0-tp0"}
    first.release()
    assert json.loads(first.path.read_text())["leases"] == {}


def test_pending_reservations_cannot_all_spend_the_same_memavailable(tmp_path):
    first = AuditCpuBudget(tmp_path, "a", limit_bytes=1000, margin_bytes=10, available=lambda: 100)
    second = AuditCpuBudget(tmp_path, "b", limit_bytes=1000, margin_bytes=10, available=lambda: 100)
    first.reserve(60)
    with pytest.raises(ValueError, match="MemAvailable"):
        second.reserve(40)
    first.allocated()
    # MemAvailable now represents the remaining physical RAM; allocated
    # snapshots must not be subtracted a second time.
    second.available = lambda: 40
    second.reserve(30)
    with pytest.raises(RuntimeError, match="already owns"):
        first.reserve(1)
    first.release()
    second.release()
