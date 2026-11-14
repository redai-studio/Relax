# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Every TP worker shares the node budget, including not-yet-resident
memory."""

import json

import pytest

from task_delta_weight_sync.audit_cpu_budget import AuditCpuBudget, host_memory_budget


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


@pytest.mark.parametrize("kind", ["cgroup", "cgroup2"])
def test_effective_memory_uses_tighter_parent_even_when_leaf_is_unlimited(tmp_path, kind):
    proc, mount = tmp_path / "proc", tmp_path / "memory"
    (proc / "self").mkdir(parents=True)
    (mount / "worker").mkdir(parents=True)
    (proc / "meminfo").write_text("MemAvailable: 1000 kB\n")
    membership = "1:memory:/pod/worker\n" if kind == "cgroup" else "0::/pod/worker\n"
    (proc / "self/cgroup").write_text(membership)
    (proc / "self/mountinfo").write_text(f"1 0 0:1 /pod {mount} rw - {kind} cgroup rw,memory\n")
    limit_name, usage_name = (
        ("memory.limit_in_bytes", "memory.usage_in_bytes") if kind == "cgroup" else ("memory.max", "memory.current")
    )
    (mount / limit_name).write_text("100000")
    (mount / usage_name).write_text("90000")
    (mount / "worker" / limit_name).write_text(str((1 << 63) - 4096) if kind == "cgroup" else "max")
    (mount / "worker" / usage_name).write_text("1000")
    peak_name = "memory.max_usage_in_bytes" if kind == "cgroup" else "memory.peak"
    (mount / peak_name).write_text("95000")
    (mount / "worker" / peak_name).write_text("2000")
    result = host_memory_budget(proc)
    assert result["effective_available_bytes"] == 10000
    assert len(result["cgroups"]) == 2
    assert [row["peak_bytes"] for row in result["cgroups"]] == [2000, 95000]
    assert (mount / peak_name).read_text() == "95000"  # Shared lifetime counters must never be reset.
    budget = AuditCpuBudget(
        tmp_path / "leases",
        "worker",
        limit_bytes=1000000,
        margin_bytes=0,
        available=lambda: host_memory_budget(proc)["effective_available_bytes"],
    )
    with pytest.raises(ValueError, match="Insufficient host"):
        budget.reserve(10001)
    (mount / usage_name).write_text("100001")
    assert host_memory_budget(proc)["effective_available_bytes"] == 0


def test_v2_root_without_limit_uses_node_availability_and_missing_accounting_refuses(tmp_path):
    proc, mount = tmp_path / "proc", tmp_path / "memory"
    (proc / "self").mkdir(parents=True)
    mount.mkdir()
    (proc / "meminfo").write_text("MemAvailable: 123 kB\n")
    (proc / "self/cgroup").write_text("0::/\n")
    (proc / "self/mountinfo").write_text(f"1 0 0:1 / {mount} rw - cgroup2 cgroup rw\n")
    (mount / "memory.current").write_text("500")
    assert host_memory_budget(proc)["effective_available_bytes"] == 123 * 1024
    (mount / "memory.current").unlink()
    with pytest.raises(FileNotFoundError):
        host_memory_budget(proc)
