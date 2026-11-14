"""Host cache accounting must not hide usable RAM or bypass job limits."""

from pathlib import Path

import pytest

from relax.distributed.weight_sync.source_budget import memory_capacity


def fixture(tmp_path, *, arc=True):
    proc, sys = tmp_path / "proc", tmp_path / "sys"
    (proc / "self").mkdir(parents=True)
    (proc / "meminfo").write_text(
        "MemTotal: 256000 kB\nMemAvailable: 12000 kB\nCached: 8000 kB\nBuffers: 1000 kB\nSwapCached: 1000 kB\n"
    )
    (proc / "self/cgroup").write_text("0::/job/task\n")
    task = sys / "fs/cgroup/job/task"
    task.mkdir(parents=True)
    (task / "memory.max").write_text("max")
    (task.parent / "memory.max").write_text(str(192000 * 1024))
    (task.parent / "memory.current").write_text(str(110000 * 1024))
    if arc:
        stats = proc / "spl/kstat/zfs/arcstats"
        stats.parent.mkdir(parents=True)
        values = {
            "size": 128000,
            "c_min": 8000,
            "c_max": 128000,
            "mru_evictable_data": 60000,
            "mfu_evictable_data": 50000,
            "mru_evictable_metadata": 7000,
            "mfu_ghost_evictable_data": 999999,
        }
        stats.write_text("\n".join(f"{key} 4 {value * 1024}" for key, value in values.items()))
        percent = sys / "module/zfs/parameters/zfs_arc_pc_percent"
        percent.parent.mkdir(parents=True)
        percent.write_text("0")
    return proc, sys


def replace_counter(proc: Path, key: str, value: int):
    path = proc / "spl/kstat/zfs/arcstats"
    rows = [row for row in path.read_text().splitlines() if not row.startswith(key + " ")]
    path.write_text("\n".join(rows + [f"{key} 4 {value}"]))


def test_clean_arc_recovers_headroom_without_exceeding_job_limit(tmp_path):
    proc, sys = fixture(tmp_path)
    report = memory_capacity(proc, sys)
    assert report == {
        "available_bytes": 82000 * 1024,
        "linux_available_bytes": 12000 * 1024,
        "reclaimable_arc_data_bytes": 110000 * 1024,
        "cgroup_available_bytes": 82000 * 1024,
    }
    # A nearly full job still cannot borrow host cache to exceed its own limit.
    (sys / "fs/cgroup/job/memory.current").write_text(str(191000 * 1024))
    assert memory_capacity(proc, sys)["available_bytes"] == 1000 * 1024


def test_arc_floor_and_page_cache_policy_are_preserved(tmp_path):
    proc, sys = fixture(tmp_path)
    replace_counter(proc, "size", 15000 * 1024)
    assert memory_capacity(proc, sys)["reclaimable_arc_data_bytes"] == 7000 * 1024
    (sys / "module/zfs/parameters/zfs_arc_pc_percent").write_text("100")
    assert memory_capacity(proc, sys)["reclaimable_arc_data_bytes"] == 5000 * 1024
    replace_counter(proc, "size", 7000 * 1024)
    assert memory_capacity(proc, sys)["reclaimable_arc_data_bytes"] == 0


def test_metadata_and_ghost_entries_are_not_available_data(tmp_path):
    proc, sys = fixture(tmp_path)
    replace_counter(proc, "mru_evictable_data", 0)
    replace_counter(proc, "mfu_evictable_data", 0)
    assert memory_capacity(proc, sys)["available_bytes"] == 12000 * 1024


@pytest.mark.parametrize(
    "damage", ["missing", "truncated", "negative", "oversized", "policy_missing", "policy_invalid"]
)
def test_unusable_arc_evidence_preserves_linux_estimate(tmp_path, damage):
    proc, sys = fixture(tmp_path, arc=damage != "missing")
    if damage == "truncated":
        (proc / "spl/kstat/zfs/arcstats").write_text("size 4 1000")
    elif damage == "negative":
        replace_counter(proc, "mru_evictable_data", -1)
    elif damage == "oversized":
        replace_counter(proc, "size", 9999999999)
    elif damage == "policy_missing":
        (sys / "module/zfs/parameters/zfs_arc_pc_percent").unlink()
    elif damage == "policy_invalid":
        (sys / "module/zfs/parameters/zfs_arc_pc_percent").write_text("invalid")
    assert memory_capacity(proc, sys)["available_bytes"] == 12000 * 1024


def test_missing_linux_meminfo_remains_unknown(tmp_path):
    assert memory_capacity(tmp_path / "proc", tmp_path / "sys") == {"available_bytes": None}
