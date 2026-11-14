import copy

import pytest

from relax.distributed.weight_sync.source_budget import (
    cgroup_available,
    check_source_preflight,
    gpu_capacity,
    source_demand,
    source_preflight_record,
)
from relax.distributed.weight_sync.store import VersionStore


def record(host="node", available=10**12):
    return {
        "host": host,
        "filesystem": 1,
        "demand": source_demand(1024, 4096, 512, adaptive=True, bridge=True),
        "limits": {},
        "available_gpu_bytes": available,
        "available_ram_bytes": available,
        "available_disk_bytes": available,
        "scratch_used_bytes": 0,
    }


def test_shared_ram_estimate_reports_pressure_without_rejecting_publication():
    first = record(available=400 * 1024**2)
    report = check_source_preflight([first, copy.deepcopy(first)])
    assert report["status"] == "ADMITTED"
    assert report["shortages"] == []
    assert {item["resource"] for item in report["warnings"]} == {"available_ram"}
    assert report["actions"] == []
    first["available_ram_bytes"] *= 2
    assert check_source_preflight([first, copy.deepcopy(first)])["status"] == "ADMITTED"


def test_separate_hosts_have_separate_ram():
    assert check_source_preflight([record("a", 400 * 1024**2), record("b", 400 * 1024**2)])["status"] == "ADMITTED"


def test_memory_backed_scratch_also_requires_host_ram():
    row = record(available=400 * 1024**2)
    row["scratch_memory_bytes"] = 400 * 1024**2
    result = check_source_preflight([row])
    assert {item["resource"] for item in result["warnings"]} == {"available_ram"}


def test_existing_bases_and_candidate_cannot_fit_independently():
    row = record()
    additional = row["demand"]["scratch_additional_bytes"]
    row["scratch_used_bytes"] = additional
    row["limits"] = {"scratch_bytes": additional + 1}
    assert {item["resource"] for item in check_source_preflight([row])["shortages"]} == {"scratch_limit"}


def test_store_and_source_share_physical_disk(tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    store = VersionStore(tmp_path / "store", disk_budget=1024**3, max_weight_bytes=1024)
    row = source_preflight_record(
        scratch, source_demand(1024, 4096, 512, adaptive=True, bridge=True), store=store, logical_bytes=4096
    )
    volume = row["publication_volumes"][0]
    row["available_disk_bytes"] = volume["additional_bytes"] + 1
    volume["available_bytes"] = row["available_disk_bytes"]
    assert any(item["resource"] == "combined_filesystem" for item in check_source_preflight([row])["shortages"])


def test_bridge_buffers_include_raw_and_logical():
    direct = source_demand(100, 200, 30, adaptive=True, bridge=False)
    bridge = source_demand(100, 200, 30, adaptive=True, bridge=True)
    assert bridge["scratch_additional_bytes"] - direct["scratch_additional_bytes"] == 200
    assert bridge["ram_components"]["frozen_tensors"] == 100
    assert bridge["gpu_bytes"] == 90


def test_separate_adaptive_cache_volume_is_admitted_before_capture(tmp_path):
    scratch, cache = tmp_path / "scratch", tmp_path / "cache"
    scratch.mkdir()
    cache.mkdir()
    row = source_preflight_record(
        scratch, source_demand(1024, 4096, 512, adaptive=True, bridge=True), source_cache=cache
    )
    volume = row["publication_volumes"][0]
    assert volume["kind"] == "source_cache"
    assert volume["path"] == str(cache)
    volume["available_bytes"] = 1
    assert any(item["resource"] == "combined_filesystem" for item in check_source_preflight([row])["shortages"])


def test_bridge_source_padding_has_its_own_capacity_bound():
    demand = source_demand(500, 800, 40, adaptive=True, bridge=True, gathered_source_bytes=1000)
    assert demand["scratch_components"]["gathered_source"] == 1000
    assert demand["scratch_components"]["logical_export"] == 800
    # Local shards, frames, logical export, immutable candidate and raw cache
    # coexist before source confirmation releases the old base.
    assert demand["scratch_additional_bytes"] == 500 + 500 + 800 + 800 + 1000


@pytest.mark.parametrize("same_volume", [True, False])
def test_relocated_source_files_reserve_each_byte_on_its_destination(tmp_path, monkeypatch, same_volume):
    from relax.distributed.weight_sync import source_budget

    scratch, cache = tmp_path / "publication", tmp_path / "source"
    scratch.mkdir()
    cache.mkdir()
    monkeypatch.setattr(
        source_budget, "filesystem_key", lambda path: "disk-a" if path == scratch or same_volume else "disk-b"
    )
    demand = source_demand(100, 200, 30, adaptive=True, bridge=True)
    row = source_preflight_record(scratch, demand, source_cache=cache)
    volume = row["publication_volumes"][0]
    # The 200-byte publication candidate and 600-byte source workspace coexist.
    # Moving that workspace used to reserve all 800 bytes on both disks.
    assert row["scratch_volume_additional_bytes"] == 200
    assert volume["additional_bytes"] == 600
    row["available_disk_bytes"] = 800 if same_volume else 200
    volume["available_bytes"] = 800 if same_volume else 600
    assert check_source_preflight([row])["status"] == "ADMITTED"
    volume["available_bytes"] -= 1
    assert {x["resource"] for x in check_source_preflight([row])["shortages"]} == {"combined_filesystem"}


def test_external_source_base_counts_toward_explicit_workspace_limit(tmp_path):
    scratch, cache = tmp_path / "publication", tmp_path / "source"
    scratch.mkdir()
    cache.mkdir()
    (cache / "confirmed-shard.bin").write_bytes(bytes(100))
    demand = source_demand(100, 200, 30, adaptive=True, bridge=True)
    row = source_preflight_record(scratch, demand, source_cache=cache, limits={"scratch_bytes": 899})
    report = check_source_preflight([row])
    assert report["scratch_workspace"]["retained_bytes"] == 100
    assert {x["resource"] for x in report["shortages"]} == {"scratch_limit"}


def test_parent_job_memory_limit_applies_to_unlimited_task(tmp_path):
    job = tmp_path / "job"
    task = job / "task"
    task.mkdir(parents=True)
    (job / "memory.max").write_text("1000")
    (job / "memory.current").write_text("800")
    (task / "memory.max").write_text("max")
    assert cgroup_available(task, tmp_path) == 200
    (job / "memory.current").write_text("1001")
    assert cgroup_available(task, tmp_path) == 0


@pytest.mark.parametrize(
    "stats, expected",
    [
        ({"inactive_file": 600, "file": 700}, 800),
        ({"inactive_file": 600, "file": 700, "shmem": 400}, 500),
        ({"inactive_file": 600, "file": 700, "file_dirty": 100, "file_writeback": 50}, 650),
        ({"inactive_file": 600, "file": 700, "file_dirty": 900}, 200),
        ({"inactive_file": 0, "file": 700, "active_file": 700}, 200),
        ({"inactive_file": 9000, "file": 9000}, 1000),
        ({"inactive_file": -1, "file": 700}, 200),
    ],
)
def test_only_clean_inactive_disk_cache_adds_headroom(tmp_path, stats, expected):
    (tmp_path / "memory.max").write_text("1000")
    (tmp_path / "memory.current").write_text("800")
    fields = dict.fromkeys(("inactive_file", "file", "shmem", "file_dirty", "file_writeback"), 0)
    fields.update(stats)
    (tmp_path / "memory.stat").write_text("\n".join(f"{key} {value}" for key, value in fields.items()))
    assert cgroup_available(tmp_path, tmp_path) == expected


@pytest.mark.parametrize("stats", ["", "file 700\ninactive_file 600", "file invalid", "malformed line extra"])
def test_unavailable_cache_stats_keep_strict_headroom(tmp_path, stats):
    (tmp_path / "memory.max").write_text("1000")
    (tmp_path / "memory.current").write_text("800")
    (tmp_path / "memory.stat").write_text(stats)
    assert cgroup_available(tmp_path, tmp_path) == 200


def test_reclaimable_child_cache_still_obeys_parent_limit(tmp_path):
    child = tmp_path / "child"
    child.mkdir()
    for directory, limit, current in ((tmp_path, 1000, 900), (child, 800, 700)):
        (directory / "memory.max").write_text(str(limit))
        (directory / "memory.current").write_text(str(current))
        (directory / "memory.stat").write_text(
            "inactive_file 400\nfile 500\nshmem 0\nfile_dirty 0\nfile_writeback 0\n"
        )
    assert cgroup_available(child, tmp_path) == 500
    (tmp_path / "memory.stat").write_text("inactive_file 0\nfile 0\nshmem 0\nfile_dirty 0\nfile_writeback 0\n")
    assert cgroup_available(child, tmp_path) == 100


def test_other_rank_scratch_and_owner_store_share_disk():
    owner, other = record(), record()
    owner["filesystem"] = 1
    other["filesystem"] = 2
    other["available_disk_bytes"] = 20000
    owner["publication_volumes"] = [
        {
            "kind": "encoded_store",
            "filesystem": 2,
            "additional_bytes": 16000,
            "available_bytes": 20000,
            "budget": 10**9,
            "used_bytes": 0,
            "path": "/store",
        }
    ]
    assert any(
        item["resource"] == "combined_filesystem" for item in check_source_preflight([owner, other])["shortages"]
    )


def test_shared_remote_mount_groups_different_host_device_numbers():
    first, second = record("a"), record("b")
    first["filesystem"], second["filesystem"] = 1, 99
    for row in (first, second):
        row["filesystem_key"] = "shared:export-id"
        row["available_disk_bytes"] = row["demand"]["scratch_additional_bytes"] + 1
    assert any(
        item["resource"] == "combined_filesystem" for item in check_source_preflight([first, second])["shortages"]
    )


def test_separate_local_disks_only_reserve_their_own_scratch():
    first, second = record("a"), record("b")
    for row in (first, second):
        row["available_disk_bytes"] = row["demand"]["scratch_additional_bytes"] + 1
    assert check_source_preflight([first, second])["status"] == "ADMITTED"


def test_workspace_limit_includes_bases_on_each_distinct_host():
    first, second = record("a"), record("b")
    additional = 2 * first["demand"]["scratch_additional_bytes"]
    for row in (first, second):
        row["scratch"] = "/scratch"
        row["scratch_used_bytes"] = 1000
        row["limits"] = {"scratch_bytes": additional + 1500}
    result = check_source_preflight([first, second])
    assert result["scratch_workspace"]["retained_bytes"] == 2000
    assert {item["resource"] for item in result["shortages"]} == {"scratch_limit"}
    # Both ranks observing the same shared directory see the same retained
    # files. Future writes still reserve space independently for both ranks.
    for row in (first, second):
        row["filesystem_key"] = "shared:one-export"
    result = check_source_preflight([first, second])
    assert result["scratch_workspace"]["retained_bytes"] == 1000
    assert result["status"] == "ADMITTED"


def test_nested_scratch_paths_do_not_count_existing_files_twice():
    parent, child = record(), record()
    parent.update(scratch="/scratch", scratch_used_bytes=1000)
    child.update(scratch="/scratch/rank1", scratch_used_bytes=300)
    result = check_source_preflight([parent, child])
    assert result["scratch_workspace"]["retained_bytes"] == 1000


def test_gpu_rechecks_driver_after_releasing_only_unused_cache():
    class CUDA:
        free = 100

        def mem_get_info(self):
            return self.free, 1000

        def memory_reserved(self):
            return 800

        def memory_allocated(self):
            return 500

        def empty_cache(self):
            self.free = 400

    cuda = CUDA()
    assert not gpu_capacity(cuda, 50)["released_unused_cache"]
    report = gpu_capacity(cuda, 300)
    assert report["driver_free_before_bytes"] == 100 and report["available_bytes"] == 400
    assert report["released_unused_cache"]


def test_observed_gpu_pressure_is_advisory_but_explicit_limits_remain_effective():
    row = record()
    row["available_gpu_bytes"] = 1
    row["available_ram_bytes"] = 1
    result = check_source_preflight([row])
    assert result["status"] == "ADMITTED"
    assert result["shortages"] == []
    assert {x["resource"] for x in result["warnings"]} == {"available_gpu", "available_ram"}
    row["limits"] = {"ram_bytes": 1, "gpu_bytes": 1}
    result = check_source_preflight([row])
    assert result["status"] == "WAITING_ACTION"
    assert {x["resource"] for x in result["shortages"]} == {"ram_limit", "gpu_limit"}
    assert "retry the same update" in result["actions"]
