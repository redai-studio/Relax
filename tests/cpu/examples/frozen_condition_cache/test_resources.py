import os

from experiments.resources import cache_storage_contract, directory_usage, storage_snapshot


def test_stage_storage_counts_task_files_without_following_external_links(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "evidence").mkdir()
    payload = run / "evidence" / "gradient.bin"
    payload.write_bytes(b"gradient" * 5)
    os.link(payload, run / "same-inode.bin")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "not-task.bin").write_bytes(b"x" * 1000)
    (run / "outside-link").symlink_to(outside, target_is_directory=True)
    sample = storage_snapshot(run, shm_path=tmp_path)
    task = sample["run_directory"]
    assert task["logical_file_bytes"] == 40
    assert task["unique_regular_files"] == 1
    assert task["symlinks_not_followed"] == 1 and task["complete"]
    assert task["allocated_file_bytes"] >= 40
    for name in ("run_filesystem", "shared_memory_filesystem"):
        usage = sample[name]
        assert usage["scope"] == "filesystem_global" and usage["available"]
        assert usage["total_bytes"] > 0 and usage["free_bytes"] >= 0 and usage["used_bytes"] >= 0


def test_missing_shm_is_unavailable_instead_of_fabricated_zero(tmp_path):
    sample = storage_snapshot(tmp_path, shm_path=tmp_path / "missing-shm")
    assert sample["shared_memory_filesystem"]["available"] is False
    assert "used_bytes" not in sample["shared_memory_filesystem"]
    assert directory_usage(tmp_path / "missing-run")["complete"] is False


def test_cache_storage_declaration_distinguishes_backend_from_global_filesystems():
    contract = cache_storage_contract(True, "cpu")
    assert contract["storage"] == "process_private_cpu_ram"
    assert not contract["persistent_files_created_by_cache_backend"]
    assert not contract["shared_memory_segments_created_by_cache_backend"]
    assert "implementation uses" in contract["basis"]
    assert cache_storage_contract(False, "cpu")["storage"] == "disabled"
