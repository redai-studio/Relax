import json
import multiprocessing as mp
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from relax.distributed.weight_sync.publication import VersionPublisher
from relax.distributed.weight_sync.scratch import ScratchWorkspace
from relax.distributed.weight_sync.store import VersionStore


def crash_writer(root, point, full_only):
    root = Path(root)
    store = VersionStore(root / "store", disk_budget=1024**2, max_weight_bytes=1024)
    publisher = VersionPublisher(
        store, "run", root / "scratch", model_config={}, converter_revision="test", full_only=full_only
    )
    if point == "before-journal":
        import shutil

        source = root / "source.bin"
        source.write_bytes(bytes(256))
        copy = shutil.copyfile

        def crash_during_copy(*args, **kwargs):
            copy(*args, **kwargs)
            os._exit(23)

        with patch("relax.distributed.weight_sync.publication.shutil.copyfile", crash_during_copy):
            publisher.submit("update", {"w": ("U8", [256], source)})
    else:
        source = root / "source.bin"
        source.write_bytes(bytes(range(64)))
        handle = publisher.submit("update", {"w": ("U8", [64], source)})
        published = publisher.wait_committed(handle, timeout=10)
        (root / "identity.json").write_text(json.dumps({"root": published.root, "candidate": handle.candidate_id}))
        if point == "source-confirmed":
            original_unlink = Path.unlink

            def crash_before_journal_release(path, *args, **kwargs):
                if path == publisher._journal:
                    os._exit(23)
                return original_unlink(path, *args, **kwargs)

            with patch.object(Path, "unlink", crash_before_journal_release):
                publisher.confirm(handle, published)
    os._exit(23)


@pytest.mark.parametrize("full_only", [True, False])
@pytest.mark.parametrize("point", ["before-journal", "store-committed", "source-confirmed"])
def test_hard_exit_reclaims_unjournaled_candidate_and_preserves_pending(tmp_path, point, full_only):
    other = ScratchWorkspace(tmp_path / "scratch", "other-live-stream")
    other_file = other.temporary() / "keep.bin"
    other_file.write_bytes(b"live")
    try:
        for _ in range(2):
            process = mp.get_context("spawn").Process(target=crash_writer, args=(str(tmp_path), point, full_only))
            process.start()
            process.join(20)
            assert process.exitcode == 23
            store = VersionStore(tmp_path / "store", disk_budget=1024**2, max_weight_bytes=1024)
            publisher = VersionPublisher(
                store, "run", tmp_path / "scratch", model_config={}, converter_revision="test", full_only=full_only
            )
            if point == "store-committed":
                expected = json.loads((tmp_path / "identity.json").read_text())
                assert publisher.pending.candidate_id == expected["candidate"]
                result = publisher.finalize_pending(publisher.pending)
                assert result.root == expected["root"]
            else:
                assert publisher.pending is None
            assert list(publisher.workspace.root.glob(publisher.workspace.prefix + "*")) == []
            publisher.close()
            assert other_file.read_bytes() == b"live"
    finally:
        other.close()


def orphan_source(root, scope):
    workspace = ScratchWorkspace(root, scope)
    (workspace.temporary() / "export.bin").write_bytes(bytes(1024))
    os._exit(24)


@pytest.mark.parametrize("scope", ["export:run:0", "source-cache:run:1"])
def test_export_and_source_cache_recover_after_real_hard_exit(tmp_path, scope):
    process = mp.get_context("spawn").Process(target=orphan_source, args=(str(tmp_path), scope))
    process.start()
    process.join(20)
    assert process.exitcode == 24
    workspace = ScratchWorkspace(tmp_path, scope)
    assert list(tmp_path.glob(workspace.prefix + "*")) == []
    with pytest.raises((OSError, TimeoutError)):
        ScratchWorkspace(tmp_path, scope)
    workspace.close()


def test_failed_delete_has_durable_record_and_retries(tmp_path):
    workspace = ScratchWorkspace(tmp_path, "export:run:0")
    directory = workspace.temporary()
    (directory / "data").write_bytes(b"test")
    with patch("relax.distributed.weight_sync.scratch.shutil.rmtree", side_effect=PermissionError("busy")):
        assert workspace.remove(directory) is False
    assert directory.name in json.loads(workspace.record.read_text())["pending_cleanup"]
    workspace.close()
    restored = ScratchWorkspace(tmp_path, "export:run:0")
    assert not directory.exists()
    assert json.loads(restored.record.read_text())["pending_cleanup"] == []
    restored.close()


def test_worker_schedule_failure_keeps_journal_and_same_candidate(tmp_path):
    store = VersionStore(tmp_path / "store", disk_budget=1024**2, max_weight_bytes=1024)
    publisher = VersionPublisher(store, "run", tmp_path / "scratch", model_config={}, converter_revision="test")
    source = tmp_path / "w.bin"
    source.write_bytes(bytes(64))
    try:
        with patch.object(publisher, "_schedule_publish", side_effect=RuntimeError("schedule failed")):
            with pytest.raises(RuntimeError, match="schedule failed"):
                publisher.submit("same-update", {"w": ("U8", [64], source)})
        handle = publisher.pending
        assert handle is not None and publisher._directory.exists() and publisher._journal.exists()
        publisher.retry(handle)
        result = publisher.finalize_pending(handle)
        assert result.update_id == "same-update" and result.version == 0
    finally:
        publisher.close()
