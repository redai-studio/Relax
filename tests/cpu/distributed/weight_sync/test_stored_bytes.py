import os

import pytest

from relax.distributed.weight_sync.store import VersionStore, stored_bytes


def test_capacity_scan_tolerates_entries_removed_during_metadata_read(tmp_path, monkeypatch):
    from contextlib import contextmanager

    from relax.distributed.weight_sync import store

    (tmp_path / "retained").write_bytes(b"retained")
    (tmp_path / ".partial-test").write_bytes(b"temporary")
    empty = tmp_path / "removed-directory"
    empty.mkdir()
    scan = store.os.scandir

    class Entry:
        def __init__(self, entry):
            self.entry = entry
            self.path = entry.path

        def is_dir(self, **kwargs):
            result = self.entry.is_dir(**kwargs)
            if self.entry.name == "removed-directory":
                empty.rmdir()
            return result

        def is_file(self):
            result = self.entry.is_file()
            if self.entry.name == ".partial-test":
                (tmp_path / ".partial-test").unlink()
            return result

        def stat(self):
            return self.entry.stat()

    def racing_scan(path):
        entries = scan(path)

        @contextmanager
        def wrapped():
            with entries:
                yield (Entry(entry) for entry in entries)

        return wrapped()

    monkeypatch.setattr(store.os, "scandir", racing_scan)
    assert stored_bytes(tmp_path) == len(b"retained")


def test_links_count_target_inodes_once_without_directory_traversal(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    original = store / "original"
    original.write_bytes(bytes(31))
    os.link(original, store / "hardlink")
    (store / "file-link").symlink_to(original)
    nested = store / "nested"
    nested.mkdir()
    (nested / "other").write_bytes(bytes(17))
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "excluded").write_bytes(bytes(1000))
    (store / "directory-link").symlink_to(outside, target_is_directory=True)
    (store / "loop").symlink_to(store, target_is_directory=True)
    (store / "broken-link").symlink_to(store / "missing")
    assert stored_bytes(store) == 48


def test_external_file_symlink_counts_its_target(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    external = tmp_path / "external"
    external.write_bytes(bytes(23))
    (store / "external-link").symlink_to(external)
    assert stored_bytes(store) == 23


@pytest.mark.parametrize("kind", ["missing", "file", "empty", "directory-link"])
def test_empty_or_non_directory_roots(tmp_path, kind):
    root = tmp_path / "root"
    if kind == "file":
        root.write_bytes(bytes(23))
    elif kind == "empty":
        root.mkdir()
    elif kind == "directory-link":
        target = tmp_path / "target"
        target.mkdir()
        root.symlink_to(target, target_is_directory=True)
    assert stored_bytes(root) == 0


def test_new_files_are_seen_on_every_admission_scan(tmp_path):
    store = VersionStore(tmp_path, disk_budget=4096, max_weight_bytes=1024)
    baseline = stored_bytes(tmp_path)
    store._reserve(1)
    (tmp_path / "retained-object").write_bytes(bytes(4096))
    assert stored_bytes(tmp_path) == baseline + 4096
    store.refresh_budget()
    with pytest.raises(OSError, match="disk budget exhausted"):
        store._reserve(1)
