from types import SimpleNamespace

from relax.distributed.weight_sync.provenance import record_optimizer_step, source_progress
from relax.distributed.weight_sync.publication import VersionPublisher
from relax.distributed.weight_sync.store import VersionStore


def test_optimizer_attempts_distinguish_skips_from_unchanged_content():
    args = SimpleNamespace()
    assert source_progress(args)["optimizer_attempts"] == 0
    record_optimizer_step(args, 0, 0, True)
    record_optimizer_step(args, 0, 1, False)
    assert source_progress(args) == {
        "optimizer_attempts": 2,
        "optimizer_updates": 1,
        "last_step_updated": False,
        "rollout_id": 0,
        "step_in_rollout": 1,
    }
    # This hook is called once per optimizer attempt, after all microbatches.
    record_optimizer_step(args, 1, 0, True)
    assert source_progress(args)["optimizer_updates"] == 2
    assert source_progress(SimpleNamespace())["optimizer_attempts"] == 0


def test_source_identity_survives_retry_and_same_content(tmp_path):
    store = VersionStore(tmp_path / "store", disk_budget=1 << 20, max_weight_bytes=1024)
    path = tmp_path / "w"
    path.write_bytes(bytes(256))
    table = {"w": ("BF16", [128], path)}
    publisher = VersionPublisher(store, "run", tmp_path / "scratch", model_config={}, converter_revision="test")
    try:
        source = {"source_epoch": "generation", "optimizer_updates": 0, "snapshot_kind": "init"}
        first = publisher.submit("generation:0", table, source=source)
        publisher.finalize_pending(first)
        source["optimizer_updates"] = 1
        source["snapshot_kind"] = "update"
        second = publisher.submit("generation:1", table, source=source)
        publisher.finalize_pending(second)
        a, b = store.manifest("run", 0), store.manifest("run", 1)
        assert a["root"] == b["root"]
        assert a["update_id"] != b["update_id"]
        assert a["source"]["optimizer_updates"] == 0
        assert b["source"]["optimizer_updates"] == 1
        assert store.ensure_full("run", 1)["source"] == b["source"]
    finally:
        publisher.close()
