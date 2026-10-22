# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import base64
import json
import multiprocessing
import threading
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import pytest
from PIL import Image

from relax.agentic.archive.client import Recorder
from relax.agentic.archive.directory import DirectoryTransport
from relax.agentic.archive.export import convert_sft, export_archive
from relax.agentic.archive.schema import encode
from relax.agentic.archive.server import make_server
from relax.agentic.archive.store import ArchiveStore


def png(color="red"):
    buf = BytesIO()
    Image.new("RGB", (32, 32), color).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def metadata(task="task", **kwargs):
    return dict(
        run_id="run",
        session_id="session",
        task_source="browser-mock",
        task_version="v1",
        task_id=task,
        instruction="Open settings",
        task_config={"target": "settings"},
        model="mock",
        policy_version="v1",
        agent_config={},
        environment_version="mock-browser-v2",
        started_at=1,
        **kwargs,
    )


def candidate(store, task="task", path="settings", *, status="success", errors=0, image=None):
    tid = store.begin(metadata(task))

    def put(kind, **data):
        store.append(tid, {"kind": kind, "data": data})

    put(
        "observation",
        observation_id="o0",
        step=0,
        time=1,
        screenshot=image or png(),
        valid=True,
        fields={"dom": "<button>Settings</button>"},
    )
    request = {
        "messages": [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": image or png()}}]},
            {"role": "assistant", "content": "prior history"},
            {"role": "user", "content": "Continue"},
        ],
        "tools": [{"type": "function", "function": {"name": "click"}}],
        "chat_template_kwargs": {"enable_thinking": True},
    }
    put(
        "model_call",
        call_id="c0",
        step=0,
        transport_retry=0,
        parse_retry=0,
        request=request,
        observation_ids=["o0"],
        parse_status="parsed",
        selected=True,
    )
    reply = {"role": "assistant", "content": "click settings"}
    put("call_result", call_id="c0", response=reply, assistant_message=reply)
    for index in range(1 + errors):
        put(
            "action",
            action_id=f"a{index}",
            call_id="c0",
            step=0,
            order=index,
            type="browser_element",
            payload={"selector": path, "parameters": {"button": "left"}},
            before_observation_id="o0",
        )
        put(
            "action_result",
            action_id=f"a{index}",
            after_observation_id="o0",
            status="failure" if index < errors else "success",
            error="click failed" if index < errors else None,
        )
    outcome = dict(
        status=status,
        raw_score=1,
        basis="mock DOM assertion",
        evaluation_error=None,
        ended_at=2,
        termination_reason="done",
        final_observation_id="o0",
    )
    return tid, outcome


class DirectStore:
    """Exercise complete candidates over the same transport used by Agents."""

    def __init__(self, root, **kwargs):
        self.transport = DirectoryTransport(root, **kwargs)

    def begin(self, meta):
        return self.transport("begin", {"metadata": meta})["trajectory_id"]

    def append(self, tid, event):
        self.transport("event", {"trajectory_id": tid, "event": event})

    def finish(self, tid, outcome):
        return self.transport("finish", {"trajectory_id": tid, "outcome": outcome})


def _direct_producer(root, index):
    store = DirectStore(root, max_bytes=100_000, per_task_limit=1, task_limits={"task": 2})
    tid, outcome = candidate(store, path=f"button-{index}")
    store.finish(tid, outcome)


def test_archive_direct_processes_share_quota_assets_and_export(tmp_path):
    root = tmp_path / "sessions"
    processes = [
        multiprocessing.get_context("spawn").Process(target=_direct_producer, args=(root, i)) for i in range(4)
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(30)
            assert process.exitcode == 0
        store = ArchiveStore(root, allow_shared_filesystem=True)
        try:
            assert len(store.retained()) == 2  # Override, not one quota per producer.
            assert store.task_limits == {"task": 2}
            assert len(list((root / "assets/task").iterdir())) == 2
            assert len(list((root / "assets").rglob("*.png"))) == 4
            assert not list((root / "staging").iterdir())
            assert store.usage() <= 100_000
            manifest = export_archive(store, tmp_path / "export")
            assert manifest["trajectories_selected"] == manifest["sft_samples"] == 2
        finally:
            store.close()
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(5)


def _direct_pending_producer(root, index, barrier):
    recorder = Recorder("", metadata(str(index)), transport=DirectoryTransport(root, 6000))
    recorder.event("incomplete", reason="x" * 3000)
    barrier.wait(timeout=20)  # Leave all successful candidates in progress together.


def test_archive_direct_capacity_counts_all_processes_in_progress(tmp_path):
    root = tmp_path / "sessions"
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(3)
    processes = [context.Process(target=_direct_pending_producer, args=(root, i, barrier)) for i in range(3)]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(30)
            assert process.exitcode == 0
        store = ArchiveStore(root, allow_shared_filesystem=True)
        try:
            assert store.usage() <= 6000
            assert len(list((root / "staging").iterdir())) == 1
            assert store.recover() == 1
        finally:
            store.close()
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(5)


def test_archive_direct_lock_timeout_leaves_other_active_candidate(store):
    active = store.begin(metadata())
    transport = DirectoryTransport(store.root, 1_000_000, lock_timeout=0.01)
    recorder = Recorder("", metadata("other"), transport=transport)
    assert recorder.trajectory_id is None
    assert "owned by another process" in recorder.reason
    assert store._path(active).exists()


def test_archive_direct_shared_filesystem_opt_in_and_fixed_task_config(tmp_path, monkeypatch):
    from relax.agentic.archive import store as module

    monkeypatch.setattr(module, "filesystem_type", lambda p: "nfs")
    root = tmp_path / "shared"
    transport = DirectoryTransport(root, 10000, task_limits={"disabled": 0})
    recorder = Recorder("", metadata("disabled"), transport=transport)
    assert recorder.trajectory_id is None and "task disabled" in recorder.reason
    other = Recorder("", metadata("enabled"), transport=transport)
    assert other.trajectory_id is not None
    conflicting = Recorder("", metadata(), transport=DirectoryTransport(root, 10000))
    assert conflicting.trajectory_id is None and "task limits differ" in conflicting.reason
    other.abort()


def test_archive_task_images_follow_observation_and_call_order(tmp_path):
    from urllib.parse import quote

    root = tmp_path / "archive"
    store = ArchiveStore(root, 1_000_000, new_asset_layout="task")
    try:
        task = "task /中文"
        identity = store.begin(metadata(task))
        for i in range(3):
            store.append(
                identity,
                {
                    "kind": "observation",
                    "data": {
                        "observation_id": f"o{i}",
                        "step": 0,
                        "screenshot": png(),
                        "valid": True,
                    },
                },
            )
        jpeg = BytesIO()
        Image.new("RGB", (8, 8), "blue").save(jpeg, format="JPEG")
        jpeg_uri = "data:image/png;base64," + base64.b64encode(jpeg.getvalue()).decode()
        for i in range(2):
            store.append(
                identity,
                {
                    "kind": "model_call",
                    "data": {
                        "call_id": f"c{i}",
                        "step": 0,
                        "request": {
                            "messages": [
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "image_url", "image_url": {"url": png()}},
                                        {"type": "image_url", "image_url": {"url": jpeg_uri}},
                                    ],
                                }
                            ]
                        },
                    },
                },
            )
        images = root / "assets" / quote(task, safe="") / identity
        shots = sorted((images / "screenshots").glob("*.png"))
        assert [p.name.split("-", 1)[0] for p in shots] == ["000000", "000001", "000002"]
        for call in ("000000", "000001"):
            files = sorted((images / "model_inputs" / call).iterdir())
            assert [p.name.split("-", 1)[0] for p in files] == ["000000", "000001"]
            assert [p.suffix for p in files] == [".png", ".jpg"]
            assert files[1].read_bytes() == jpeg.getvalue()
        store.verify_assets(store._events(store._path(identity)))
        store.abort(identity)
        assert not list((root / "assets").iterdir())
    finally:
        store.close()


def test_archive_task_layout_deduplicates_content_and_removes_evicted_images(tmp_path):
    store = ArchiveStore(tmp_path / "archive", 1_000_000, 1, new_asset_layout="task")
    try:
        first, outcome = candidate(store, errors=1)
        assert store.finish(first, outcome)["retained"]
        duplicate, outcome = candidate(store, errors=1)
        assert store.finish(duplicate, outcome)["reason"] == "duplicate_path"
        assert not (store.root / "assets/task" / duplicate).exists()
        better, outcome = candidate(store, path="shorter")
        assert store.finish(better, outcome)["retained"]
        assert not (store.root / "assets/task" / first).exists()
        assert (store.root / "assets/task" / better).is_dir()
        store.verify_assets(store.retained()[0])
        assert store.usage() <= store.max_bytes
    finally:
        store.close()


def test_archive_directory_keeps_existing_flat_layout(tmp_path):
    root = tmp_path / "archive"
    old = ArchiveStore(root, 1_000_000)
    old.close()
    direct = DirectStore(root, max_bytes=1_000_000)
    identity, outcome = candidate(direct)
    assert direct.finish(identity, outcome)["retained"]
    with (root / "archive.json").open() as handle:
        assert json.load(handle).get("asset_layout", "flat") == "flat"
    assert len(list((root / "assets").iterdir())) == 1
    assert next((root / "assets").iterdir()).is_file()


@pytest.mark.parametrize(
    "key",
    ["../escape", "/absolute", "task/../image.png", "a/" + "0" * 32 + "/screenshots/000000-" + "0" * 64 + ".png/.."],
)
def test_archive_asset_references_reject_escaping_paths(store, key):
    with pytest.raises(ValueError, match="invalid asset reference"):
        store.verify_assets({"image": "asset://" + key})


@pytest.fixture
def store(tmp_path):
    archive = ArchiveStore(tmp_path / "archive", 1_000_000, 5)
    yield archive
    archive.close()


@pytest.mark.parametrize(
    "status,expected", [("success", True), ("failure", False), ("partial", False), ("unknown", False)]
)
def test_archive_accepts_only_complete_confirmed_success(store, status, expected):
    tid, outcome = candidate(store, status=status)
    assert store.finish(tid, outcome)["retained"] is expected
    assert len(store.retained()) == int(expected)
    assert not list((store.root / "staging").iterdir())
    assert len(list((store.root / "assets").iterdir())) == int(expected)


@pytest.mark.parametrize("fault", ["missing_image", "placeholder", "missing_result", "evaluation", "broken_link"])
def test_archive_rejects_incomplete_evidence(store, fault):
    tid, outcome = candidate(store)
    path = store._path(tid)
    events = store._events(path)
    if fault == "missing_image":
        next((store.root / "assets").iterdir()).unlink()
    elif fault == "placeholder":
        events[1]["data"]["valid"] = False
    elif fault == "missing_result":
        events = [e for e in events if e["kind"] != "call_result"]
    elif fault == "evaluation":
        outcome["evaluation_error"] = "timeout"
    else:
        events[-1]["data"]["after_observation_id"] = "absent"
    path.write_bytes(b"".join(encode(e) + b"\n" for e in events))
    assert not store.finish(tid, outcome)["retained"]
    assert not store.retained()


def test_archive_deduplicates_behavior_ignoring_ids_time_and_response(store):
    tid, outcome = candidate(store)
    assert store.finish(tid, outcome)["retained"]
    again, outcome = candidate(store)
    events = store._events(store._path(again))
    events[0]["data"].update(session_id="different", started_at=999)
    events[3]["data"]["response"] = "different reasoning, same action"
    store._path(again).write_bytes(b"".join(encode(e) + b"\n" for e in events))
    assert store.finish(again, outcome)["reason"] == "duplicate_path"
    assert len(store.retained()) == 1
    assert len(list((store.root / "assets").iterdir())) == 1


def test_archive_task_quota_quality_and_shared_assets(tmp_path):
    store = ArchiveStore(tmp_path / "archive", 1_000_000, 1)
    try:
        tid, outcome = candidate(store, errors=2)
        assert store.finish(tid, outcome)["retained"]
        better, outcome = candidate(store, path="short")
        assert store.finish(better, outcome)["retained"]
        assert [t["trajectory_id"] for t in store.retained()] == [better]
        assert len(list((store.root / "assets").iterdir())) == 1
        other, outcome = candidate(store, task="other")
        assert store.finish(other, outcome)["retained"]
        store._delete(store.retained()[0])
        store.verify_assets(store.retained()[0])
    finally:
        store.close()


def test_archive_capacity_includes_concurrent_staging_and_preserves_coverage(tmp_path):
    store = ArchiveStore(tmp_path / "archive", 8000)
    try:
        tid, outcome = candidate(store)
        assert store.finish(tid, outcome)["retained"]
        active = store.begin(metadata("other"))
        with pytest.raises(ValueError, match="capacity_exhausted"):
            store.append(active, {"kind": "incomplete", "data": {"reason": "x" * 8000}})
        assert store.usage() <= 8000
        assert len(store.retained()) == 1
        active = store.begin(metadata("third"))
        store.append(active, {"kind": "incomplete", "data": {"reason": "x" * 3500}})
        with pytest.raises(ValueError, match="capacity_exhausted"):
            other = store.begin(metadata("fourth"))
            store.append(other, {"kind": "incomplete", "data": {"reason": "x" * 3500}})
        assert store._path(active).exists()  # Never evict active staging.
        assert store.usage() <= 8000
    finally:
        store.close()


def test_archive_evicts_extras_before_rejecting_new_task(store):
    for path in ("one", "two"):
        tid, outcome = candidate(store, path=path)
        assert store.finish(tid, outcome)["retained"]
    used = store.usage()
    store.max_bytes = used + 300
    new = store.begin(metadata("new-task"))
    assert len(store.retained()) == 1
    assert store.usage() <= store.max_bytes
    store.abort(new)


def _crash(root):
    import os

    store = ArchiveStore(root, 10000)
    store.begin(metadata())
    os._exit(0)


def test_archive_recovery_requires_exclusive_owner_and_leaves_live_staging(tmp_path):
    root = tmp_path / "archive"
    process = multiprocessing.get_context("fork").Process(target=_crash, args=(root,))
    process.start()
    process.join(10)
    assert process.exitcode == 0
    store = ArchiveStore(root)
    try:
        assert len(list((root / "staging").iterdir())) == 1
        with pytest.raises(RuntimeError, match="owned"):
            ArchiveStore(root)
        assert store.recover() == 1
        assert store.retained() == []
    finally:
        store.close()


def test_archive_rejects_shared_fs_and_unowned_paths(tmp_path, monkeypatch):
    from relax.agentic.archive import store as module

    directory = tmp_path / "user-data"
    directory.mkdir()
    precious = directory / "keep.txt"
    precious.write_text("keep")
    with pytest.raises(ValueError, match="dedicated"):
        ArchiveStore(directory, 10000)
    monkeypatch.setattr(module, "filesystem_type", lambda p: "nfs")
    with pytest.raises(ValueError, match="local filesystem"):
        ArchiveStore(tmp_path / "new", 10000)
    assert precious.read_text() == "keep"
    assert not (tmp_path / "new").exists()


def test_archive_export_has_self_contained_images_and_target_only_masks(store, tmp_path):
    tid, outcome = candidate(store)
    assert store.finish(tid, outcome)["retained"]
    pending = store.begin(metadata("pending"))
    target = tmp_path / "export"
    manifest = export_archive(store, target)
    assert manifest["trajectories_selected"] == manifest["sft_samples"] == 1
    assert manifest["incomplete_staging_excluded"] == 1
    row = json.loads((target / "samples.jsonl").read_text())
    assert [m["learn"] for m in row["messages"]] == [False, False, False, True]
    url = row["messages"][0]["content"][0]["image_url"]["url"]
    assert (target / url).is_file()
    assert row["metadata"]["apply_chat_template_kwargs"] == {"enable_thinking": True}
    assert row["tools"][0]["function"]["name"] == "click"
    convert_sft(target / "samples.jsonl", target / "sft.local.jsonl")
    local = json.loads((target / "sft.local.jsonl").read_text())
    assert local["messages"][0]["content"][0]["image_url"]["url"] == str(target / url)
    with pytest.raises(FileExistsError):
        export_archive(store, target)
    with pytest.raises(FileExistsError):
        convert_sft(target / "samples.jsonl", target / "sft.local.jsonl")
    store.abort(pending)


def test_archive_export_reports_missing_and_skips_corrective_targets(store, tmp_path):
    tid, outcome = candidate(store, errors=1)
    assert store.finish(tid, outcome)["retained"]
    manifest = export_archive(store, tmp_path / "corrected")
    assert manifest["trajectories_selected"] == 1 and manifest["sft_samples"] == 0
    assert manifest["sft_skipped"]["execution_failure"] == 1
    next((store.root / "assets").iterdir()).unlink()
    manifest = export_archive(store, tmp_path / "missing")
    assert manifest["missing"] == 1 and manifest["trajectories_selected"] == 0


def test_archive_http_serializes_concurrent_clients_and_fail_open(store, monkeypatch):
    server = make_server(store, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:

        def write(i):
            recorder = Recorder(url, metadata(str(i)))
            recorder.event("observation", observation_id="o", step=0, screenshot=png(), valid=True)
            return recorder

        with ThreadPoolExecutor(max_workers=5) as pool:
            recorders = list(pool.map(write, range(10)))
        assert all(r.trajectory_id for r in recorders)
        assert len(list((store.root / "staging").iterdir())) == 10
        assert len(list((store.root / "assets").iterdir())) == 1
        assert store.usage() <= store.max_bytes
        original = store._append

        def disk_full(*args, **kwargs):
            raise OSError("No space left on device")

        monkeypatch.setattr(store, "_append", disk_full)
        recorders[0].event("incomplete", reason="still training")
        assert recorders[0].trajectory_id is None
        assert "No space" in recorders[0].reason
        monkeypatch.setattr(store, "_append", original)
        for recorder in recorders:
            recorder.abort()
        assert not list((store.root / "assets").iterdir())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def test_archive_partial_json_never_exports_and_unknown_image_sources_abort(store):
    tid = store.begin(metadata())
    with store._path(tid).open("ab") as handle:
        handle.write(b'{"partial":')
    assert store.retained() == []
    store.recover()
    tid = store.begin(metadata())
    with pytest.raises(ValueError, match="not captured"):
        store.append(
            tid,
            {
                "kind": "model_call",
                "data": {
                    "request": {
                        "messages": [{"type": "image_url", "image_url": {"url": "https://expired.invalid/image.png"}}]
                    }
                },
            },
        )
    assert not store._path(tid).exists()


def test_archive_export_loads_with_existing_sft_loader(store, tmp_path):
    from relax.engine.sft.dataset.streaming import _build_canonical_sample_from_row

    tid, outcome = candidate(store)
    assert store.finish(tid, outcome)["retained"]
    target = tmp_path / "sft-export"
    export_archive(store, target)
    convert_sft(target / "samples.jsonl", target / "local.jsonl")
    row = json.loads((target / "local.jsonl").read_text())
    sample = _build_canonical_sample_from_row(
        row,
        row_index=0,
        prompt_key="messages",
        label_key=None,
        multimodal_keys={"image": "images"},
        metadata_key="metadata",
        tool_key="tools",
        system_prompt=None,
        require_response=True,
        source_name=str(target / "local.jsonl"),
    )
    assert [message.learn for message in sample.messages] == [False, False, False, True]
    assert sample.messages[1].content == [{"type": "text", "text": "prior history"}]
    assert sample.messages[-1].content == [{"type": "text", "text": "click settings"}]
    assert len(sample.images) == 1
    assert sample.tools == row["tools"]
    assert sample.metadata["apply_chat_template_kwargs"] == {"enable_thinking": True}


def _remote_producer(url, task):
    recorder = Recorder(url, metadata(task))
    recorder.event("observation", observation_id="o", step=0, screenshot=png(), valid=True)
    assert recorder.trajectory_id is not None


def test_archive_http_multiple_processes_share_one_budget_and_asset(store):
    server = make_server(store, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    processes = [
        multiprocessing.get_context("spawn").Process(target=_remote_producer, args=(url, str(i))) for i in range(3)
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(20)
            assert process.exitcode == 0
        assert len(list((store.root / "staging").iterdir())) == 3
        assert len(list((store.root / "assets").iterdir())) == 1
        assert store.usage() <= store.max_bytes
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(5)
        server.shutdown()
        server.server_close()
        thread.join(5)


def test_archive_ranking_prefers_fewer_known_errors_to_shorter_unknown_path():
    from relax.agentic.archive.schema import quality

    clean = {
        "actions": [{"status": "unknown", "step": i} for i in range(3)],
        "model_calls": [1, 2, 3],
        "trajectory_id": "clean",
    }
    failed = {
        "actions": [{"status": "failure", "step": 0}],
        "model_calls": [1],
        "trajectory_id": "failed",
    }
    assert quality(clean) < quality(failed)


def test_archive_pending_asset_survives_eviction_during_reservation(store):
    first, outcome = candidate(store, path="keep")
    assert store.finish(first, outcome)["retained"]
    victim, outcome = candidate(store, path="evict", errors=1, image=png("blue"))
    assert store.finish(victim, outcome)["retained"]
    active = store.begin(metadata("new-task"))
    store.max_bytes = store.usage() + 1
    store.append(active, {"kind": "observation", "data": {"screenshot": png("blue")}})
    assert [t["trajectory_id"] for t in store.retained()] == [first]
    assert len(list((store.root / "assets").iterdir())) == 2
    store.verify_assets(store._events(store._path(active)))
    assert store.usage() <= store.max_bytes
    store.abort(active)
    assert len(list((store.root / "assets").iterdir())) == 1


def test_archive_partial_asset_write_is_fail_open_and_keeps_shared_assets(store, monkeypatch):
    from relax.agentic.archive import store as store_module

    kept, outcome = candidate(store)
    assert store.finish(kept, outcome)["retained"]

    def rpc(self, operation, payload):
        if operation == "begin":
            return {"trajectory_id": store.begin(payload["metadata"])}
        if operation == "event":
            store.append(payload["trajectory_id"], payload["event"])
        else:
            store.abort(payload["trajectory_id"])
        return {"ok": True}

    monkeypatch.setattr(Recorder, "_rpc", rpc)
    recorder = Recorder("http://mock-only", metadata("candidate"))

    def interrupted_fsync(fd):
        raise OSError("injected disk failure after asset write")

    monkeypatch.setattr(store_module.os, "fsync", interrupted_fsync)
    recorder.event("observation", screenshot=png("blue"))
    assert recorder.trajectory_id is None
    assert "injected disk failure" in recorder.reason
    assert not list((store.root / "staging").iterdir())
    assert len(list((store.root / "assets").iterdir())) == 1
    store.verify_assets(store.retained()[0])


@pytest.mark.parametrize("fault", ["commit_missing", "malformed_record"])
def test_archive_export_skips_damaged_records_with_reason(store, tmp_path, fault):
    tid, outcome = candidate(store)
    assert store.finish(tid, outcome)["retained"]
    path = store._path(tid, "trajectories")
    events = store._events(path)
    if fault == "commit_missing":
        events = [event for event in events if event["kind"] != "commit"]
    else:
        events[1]["data"] = None
    path.write_bytes(b"".join(encode(event) + b"\n" for event in events))
    manifest = export_archive(store, tmp_path / "damaged-export")
    assert manifest["trajectories_selected"] == 0
    assert manifest["trajectories_skipped"] == 1
    assert sum(manifest["skip_reasons"].values()) == 1
    assert (tmp_path / "damaged-export" / "manifest.json").exists()


def test_archive_cli_export_requires_snapshot_lock_and_converts_sft(store, tmp_path):
    import subprocess
    import sys

    tid, outcome = candidate(store)
    assert store.finish(tid, outcome)["retained"]
    output = tmp_path / "cli-export"
    command = [
        sys.executable,
        "-m",
        "relax.agentic.archive.export",
        "export",
        "--archive-dir",
        str(store.root),
        "--output-dir",
        str(output),
    ]
    blocked = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert blocked.returncode != 0 and "owned by another process" in blocked.stderr
    assert not output.exists()
    store.close()
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["trajectories_selected"] == manifest["sft_samples"] == 1
    subprocess.run(
        [
            sys.executable,
            "-m",
            "relax.agentic.archive.export",
            "convert-sft",
            "--samples",
            str(output / "samples.jsonl"),
            "--output",
            str(output / "local.jsonl"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    sample = json.loads((output / "local.jsonl").read_text())
    assert sample["messages"][0]["content"][0]["image_url"]["url"].startswith(str(output))
