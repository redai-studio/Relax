# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib.util
import json
import sys
from io import BytesIO
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from PIL import Image

from examples.osworld_agentic.app import sandbox_health as health


def event(kind, t=0, session="s", attempt=1, **fields):
    return dict(
        run_id="run",
        session_id=session,
        event_id=f"{session}:{kind}:{t}:{attempt}",
        time=t,
        attempt=attempt,
        kind=kind,
        **fields,
    )


def snapshot(events, query=None, now=300):
    agg = health.Aggregate("run")
    for item in events:
        agg.add(item)
        agg.add(dict(item))  # Network/file replay is harmless.
    return agg.snapshot(now, grace=120, stale=90, query=query)


def row(result, name):
    return next(r for r in result["summary"] if r["metric"] == f"sandbox/{name}")


def test_session_rates_include_failures_without_exports_and_global_p95():
    events = []
    for sid, attempt, duration in (("first", 1, 2), ("retry", 2, 12)):
        events += [
            event("ready", duration, sid, attempt, ready_seconds=duration),
            event("init_done", duration + 1, sid, attempt, outcome="ready"),
            event("session_end", 20, sid, outcome="completed"),
        ]
    events += [
        event("attempt_failed", 1, "retry"),
        event("attempt_failed", 1, "failed"),
        event("init_done", 19, "failed", 3, outcome="exhausted"),
        event("attempt_started", 21, "pending", 2),
    ]
    result = snapshot(events)
    assert row(result, "first_ready_rate")["value"] == 1 / 3
    assert row(result, "retry_recovery_rate")["value"] == 1 / 2
    assert row(result, "init_failure_rate")["value"] == 1 / 3
    assert row(result, "ready_seconds_p95")["value"] == 12
    assert row(result, "session_accounting")["init_incomplete"] == 1


@pytest.mark.parametrize("answer", ["exists", "unknown", "released", None, "raises"])
def test_cleanup_requires_remote_evidence_and_retains_each_instance(answer):
    def query(sid):
        assert sid in {"old", "new"}
        if answer == "raises":
            raise OSError("query failed")
        return answer

    events = [
        event("created", 0, sandbox_id="old"),
        event("closing", 1, sandbox_id="old"),
        event("close_error", 2, sandbox_id="old", error_category="cleanup_error"),
        event("created", 5, attempt=2, sandbox_id="new"),
        event("closing", 6, attempt=2, sandbox_id="new"),
        event("close_returned", 7, attempt=2, sandbox_id="new"),
    ]
    result = snapshot(events, query if answer else None)
    expected = 0 if answer == "released" else 2
    assert row(result, "cleanup_unconfirmed")["value"] == expected
    if expected:
        assert {r["sandbox_id"] for r in result["instances"] if r["sandbox_id"]} == {"old", "new"}
        assert next(r for r in result["instances"] if r["sandbox_id"] == "old")["error_category"] == "cleanup_error"
    assert row(snapshot(events, query if answer else None, now=10), "cleanup_unconfirmed").get("unconfirmed", 0) == 0


def test_no_data_timeout_cancellation_and_unknown_allocation_are_explicit():
    events = [
        event("creating", creation_id="lost"),
        event("attempt_failed", 10, error_category="init_timeout"),
        event("session_end", 11, outcome="cancelled"),
    ]
    result = snapshot(events)
    assert row(result, "first_ready_rate")["value"] is None
    assert row(result, "retry_recovery_rate")["value"] is None
    assert row(result, "ready_seconds_p95")["value"] is None
    assert row(result, "cleanup_unconfirmed")["value"] is None
    accounting = row(result, "session_accounting")
    assert accounting["cancelled"] == accounting["init_timeout"] == accounting["unknown_identity"] == 1
    assert row(result, "init_failure_rate")["denominator"] == 0


def test_runtime_action_error_is_not_confirmed_environment_failure():
    events = [
        event("ready", 1, ready_seconds=1),
        event("init_done", 2, outcome="ready"),
        event("runtime_error", 3, error_category="unknown"),
        event("session_end", 4, outcome="process_exit"),
    ]
    result = snapshot(events)
    assert row(result, "runtime_failure_rate")["value"] == 0
    assert row(result, "runtime_failure_rate")["status"] == "partial_unknown"
    assert row(result, "ready_seconds_p95")["value"] is None
    events.append(event("runtime_failure", 3.5))
    assert row(snapshot(events), "runtime_failure_rate")["value"] == 1


def test_confirmed_disappearance_while_ready_counts_only_when_session_ends():
    agg = health.Aggregate("run")
    agg.add(event("ready", 1, ready_seconds=1, sandbox_id="owned"))
    agg.add(event("init_done", 2, outcome="ready"))
    assert row(agg.snapshot(3, 120, 90, lambda sid: "released"), "runtime_failure_rate")["value"] is None
    agg.add(event("session_end", 4, outcome="completed"))
    assert row(agg.snapshot(5, 120, 90), "runtime_failure_rate")["value"] == 1


def test_incomplete_writer_tail_and_other_run_are_not_counted(tmp_path):
    path = tmp_path / "writer.jsonl"
    first = event("attempt_started")
    path.write_text(json.dumps(first) + "\n" + '{"run_id":')
    agg = health.Aggregate("run")
    agg.read(tmp_path)
    agg.read(tmp_path)
    assert len(agg.events) == 1
    path.write_text(json.dumps(first) + "\n" + json.dumps(event("session_end", 1, outcome="process_exit")) + "\n")
    agg.read(tmp_path)
    agg.add(dict(first, run_id="another", event_id="foreign"))
    assert len(agg.events) == 2
    assert row(agg.snapshot(300, 120, 90), "session_accounting")["process_exit"] == 1
    assert any(r["stage"] == "process_exit" for r in agg.snapshot(300, 120, 90)["instances"])


def test_stale_process_is_unknown_not_closed_or_completed():
    result = snapshot([event("created", sandbox_id="alive-or-dead")])
    assert result["instances"][0]["stage"] == "unknown"
    assert row(result, "init_failure_rate")["denominator"] == 0


def test_observer_captures_identity_before_constructor_failure_and_recreation():
    events = []
    writer = SimpleNamespace(emit=lambda kind, **kw: events.append((kind, kw)))
    ids = iter(["first", "replacement"])
    client = SimpleNamespace(create_sandbox=lambda: next(ids), delete_sandbox=lambda sid: None)
    health.observe_nex(writer, client)
    assert client.create_sandbox() == "first"
    client.delete_sandbox("first")
    assert client.create_sandbox() == "replacement"
    assert [e[1]["sandbox_id"] for e in events if e[0] == "created"] == ["first", "replacement"]
    assert all(e[0] != "closed" for e in events)


def load_env(monkeypatch):
    stub = ModuleType("desktop_env.desktop_env")
    stub.DesktopEnv = object
    monkeypatch.setitem(sys.modules, "desktop_env", ModuleType("desktop_env"))
    monkeypatch.setitem(sys.modules, "desktop_env.desktop_env", stub)
    name = "examples.osworld_agentic.app.env_osworld"
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parents[2] / "examples/osworld_agentic/app/env_osworld.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("failures", [0, 1, 3])
def test_actual_reset_loop_is_unchanged_and_backoff_is_in_ready_time(monkeypatch, failures):
    module = load_env(monkeypatch)
    events, calls = [], []
    clock = [0.0]
    writer = SimpleNamespace(attempt=0)
    writer.emit = lambda kind, **kw: events.append(event(kind, clock[0], attempt=writer.attempt, **kw))
    env = object.__new__(module.OSWorldEnv)
    env._health, env._health_screenshot_valid = writer, True
    env._env = SimpleNamespace(path_to_vm="owned")

    def reset():
        calls.append("reset")
        clock[0] += 2
        if calls.count("reset") <= failures:
            raise TimeoutError("mock init timeout")
        return "unchanged observation"

    env._reset_once = reset
    env._close_sync = lambda **kw: calls.append("close")
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(module.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    if failures == 3:
        with pytest.raises(RuntimeError, match="after 3 attempts"):
            env._reset_sync()
        assert row(snapshot(events), "init_failure_rate")["value"] == 1
    else:
        assert env._reset_sync() == "unchanged observation"
        ready = next(e for e in events if e["kind"] == "ready")
        assert ready["ready_seconds"] == (2 if failures == 0 else 9)
    assert calls.count("reset") == min(failures + 1, 3)


def test_placeholder_is_not_ready_evidence_and_disabled_has_no_journal(monkeypatch):
    monkeypatch.delenv("OSWORLD_SANDBOX_HEALTH", raising=False)
    monkeypatch.setattr(health, "Journal", lambda *a: pytest.fail("must not create writer"))
    assert health.journal() is None
    module = load_env(monkeypatch)
    assert not module._real_screenshot(None)
    assert not module._real_screenshot(b"broken")
    image = BytesIO()
    Image.new("RGB", (40, 40)).save(image, format="PNG")
    assert module._real_screenshot(image.getvalue())
    # Existing fallback stays available to the agent, but readiness checks raw obs first.
    assert module.OSWorldEnv._pack_obs({}, done=False, exec_ok=True, exec_result={}).screenshot_bytes


def test_clearml_table_marker_uses_existing_adapter_without_creating_task():
    from relax.utils.metrics.adapters.clearml import _ClearMLAdapter

    tables, scalars = [], []
    adapter = object.__new__(_ClearMLAdapter)
    adapter._get_logger = lambda: SimpleNamespace(
        report_table=lambda **kw: tables.append(kw), report_scalar=lambda **kw: scalars.append(kw)
    )
    adapter.log(
        {"sandbox/summary": {"__relax_table__": [{"value": None, "denominator": 0}]}, "sandbox/first_ready_rate": 0.5},
        30,
    )
    assert len(tables) == len(scalars) == 1
    assert tables[0]["table_plot"].iloc[0]["denominator"] == 0
    assert scalars[0]["iteration"] == 30


@pytest.mark.parametrize("terminate", [False, True])
def test_agent_supervisor_records_exit_without_any_training_export(tmp_path, terminate):
    import os
    import signal
    import subprocess
    import time

    app = tmp_path / "app"
    app.mkdir()
    (app / "__init__.py").write_text("")
    (app / "agent.py").write_text(
        "import os, time\n"
        "from examples.osworld_agentic.app.sandbox_health import emit, journal\n"
        "emit('creating', creation_id='partial')\n"
        "journal().flush(1)\n" + ("time.sleep(60)\n" if terminate else "os._exit(7)\n")
    )
    root = Path(__file__).resolve().parents[2]
    env = dict(
        os.environ,
        PYTHONPATH=f"{tmp_path}:{root}",
        OSWORLD_SANDBOX_HEALTH="1",
        OSWORLD_HEALTH_DIR=str(tmp_path / "events"),
        OSWORLD_RUN_TAG="run",
        RELAX_SESSION_ID="s",
    )
    child = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "examples.osworld_agentic.app.health_runner",
            "--agent",
            sys.executable,
            "-m",
            "app.agent",
        ],
        cwd=tmp_path,
        env=env,
        start_new_session=True,
    )
    try:
        if terminate:
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                if any('"kind": "creating"' in p.read_text() for p in (tmp_path / "events").glob("*.jsonl")):
                    break
                time.sleep(0.05)
            else:
                pytest.fail("mock agent did not start")
            os.killpg(child.pid, signal.SIGTERM)  # Same contract as the actual Relax launcher.
        code = child.wait(timeout=15)
        assert code == (143 if terminate else 7)
        agg = health.Aggregate("run")
        agg.read(tmp_path / "events")
        result = agg.snapshot(time.time(), 120, 90)
        assert row(result, "session_accounting")["cancelled" if terminate else "process_exit"] == 1
        assert row(result, "first_ready_rate")["denominator"] == 0
        assert row(result, "cleanup_unconfirmed")["value"] is None
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def test_monitoring_disabled_training_wrapper_starts_no_threads(monkeypatch):
    from examples.osworld_agentic.app import health_runner

    monkeypatch.delenv("OSWORLD_SANDBOX_HEALTH", raising=False)
    monkeypatch.setattr(sys, "argv", ["health_runner"])
    monkeypatch.setattr(health_runner.threading, "Thread", lambda *a, **kw: pytest.fail("monitor thread"))
    calls = []
    monkeypatch.setattr(health_runner.runpy, "run_module", lambda *a, **kw: calls.append(a))
    health_runner.main()
    assert calls == [("relax.entrypoints.train",)]


def test_monitor_without_rollout_keeps_local_snapshot_without_reporting(tmp_path, monkeypatch):
    import threading

    from examples.osworld_agentic.app import health_runner

    monkeypatch.setenv("OSWORLD_HEALTH_DIR", str(tmp_path))
    monkeypatch.setenv("OSWORLD_RUN_TAG", "run")
    (tmp_path / "agent.jsonl").write_text(json.dumps(event("init_done", 1, outcome="exhausted")) + "\n")
    stop = threading.Event()
    stop.set()
    monkeypatch.setattr(health_runner, "report", lambda *args: pytest.fail("periodic ClearML report"))
    monkeypatch.setattr(health_runner.time, "monotonic", iter([100.0, 2779.0]).__next__)
    health_runner.monitor(stop)
    result = json.loads((tmp_path / "snapshot.json").read_text())
    assert row(result, "init_failure_rate")["value"] == 1
    assert result["scalars"]["sandbox/elapsed_seconds"] == 2679.0


def test_real_reset_placeholder_does_not_change_training_behavior(monkeypatch):
    module = load_env(monkeypatch)
    env = object.__new__(module.OSWorldEnv)
    events = []
    writer = SimpleNamespace(attempt=0)
    writer.emit = lambda kind, **kw: events.append(event(kind, attempt=writer.attempt, **kw))
    env._health, env._health_screenshot_valid = writer, False
    env.task_config = {}
    env._env = SimpleNamespace(path_to_vm="owned", reset=lambda **kw: {})
    obs = env._reset_sync()
    assert obs.screenshot_bytes  # Still returns the original fallback; no new retry.
    result = snapshot(events)
    assert row(result, "first_ready_rate")["value"] == 0
    assert row(result, "init_failure_rate")["value"] == 0
    assert row(result, "session_accounting")["unhealthy_reset"] == 1
    assert row(result, "ready_seconds_p95")["value"] is None


def test_monitor_failure_does_not_replace_original_reset_exception(monkeypatch, tmp_path):
    # Storage failure is isolated in the writer; the adapter's three-attempt
    # failure remains the training-visible exception.
    broken = tmp_path / "not_a_directory"
    broken.write_text("file")
    writer = health.Journal(str(broken), "run", "s")
    module = load_env(monkeypatch)
    env = object.__new__(module.OSWorldEnv)
    env._health = writer
    env._reset_once = lambda: (_ for _ in ()).throw(ValueError("original reset error"))
    env._close_sync = lambda **kw: None
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="original reset error"):
        env._reset_sync()


def test_export_count_and_event_replay_do_not_weight_sessions():
    events = [event("ready", 1, ready_seconds=1), event("init_done", 2, outcome="ready")]
    baseline = snapshot(events)
    events += [event("export_record", i + 3) for i in range(20)]
    result = snapshot(events)
    for name in ("first_ready_rate", "retry_recovery_rate", "init_failure_rate"):
        assert row(result, name) == row(baseline, name)


def test_late_cleanup_event_invalidates_disappearance_as_runtime_failure():
    agg = health.Aggregate("run")
    agg.add(event("ready", 1, ready_seconds=1, sandbox_id="owned"))
    agg.add(event("init_done", 2, outcome="ready"))
    agg.snapshot(5, 120, 90, lambda sid: "released")
    # A queued closing event becomes visible after the query: it was not a
    # spontaneous disappearance. Cross-file arrival order must not change this.
    agg.add(event("closing", 4, sandbox_id="owned"))
    agg.add(event("session_end", 6, outcome="completed"))
    assert row(agg.snapshot(10, 120, 90), "runtime_failure_rate")["value"] == 0


@pytest.mark.parametrize("other_backend", [False, True])
def test_health_report_reuses_service_and_never_advances_other_backend_steps(monkeypatch, other_backend):
    from examples.osworld_agentic.app import health_runner

    calls = []
    client_module = ModuleType("relax.utils.metrics.client")
    client_module.MetricsClient = lambda url: SimpleNamespace(
        report_backend=lambda backend, step, payload: (
            calls.append((url, step, payload, backend)) or {"status": "success"}
        ),
    )
    adapter_module = ModuleType("relax.utils.metrics.metrics_service_adapter")
    adapter_module.get_metrics_service_adapter = lambda: SimpleNamespace(
        args=SimpleNamespace(use_wandb=other_backend, use_tensorboard=False),
        client=SimpleNamespace(service_url="http://mock-only/metrics"),
    )
    monkeypatch.setitem(sys.modules, client_module.__name__, client_module)
    monkeypatch.setitem(sys.modules, adapter_module.__name__, adapter_module)
    result = snapshot([event("init_done", 1, outcome="exhausted")])
    health_runner.report(result, 30)
    assert calls[0][3] == "clearml"
    assert calls[0][2]["sandbox/init_failure_rate"] == 1
    assert "sandbox/ready_seconds_p95" not in calls[0][2]
    assert "__relax_table__" in calls[0][2]["sandbox/summary"]


def test_monitor_flushes_when_stopped_before_first_interval(monkeypatch, tmp_path):
    import threading

    from examples.osworld_agentic.app import health_runner

    monkeypatch.setenv("OSWORLD_HEALTH_DIR", str(tmp_path))
    monkeypatch.setenv("OSWORLD_RUN_TAG", "run")
    stop = threading.Event()
    stop.set()
    reports = []
    monkeypatch.setattr(health_runner, "report", lambda snapshot, step: reports.append(snapshot))
    health_runner.monitor(stop)
    assert reports == []
    assert (tmp_path / "snapshot.json").exists()


def test_rollout_report_uses_rollout_step_and_retains_snapshot_on_failure(monkeypatch, tmp_path):
    from examples.osworld_agentic.app import health_runner

    monkeypatch.setenv("OSWORLD_SANDBOX_HEALTH", "1")
    monkeypatch.setenv("OSWORLD_HEALTH_DIR", str(tmp_path))
    result = snapshot([event("init_done", 1, outcome="exhausted")])
    result["scalars"]["sandbox/elapsed_seconds"] = 2679
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(result))
    calls = []
    monkeypatch.setattr(health_runner, "report", lambda data, step: calls.append((data, step)))
    health_runner.report_rollout(0)
    health_runner.report_rollout(1)
    assert [step for _, step in calls] == [0, 1]
    assert calls[0][0]["scalars"]["sandbox/elapsed_seconds"] == 2679

    def fail(*args):
        raise OSError("metrics unavailable")

    monkeypatch.setattr(health_runner, "report", fail)
    health_runner.report_rollout(2)
    assert json.loads(path.read_text()) == result
    path.unlink()
    health_runner.report_rollout(3)  # Missing snapshot must not interrupt training.
    monkeypatch.setenv("OSWORLD_SANDBOX_HEALTH", "0")
    monkeypatch.setattr(health_runner, "report", lambda *args: pytest.fail("disabled monitor"))
    health_runner.report_rollout(4)
