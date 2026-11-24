"""Exercise the standalone publication wait loop without starting a GPU
engine."""

import io
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from relax.distributed.weight_sync import cli_consume


@pytest.mark.parametrize("wait_seconds", [5, 100000])
def test_each_target_gets_its_own_publication_wait(tmp_path, monkeypatch, wait_seconds):
    clock, queries, applied = [0.0], {}, []
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    configuration = {
        "start_version": 0,
        "end_version": 1,
        "wait_seconds": wait_seconds,
        "model_path": str(model),
        "checkpoint_root": str(tmp_path / "checkpoints"),
        "checkpoint_budget": 1024,
        "session_directory": str(tmp_path / "session"),
        "consumer": "test",
        "stream": "s",
        "transport": {"type": "tcp", "host": "unused", "port": 1},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(configuration))

    def versions(stream, after, limit):
        version = after + 1
        queries[version] = queries.get(version, 0) + 1
        if queries[version] == 1:
            clock[0] += 3
            return []
        return [{"version": version}]

    transport = SimpleNamespace(versions=versions, consumer_status=lambda name: {"cursor": -1})

    def apply(version, **kwargs):
        applied.append(version)
        clock[0] += 3
        return {"version": version}

    runner = SimpleNamespace(transport=transport, stream="s", mode="all_versions", name="test", apply=apply)
    engine = Mock()
    monkeypatch.setattr(cli_consume, "time", SimpleNamespace(monotonic=lambda: clock[0], sleep=lambda n: None))
    monkeypatch.setattr(cli_consume, "sys", SimpleNamespace(stdout=SimpleNamespace(buffer=io.BytesIO())))
    monkeypatch.setattr(cli_consume, "TCPTransport", lambda *args, **kw: transport)
    for name in ("VersionConsumer", "EmbeddedSGLangGroup", "InstallSession"):
        monkeypatch.setattr(cli_consume, name, Mock())
    monkeypatch.setattr(cli_consume, "ConsumptionRunner", lambda *args, **kw: runner)
    monkeypatch.setattr(cli_consume, "create_engine", lambda **kw: engine)
    cli_consume.consume_versions(SimpleNamespace(root=tmp_path), path)
    assert applied == [0, 1]
    assert queries == {0: 2, 1: 2}
    engine.shutdown.assert_called_once()
