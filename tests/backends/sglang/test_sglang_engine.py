# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    ("enable_mtp_training", "speculative_algorithm", "overrides", "expected"),
    [
        (True, "EAGLE", None, False),
        (False, None, None, False),
        (False, "EAGLE", None, True),
        (False, None, {"speculative_algorithm": "EAGLE"}, True),
        (False, "EAGLE", {"speculative_algorithm": None}, False),
    ],
)
def test_draft_weights_cpu_backup_follows_mtp_and_speculative_config(
    enable_mtp_training, speculative_algorithm, overrides, expected
):
    pytest.importorskip("sglang.srt.server_args", exc_type=ImportError)

    from relax.backends.sglang.sglang_engine import _enable_draft_weights_cpu_backup

    args = SimpleNamespace(
        enable_mtp_training=enable_mtp_training,
        sglang_speculative_algorithm=speculative_algorithm,
    )

    assert _enable_draft_weights_cpu_backup(args, overrides) is expected


def test_managed_startup_health_does_not_flush_cache(monkeypatch):
    pytest.importorskip("sglang.srt.server_args", exc_type=ImportError)
    from relax.backends.sglang import sglang_engine as module

    calls = []

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url, **kwargs):
            calls.append(url)
            assert not url.endswith("/flush_cache")
            return SimpleNamespace(status_code=200)

    monkeypatch.setattr(module.requests, "Session", Client)
    module._wait_server_healthy("http://engine", None, lambda: True, timeout=1, flush_cache=False)
    assert calls == ["http://engine/health_generate"]


@pytest.mark.parametrize("state", ["zombie", "dead", "gone", "running", "sleeping", "stopped", "disk-sleep", "denied"])
def test_managed_shutdown_distinguishes_exited_processes_with_one_deadline(monkeypatch, state):
    pytest.importorskip("sglang.srt.server_args", exc_type=ImportError)
    import threading

    import psutil

    from relax.backends.sglang import sglang_engine as module

    killed, joined = [], []
    clock = iter([0, 2, 11])
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: next(clock), sleep=lambda _: None))

    def status():
        if state == "denied":
            raise psutil.AccessDenied(1001)
        return state

    root = SimpleNamespace(pid=1000, kill=lambda: killed.append(1000), is_running=lambda: False)
    child = SimpleNamespace(
        pid=1001, kill=lambda: killed.append(1001), is_running=lambda: state != "gone", status=status
    )
    engine = SimpleNamespace(
        args=SimpleNamespace(rollout_external=False, _lora_publication_launch=True),
        _lora_closing=threading.Event(),
        _lora_shutdown_children=[root, child],
        process=SimpleNamespace(join=lambda timeout: joined.append(timeout)),
    )
    if state == "denied":
        with pytest.raises(psutil.AccessDenied):
            module.SGLangEngine.shutdown(engine)
    elif state in ("zombie", "dead", "gone"):
        assert module.SGLangEngine.shutdown(engine) == {"processes_exited": True}
    else:
        with pytest.raises(RuntimeError, match=f"1001.*{state}"):
            module.SGLangEngine.shutdown(engine)
    assert engine._lora_closing.is_set()
    assert killed == [1001, 1000] and joined == [8]
