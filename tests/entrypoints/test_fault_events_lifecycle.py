# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Actual train module control flow with explicit CPU dependency
substitutes."""

import argparse
import importlib.util
import signal
import socket
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

import relax.utils
from relax.utils.fault_events import runtime as runtime_module
from relax.utils.fault_events.models import EventDraft
from tests.utils.test_arguments_opd_teacher_colocate import arguments_module as _arguments_module_fixture


ROOT = Path(__file__).resolve().parents[2]
arguments_module = _arguments_module_fixture


@pytest.fixture
def train_loader(monkeypatch):
    modules = []
    monkeypatch.delenv("RELAX_KERNEL_CACHE_DIR", raising=False)
    monkeypatch.setattr(relax.utils, "try_import_telemetry_hook", lambda: None)

    def load(*, fail_at=None, on_train=None, metrics_enabled=False):
        events, timers = [], []
        initialized = False
        serve = ModuleType("ray.serve")
        serve.start = lambda **_kwargs: events.append("serve_start")
        serve.shutdown = lambda: events.append("serve_shutdown")
        serve.get_app_handle = Mock()
        finish_tracking = serve.get_app_handle.return_value.finish_tracking.remote.return_value.result
        finish_tracking.side_effect = lambda **_kwargs: events.append("finish_tracking") or {"status": "success"}
        ray = ModuleType("ray")
        ray.serve = serve

        def init(**_kwargs):
            nonlocal initialized
            initialized = True
            events.append("ray_init")

        def shutdown():
            nonlocal initialized
            initialized = False
            events.append("ray_shutdown")

        ray.init = init
        ray.shutdown = shutdown
        ray.is_initialized = lambda: initialized

        class CPUController:
            def __init__(self, config, runtime_env):
                events.append("controller_init")
                assert all(not isinstance(value, runtime_module.FaultEventRuntime) for value in vars(config).values())
                self.config, self.runtime_env = config, runtime_env
                self._metrics_service_enabled = metrics_enabled
                if fail_at == "startup":
                    raise RuntimeError("controller startup failed")

            def training_loop(self):
                events.append("training_loop")
                if on_train is not None:
                    on_train(module, self.config)
                if fail_at == "training":
                    raise RuntimeError("training loop failed")

            def shutdown(self):
                events.append("controller_shutdown")

        def tracking(_args):
            events.append("tracking")
            if fail_at == "tracking":
                raise RuntimeError("tracking failed")

        replacements = {
            "ray": ray,
            "ray.serve": serve,
            "relax.core.controller": {"Controller": CPUController},
            "relax.core.node_group_affinity": {"maybe_pin_baseline_to_stable": lambda _args: events.append("pin")},
            "relax.utils.arguments": {"parse_args": Mock()},
            "relax.utils.tracking_utils": {"init_tracking": tracking},
            "relax.utils.utils": {"post_process_env": lambda _args, env: env},
        }
        for name, value in replacements.items():
            if isinstance(value, dict):
                replacement = ModuleType(name)
                replacement.__dict__.update(value)
            else:
                replacement = value
            monkeypatch.setitem(sys.modules, name, replacement)

        spec = importlib.util.spec_from_file_location(
            "_fault_events_train_under_test", ROOT / "relax/entrypoints/train.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
        module.logger = Mock()
        module.atexit = SimpleNamespace(register=lambda _callback: events.append("atexit_registered"))
        module.signal = SimpleNamespace(
            SIGTERM=signal.SIGTERM,
            SIGINT=signal.SIGINT,
            Signals=signal.Signals,
            signal=lambda sig, _callback: events.append(("signal_registered", sig)),
        )

        class Watchdog:
            def __init__(self, timeout, _function, args):
                self.timeout, self.args = timeout, args
                self.daemon = False

            def start(self):
                timers.append(("start", self.timeout, self.args))

            def cancel(self):
                timers.append(("cancel", self.timeout, self.args))

        def hard_exit(code):
            events.append(("exit", code))
            raise SystemExit(code)

        module.threading = SimpleNamespace(Timer=Watchdog)
        module._hard_exit = hard_exit
        return module, events, timers

    yield load
    for module in modules:
        runtime = module._fault_runtime
        if not isinstance(runtime, runtime_module.FaultEventRuntime):
            continue
        runtime.close()
        runtime._consumer.join(3)
        assert not runtime._consumer.is_alive()
        server = runtime.query_server
        if server is not None and server._thread is not None:
            server._thread.join(3)
            assert not server._thread.is_alive()


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("fail_at", [None, "tracking", "startup", "training"])
def test_train_runtime_preserves_success_and_failure_cleanup(train_loader, monkeypatch, enabled, fail_at):
    module, events, timers = train_loader(fail_at=fail_at, metrics_enabled=True)
    original_factory = runtime_module.start_fault_events

    def factory(**kwargs):
        runtime = original_factory(**kwargs)
        if runtime is not None:
            close = runtime.close

            def tracked_close():
                events.append("fault_close")
                close()

            monkeypatch.setattr(runtime, "close", tracked_close)
        return runtime

    monkeypatch.setattr(runtime_module, "start_fault_events", factory)
    args = SimpleNamespace(fault_events_enabled=enabled, use_health_check=True, fault_events_port=0)
    with pytest.raises(SystemExit) as exc:
        module.main(args)
    expected = int(fail_at is not None)
    assert exc.value.code == expected
    assert events[:3] == [
        ("signal_registered", signal.SIGTERM),
        ("signal_registered", signal.SIGINT),
        "atexit_registered",
    ]
    assert events[-3:] == ["serve_shutdown", "ray_shutdown", ("exit", expected)]
    assert events.count("controller_shutdown") == int(fail_at != "tracking")
    assert events.count("training_loop") == int(fail_at not in ("tracking", "startup"))
    assert module._training_completed is (fail_at is None)
    assert events.count("finish_tracking") == int(fail_at is None)
    if fail_at is None:
        assert events.index("controller_shutdown") < events.index("finish_tracking") < events.index("serve_shutdown")
    assert (module._fault_runtime is not None) is enabled
    assert set(vars(args)) == {"fault_events_enabled", "use_health_check", "fault_events_port"}
    if enabled:
        assert events.index("fault_close") < events.index("serve_shutdown")
        if fail_at != "tracking":
            assert events.index("fault_close") < events.index("controller_shutdown")
    timeout = 60 if expected else 1800
    assert timers == [("start", timeout, (expected,)), ("cancel", timeout, (expected,))]


def test_train_default_off_does_not_start_runtime(train_loader, monkeypatch):
    module, events, _ = train_loader()
    factory = Mock(side_effect=AssertionError("default-off observer must not start"))
    monkeypatch.setattr(runtime_module, "start_fault_events", factory)
    with pytest.raises(SystemExit) as exc:
        module.main(SimpleNamespace())
    assert exc.value.code == 0
    factory.assert_not_called()
    assert module._fault_runtime is None
    assert "training_loop" in events


def test_train_health_gate_emits_fixed_hint_and_does_not_start(train_loader, monkeypatch):
    module, _, _ = train_loader()
    logger = Mock()
    monkeypatch.setattr(runtime_module, "logger", logger)
    with pytest.raises(SystemExit) as exc:
        module.main(SimpleNamespace(fault_events_enabled=True, use_health_check=False, fault_events_port=0))
    assert exc.value.code == 0
    assert module._fault_runtime is None
    logger.warning.assert_called_once_with("Fault events require --use-health-check; collection disabled")


def test_train_observer_start_failure_is_fixed_and_keeps_training(train_loader, monkeypatch):
    module, events, _ = train_loader()
    monkeypatch.setattr(runtime_module, "start_fault_events", Mock(side_effect=RuntimeError("private failure text")))
    with pytest.raises(SystemExit) as exc:
        module.main(SimpleNamespace(fault_events_enabled=True, use_health_check=True, fault_events_port=0))
    assert exc.value.code == 0
    assert "training_loop" in events
    module.logger.warning.assert_called_once_with("Fault event runtime unavailable")


def test_train_occupied_query_port_keeps_internal_recording(train_loader, monkeypatch):
    committed = threading.Event()
    queried = []

    def training(module, _args):
        runtime = module._fault_runtime
        assert runtime is not None
        assert runtime.query_server is not None
        assert runtime.query_server.available is False
        assert runtime.query_server.url is None
        append = runtime.store.append

        def tracked_append(draft):
            result = append(draft)
            committed.set()
            return result

        monkeypatch.setattr(runtime.store, "append", tracked_append)
        assert runtime.record(
            EventDraft(runtime.run_id, "1" * 32, "2" * 32, "actor", 0, "fault_detected", "reported_error")
        )
        assert committed.wait(3)
        queried.extend(runtime.query()["events"])
        assert runtime.status()["query_available"] is False
        assert runtime.diagnostics()["coverage_unknown"] is False

    module, events, _ = train_loader(on_train=training)
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen(1)
        port = occupied.getsockname()[1]
        with pytest.raises(SystemExit) as exc:
            module.main(SimpleNamespace(fault_events_enabled=True, use_health_check=True, fault_events_port=port))
    assert exc.value.code == 0
    assert "controller_shutdown" in events
    assert len(queried) == 1


def test_train_signal_cleanup_isolates_observer_close_and_preserves_exit(train_loader):
    module, events, timers = train_loader()

    def fail_close():
        events.append("fault_close")
        raise OSError("private failure text")

    module._fault_runtime = SimpleNamespace(close=fail_close)
    module._ctrl = SimpleNamespace(
        shutdown=lambda: events.append("controller_shutdown"), _metrics_service_enabled=True
    )
    module._training_completed = True
    module.ray.is_initialized = lambda: True
    with pytest.raises(SystemExit) as exc:
        module._graceful_shutdown(sig=signal.SIGTERM)
    assert exc.value.code == 143
    assert events == ["fault_close", "controller_shutdown", "serve_shutdown", "ray_shutdown", ("exit", 143)]
    assert timers == [("start", 60, (143,)), ("cancel", 60, (143,))]
    module.logger.warning.assert_called_once_with("Fault event runtime shutdown unavailable")
    module.serve.get_app_handle.assert_not_called()
    with pytest.raises(SystemExit) as exc:
        module._graceful_shutdown(exit_code=0)
    assert exc.value.code == 0
    assert events.count("fault_close") == events.count("controller_shutdown") == 1


def test_runtime_factory_gate_and_start_failure_are_isolated(monkeypatch):
    constructor = Mock(side_effect=RuntimeError("private startup details"))
    logger = Mock()
    monkeypatch.setattr(runtime_module, "FaultEventRuntime", constructor)
    monkeypatch.setattr(runtime_module, "logger", logger)
    assert runtime_module.start_fault_events(enabled=False, health_enabled=True) is None
    assert runtime_module.start_fault_events(enabled=True, health_enabled=False) is None
    constructor.assert_not_called()
    assert runtime_module.start_fault_events(enabled=True, health_enabled=True) is None
    assert logger.warning.call_args.args == ("Fault event runtime unavailable",)
    logger.warning.side_effect = OSError("logging failed")
    assert runtime_module.start_fault_events(enabled=True, health_enabled=True) is None


@pytest.fixture
def fault_parser(arguments_module):
    arguments_module.RouterArgs = SimpleNamespace(add_cli_args=lambda parser, **_kwargs: parser)
    parser = argparse.ArgumentParser(allow_abbrev=False)
    arguments_module.get_slime_extra_args_provider()(parser)
    return parser


def test_fault_event_cli_is_opt_in_with_ephemeral_port(fault_parser):
    args = fault_parser.parse_args([])
    assert args.fault_events_enabled is False
    assert args.fault_events_port == 0
    assert args.use_health_check is False


def test_fault_event_cli_preserves_upstream_training_options(fault_parser):
    args = fault_parser.parse_args(
        [
            "--fault-events-enabled",
            "--use-health-check",
            "--fault-events-port",
            "65535",
            "--qsa-indexer-loss-coeff",
            "0.125",
            "--tq-rdma-mode",
            "auto",
            "--sft-wholepack",
            "--sft-packing-strategy",
            "binpack",
            "--sft-cpt-template",
            "multimodal",
        ]
    )
    assert args.fault_events_enabled and args.use_health_check
    assert args.fault_events_port == 65535
    assert args.qsa_indexer_loss_coeff == 0.125
    assert args.tq_rdma_mode == "auto"
    assert args.sft_wholepack
    assert args.sft_packing_strategy == "binpack"
    assert args.sft_cpt_template == "multimodal"


@pytest.mark.parametrize("port", [0, 1, 65535])
def test_fault_event_cli_accepts_port_boundaries(fault_parser, port):
    args = fault_parser.parse_args(["--fault-events-enabled", "--use-health-check", "--fault-events-port", str(port)])
    assert args.fault_events_enabled is True
    assert args.use_health_check is True
    assert args.fault_events_port == port


@pytest.mark.parametrize("port", ["-1", "65536", "1.5", "not-a-port"])
def test_fault_event_cli_rejects_invalid_ports(fault_parser, port):
    with pytest.raises(SystemExit) as exc:
        fault_parser.parse_args(["--fault-events-port", port])
    assert exc.value.code == 2
