import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import ray

from relax.core.controller import Controller
from relax.utils.failure_events import FailureEvent, FailureEventStore


NAMESPACE = "relax_failure_events"


@pytest.fixture(scope="module")
def ray_runtime():
    connected_here = False

    if not ray.is_initialized():
        try:
            ray.init(address="auto")
        except ConnectionError:
            ray.init(num_cpus=1, include_dashboard=False)
        connected_here = True

    address = ray.get_runtime_context().gcs_address
    yield address

    if connected_here:
        ray.shutdown()


def create_store(name: str, capacity: int = 16):
    store_actor = ray.remote(num_cpus=0)(FailureEventStore)
    return store_actor.options(
        name=name,
        namespace=NAMESPACE,
        lifetime="detached",
        get_if_exists=True,
    ).remote(capacity)


def kill_store(name: str):
    try:
        store = ray.get_actor(name, namespace=NAMESPACE)
    except ValueError:
        return
    ray.kill(store)


def test_detached_store_survives_driver_exit(ray_runtime):
    name = "failure_event_test_reconnect"
    kill_store(name)

    code = f"""
import ray

from relax.utils.failure_events import FailureEvent, FailureEventStore

ray.init(address={ray_runtime!r})

store_actor = ray.remote(num_cpus=0)(FailureEventStore)
store = store_actor.options(
    name={name!r},
    namespace={NAMESPACE!r},
    lifetime="detached",
    get_if_exists=True,
).remote(16)

ray.get(
    store.append.remote(
        FailureEvent(
            fault_id="fault-1",
            role="actor",
            phase="detected",
            occurred_at_ms=1,
        )
    )
)
"""

    subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
    )

    store = ray.get_actor(name, namespace=NAMESPACE)
    page = ray.get(store.query.remote())

    assert [event["fault_id"] for event in page["events"]] == ["fault-1"]

    ray.kill(store)


def test_different_store_names_do_not_share_history(ray_runtime):
    first_name = "failure_event_test_run_a"
    second_name = "failure_event_test_run_b"

    kill_store(first_name)
    kill_store(second_name)

    first = create_store(first_name)
    second = create_store(second_name)

    ray.get(
        first.append.remote(
            FailureEvent(
                fault_id="fault-a",
                role="actor",
                phase="detected",
                occurred_at_ms=1,
            )
        )
    )

    first_page = ray.get(first.query.remote())
    second_page = ray.get(second.query.remote())

    assert [event["fault_id"] for event in first_page["events"]] == ["fault-a"]
    assert second_page["events"] == []

    ray.kill(first)
    ray.kill(second)


def test_controller_shutdown_kills_failure_event_store(ray_runtime):
    name = "failure_event_test_shutdown"

    kill_store(name)
    store = create_store(name)

    controller = Controller.__new__(Controller)
    controller._failure_event_store = store
    controller._health_manager = SimpleNamespace(stop=Mock())
    controller.serve_dict = {}
    controller._teacher_manager = None
    controller._shutdown_agentic_rollout_services = Mock()
    controller._cleanup_s3_model_weights_after_init = Mock()
    controller.stop_health_check = Mock()

    controller.shutdown()

    with pytest.raises(ValueError):
        ray.get_actor(name, namespace=NAMESPACE)


def test_failure_event_reporting_does_not_change_recovery():
    controller = Controller.__new__(Controller)

    append = Mock(side_effect=RuntimeError("store unavailable"))
    controller._failure_event_store = SimpleNamespace(append=SimpleNamespace(remote=append))

    controller._emit_failure_event(
        fault_id="fault-1",
        role="actor",
        phase="detected",
        reason="reported_error",
        step=1,
    )

    append.assert_called_once()
