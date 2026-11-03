# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Dead-engine classification for GenRM single-engine recovery.

``_is_engine_dead`` decides whether a failure from an engine means "the process
is gone, retire and rebuild it" or "this is a real bug, let it propagate". Both
directions matter: too broad and a genuine bug is silently swallowed into a
pointless ~1.5 min engine rebuild; too narrow and a dead engine escalates to the
global restart the whole recovery path exists to avoid.

The subtle case is the resume path. ``resume_memory_occupation`` goes through
``GenRMEngine._make_request``, which uses ``requests`` -- and
``requests.exceptions.ConnectionError`` is an ``OSError`` subclass but *not* a
builtin ``ConnectionError``, so it has to be listed explicitly.
"""

from __future__ import annotations

import pytest
import requests


try:
    import ray

    from relax.distributed.ray.genrm import _is_engine_dead

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="requires ray + the full relax training deps")


def _ray_task_error(cause: BaseException) -> BaseException:
    """Build what ``ray.get`` actually raises for a failed remote call.

    Ray re-raises as a class inheriting from *both* ``RayTaskError`` and the
    original cause (``ray/exceptions.py::as_instanceof_cause``), so a
    classifier based on ``isinstance`` only works if it survives that wrapping.
    """
    err = ray.exceptions.RayTaskError(
        function_name="release_memory_occupation",
        traceback_str="Traceback (most recent call last):\n" + repr(cause),
        cause=cause,
    )
    return err.as_instanceof_cause()


def test_drain_connection_error_is_dead():
    """``GenRMEngine.release_memory_occupation`` raises builtin ConnectionError
    from its dead-server fast-fail; it must survive the trip through
    ray.get."""
    cause = ConnectionError(
        "GenRM engine unreachable while draining before release "
        "(3 consecutive connection errors) — the server process is most likely dead."
    )
    exc = _ray_task_error(cause)
    assert isinstance(exc, ray.exceptions.RayTaskError), "sanity: still a RayTaskError"
    assert _is_engine_dead(exc)


def test_drain_timeout_is_dead():
    assert _is_engine_dead(_ray_task_error(TimeoutError("Timeout while draining GenRM before release.")))


def test_dead_ray_actor_is_dead():
    assert _is_engine_dead(ray.exceptions.RayActorError())


def test_requests_transport_errors_are_dead():
    """The resume path fails with ``requests`` exceptions, not builtin ones.

    Without these listed, an engine that died during the offloaded window --
    only observable at onload, because offload short-circuits when already
    offloaded -- is misclassified as a real bug and re-raised, taking the whole
    job down.
    """
    assert not issubclass(requests.exceptions.ConnectionError, ConnectionError), "premise of this test"
    assert _is_engine_dead(_ray_task_error(requests.exceptions.ConnectionError("Connection refused")))
    assert _is_engine_dead(_ray_task_error(requests.exceptions.ReadTimeout("read timed out")))


@pytest.mark.parametrize(
    "cause",
    [
        ValueError("bad tags"),
        KeyError("missing"),
        RuntimeError("CUDA error: an illegal memory access was encountered"),
        AssertionError("invariant broken"),
        # A 5xx from a server that answered is a bug in the engine, not a
        # liveness failure -- rebuilding would only hide it.
        requests.exceptions.HTTPError("500 Server Error"),
    ],
)
def test_real_bugs_are_not_swallowed_as_dead_engines(cause):
    assert not _is_engine_dead(cause)
    assert not _is_engine_dead(_ray_task_error(cause))


def test_oserror_does_not_over_match():
    """ConnectionError is an OSError subclass, so the check must not widen to
    OSError -- a disk problem inside a healthy engine is not a dead process."""
    assert not _is_engine_dead(OSError("No space left on device"))
    assert _is_engine_dead(ConnectionResetError("peer reset"))  # is a ConnectionError
