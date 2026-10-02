# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""TransferQueue controller/store lifecycle helpers.

Extracted from :mod:`relax.core.controller` so the behavior can be unit-tested
without importing the whole Controller (which drags in Ray Serve and Megatron).
The Controller keeps thin wrappers that delegate here.

Two problems these helpers exist for:

* **F10 anti-hang** -- ``tq.init`` attaches to the existing named actor and polls
  ``get_config`` forever while it returns ``None`` (TQ ``interface.py:109-118``),
  so a controller left behind by a run that died between actor creation and
  ``store_config`` turns the next start into a hang.
* **Segment leak on teardown** -- ``tq.close()`` only tears down ZMQ (TQ
  ``storage/managers/base.py:378``); it never calls ``storage_client.close()``,
  so a MooncakeStore segment stays mounted and registered in the master until
  ``client_ttl`` (30 s) expires, and puts from the restarted job fail with
  "Failed to open segment ... Connection refused".
"""

from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

import ray
import transfer_queue as tq

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

CONTROLLER_NAME = "TransferQueueController"
CONTROLLER_NAMESPACE = "transfer_queue"
DEFAULT_TQ_INIT_TIMEOUT_SECONDS = 60.0
DEFAULT_TQ_ATTACH_TIMEOUT_SECONDS = 60.0


@dataclass(frozen=True)
class TqInitResult:
    """Result of an exclusive-owner TransferQueue initialization."""

    config: Any
    owner: Any
    fallback_reason: str = ""


class TqInitializationTimeout(TimeoutError):
    """Raised when ``tq.init`` does not finish within the bounded timeout."""


class TqInitializationError(RuntimeError):
    """Raised after a failed ``tq.init`` candidate is fully cleaned."""


class TqAttachTimeout(TimeoutError):
    """Raised when a worker cannot attach to TransferQueue within the bound."""


class TqCleanupTimeout(TimeoutError):
    """Raised when an owner or controller cannot be confirmed gone."""


class TqHandshakeIsolationError(RuntimeError):
    """Raised when timed-out handshake workers cannot be confirmed stopped."""


def safe_exception_kind(error: BaseException) -> str:
    """Return a log-safe exception category without rendering its payload.

    Ray exception strings can contain worker IPs, process IDs and remote
    traceback paths.  Lifecycle logs and job-level failure summaries therefore
    record only the stable exception class.
    """
    ray_task_error_type = getattr(getattr(ray, "exceptions", None), "RayTaskError", None)
    if isinstance(ray_task_error_type, type) and isinstance(error, ray_task_error_type):
        return "RayTaskError"
    name = type(error).__name__
    return name if name.isidentifier() else "Exception"


def _get_config_value(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def uses_mooncake(conf: Any) -> bool:
    """True when ``conf`` selects the MooncakeStore backend.

    Public so the Controller can decide whether the cluster-wide attach
    handshake is required for the stored job-level config.
    """
    backend = _get_config_value(conf, "backend", {})
    return _get_config_value(backend, "storage_backend", "SimpleStorage") == "MooncakeStore"


def _prepare_mooncake_runtime(conf: Any) -> None:
    if not uses_mooncake(conf):
        return
    from relax.utils.tq.config import validate_mooncake_runtime_contract

    validate_mooncake_runtime_contract()


def kill_tq_controller_and_wait(timeout: float = 20.0) -> None:
    """Kill the TransferQueueController named actor, then wait for GCS
    deregistration.

    Waiting matters because ``ray.kill`` is asynchronous: the handle stays
    resolvable for a short window, and a re-init landing in that window
    attaches to a dying actor and fails with ``ActorDiedError``.
    """
    try:
        stale = ray.get_actor(CONTROLLER_NAME, namespace=CONTROLLER_NAMESPACE)
        ray.kill(stale)
        logger.info("[dataplane] Killed TransferQueueController actor (F10 guard).")
    except ValueError:
        return  # actor does not exist — nothing to kill or wait for.
    except Exception as e:
        raise RuntimeError(f"Failed to kill TransferQueueController ({safe_exception_kind(e)})") from None

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            ray.get_actor(CONTROLLER_NAME, namespace=CONTROLLER_NAMESPACE)
        except ValueError:
            return
        except ray.exceptions.RayError as error:
            raise RuntimeError(
                f"Failed to confirm TransferQueueController cleanup ({safe_exception_kind(error)})"
            ) from None
        time.sleep(0.4)
    raise TqCleanupTimeout(f"TransferQueueController still resolvable after {timeout}s")


def reap_unusable_tq_controller(get_config_timeout: float = 10.0) -> bool:
    """Require a clean exclusive cluster, reaping only unusable TQ state.

    Returns ``True`` when a half-initialised or unresponsive controller was
    reaped.  A healthy controller is never attached or killed: the initial RDMA
    release supports one Relax job per Ray cluster, so healthy existing state
    means the cluster is not clean and startup fails explicitly.
    """
    try:
        existing = ray.get_actor(CONTROLLER_NAME, namespace=CONTROLLER_NAMESPACE)
    except ValueError:
        return False  # nothing there — nothing to reap.
    except ray.exceptions.RayError as error:
        raise RuntimeError(f"Failed to query TransferQueueController ({safe_exception_kind(error)})") from None

    try:
        conf = ray.get(existing.get_config.remote(), timeout=get_config_timeout)
    except (ray.exceptions.GetTimeoutError, ray.exceptions.RayActorError) as error:
        # Only a proven timeout/dead actor is safe to classify as unusable.
        # Other RayError subclasses can be transient GCS/control-plane failures;
        # killing a healthy controller on those would violate the exclusive-job
        # ownership boundary.
        logger.warning(
            f"[dataplane] Existing TransferQueueController is unusable ({safe_exception_kind(error)}); reaping it."
        )
        conf = None
    except ray.exceptions.RayError as error:
        raise RuntimeError(f"Failed to inspect TransferQueueController ({safe_exception_kind(error)})") from None

    if conf is not None:
        raise RuntimeError(
            "A healthy TransferQueueController already exists. The initial RDMA release requires an exclusive, "
            "clean Ray cluster; stop the previous Relax job or remove its TQ state before retrying."
        )

    logger.warning("[dataplane] TransferQueueController has no stored config (half-initialised); reaping it.")
    kill_tq_controller_and_wait()
    return True


def assert_mooncake_rdma_configured() -> None:
    """Fail unless the attached client configuration requests Mooncake/RDMA.

    A successful :func:`attach_tq_client` only proves ``tq.init`` returned
    without raising.  This check also establishes that the stored controller
    config produced a Mooncake manager and an RDMA-configured storage client:
    ``tq.init`` ignores the caller's conf when attaching to an existing
    controller (``interface.py:130-135``).  ``storage_client.protocol`` is the
    configured request, not a negotiated transport signal, so this is not
    proof that bytes traversed an HCA or that a native transport fallback is
    impossible.  Wire-level proof remains the benchmark's counter check.

    Raises
    ------
    RuntimeError
        When the storage manager is not MooncakeStore, when no storage client
        is present, or when the client's configured protocol is not ``rdma``.
    """
    manager = tq.get_client().storage_manager
    if type(manager).__name__ != "MooncakeStorageManager":
        raise RuntimeError("attached storage manager is not MooncakeStorageManager")

    store_client = getattr(manager, "storage_client", None)
    if store_client is None:
        raise RuntimeError("MooncakeStorageManager exposes no storage_client after attach")

    protocol = getattr(store_client, "protocol", None)
    if protocol != "rdma":
        raise RuntimeError("MooncakeStore client is not configured for protocol=rdma")


def _get_stored_config(timeout: float = 10.0) -> Any:
    try:
        controller = ray.get_actor(CONTROLLER_NAME, namespace=CONTROLLER_NAMESPACE)
        conf = ray.get(controller.get_config.remote(), timeout=timeout)
    except ValueError:
        raise
    except ray.exceptions.RayError as error:
        raise RuntimeError(f"Failed to read TransferQueueController config ({safe_exception_kind(error)})") from None
    if conf is None:
        raise RuntimeError("TransferQueueController returned no config after tq.init completed")
    return conf


def _resolve_attach_timeout() -> float:
    """Attach deadline in seconds; override via
    ``RELAX_TQ_ATTACH_TIMEOUT_SECONDS``."""
    raw = os.environ.get("RELAX_TQ_ATTACH_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return DEFAULT_TQ_ATTACH_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        raise RuntimeError("RELAX_TQ_ATTACH_TIMEOUT_SECONDS must be a finite positive number of seconds") from None
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError("RELAX_TQ_ATTACH_TIMEOUT_SECONDS must be a finite positive number of seconds")
    return value


def _await_controller_config(deadline: float) -> None:
    """Bounded wait until the named controller serves a non-``None`` config.

    ``tq.init`` polls ``get_config`` forever while it returns ``None`` (the F10
    hang), so a worker refuses to enter that loop unless a config is provably
    served before the deadline.
    """
    last_error = "TransferQueueController named actor not found"
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TqAttachTimeout(f"TransferQueue attach timed out: {last_error}")
        try:
            controller = ray.get_actor(CONTROLLER_NAME, namespace=CONTROLLER_NAMESPACE)
        except ValueError:
            time.sleep(min(0.5, remaining))
            continue
        try:
            conf = ray.get(controller.get_config.remote(), timeout=max(min(remaining, 10.0), 0.1))
        except Exception as e:
            last_error = f"get_config failed ({safe_exception_kind(e)})"
            time.sleep(min(0.5, max(deadline - time.monotonic(), 0.0)))
            continue
        if conf is not None:
            return
        last_error = "controller exists but has stored no config yet"
        time.sleep(min(0.5, max(deadline - time.monotonic(), 0.0)))


def _bounded_tq_init(conf: Any, deadline: float, *, role: str) -> None:
    """Run ``tq.init`` under the remaining deadline.

    ``tq.init`` takes no timeout and its mooncake client setup blocks in native
    code, so it runs on a daemon watchdog thread.  On expiry the caller fails
    fast with :class:`TqAttachTimeout`; the abandoned thread dies with the
    failed worker process (Serve tears the replica down).
    """
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TqAttachTimeout(f"TransferQueue attach for role={role} timed out before tq.init")
    error: list[BaseException] = []

    def _run() -> None:
        try:
            tq.init(conf=conf)
        except BaseException as e:  # propagated to the attaching caller below
            error.append(e)

    thread = threading.Thread(target=_run, name=f"tq-attach-{role}", daemon=True)
    thread.start()
    thread.join(remaining)
    if thread.is_alive():
        raise TqAttachTimeout(
            f"tq.init for role={role} did not finish within {remaining:.0f}s "
            "(mooncake setup or controller poll hung); failing this worker fast."
        )
    if error:
        raise error[0]


def attach_tq_client(
    conf: Any,
    *,
    role: str,
    timeout: float | None = None,
) -> Any:
    """Attach a component process within a bounded deadline.

    The deadline covers both waiting for a served controller config and
    ``tq.init`` itself, because either phase can hang unboundedly (get_config
    polling and mooncake endpoint setup respectively).  ``None`` resolves the
    deadline from ``RELAX_TQ_ATTACH_TIMEOUT_SECONDS`` (default 60 s).

    The initial release assumes one component/replica per Ray actor process.
    Components therefore own the one process-global TQ client they attach and
    unconditionally detach it during teardown.
    """
    if timeout is None:
        timeout = _resolve_attach_timeout()
    deadline = time.monotonic() + timeout
    _prepare_mooncake_runtime(conf)
    _await_controller_config(deadline)
    _bounded_tq_init(conf, deadline, role=role)
    return tq.get_client()


def detach_tq_client() -> None:
    """Detach this worker's TQ client (attach-only inverse of
    :func:`attach_tq_client`).

    Closes the attached Mooncake storage client so its segment deregisters
    from the master immediately instead of lingering until ``client_ttl``
    expires — a stale endpoint breaks fast restarts with "Failed to open
    segment".  Only process-local handles are touched (never the named
    controller or globally stored data).  The initial release assumes one
    component/replica per Ray actor process, so there is no cross-instance
    generation lease.  Force-killed workers still fall back to the master-side
    TTL.
    """
    client = None
    store_client = None
    try:
        client = tq.get_client()
        store_client = getattr(client.storage_manager, "storage_client", None)
    except (AssertionError, AttributeError):
        pass

    if store_client is not None and hasattr(store_client, "close"):
        try:
            store_client.close()
        except Exception as e:  # pragma: no cover - best-effort local cleanup
            logger.warning(f"[dataplane] Failed to close attached MooncakeStore client ({safe_exception_kind(e)}).")

    if client is not None and hasattr(client, "close"):
        try:
            client.close()
        except Exception as e:  # pragma: no cover - best-effort local cleanup
            logger.warning(f"[dataplane] Failed to close attached TransferQueue client ({safe_exception_kind(e)}).")

    # TransferQueue has no public detach-only API. Reset only process-local
    # handles; never touch _TQ_STORAGE or the named controller actor.
    try:
        from transfer_queue import interface as tq_interface

        tq_interface._TQ_CLIENT = None
        tq_interface._TQ_CONTROLLER = None
    except (ImportError, AttributeError):  # pragma: no cover - version dependent
        pass


def _alive_node_ids() -> list[str]:
    """Every alive node: TQ endpoints (Serve replicas and 0-CPU actors) carry
    no placement binding, so any alive node may end up hosting one."""
    node_ids: list[str] = []
    for node in ray.nodes():
        if not node.get("Alive"):
            continue
        node_id = node.get("NodeID")
        if not isinstance(node_id, str) or not node_id:
            raise RuntimeError("an alive Ray node has no valid NodeID")
        node_ids.append(node_id)
    return node_ids


def _cancel_handshake_tasks(refs: list[Any], *, timeout: float = 10.0) -> bool:
    """Force-cancel submitted one-shot workers and confirm every ref is ready.

    A timed-out worker may still own a daemon ``tq.init`` thread.  The caller
    may only tear down Mooncake and start SimpleStorage after Ray confirms the
    force-cancelled task refs have reached a terminal state.
    """
    cancellation_failed = False
    for ref in refs:
        try:
            ray.cancel(ref, force=True)
        except Exception as error:  # pragma: no cover - Ray control-plane failure
            cancellation_failed = True
            logger.warning(f"[dataplane] Failed to force-cancel a TQ handshake worker ({safe_exception_kind(error)}).")
    if cancellation_failed:
        return False
    try:
        _ready, pending = ray.wait(refs, num_returns=len(refs), timeout=timeout)
    except Exception as error:  # pragma: no cover - Ray control-plane failure
        logger.warning(
            f"[dataplane] Could not confirm TQ handshake worker cancellation ({safe_exception_kind(error)})."
        )
        return False
    return not pending


def verify_cluster_attach(conf: Any, *, timeout: float | None = None) -> list[str]:
    """Bounded attach handshake from every alive node; returns failure
    summaries.

    Each one-shot task performs the same bounded :func:`attach_tq_client` a
    worker would perform (real stored config, real storage client), confirms a
    Mooncake manager whose client is configured for ``protocol=rdma``, and
    detaches immediately.  This validates actual attach/setup and configuration
    agreement rather than a ``/sys`` heuristic; it is not negotiated-transport
    or wire proof.  The worker is not reused because a timed-out ``tq.init``
    daemon thread cannot be stopped safely in-process.  An empty return means
    every alive node attached within the deadline; the driver aggregates
    failures and decides one job-level outcome.
    """
    if timeout is None:
        timeout = _resolve_attach_timeout()

    try:
        node_ids = _alive_node_ids()
    except Exception as error:
        return [f"cluster: node discovery failed ({safe_exception_kind(error)})"]
    if not node_ids:
        return ["cluster: no alive Ray nodes discovered"]

    expects_mooncake = uses_mooncake(conf)

    refs: list[Any] = []
    node_number_by_ref: dict[Any, int] = {}
    try:
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        @ray.remote(num_cpus=0, max_retries=0, max_calls=1)
        def _handshake(handshake_conf: Any, check_mooncake: bool, attach_timeout: float) -> None:
            from relax.utils.tq.lifecycle import (
                assert_mooncake_rdma_configured,
                attach_tq_client,
                detach_tq_client,
            )

            attach_tq_client(handshake_conf, role="attach-handshake", timeout=attach_timeout)
            try:
                if check_mooncake:
                    assert_mooncake_rdma_configured()
            finally:
                # A failed assertion must not leak this node's registered
                # segment; without the detach it lingers until client_ttl.
                detach_tq_client()

        for node_number, node_id in enumerate(node_ids, start=1):
            strategy = NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
            ref = _handshake.options(scheduling_strategy=strategy).remote(conf, expects_mooncake, timeout)
            refs.append(ref)
            # Keep the raw Ray NodeID only in the scheduling strategy.  Failure
            # summaries may reach public CI/PR logs, so identify nodes by a
            # stable per-handshake ordinal instead.
            node_number_by_ref[ref] = node_number
    except Exception as error:
        if refs and not _cancel_handshake_tasks(refs):
            raise TqHandshakeIsolationError(
                "submitted TQ handshake workers could not be confirmed stopped after scheduling failed"
            ) from None
        return [f"cluster: handshake scheduling failed ({safe_exception_kind(error)})"]

    # Grace beyond the per-node attach deadline covers task scheduling and
    # worker startup on a busy cluster.
    wait_bound = timeout + 30.0
    try:
        ready, pending = ray.wait(refs, num_returns=len(refs), timeout=wait_bound)
    except Exception as error:
        if not _cancel_handshake_tasks(refs):
            raise TqHandshakeIsolationError(
                "submitted TQ handshake workers could not be confirmed stopped after wait failed"
            ) from None
        return [f"cluster: handshake wait failed ({safe_exception_kind(error)})"]
    failures: list[str] = []
    for ref in ready:
        try:
            ray.get(ref)
        except Exception as error:
            failures.append(f"node#{node_number_by_ref[ref]}: handshake task failed ({safe_exception_kind(error)})")
    if pending and not _cancel_handshake_tasks(pending):
        raise TqHandshakeIsolationError("timed-out TQ handshake workers could not be confirmed stopped")
    for ref in pending:
        failures.append(f"node#{node_number_by_ref[ref]}: handshake did not return within {wait_bound:.0f}s")
    return failures


def close_tq_and_unmount() -> None:
    """Close TransferQueue and unmount the MooncakeStore segment.

    Order matters: ``tq.close()`` still needs the store alive for its
    ``remove_all()``, so the client handle is captured first and unmounted
    after. SimpleStorage has no ``storage_client``, so this is a no-op there.

    This function is reachable only inside the dedicated owner actor. Workers
    call :func:`detach_tq_client` instead because upstream ``tq.close()`` kills
    the named controller even from an attached process.
    """
    store_client = None
    try:
        store_client = getattr(tq.get_client().storage_manager, "storage_client", None)
    except (AssertionError, AttributeError):
        pass  # TQ not initialised in this process, or no KV client.

    try:
        tq.close()
    finally:
        # ``tq.close`` may fail while removing queued data.  Segment teardown
        # is still mandatory; otherwise the failed owner leaves registered and
        # locked memory behind until the Mooncake TTL expires.
        if store_client is not None and hasattr(store_client, "close"):
            try:
                store_client.close()
                logger.info("[dataplane] Unmounted MooncakeStore segment on teardown.")
            except Exception as e:  # pragma: no cover - best-effort cleanup
                logger.warning(f"[dataplane] Failed to unmount MooncakeStore segment ({safe_exception_kind(e)}).")


@ray.remote(num_cpus=0)
class _TransferQueueOwner:
    """Process boundary for first-time TQ initialization and global cleanup."""

    def ready(self) -> None:
        """Scheduling barrier used before ``tq.init`` enters fallback scope."""

    def termination_probe(self) -> None:
        """Never return; a terminal ref proves this actor process was
        killed."""
        threading.Event().wait()

    def initialize(self, conf: Any) -> Any:
        _prepare_mooncake_runtime(conf)
        tq.init(conf=conf)
        return _get_stored_config()

    def close(self) -> None:
        close_tq_and_unmount()


def _stop_owner_actor(owner: Any, *, timeout: float = 10.0) -> None:
    """Force-kill ``owner`` and prove its process reached a terminal state."""
    probe_ref = None
    control_error = ""
    try:
        # Queue behind any blocked initialize/close call. It can only become
        # ready after the actor dies because the method never returns.
        probe_ref = owner.termination_probe.remote()
    except ray.exceptions.RayActorError:
        return  # Already terminal before cleanup reached it.
    except Exception as error:
        control_error = f"termination probe submission ({safe_exception_kind(error)})"

    try:
        ray.kill(owner, no_restart=True)
    except ray.exceptions.RayActorError:
        pass
    except Exception as error:
        kind = safe_exception_kind(error)
        control_error = f"{control_error}, " if control_error else ""
        control_error += f"actor kill ({kind})"

    pending: list[Any] = []
    if probe_ref is not None:
        try:
            ready, pending = ray.wait([probe_ref], num_returns=1, timeout=timeout)
        except Exception as error:
            ready = []
            control_error = f"{control_error}, " if control_error else ""
            control_error += f"termination wait ({safe_exception_kind(error)})"
        for ref in ready:
            try:
                ray.get(ref)
            except ray.exceptions.RayActorError:
                pass
            except Exception as error:
                control_error = f"{control_error}, " if control_error else ""
                control_error += f"termination probe result ({safe_exception_kind(error)})"
            else:
                control_error = f"{control_error}, " if control_error else ""
                control_error += "termination probe returned normally"

    if control_error or pending or probe_ref is None:
        detail = control_error or "termination could not be observed"
        if pending:
            detail = f"{detail}, " if detail else ""
            detail += "termination probe remained pending"
        raise TqCleanupTimeout(f"Failed to confirm TransferQueue owner cleanup: {detail}") from None


def _cleanup_failed_owner(owner: Any, *, timeout: float = 10.0) -> None:
    """Stop a failed initializer and remove its exclusive-cluster state."""
    try:
        ray.get(owner.close.remote(), timeout=timeout)
    except Exception as e:
        logger.warning(f"[dataplane] TQ owner cleanup RPC failed; killing owner actor ({safe_exception_kind(e)}).")
    _stop_owner_actor(owner, timeout=timeout)

    # Only remove global state after the actor is confirmed terminal. A
    # pre-kill existence query races with a late controller creation by the
    # blocked native initializer. The exclusive-cluster precondition proves
    # any controller visible here belongs to this attempt.
    kill_tq_controller_and_wait(timeout=timeout)


def _start_owner(*, timeout: float) -> Any:
    """Create and schedule the isolated owner before candidate fallback."""
    try:
        owner = _TransferQueueOwner.remote()
        ray.get(owner.ready.remote(), timeout=timeout)
    except Exception as error:
        if "owner" in locals():
            _stop_owner_actor(owner, timeout=timeout)
        raise RuntimeError(f"TransferQueue owner creation failed ({safe_exception_kind(error)})") from None
    return owner


def _initialize_owner(owner: Any, conf: Any, *, timeout: float) -> TqInitResult:
    """Initialize one candidate and fully clean its owner on failure."""
    try:
        stored_conf = ray.get(owner.initialize.remote(conf), timeout=timeout)
    except ray.exceptions.GetTimeoutError:
        _cleanup_failed_owner(owner)
        raise TqInitializationTimeout(f"tq.init did not finish within {timeout:.0f}s") from None
    except Exception as error:
        _cleanup_failed_owner(owner)
        raise TqInitializationError(
            f"TransferQueue owner initialization failed ({safe_exception_kind(error)})"
        ) from None
    return TqInitResult(config=stored_conf, owner=owner)


def close_tq_owner(owner: Any | None, *, timeout: float = 30.0) -> None:
    """Ask the exclusive owner process to close global TQ state."""
    if owner is None:
        return
    close_error: Exception | None = None
    try:
        ray.get(owner.close.remote(), timeout=timeout)
    except Exception as e:
        close_error = e
    _stop_owner_actor(owner, timeout=timeout)
    # Ensure the controller has left GCS even when owner.close() failed before
    # reaching upstream tq.close(). This is a no-op when it is already gone.
    kill_tq_controller_and_wait(timeout=timeout)
    if close_error is not None:
        raise RuntimeError(f"TransferQueue owner cleanup failed ({safe_exception_kind(close_error)})") from None


def initialize_tq_with_fallback(
    conf: Any,
    *,
    mode: str,
    fallback_conf: Any | None = None,
    timeout: float = DEFAULT_TQ_INIT_TIMEOUT_SECONDS,
) -> TqInitResult:
    """Initialize TQ with one exclusive owner and one safe auto fallback.

    ``fallback_conf`` must be the equivalent SimpleStorage configuration.  It
    is used only for ``mode='auto'`` after a real Mooncake ``tq.init`` failure.
    ``required`` always cleans up and re-raises the original error.
    """

    def _attempt(owner: Any, attempt_conf: Any) -> TqInitResult:
        return _initialize_owner(owner, attempt_conf, timeout=timeout)

    # Keep the exclusive-cluster gate outside the auto fallback boundary. A
    # healthy pre-existing controller is not an RDMA candidate failure and
    # must never be converted into a SimpleStorage attach attempt.
    reap_unusable_tq_controller()
    owner = _start_owner(timeout=timeout)
    try:
        return _attempt(owner, conf)
    except (TqInitializationTimeout, TqInitializationError) as primary_error:
        if mode != "auto" or fallback_conf is None:
            raise

        reason = f"mooncake_init_failed:{type(primary_error).__name__}"
        logger.warning(
            f"[dataplane] Mooncake tq.init failed ({safe_exception_kind(primary_error)}); "
            "cleaned partial state and retrying once with SimpleStorage."
        )
        try:
            # Failed-owner cleanup must leave a clean cluster. Re-check before
            # starting the fallback so unknown residual state fails closed.
            reap_unusable_tq_controller()
            fallback_owner = _start_owner(timeout=timeout)
            result = _attempt(fallback_owner, fallback_conf)
        except Exception as fallback_error:
            raise RuntimeError(
                "TransferQueue SimpleStorage fallback initialization failed after "
                f"Mooncake initialization error ({safe_exception_kind(primary_error)}); "
                f"fallback error ({safe_exception_kind(fallback_error)})"
            ) from None
        return TqInitResult(
            config=result.config,
            owner=result.owner,
            fallback_reason=reason,
        )
