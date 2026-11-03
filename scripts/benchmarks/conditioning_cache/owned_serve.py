#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Strict name-scoped Serve ownership, for dedicated single-node experiment
jobs.

No Ray imports or state changes occur at import time. Expected app/deployment
names are supplied by the workload, never inferred from another model's run.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path


class IncompleteServeDeploymentConfig(RuntimeError):
    """Ray may briefly omit deployment config while an application is
    deleting."""


def active_other_jobs(jobs: dict | list, submission_id: str | None = None, probe_job_id: str | None = None) -> list:
    values = jobs.values() if isinstance(jobs, dict) else jobs
    return [
        job
        for job in values
        if job.get("status") in {"PENDING", "RUNNING"}
        and not (submission_id is not None and job.get("submission_id") == submission_id)
        and not (
            probe_job_id is not None
            and job.get("job_id") == probe_job_id
            and job.get("type") == "DRIVER"
            and job.get("submission_id") is None
        )
    ]


def serve_identities(response: dict) -> dict:
    """Keep deployment identity, excluding changing health and replica
    state."""
    applications = response.get("applications")
    if not isinstance(applications, dict):
        raise RuntimeError("Serve response lacks an application inventory")
    identities = {}
    for name, application in applications.items():
        stamp = application.get("last_deployed_time_s")
        deployments = application.get("deployments")
        if (
            application.get("name") != name
            or not isinstance(stamp, (int, float))
            or not math.isfinite(stamp)
            or stamp <= 0
            or not isinstance(deployments, dict)
        ):
            raise RuntimeError(f"Incomplete Serve identity for {name}")
        configs = {key: value.get("deployment_config") for key, value in deployments.items()}
        if not configs or any(not isinstance(value, dict) for value in configs.values()):
            raise IncompleteServeDeploymentConfig(f"Incomplete Serve deployment config for {name}")
        # Configs may contain environment credentials. Keep only their digest and
        # the nonsecret run-directory marker in evidence / subprocess arguments.
        config_digests = {
            key: hashlib.sha256(
                json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            ).hexdigest()
            for key, value in configs.items()
        }
        run_directories = {
            key: ((value.get("ray_actor_options") or {}).get("runtime_env") or {})
            .get("env_vars", {})
            .get("RELAX_CONDITION_CACHE_RUN_DIR")
            for key, value in configs.items()
        }
        identities[name] = {
            "last_deployed_time_s": stamp,
            "route_prefix": application.get("route_prefix"),
            "source": application.get("source"),
            "deployment_config_sha256": config_digests,
            "run_directories": run_directories,
        }
    return identities


def claim_owned_serve(
    response: dict,
    submitted_unix: float,
    run_directory: Path,
    proc_root: Path = Path("/proc"),
    *,
    expected_deployments: dict[str, str],
) -> dict:
    """Only claim declared apps after an empty preflight; verify every live
    replica marker."""
    identities = serve_identities(response)
    if set(identities) != set(expected_deployments):
        raise RuntimeError("Unexpected Serve applications; no cancellation or cleanup was attempted")
    for name, identity in identities.items():
        if identity["last_deployed_time_s"] < submitted_unix:
            raise RuntimeError(f"Serve application {name} predates this submission")
        if set(identity["deployment_config_sha256"]) != {expected_deployments[name]}:
            raise RuntimeError(f"Unexpected Serve deployment for {name}")
        for marker in identity["run_directories"].values():
            if marker is not None and marker != str(run_directory):
                raise RuntimeError(f"Serve application {name} has another run's config ownership marker")
        # Ray 2.56 exposes the explicit environment in deployment_config, but
        # Actor / ImageSFT inherit this marker and omit it there. Validate the
        # effective env of every live replica on this dedicated single node.
        for deployment in response["applications"][name]["deployments"].values():
            replicas = deployment.get("replicas")
            if not isinstance(replicas, list) or not replicas:
                raise RuntimeError(f"No live Serve replicas to establish ownership of {name}")
            for replica in replicas:
                pid = replica.get("pid")
                if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
                    raise RuntimeError(f"No local replica PID to establish ownership of {name}")
                try:
                    entries = (proc_root / str(pid) / "environ").read_bytes().split(b"\0")
                except OSError as error:
                    raise RuntimeError(f"Cannot read local replica ownership marker for {name}") from error
                markers = [
                    entry.split(b"=", 1)[1] for entry in entries if entry.startswith(b"RELAX_CONDITION_CACHE_RUN_DIR=")
                ]
                if markers != [str(run_directory).encode()]:
                    raise RuntimeError(f"Serve application {name} lacks this run's effective replica ownership marker")
    return identities


def validate_owned_serve(response: dict, owned: dict) -> list[str]:
    applications = response.get("applications")
    if not isinstance(applications, dict):
        raise RuntimeError("Serve response lacks an application inventory")
    if set(applications) - set(owned):
        raise RuntimeError("Unrecorded Serve applications appeared; refusing cleanup")
    incomplete = None
    for name, application in applications.items():
        # Detect replacement even if some application is in the transient state.
        if any(application.get(key) != owned[name][key] for key in ("last_deployed_time_s", "route_prefix", "source")):
            raise RuntimeError(f"Serve application {name} was replaced or modified; refusing cleanup")
        try:
            identity = serve_identities({"applications": {name: application}})[name]
        except IncompleteServeDeploymentConfig as error:
            incomplete = error
            continue
        if identity != owned[name]:
            raise RuntimeError(f"Serve application {name} was replaced or modified; refusing cleanup")
    if incomplete is not None:
        raise incomplete
    return sorted(applications)


def wait_owned_serve(
    snapshot, owned: dict, deadline: float, retry_interval: float = 0.5, retry_events: list | None = None
) -> list[str]:
    """Retry only missing deployment config; snapshot rechecks active jobs
    too."""
    while True:
        try:
            return validate_owned_serve(snapshot(), owned)
        except IncompleteServeDeploymentConfig as error:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    "Timed out waiting for complete owned Serve deployment config; no unchecked deletion"
                ) from error
            if retry_events is not None:
                retry_events.append({"unix": time.time(), "error": str(error), "remaining_seconds": remaining})
            time.sleep(min(retry_interval, remaining))


def delete_owned_serve(
    snapshot,
    delete,
    owned: dict,
    incomplete_timeout: float = 30.0,
    retry_interval: float = 0.5,
    retry_events: list | None = None,
) -> list[str]:
    """Validate the whole inventory, then recheck before each name-scoped
    delete."""
    if (
        not math.isfinite(incomplete_timeout)
        or not math.isfinite(retry_interval)
        or min(incomplete_timeout, retry_interval) <= 0
    ):
        raise ValueError("Serve transition timeout and retry interval must be finite and positive")
    # One deadline covers the entire deletion sequence; it never resets per app.
    deadline = time.monotonic() + incomplete_timeout
    names = wait_owned_serve(snapshot, owned, deadline, retry_interval, retry_events)
    deleted = []
    for name in names:
        if name in wait_owned_serve(snapshot, owned, deadline, retry_interval, retry_events):
            delete(name)
            deleted.append(name)
    return deleted
