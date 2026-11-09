# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Case-independent release contracts for managed Session resources.

The launcher owns command start/cancel/join. A resource owner declares what
else that command owns and verifies its terminal release proof. These methods
must be bounded local operations: remote cleanup belongs to the agent's cancel
path, which remains alive under the launcher's cooperative termination budget.
Never start resources in an owner's constructor or verifier.
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
import stat
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


DEFAULT_RESOURCE_OWNER_PATH = "relax.agentic.runner.resources.ProcessGroupResourceOwner"
_MAX_RECEIPT_BYTES = 4096


@dataclass(frozen=True)
class ResourceLease:
    """One launch's immutable identity; never reused across Session
    attempts."""

    session_id: str
    group_id: str
    nonce: str
    release_path: Path


@dataclass(frozen=True)
class ResourceReleaseReceipt:
    """A terminal proof bound to the exact lease, not just a reusable slot."""

    session_id: str
    nonce: str

    def matches(self, lease: ResourceLease) -> bool:
        return self.session_id == lease.session_id and self.nonce == lease.nonce

    def to_payload(self) -> dict[str, Any]:
        return {"schema": 1, "session_id": self.session_id, "nonce": self.nonce, "status": "released"}

    @classmethod
    def from_payload(cls, value: Any) -> ResourceReleaseReceipt | None:
        if (
            not isinstance(value, dict)
            or set(value) != {"schema", "session_id", "nonce", "status"}
            or type(value["schema"]) is not int
            or value["schema"] != 1
            or value["status"] != "released"
            or not isinstance(value["session_id"], str)
            or not value["session_id"]
            or not isinstance(value["nonce"], str)
            or not value["nonce"]
        ):
            return None
        return cls(session_id=value["session_id"], nonce=value["nonce"])


def read_release_receipt(path: Path) -> ResourceReleaseReceipt | None:
    """Read a bounded regular file; absent, partial or invalid proofs fail
    closed."""

    descriptor = None
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            return None
        payload = os.read(descriptor, _MAX_RECEIPT_BYTES + 1)
        if len(payload) > _MAX_RECEIPT_BYTES:
            return None
        return ResourceReleaseReceipt.from_payload(json.loads(payload))
    except (OSError, ValueError):
        return None
    finally:
        if descriptor is not None:
            os.close(descriptor)


class SessionResourceOwner(ABC):
    """Extension boundary for resource scope and identity-bound release proof.

    Subclasses receive a lease before the command starts. The command receives
    ``launch_environment`` and must clean up its external resources on normal
    exit or cancellation. ``confirm_release`` runs only after launcher-owned
    process-group cleanup; it may return a matching receipt or ``None``. An
    exception, invalid proof, or expired cleanup deadline never returns credit.

    Confirmation is idempotent and must not allocate, launch or block on remote
    cleanup. Custom backends can use a local receipt written by their remote
    resource adapter, without introducing another asynchronous cleanup owner.
    """

    requires_external_cleanup: bool = False

    def __init__(self, lease: ResourceLease) -> None:
        self.lease = lease

    def launch_environment(self) -> Mapping[str, str]:
        return {
            "RELAX_REQUIRE_RESOURCE_RELEASE": "1" if self.requires_external_cleanup else "0",
            "RELAX_RESOURCE_RELEASE_PATH": str(self.lease.release_path),
            "RELAX_RESOURCE_RELEASE_NONCE": self.lease.nonce,
        }

    @abstractmethod
    def confirm_release(self) -> ResourceReleaseReceipt | None:
        """Return this scope's terminal proof after the command group is
        closed."""

    def diagnostics(self) -> Mapping[str, Any]:
        return {"owner": f"{type(self).__module__}.{type(self).__qualname__}"}


class ProcessGroupResourceOwner(SessionResourceOwner):
    """The complete resource scope is the launcher's local process group.

    This default needs no application receipt. It does not own remote services,
    containers managed elsewhere, or descendants that escape with ``setsid``.
    Such applications must select an external resource owner explicitly.
    """

    def confirm_release(self) -> ResourceReleaseReceipt:
        return ResourceReleaseReceipt(self.lease.session_id, self.lease.nonce)


class ReceiptResourceOwner(SessionResourceOwner):
    """External adapters acknowledge cleanup through the common file
    protocol."""

    requires_external_cleanup = True

    def confirm_release(self) -> ResourceReleaseReceipt | None:
        return read_release_receipt(self.lease.release_path)

    def diagnostics(self) -> Mapping[str, Any]:
        receipt = read_release_receipt(self.lease.release_path)
        return {
            **super().diagnostics(),
            "receipt_present": receipt is not None,
            "receipt_matches": receipt is not None and receipt.matches(self.lease),
        }


def load_resource_owner_type(path: str) -> type[SessionResourceOwner]:
    """Load an explicit resource contract; never infer ownership from case
    names."""

    if not isinstance(path, str):
        raise TypeError("Agent resource owner class path must be a string")
    module_name, separator, name = path.rpartition(".")
    if not separator or not module_name or not name:
        raise ValueError("Agent resource owner must be a fully qualified class path")
    owner_type = getattr(importlib.import_module(module_name), name)
    if not isinstance(owner_type, type) or not issubclass(owner_type, SessionResourceOwner):
        raise TypeError("Agent resource owner must subclass SessionResourceOwner")
    if type(owner_type.requires_external_cleanup) is not bool:
        raise TypeError("Resource owner requires_external_cleanup must be boolean")
    if any(
        not callable(getattr(owner_type, name)) for name in ("launch_environment", "confirm_release", "diagnostics")
    ):
        raise TypeError("Resource owner hooks must be callable")
    if any(
        inspect.iscoroutinefunction(getattr(owner_type, name))
        for name in ("launch_environment", "confirm_release", "diagnostics")
    ):
        raise TypeError("Resource owner hooks must be bounded synchronous local operations")
    if owner_type.__abstractmethods__:
        raise TypeError("Agent resource owner must implement confirm_release")
    return owner_type
