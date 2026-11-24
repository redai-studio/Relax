# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Node-local reservations spanning preparation and device installation."""

from contextlib import contextmanager
from pathlib import Path

from relax.utils.logging_utils import get_logger

from .store import atomic_write, canonical, load_json, sync_directory, valid_id, writer_lock


logger = get_logger(__name__)


class NodeBudget:
    """Deployment limits include resident models, KV/graphs and recovery space.

    An interrupted installation retains its reservation until its owning
    consumer successfully recovers. A process exit alone does not prove that
    asynchronous worker/device operations have finished.
    """

    def __init__(self, directory: str | Path, limits: dict[str, int], resident: dict[str, int]) -> None:
        if not limits or any(type(value) is not int or value <= 0 for value in limits.values()):
            raise ValueError("Node budget requires positive resource limits")
        if any(
            key not in limits or type(value) is not int or not 0 <= value <= limits[key]
            for key, value in resident.items()
        ):
            raise ValueError("Resident resources exceed node capacity")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.limits, self.resident = dict(limits), dict(resident)
        with writer_lock(self.directory / ".budget", timeout=120):
            path = self.directory / "configuration.json"
            expected = {"limits": self.limits, "resident": self.resident}
            retained = load_json(path.read_bytes()) if path.exists() else None
            if retained is not None and retained != expected:
                expansion = (
                    retained["resident"] == self.resident
                    and retained["limits"].keys() == self.limits.keys()
                    and all(self.limits[key] >= value for key, value in retained["limits"].items())
                )
                if not expansion:
                    raise ValueError(
                        "Consumers sharing a node budget must use the same limits and resident allocation"
                    )
            if retained != expected:
                atomic_write(path, canonical(expected))

    @contextmanager
    def operation(self, owner: str, demand: dict[str, int], *, quiescent):
        valid_id(owner)
        if any(key not in self.limits or type(value) is not int or value < 0 for key, value in demand.items()):
            raise ValueError("Invalid operation resource request")
        # Different consumers may progress together if their summed reservations fit.
        # A recovering owner remains exclusive and keeps its previous allocation.
        with writer_lock(self.directory / f".operation-{owner}", timeout=120):
            path = self.directory / f"reservation-{owner}.json"
            with writer_lock(self.directory / ".budget", timeout=120):
                # Existing consumers observe an explicitly configured expansion
                # without discarding any in-flight owner's reservation.
                current = load_json((self.directory / "configuration.json").read_bytes())
                self.limits, self.resident = current["limits"], current["resident"]
                used = dict(self.resident)
                for existing in self.directory.glob("reservation-*.json"):
                    if existing != path:
                        for key, value in load_json(existing.read_bytes())["demand"].items():
                            used[key] = used.get(key, 0) + value
                # Recovering an owner retains at least its original allocation.
                retained = load_json(path.read_bytes())["demand"] if path.exists() else {}
                demand = {key: max(demand.get(key, 0), retained.get(key, 0)) for key in self.limits}
                shortages = {
                    key: {"required": used.get(key, 0) + value, "limit": self.limits[key]}
                    for key, value in demand.items()
                    if used.get(key, 0) + value > self.limits[key]
                }
                if shortages:
                    raise OSError(
                        f"Node capacity unavailable: {shortages}. Recover retained installations or increase the deployment budget, then retry."
                    )
                atomic_write(path, canonical({"owner": owner, "demand": demand}))
            try:
                yield
            finally:
                if quiescent():
                    try:
                        with writer_lock(self.directory / ".budget", timeout=120):
                            path.unlink(missing_ok=True)
                            sync_directory(path.parent)
                    except (OSError, TimeoutError):
                        logger.warning(
                            "Operation completed; resource reservation cleanup is pending and will retry for %s",
                            owner,
                            exc_info=True,
                        )
