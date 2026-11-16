# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""One background thread that runs queued jobs in submission order."""

from __future__ import annotations

import queue
import threading
from typing import Any, Callable


class AnalysisWorker:
    """Hand each submitted job to ``handle`` on a single daemon thread.

    Daemon so that a stuck job can never hold up process exit; ``handle`` must
    not raise.
    """

    def __init__(self, handle: Callable[[Any], None], name: str = "straggler-analyze") -> None:
        self._handle = handle
        self._queue: queue.Queue[Any] = queue.Queue()
        self._pending = 0
        self._cond = threading.Condition()
        self._stopped = False
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    @property
    def backlog(self) -> int:
        return self._pending

    def submit(self, job: Any) -> None:
        with self._cond:
            self._pending += 1
        self._queue.put(job)

    def flush(self, timeout: float | None) -> bool:
        """Wait until every submitted job has been handled."""
        with self._cond:
            return self._cond.wait_for(lambda: self._pending == 0, timeout)

    def stop(self) -> None:
        """Let already-queued jobs finish, then end the thread."""
        if not self._stopped:
            self._stopped = True
            self._queue.put(_STOP)

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            if job is _STOP:
                return
            try:
                self._handle(job)
            finally:
                with self._cond:
                    self._pending -= 1
                    self._cond.notify_all()


_STOP = object()
