# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Switching the profiler's level while training runs, through the metrics
service: an operator sets a level on a small HTTP page, the primary rank polls
for it on a daemon thread, and the window gather hands it to every rank. The
levels are ``core.stats.LEVELS``.
"""

from __future__ import annotations

import json
import logging
import threading
from time import time
from typing import Any, Callable

from relax.utils.straggler.core.stats import LEVELS


logger = logging.getLogger(__name__)


class ControlState:
    """Metrics-service side: the level last requested (``None`` until someone
    does) with a version that counts requests, and the status the primary rank
    reported in its last poll."""

    def __init__(self, clock: Callable[[], float] = time) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._requested: int | None = None
        self._version = 0
        self._report: dict[str, Any] | None = None

    def request(self, level: int) -> dict[str, Any]:
        if not 0 <= level < len(LEVELS):
            raise ValueError(f"straggler level must be one of 0..{len(LEVELS) - 1} ({', '.join(LEVELS)}), got {level}")
        with self._lock:
            self._requested = level
            self._version += 1
        return self.state()

    def poll(self, status: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._report = {**status, "time": self._clock()}
            return {"requested": self._requested, "version": self._version}

    def state(self) -> dict[str, Any]:
        with self._lock:
            return {
                "requested": self._requested,
                "version": self._version,
                "report": self._report,
                "now": self._clock(),
            }


class LevelPoller:
    """Primary rank: every ``interval`` seconds, report ``status()`` to the
    metrics service and keep the level it answers with in ``requested``.

    Runs on a daemon thread so the training thread never waits on HTTP; a
    failed poll keeps the previous answer.
    """

    def __init__(
        self,
        service_url: str,
        status: Callable[[], dict[str, Any]],
        interval: float = 5.0,
        timeout: float = 2.0,
        post: Callable[..., Any] | None = None,
    ) -> None:
        if post is None:
            import requests

            post = requests.post
        self.requested: int | None = None
        self._url = f"{service_url.rstrip('/')}/straggler/poll"
        self._status = status
        self._interval = interval
        self._timeout = timeout
        self._post = post
        self._failing = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="straggler-level-poller", daemon=True)
        self._thread.start()

    def poll_once(self) -> None:
        try:
            reply = self._post(self._url, json=self._status(), timeout=self._timeout)
            reply.raise_for_status()
            self.requested = reply.json()["requested"]
            self._failing = False
        except Exception as exc:
            if not self._failing:
                logger.warning(
                    "[straggler] level poll to %s failed (%s: %s); keeping the last level",
                    self._url,
                    type(exc).__name__,
                    exc,
                )
            self._failing = True

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self.poll_once()

    def stop(self) -> None:
        self._stop.set()


CONTROL_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Straggler profiler</title>
<style>
body { font: 14px sans-serif; margin: 2em; max-width: 44em; }
td { padding: .25em 1.5em .25em 0; }
button { font-size: 14px; margin-right: .6em; padding: .4em 1em; }
button.current { font-weight: bold; }
</style></head>
<body>
<h2>Straggler profiler</h2>
<table>
<tr><td>Requested level</td><td id="requested">-</td></tr>
<tr><td>Level in effect</td><td id="level">-</td></tr>
<tr><td>Latest finished rollout</td><td id="rollout">-</td></tr>
<tr><td>Profiler health</td><td id="health">-</td></tr>
<tr><td>Last report from the primary rank</td><td id="age">-</td></tr>
</table>
<p>
<button data-level="0">Pause</button>
<button data-level="1">Coarse</button>
<button data-level="2">Detail (sampled-step trace)</button>
<button data-level="3">Modules (detail + per-layer blocks)</button>
</p>
<p>A change takes effect at the next window boundary, at most RELAX_STRAGGLER_REPORT_INTERVAL rollouts
later. Detail and modules need a trace directory (RELAX_STRAGGLER_TRACE_DIR or --timeline-dump-dir);
without one they stay coarse. Modules times the blocks of every layer during the sampled step only.</p>
<script>
const LEVELS = __LEVELS__;
const api = location.pathname.replace(/\\/straggler\\/?$/, "") + "/straggler/control";
const name = (level) => (level === null || level === undefined ? "-" : LEVELS[level] || String(level));
async function refresh() {
  const state = await (await fetch(api)).json();
  const report = state.report || {};
  document.getElementById("requested").textContent = name(state.requested);
  document.getElementById("level").textContent = name(report.level);
  document.getElementById("rollout").textContent = report.rollout ?? "-";
  document.getElementById("health").textContent = report.health ?? "-";
  document.getElementById("age").textContent = report.time ? `${Math.round(state.now - report.time)} s ago` : "never";
  document.querySelectorAll("button").forEach((b) => b.classList.toggle("current", +b.dataset.level === report.level));
}
document.querySelectorAll("button").forEach((b) => {
  b.onclick = async () => {
    await fetch(api, {method: "POST", headers: {"Content-Type": "application/json"},
                      body: JSON.stringify({level: +b.dataset.level})});
    refresh();
  };
});
refresh();
setInterval(refresh, 5000);
</script>
</body></html>
""".replace("__LEVELS__", json.dumps(LEVELS))
