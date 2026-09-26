# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import threading

import ray
from ray.exceptions import RayActorError

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


class RolloutHealthMonitor:
    """Health monitor for rollout engines.

    The monitor runs continuously once started, but can be paused/resumed
    based on whether the engines are offloaded (cannot health check when offloaded).

    Lifecycle:
    - start(): Start the monitor thread (called once during initialization)
    - pause(): Pause health checking (called when offloading engines)
    - resume(): Resume health checking (called when onloading engines)
    - stop(): Stop the monitor thread completely (called during dispose)
    """

    def __init__(self, engine_group, args):
        self._engine_group = engine_group

        self._thread = None
        self._stop_event = None
        self._pause_event = None  # When set, health checking is paused
        self._check_interval = args.rollout_health_check_interval
        self._check_timeout = args.rollout_health_check_timeout
        self._check_first_wait = args.rollout_health_check_first_wait
        self._max_consecutive_failures = args.rollout_health_check_max_consecutive_failures
        self._need_first_wait = True  # Need to wait after each resume
        self._is_checking_enabled = False  # Track if health checking should be active
        self._intentionally_removed: set = set()  # Engine indices being intentionally removed (scale-in)
        self._consecutive_failures: dict[int, int] = {}  # engine_id -> consecutive failure count

    def start(self) -> bool:
        """Start the health monitor thread. Called once during initialization.

        Returns:
            True if the monitor was started, False if there are no engines to monitor.
        """
        if not self._engine_group.all_engines:
            return False

        if self._thread is not None:
            logger.warning("Health monitor thread is already running.")
            return True

        logger.info("Starting RolloutHealthMonitor...")
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # Start in paused state until resume() is called
        self._thread = threading.Thread(
            target=self._health_monitor_loop,
            name="RolloutHealthMonitor",
            daemon=True,
        )
        self._thread.start()
        logger.info("RolloutHealthMonitor started (in paused state).")
        return True

    def stop(self) -> None:
        """Stop the health monitor thread completely.

        Called during dispose.
        """
        if not self._thread:
            return

        logger.info("Stopping RolloutHealthMonitor...")
        assert self._stop_event is not None
        self._stop_event.set()
        # Also clear pause to let the thread exit
        if self._pause_event:
            self._pause_event.clear()
        timeout = self._check_timeout + self._check_interval + 5
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            logger.warning("Rollout health monitor thread did not terminate within %.1fs", timeout)
        else:
            logger.info("RolloutHealthMonitor stopped.")

        self._thread = None
        self._stop_event = None
        self._pause_event = None
        self._is_checking_enabled = False

    def pause(self) -> None:
        """Pause health checking.

        Called when engines are offloaded.
        """
        if self._pause_event is None:
            return
        logger.info("Pausing health monitor...")
        self._pause_event.set()
        self._is_checking_enabled = False

    def resume(self) -> None:
        """Resume health checking.

        Called when engines are onloaded.
        """
        if self._pause_event is None:
            return
        logger.info("Resuming health monitor...")
        self._need_first_wait = True  # Need to wait after each resume
        self._pause_event.clear()
        self._is_checking_enabled = True

    def is_checking_enabled(self) -> bool:
        """Return whether health checking is currently enabled (not paused)."""
        return self._is_checking_enabled

    def mark_intentionally_removed(self, rollout_engine_id: int) -> None:
        self._intentionally_removed.add(rollout_engine_id)

    def _health_monitor_loop(self) -> None:
        assert self._stop_event is not None
        assert self._pause_event is not None

        while not self._stop_event.is_set():
            # Wait while paused
            while self._pause_event.is_set() and not self._stop_event.is_set():
                self._stop_event.wait(timeout=0.5)

            if self._stop_event.is_set():
                break

            # Do first wait after each resume (for large MoE models to be ready)
            if self._need_first_wait:
                logger.info(f"Health monitor doing first wait after resume: {self._check_first_wait}s")
                if self._stop_event.wait(self._check_first_wait):
                    logger.info("Health monitor stopped during first wait.")
                    break
                if self._pause_event.is_set():
                    # Got paused during first wait, skip this round and wait again next resume
                    logger.info("Health monitor paused during first wait, will wait again next resume.")
                    continue
                self._need_first_wait = False

            # Run health checks
            if not self._pause_event.is_set() and not self._stop_event.is_set():
                self._run_health_checks()

            # Wait for next check interval
            if self._stop_event.wait(self._check_interval):
                break

    def _run_health_checks(self) -> None:
        for rollout_engine_id, engine in enumerate(self._engine_group.engines):
            if self._stop_event is not None and self._stop_event.is_set():
                break
            if self._pause_event is not None and self._pause_event.is_set():
                break
            self._check_engine_health(rollout_engine_id, engine)

    def _check_engine_health(self, rollout_engine_id, engine) -> None:
        if rollout_engine_id in self._intentionally_removed:
            logger.debug(f"Skipping health check for engine {rollout_engine_id} (intentionally removed)")
            return

        if engine is None:
            logger.info(f"Skipping health check for engine {rollout_engine_id} (None)")
            return

        try:
            ray.get(engine.health_generate.remote(timeout=self._check_timeout))
        except RayActorError as e:
            # Hard death: the whole rollout actor (or its node) is gone. The
            # consecutive-failure counter can never recover from this, so bypass
            # it and kill/rebuild immediately.
            logger.error(
                f"Health check failed for rollout engine {rollout_engine_id} "
                f"(actor is dead). Killing actor immediately. Exception: {e}"
            )
            self._kill_engine(rollout_engine_id=rollout_engine_id)
        except Exception as e:
            # Soft failure: the actor is alive but the sglang subprocess is
            # unreachable (RayTaskError wrapping ConnectionError/Timeout, etc.).
            # Accumulate across resume windows until the threshold is hit.
            self._consecutive_failures[rollout_engine_id] = self._consecutive_failures.get(rollout_engine_id, 0) + 1
            failure_count = self._consecutive_failures[rollout_engine_id]
            if failure_count >= self._max_consecutive_failures:
                logger.error(
                    f"Health check failed for rollout engine {rollout_engine_id} "
                    f"({failure_count} consecutive failures, threshold={self._max_consecutive_failures}). "
                    f"Killing actor. Exception: {e}"
                )
                self._kill_engine(rollout_engine_id=rollout_engine_id)
            else:
                logger.warning(
                    f"Health check failed for rollout engine {rollout_engine_id} "
                    f"({failure_count}/{self._max_consecutive_failures} consecutive failures, "
                    f"will retry). Exception: {e}"
                )
        else:
            # Only a successful check clears the counter. This keeps "consecutive"
            # literal (a single success rescues an intermittently slow engine) and
            # prevents overload false-kills now that resume() no longer resets it.
            if self._consecutive_failures.get(rollout_engine_id, 0) > 0:
                logger.info(
                    f"Health check recovered for rollout engine {rollout_engine_id} "
                    f"after {self._consecutive_failures[rollout_engine_id]} consecutive failure(s)"
                )
            self._consecutive_failures[rollout_engine_id] = 0
            logger.debug(f"Health check passed for rollout engine {rollout_engine_id}")

    def _kill_engine(self, rollout_engine_id: int):
        if rollout_engine_id in self._intentionally_removed:
            logger.debug(f"Skipping kill for engine {rollout_engine_id} (intentionally removed)")
            return
        logger.info(f"Killing engine group {rollout_engine_id}...")
        for i in range(
            rollout_engine_id * self._engine_group.nodes_per_engine,
            (rollout_engine_id + 1) * self._engine_group.nodes_per_engine,
        ):
            engine = self._engine_group.all_engines[i]
            if engine:
                logger.info(f"Shutting down and killing engine at index {i}")
                try:
                    ray.get(engine.shutdown.remote())
                    ray.get(engine.unregister_dcs.remote())
                    ray.kill(engine)
                    logger.info(f"Successfully killed engine at index {i}")
                except Exception as e:
                    logger.warning(f"Fail to kill engine at index {i} (e: {e})")
            else:
                logger.info(f"Engine at index {i} is already None")
            self._engine_group.all_engines[i] = None
        # Drop any stale failure count so a rebuilt engine reusing this slot
        # starts from a clean slate instead of inheriting a near-threshold count.
        self._consecutive_failures.pop(rollout_engine_id, None)
