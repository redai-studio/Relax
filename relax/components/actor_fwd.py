# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import threading
from argparse import Namespace
from typing import Any, Optional

import ray
from fastapi import FastAPI
from ray import serve

from relax.components.base import Base
from relax.distributed.ray.placement_group import allocate_train_group
from relax.utils.tq.lifecycle import attach_tq_client


app = FastAPI()


@serve.deployment
@serve.ingress(app)
class ActorFwd(Base):
    def __init__(
        self, healthy: Any, pgs: Optional[Any], num_gpus: int, config: Namespace, role: str, runtime_env: dict = None
    ) -> None:
        super().__init__()

        self.config = config
        self._lock = threading.RLock()
        self.healthy = healthy
        self.role = role

        # Threading primitives so the computation loop doesn't block the Serve/FastAPI thread
        self._stop_event = threading.Event()
        self._run_thread = None
        self._done_event: Optional[asyncio.Event] = None
        self._thread_error: Optional[Exception] = None
        self.data_system_client = attach_tq_client(
            self.config.tq_config,
            role=self.role,
        )
        self.actor_model = allocate_train_group(args=config, num_gpus=num_gpus, pg=pgs, runtime_env=runtime_env)
        self.actor_model.init_and_wait(config, role=self.role, with_ref=False)
        self.step = 0

    async def run(self) -> None:
        if self._run_thread is not None and self._run_thread.is_alive():
            if self._done_event is not None:
                await self._done_event.wait()
            return
        self.data_system_client.reset_consumption(
            partition_id=f"train_{self.step}",
            task_name="actor_log_probs" if self.role == "actor_fwd" else "ref_log_probs",
        )
        loop = asyncio.get_running_loop()
        self._done_event = asyncio.Event()

        def _thread_target():
            try:
                self._background_run()
            except Exception as exc:
                self._thread_error = exc
            finally:
                loop.call_soon_threadsafe(self._done_event.set)

        self._thread_error = None
        self._run_thread = threading.Thread(target=_thread_target, daemon=True)
        self._run_thread.start()
        await self._done_event.wait()
        if self._thread_error is not None:
            raise self._thread_error

    def _background_run(self) -> None:
        try:
            while True:
                if self._stop_event.is_set():
                    self._logger.info(f"{self.role} background loop stopping by request")
                    break

                with self._lock:
                    local_step = self.step

                if local_step >= self.config.num_rollout:
                    self._logger.info(f"All {self.role} steps finished")
                    break

                self._logger.info(
                    f"{self.role} model computing log prob for step {local_step}/{self.config.num_rollout}"
                )
                if self.role == "actor_fwd":
                    self.compute_actor_log_prob(local_step)
                else:
                    self.compute_ref_log_prob(local_step)
                self._logger.info(
                    f"{self.role} model computed log prob for step {local_step}/{self.config.num_rollout}"
                )

                try:
                    self.healthy.update_heartbeat.remote(self.role, local_step + 1)
                except Exception:
                    pass

                with self._lock:
                    self.step += 1

                # Re-check stop flag after the long blocking compute call.
                # This minimises the window between stop() and actual termination.
                if self._stop_event.is_set():
                    self._logger.info(f"{self.role} stop detected after compute, discarding step {local_step} result")
                    break

        except Exception as e:
            error_msg = f"{self.role} failed at step {self.step}: {type(e).__name__}: {str(e)}"
            self._logger.exception(error_msg)
            self.healthy.report_error.remote(self.role, error_msg)
            if not getattr(self.config, "use_health_check", False):
                raise

    async def stop(self) -> None:
        self._stop_event.set()
        if self._run_thread is not None:
            # First try: wait a short time for the thread to notice the stop flag
            self._run_thread.join(timeout=5)
            if self._run_thread.is_alive():
                self._logger.warning(
                    f"{self.role} background thread did not stop within 5s "
                    "(likely blocked in compute/recv_weight), waiting longer..."
                )
                # Second try: give it enough time to finish the current
                # compute step or NCCL operation (up to 10 minutes)
                self._run_thread.join(timeout=600)
                if self._run_thread.is_alive():
                    self._logger.error(
                        f"{self.role} background thread still alive after 600s. "
                        "Proceeding with restart anyway — stale thread may leak."
                    )

    def recv_weight_fully_async(self) -> None:
        self.actor_model.recv_weight_fully_async(0)

    # --- HTTP endpoints for restart / recovery (bypass Ray Serve handle) ---

    @app.get("/get_step")
    def http_get_step(self) -> dict:
        return {"step": self.get_step()}

    @app.post("/set_step")
    def http_set_step(self, step: int) -> dict:
        self.set_step(step)
        return {"status": "ok"}

    @app.post("/stop_service")
    async def http_stop(self) -> dict:
        await self.stop()
        return {"status": "ok"}

    @app.post("/recv_weight_fully_async")
    def http_recv_weight_fully_async(self) -> dict:
        self.recv_weight_fully_async()
        return {"status": "ok"}

    def compute_actor_log_prob(self, step: int) -> None:
        ray.get(self.actor_model.async_compute_actor_log_prob(step))

    def compute_ref_log_prob(self, step: int) -> None:
        ray.get(self.actor_model.async_compute_ref_log_prob(step))
