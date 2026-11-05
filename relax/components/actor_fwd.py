# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import threading
from argparse import Namespace
from typing import Any, Optional

import ray
from fastapi import FastAPI
from ray import serve

from relax.components.background_loop import BackgroundLoopRunner
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
        self._loop_runner = BackgroundLoopRunner()
        self.data_system_client = attach_tq_client(
            self.config.tq_config,
            role=self.role,
        )
        self.actor_model = allocate_train_group(args=config, num_gpus=num_gpus, pg=pgs, runtime_env=runtime_env)
        self.actor_model.init_and_wait(config, role=self.role, with_ref=False)
        self.step = 0

    async def run(self) -> None:
        await self._loop_runner.run(self._background_run, before_start=self._reset_consumption)

    def _reset_consumption(self) -> None:
        self.data_system_client.reset_consumption(
            partition_id=f"train_{self.step}",
            task_name="actor_log_probs" if self.role == "actor_fwd" else "ref_log_probs",
        )

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
        if not await self._loop_runner.join(timeout=5):
            self._logger.warning(
                f"{self.role} background thread did not stop within 5s "
                "(likely blocked in compute/recv_weight), waiting longer..."
            )
            # Wait for the current compute/NCCL operation; timeout does not cancel it.
            if not await self._loop_runner.join(timeout=600):
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
