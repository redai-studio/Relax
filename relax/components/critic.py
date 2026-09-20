# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import threading
import time
from argparse import Namespace
from typing import Any, Optional

import ray
import transfer_queue as tq
from ray import serve
from ray.serve.schema import LoggingConfig

from relax.components.base import Base
from relax.distributed.coordination import RolloutOffloadBarrier
from relax.distributed.ray.placement_group import allocate_train_group
from relax.engine.sft.runtime import sft_partition_id
from relax.utils.async_utils import run


@serve.deployment(
    logging_config=LoggingConfig(enable_access_log=False),
)
class Critic(Base):
    """Critic service for training the value model.

    Under PPO the critic co-hosts with the rollout on the same GPUs, so the
    train loop must wait for both the round-N rollout partition to land AND
    SGLang to finish offloading before waking up. Non-PPO callers pass a
    non-"ppo" ``advantage_estimator`` and skip that gate.
    """

    def __init__(
        self, healthy: Any, pgs: Optional[Any], num_gpus: int, config: Namespace, role: str, runtime_env: dict = None
    ) -> None:
        super().__init__()

        self.config = config
        self._lock = threading.RLock()
        self.healthy = healthy
        self.role = role

        tq.init(self.config.tq_config)
        self.data_system_client = tq.get_client()

        self.critic_model = allocate_train_group(
            args=config, num_gpus=num_gpus, pg=pgs, role=self.role, runtime_env=runtime_env
        )

        ray.get(self.critic_model.async_init(config, role=self.role, with_ref=False))
        self.step = getattr(self.config, "start_rollout_id", None) or 0
        # Wired by controller in colocate PPO to gate wake_up on SGLang offload.
        self._rollout_barrier: Optional[RolloutOffloadBarrier] = None

        # Detach the blocking train loop to a background thread so Serve's
        # user-event-loop probe stays green; without this the sync
        # `_wait_for_rollout_data` sleep hangs the async replica and it gets
        # restarted mid-run.
        self._run_thread: Optional[threading.Thread] = None
        self._done_event: Optional[asyncio.Event] = None
        self._thread_error: Optional[Exception] = None

    def set_barriers(
        self,
        *,
        rollout: Optional[RolloutOffloadBarrier] = None,
        peers: Any = None,  # accepted for interface parity; critic has no peers to wait on
    ) -> None:
        del peers
        self._rollout_barrier = rollout

    async def run(self) -> None:
        if self._run_thread is not None and self._run_thread.is_alive():
            if self._done_event is not None:
                await self._done_event.wait()
            if self._thread_error is not None:
                raise self._thread_error
            return

        loop = asyncio.get_running_loop()
        self._done_event = asyncio.Event()
        self._thread_error = None

        def _thread_target():
            try:
                self.train()
            except Exception as exc:
                error_msg = f"Critic training failed at step {self.step}: {type(exc).__name__}: {str(exc)}"
                self._logger.exception(error_msg)
                try:
                    self.healthy.report_error.remote("critic", error_msg)
                except Exception:
                    pass
                if not getattr(self.config, "use_health_check", False):
                    self._thread_error = exc
            finally:
                loop.call_soon_threadsafe(self._done_event.set)

        self._run_thread = threading.Thread(target=_thread_target, daemon=True)
        self._run_thread.start()
        await self._done_event.wait()
        if self._thread_error is not None:
            raise self._thread_error

    def _wait_for_rollout_data(self) -> None:
        """Block until this round's rollout data is ready and rollout is
        offloaded.

        Partition readiness comes from TransferQueue (rollout has produced
        round N). SGLang offload is gated by :class:`RolloutOffloadBarrier`
        when co-hosted with rollout on the same GPUs.
        """
        partition_id = sft_partition_id(self.config, self.step)
        while True:
            partition_list = run(self.data_system_client.async_get_partition_list())
            if partition_list is not None and partition_id in partition_list:
                break
            time.sleep(1)
        if getattr(self.config, "offload_rollout", False) and self._rollout_barrier is not None:
            self._rollout_barrier.wait_offloaded_sync()

    def train(self) -> None:
        from relax.algorithms import algorithm_needs_critic

        has_critic = algorithm_needs_critic(self.config)
        while self.step < self.config.num_rollout:
            if has_critic:
                self._wait_for_rollout_data()
            # In PPO colocate the actor waits for ``self.step`` to advance
            # past the current round before waking up, so block on training
            # completion here. Non-PPO critic is not on any live service graph
            # and keeps the historical fire-and-forget.
            train_ref = self.critic_model.async_train(self.step)
            if has_critic:
                ray.get(train_ref)
            # Note: save_model runs inside ``train_critic`` (backend) while the
            # model is still awake, so no explicit save call here.

            # In critic-only warmup, actor+advantages never consume the partition,
            # so critic must clear it itself; steady-state clearing stays with actor.
            if has_critic and self.step < getattr(self.config, "num_critic_only_steps", 0):
                run(self.data_system_client.async_clear_partition(partition_id=f"train_{self.step}"))

            try:
                self.healthy.update_heartbeat.remote("critic", self.step + 1)
            except Exception:
                pass

            self.step += 1
