# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU image-edit SFT producer; media stays on the training workers."""

import asyncio
from contextlib import suppress
from typing import Any

import torch
from ray import serve
from tensordict import TensorDict

from relax.components.base import Base
from relax.engine.sft.image import ImageBatchSampler
from relax.engine.sft.runtime import sft_partition_id, sft_partition_prefix
from relax.utils.tq.lifecycle import attach_tq_client


@serve.deployment
class ImageSFT(Base):
    def __init__(self, healthy: Any, pgs: Any, num_gpus: int, config: Any, role: str, runtime_env: Any = None) -> None:
        super().__init__()
        self.config, self.role, self.healthy = config, role, healthy
        self._partition_prefix = sft_partition_prefix(config)
        self.step = getattr(config, "start_rollout_id", 0) or 0
        self.sampler = ImageBatchSampler(config.i2i_sft_config["dataset_size"], config.global_batch_size, config.seed)
        self.data_system_client = attach_tq_client(config.tq_config, role=role)
        self.client = self.data_system_client
        self._stop_event = asyncio.Event()
        self._run_task: asyncio.Task | None = None

    async def run(self) -> None:
        if self._run_task is None:
            self._run_task = asyncio.create_task(self._produce())
        await self._run_task

    async def _wait_for_capacity(self) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + float(getattr(self.config, "sft_tq_timeout_minutes", None) or 30) * 60
        while not self._stop_event.is_set():
            partitions = await self.client.async_get_partition_list()
            pending = sum(str(part).startswith(self._partition_prefix) for part in (partitions or []))
            if pending < self.config.max_staleness + 1:
                return
            if loop.time() >= deadline:
                raise TimeoutError("Image SFT consumer did not drain TransferQueue")
            await asyncio.sleep(0.5)

    async def _produce(self) -> None:
        try:
            while self.step < self.config.num_rollout and not self._stop_event.is_set():
                await self._wait_for_capacity()
                if self._stop_event.is_set():
                    return
                ids = self.sampler.batch(self.step)
                await self.client.async_put(
                    data=TensorDict({"sample_id": torch.tensor(ids, dtype=torch.int64)}, batch_size=[len(ids)]),
                    partition_id=sft_partition_id(self.config, self.step),
                    custom_meta=[{"sample_index": self.step * len(ids) + index} for index in range(len(ids))],
                )
                self.step += 1
                self.healthy.update_heartbeat.remote("sft", self.step)
        except Exception as exc:
            self._healthy = False
            self._logger.exception("Image SFT producer failed at step %s", self.step)
            self.healthy.report_error.remote("sft", f"Image SFT: {type(exc).__name__}: {exc}", fatal=True)
            raise

    async def stop(self) -> None:
        self._stop_event.set()
        if self._run_task is not None:
            self._run_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._run_task
