# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import threading
from argparse import Namespace
from typing import Any, Dict

import torch
from ray import serve
from tensordict import TensorDict

from relax.algorithms.advantages import compute_advantages_and_returns
from relax.components.base import Base
from relax.core.inference_recovery import RecoveryGate
from relax.utils.async_utils import run as run_
from relax.utils.opd.opd_utils import (
    apply_opd_to_advantages,
    consume_opd_advantage_data,
)
from relax.utils.tq.lifecycle import attach_tq_client
from relax.utils.training.ppo_utils import compute_approx_kl


@serve.deployment
class Advantages(Base):
    """Service for computing advantages and returns from rollout data."""

    def __init__(
        self, healthy: Any, pgs: Any, num_gpus: int, config: Namespace, role: str, runtime_env: dict | None = None
    ) -> None:
        super().__init__()

        self.config = config
        self._lock = threading.RLock()
        self.healthy = healthy
        self.role = role
        self._recovery = RecoveryGate(config, role)

        self.data_system_client = attach_tq_client(
            self.config.tq_config,
            role="advantages",
        )
        self.step = 0

    async def run(self) -> None:
        # Offload the blocking worker loop to a thread so it never blocks the
        # Serve replica's asyncio event loop. The loop below drives blocking
        # transfer-queue waits (run_(...).result()) and synchronous advantage
        # compute; running them inline blocks the event loop for the full tq
        # wait (20s+ under data imbalance), starving Serve's event-loop health
        # probe and triggering a false-positive "event loop unresponsive" kill.
        await asyncio.to_thread(self._run_blocking)

    def _advantage_fields(self) -> list[str]:
        fields = ["tokens", "total_lengths", "response_lengths", "loss_masks", "rollout_log_probs", "rewards"]
        if not getattr(self.config, "true_on_policy_mode", False):
            fields.append("log_probs")
        if self.config.kl_coef != 0 or self.config.use_kl_loss:
            fields.append("ref_log_probs")
        consume_opd_advantage_data(fields, self.config)
        return fields

    def _run_blocking(self) -> None:
        self.data_system_client.reset_consumption(
            partition_id=f"train_{self.step}", task_name="compute_advantages_and_returns"
        )
        fields = self._advantage_fields()
        try:
            while self.step < self.config.num_rollout:
                step = self._recovery.admit(self.step, self.data_system_client, fields)
                if step is None:
                    break
                self.step = step
                try:
                    self._compute_partition(step, fields)
                    self.step = step + 1
                except Exception as exc:
                    self._recovery.fail(exc)
                    raise
                finally:
                    self._recovery.leave()
        except Exception as e:
            error_msg = f"Advantage computation failed at step {self.step}: {type(e).__name__}: {str(e)}"
            self._logger.exception(error_msg)
            self.healthy.report_error.remote("advantage", error_msg)
            if self._recovery.state is not None or not getattr(self.config, "use_health_check", False):
                raise

    def _compute_partition(self, step: int, fields: list[str]) -> None:
        self._logger.info(f"Getting rollout_id: {step} data for compute advantages and returns.")
        # Dynamic streams use producer completion rather than preallocated consumption bits.
        stream_drain = getattr(self.config, "fully_async", False) and getattr(
            self.config, "use_dynamic_batch_size", False
        )
        drain_check = (
            self.data_system_client.async_check_stream_drained
            if stream_drain
            else self.data_system_client.async_check_consumption_status
        )
        while not run_(drain_check("compute_advantages_and_returns", f"train_{step}")):
            batch_meta = run_(
                self.data_system_client.async_get_meta(
                    data_fields=fields,
                    batch_size=self.config.global_batch_size // self.config.num_iters_per_train_update,
                    partition_id=f"train_{step}",
                    task_name="compute_advantages_and_returns",
                )
            )
            if batch_meta.size == 0:
                continue
            rollout_data = run_(self.data_system_client.async_get_data(batch_meta))
            advantages_and_returns = self.compute_advantages_and_returns(rollout_data)
            advantages_and_returns = TensorDict(advantages_and_returns, batch_size=[len(batch_meta.samples)])
            # Actor can delete the partition as soon as the final advantages become readable.
            partition_drained = getattr(self.config, "fully_async", False) and run_(
                self.data_system_client.async_check_stream_drained("compute_advantages_and_returns", f"train_{step}")
            )
            run_(self.data_system_client.async_put(data=advantages_and_returns, metadata=batch_meta))
            if partition_drained:
                break
        self._logger.info(f"Successfully run compute advantages and returns for step {step}.")

    def compute_advantages_and_returns(self, rollout_data: Dict[str, Any]) -> Dict[str, Any] | None:
        """Compute advantages and returns based on
        `self.config.advantage_estimator`.

        This function extracts rewards, log-probs, values, and masks from
        `rollout_data`, computes KL divergences, then delegates to the advantage
        implementation named by the registry for `self.config.advantage_estimator`
        (see `relax.algorithms.advantages.ADVANTAGE_FNS`).

        Early returns if both `log_probs` and `values` are None (intermediate
        pipeline stages).

        Args:
            rollout_data: Dict containing input lists ("log_probs", "ref_log_probs",
                "rewards", "values", "response_lengths", "loss_masks",
                "total_lengths"). Modified in-place to add "advantages" and
                "returns" keys, each mapping to lists of tensors per sample.

        Returns:
            A dict with keys "advantages" and "returns" containing nested tensors.
        """

        log_probs: list[torch.Tensor] = rollout_data.get(
            "rollout_log_probs" if self.config.use_rollout_logprobs else "log_probs"
        )
        ref_log_probs: list[torch.Tensor] = rollout_data.get("ref_log_probs")
        rewards: torch.Tensor = rollout_data.get("rewards")
        values: None | list[torch.Tensor] = rollout_data.get("values")
        response_lengths: list[int] = rollout_data.get("response_lengths")
        loss_masks: list[torch.Tensor] = rollout_data.get("loss_masks")
        total_lengths: list[int] = rollout_data.get("total_lengths")
        # In true on-policy mode log_probs is not fetched (actor_fwd absent);
        # rollout_log_probs has identical shape and serves as a kl-zero template.
        rollout_log_probs: list[torch.Tensor] = rollout_data.get("rollout_log_probs")

        # return when not the last pp stage.
        if log_probs is None and values is None and rollout_log_probs is None:
            return

        if self.config.kl_coef == 0 or not log_probs:
            # when kl_coef is 0, we won't compute ref_log_prob
            xs = log_probs if log_probs is not None else (values if values is not None else rollout_log_probs)
            kl = [torch.zeros_like(x, dtype=torch.float32, device=x.device) for x in xs]
        else:
            kl = [
                compute_approx_kl(
                    log_probs[i],
                    ref_log_probs[i],
                    kl_loss_type=self.config.kl_loss_type,
                )
                for i in range(len(log_probs))
            ]

        advantages, returns = compute_advantages_and_returns(
            self.config,
            rewards=rewards,
            kl=kl,
            loss_masks=loss_masks,
            response_lengths=response_lengths,
            total_lengths=total_lengths,
            values=values,
            # `padded_total_lengths` is intentionally omitted: it is derived from
            # `args.qkv_format` and the VL / unsplit-forward flags in the Megatron
            # worker, and this deployment has no equivalent. Passing nothing keeps
            # the behaviour this path already had.
        )

        # Optional pure OPD mode: remove all non-OPD reward contribution.
        if getattr(self.config, "use_opd", False) and getattr(self.config, "opd_only_reward", False):
            advantages = [torch.zeros_like(a) for a in advantages]
            returns = [torch.zeros_like(r) for r in returns]

        # Apply on-policy distillation KL penalty to advantages (orthogonal to advantage estimator)
        if getattr(self.config, "use_opd", False):
            apply_opd_to_advantages(self.config, rollout_data, advantages)

        return {
            "advantages": torch.nested.nested_tensor(advantages),
            "returns": torch.nested.nested_tensor(returns),
        }
