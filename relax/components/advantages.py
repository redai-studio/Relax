# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import threading
from argparse import Namespace
from typing import Any, Dict

import torch
import transfer_queue as tq
from megatron.core import mpu
from ray import serve
from tensordict import TensorDict

from relax.components.base import Base
from relax.utils.async_utils import run as run_
from relax.utils.opd.opd_utils import (
    apply_opd_to_advantages,
    consume_opd_advantage_data,
)
from relax.utils.training.ppo_utils import (
    compute_approx_kl,
    get_advantages_and_returns_batch,
    get_grpo_returns,
    get_reinforce_plus_plus_baseline_advantages,
    get_reinforce_plus_plus_returns,
)


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

        tq.init(self.config.tq_config)
        self.data_system_client = tq.get_client()
        self.step = 0

    async def run(self) -> None:
        # Offload the blocking worker loop to a thread so it never blocks the
        # Serve replica's asyncio event loop. The loop below drives blocking
        # transfer-queue waits (run_(...).result()) and synchronous advantage
        # compute; running them inline blocks the event loop for the full tq
        # wait (20s+ under data imbalance), starving Serve's event-loop health
        # probe and triggering a false-positive "event loop unresponsive" kill.
        await asyncio.to_thread(self._run_blocking)

    def _run_blocking(self) -> None:
        step = self.step
        self.data_system_client.reset_consumption(
            partition_id=f"train_{self.step}",
            task_name="compute_advantages_and_returns",
        )
        try:
            while step < self.config.num_rollout:
                self._logger.info(
                    f"Start to got rollout_id: {step} data from transfer queue for compute advantages and returns."
                )
                # On the fully-async dynamic-batch path the partition is not
                # pre-allocated to global_batch_size, so the tensor-wide .all()
                # consumption check is unreliable — use the producer-driven drained
                # predicate (production_completed AND all inserted samples consumed).
                _stream_drain = getattr(self.config, "fully_async", False) and getattr(
                    self.config, "use_dynamic_batch_size", False
                )
                _drain_check = (
                    self.data_system_client.async_check_stream_drained
                    if _stream_drain
                    else self.data_system_client.async_check_consumption_status
                )

                while not run_(_drain_check("compute_advantages_and_returns", f"train_{step}")):
                    adv_data_fields = [
                        "tokens",
                        "total_lengths",
                        "response_lengths",
                        "loss_masks",
                        "rollout_log_probs",
                        "rewards",
                    ]
                    # log_probs is only needed for KL divergence; in true on-policy mode
                    # the actor_fwd role is absent and log_probs is not produced upstream.
                    if not getattr(self.config, "true_on_policy_mode", False):
                        adv_data_fields.append("log_probs")
                    if self.config.kl_coef != 0 or self.config.use_kl_loss:
                        adv_data_fields.append("ref_log_probs")
                    consume_opd_advantage_data(adv_data_fields, self.config)
                    batch_meta = run_(
                        self.data_system_client.async_get_meta(
                            data_fields=adv_data_fields,
                            batch_size=self.config.global_batch_size // self.config.num_iters_per_train_update,
                            partition_id=f"train_{step}",
                            task_name="compute_advantages_and_returns",
                        )  # type: ignore
                    )

                    if batch_meta.size == 0:
                        continue
                    rollout_data = run_(self.data_system_client.async_get_data(batch_meta))
                    self._logger.info(
                        f"Successfully got rollout_id: {step} data from transfer queue for compute advantages and returns."
                    )
                    advantages_and_returns = self.compute_advantages_and_returns(rollout_data)
                    advantages_and_returns = TensorDict(advantages_and_returns, batch_size=[len(batch_meta.samples)])
                    run_(self.data_system_client.async_put(data=advantages_and_returns, metadata=batch_meta))
                self._logger.info(f"Successfully run compute advantages and returns for step {step}.")
                step += 1
        except Exception as e:
            error_msg = f"Advantage computation failed at step {self.step}: {type(e).__name__}: {str(e)}"
            self._logger.exception(error_msg)
            self.healthy.report_error.remote("advantage", error_msg)
            if not getattr(self.config, "use_health_check", False):
                raise

    def compute_advantages_and_returns(self, rollout_data: Dict[str, Any]) -> Dict[str, Any] | None:
        """Compute advantages and returns based on
        `self.config.advantage_estimator`.

        This function extracts rewards, log-probs, values, and masks from
        `rollout_data`, computes KL divergences, then applies the chosen advantage
        estimator. Supported methods: "grpo", "gspo", "sapo", "cispo",
        "rloo", "ppo", "reinforce_plus_plus", and
        "reinforce_plus_plus_baseline".

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

        if self.config.advantage_estimator in ["grpo", "gspo", "sapo", "cispo", "m2po", "rloo"]:
            returns = get_grpo_returns(rewards, kl)
            advantages = list(returns)  # make a copy

        elif self.config.advantage_estimator == "ppo":
            # TODO: optimize this
            old_rewards = rewards
            rewards = []
            for reward, k in zip(old_rewards, kl, strict=False):
                k *= -self.config.kl_coef
                cp_rank = mpu.get_context_parallel_rank()
                if cp_rank == 0:
                    k[-1] += reward
                rewards.append(k)
            advantages, returns = get_advantages_and_returns_batch(
                total_lengths, response_lengths, values, rewards, self.config.gamma, self.config.lambd
            )

        elif self.config.advantage_estimator == "reinforce_plus_plus":
            rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
            returns = get_reinforce_plus_plus_returns(
                rewards=rewards,
                kl=kl,
                loss_masks=loss_masks,
                response_lengths=response_lengths,
                total_lengths=total_lengths,
                kl_coef=self.config.kl_coef,
                gamma=self.config.gamma,
            )
            advantages = list(returns)

        elif self.config.advantage_estimator == "reinforce_plus_plus_baseline":
            rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
            advantages = get_reinforce_plus_plus_baseline_advantages(
                rewards=rewards,
                kl=kl,
                loss_masks=loss_masks,
            )
            returns = advantages

        else:
            raise NotImplementedError(f"advantage_estimator {self.config.advantage_estimator} is not supported. ")

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
