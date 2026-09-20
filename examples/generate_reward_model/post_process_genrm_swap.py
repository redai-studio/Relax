# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Post-rollout GenRM scorer with rollout/genrm sleep-wake swap.

Enables verl-style "colocate mode" reward on Relax: rollout owns all GPUs
during generation (GenRM asleep), then this function fires once per rollout
batch to offload rollout, wake GenRM, batch-score all samples, and put GenRM
back to sleep.

Wire-up (in the training script):
  --rm-type dummy                        # inline reward is a no-op
  --defer-reward-to-post-process         # actor.update_weights skips GenRM onload
  --custom-reward-post-process-path <path to this file>

Assumptions:
- Shared-bundles colocate: rollout_num_gpus == genrm_num_gpus == actor_total.
- GenRMManager is created with name="relax_genrm_manager" (see
  relax/distributed/ray/placement_group.py::create_genrm_manager).
- We run inside the RolloutManager Ray actor's process, so rollout offload
  goes through the in-process singleton (get_local_rollout_manager) to avoid
  a self-remote-call deadlock. GenRM offload/onload goes through the Ray
  handle (cross-actor call).
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor

import httpx
import ray
import torch

from relax.distributed.ray.rollout import get_local_rollout_manager
from relax.engine.rewards.dapo_genrm import (
    MAX_ANSWER_LEN,
    _extract_answer,
    _format_messages,
)
from relax.utils.logging_utils import get_logger
from relax.utils.utils import get_serve_url


logger = get_logger(__name__)


def _run_async(coro):
    """Run an async coroutine from a synchronous caller even when the calling
    thread already has a running event loop (Ray AsyncActor case).

    A fresh thread + fresh loop side-steps 'asyncio.run() cannot be called from
    a running event loop'.
    """
    with ThreadPoolExecutor(1) as ex:
        return ex.submit(lambda: asyncio.run(coro)).result()


async def _score_one(client, url, question, ground_truth, predict_str):
    """Format short-circuit → judge call → loose parse.

    Mirrors dapo_genrm.async_compute_score_genrm but takes the httpx client as
    an argument so the whole batch shares one connection pool, and the pool is
    created fresh per post_process invocation (see _score_all).
    """
    answer_text = _extract_answer(predict_str)
    if answer_text is None:
        return 0.0
    if len(answer_text) > MAX_ANSWER_LEN:
        return 0.0
    messages = _format_messages(question, ground_truth, answer_text)
    try:
        resp = await client.post(url, json={"messages": messages})
        resp.raise_for_status()
        judge_response = resp.json().get("response", "")
    except Exception as e:
        logger.error(f"GenRM judge call failed, degrading to 0: {e}")
        return 0.0
    prediction = judge_response.strip()
    if "Judgement:" in prediction:
        prediction = prediction.split("Judgement:")[-1].strip()
    head = prediction[:16]
    if "1" in head:
        return 1.0
    return 0.0


async def _score_all(samples):
    # Build a fresh AsyncClient inside the new event loop each call. Do NOT
    # reuse relax.utils.genrm_client.get_genrm_client()'s singleton: it binds
    # its httpx.AsyncClient transport to the loop it was first created on,
    # and our _run_async spins up a new loop per invocation, so any reuse of
    # the old client raises "TCPTransport closed: the handler is closed".
    url = f"{get_serve_url('genrm').rstrip('/')}/generate"
    async with httpx.AsyncClient(timeout=1800.0) as client:
        tasks = []
        for s in samples:
            metadata = s.metadata if isinstance(s.metadata, dict) else {}
            question = metadata.get("question", getattr(s, "prompt", ""))
            ground_truth = metadata.get("label", getattr(s, "label", ""))
            tasks.append(_score_one(client, url, question, ground_truth, s.response))
        return await asyncio.gather(*tasks)


def _grpo_normalize(args, raw_rewards):
    """Replicates the default group normalization in
    relax.utils.utils.post_process_rewards.

    Which algorithms normalize, and which of those also divide by the group
    standard deviation, comes from the algorithm registry rather than from a
    copy of the whitelist — a copy would silently go stale the next time an
    algorithm is added.

    The gate names the two normalizers this function actually reimplements
    rather than asking ``spec.is_group_normalized``: that property would also be
    true of a future normalizer computing something else entirely, and this
    reimplementation would then silently diverge from it. Naming normalizers
    keeps it registry-driven -- a new algorithm reusing either one is covered
    for free, and a genuinely new normalizer is exactly the case where a human
    needs to look at this function.
    """
    from relax.algorithms import get_algorithm

    spec = get_algorithm(args.advantage_estimator)
    if spec.reward_normalizer not in ("group_mean", "group_mean_std") or not args.rewards_normalization:
        return raw_rewards
    rewards = torch.tensor(raw_rewards, dtype=torch.float)
    if rewards.shape[-1] == args.n_samples_per_prompt * args.rollout_batch_size:
        rewards = rewards.reshape(-1, args.n_samples_per_prompt)
    else:
        rewards = rewards.view(-1, rewards.shape[-1])
    mean = rewards.mean(dim=-1, keepdim=True)
    rewards = rewards - mean
    if spec.reward_normalizer == "group_mean_std" and args.grpo_std_normalization:
        std = rewards.std(dim=-1, keepdim=True)
        rewards = rewards / (std + 1e-6)
    return rewards.flatten().tolist()


def custom_reward_post_process(args, samples):
    """Sync entry called by relax.utils.utils.post_process_rewards.

    Flow (one call per rollout batch):
      1. Offload rollout (in-process direct call — same actor).
      2. Onload GenRM (Ray handle — different actor).
      3. Batch-score every sample via GenRM HTTP.
      4. Offload GenRM.
      5. Return normalized rewards — leave rollout offloaded (update_weights
         re-onloads it next iteration; skipping the redundant onload here
         saves one full weights+KV round trip).
    """
    # Flatten if grouped
    if samples and isinstance(samples[0], list):
        flat_samples = [s for group in samples for s in group]
    else:
        flat_samples = list(samples)

    rollout = get_local_rollout_manager()
    rollout._offload_local()

    genrm = ray.get_actor("relax_genrm_manager")
    ray.get(genrm.onload.remote())
    try:
        raw_rewards = _run_async(_score_all(flat_samples))
    finally:
        ray.get(genrm.offload.remote())

    reward_key = getattr(args, "reward_key", None)
    for sample, score in zip(flat_samples, raw_rewards, strict=True):
        sample.reward = {reward_key: score} if reward_key else score

    return _grpo_normalize(args, raw_rewards)
