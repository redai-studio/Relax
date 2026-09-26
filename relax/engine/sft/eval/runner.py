# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""SFT periodic causal-LM PPL and sequence-classification eval driver.

Extracted from ``backends/megatron/actor.py`` so the actor file stays focused
on the generic training loop. The runner is function-style: it takes the
Megatron actor as a duck-typed handle (needs ``args``, ``model``,
``data_system_client``, ``all_consumed``, ``_get_data_from_transfer_queue``).

Backend imports are done lazily inside ``run_sft_eval`` so importing this
module never pulls in Megatron / NCCL — the controller-side bootstrap can
import ``relax.engine.sft`` safely.
"""

import re
import time

import torch
import torch.distributed as dist
from megatron.core import mpu

from relax.utils import device as device_utils
from relax.utils import tracking_utils
from relax.utils.async_utils import run
from relax.utils.distributed_utils import get_gloo_group
from relax.utils.logging_utils import get_logger
from relax.utils.metrics.metric_utils import compute_rollout_step
from relax.utils.timer import timer


logger = get_logger(__name__)


def _wait_for_eval_chunk_count(actor, rollout_id: int) -> int:
    """Discover the number of eval chunks N for ``rollout_id`` from TQ.

    The producer pushes partitions named ``sft_eval_<rollout_id>_n<N>_<i>``
    serially with backpressure. Only TP/PP/CP rank 0 polls the partition list;
    N is broadcast to the rest of the ranks so they all enter the chunk loop in
    lockstep.
    """
    pat = re.compile(rf"^sft_eval_{rollout_id}_n(\d+)_\d+$")
    is_query_rank = (
        mpu.get_tensor_model_parallel_rank() == 0
        and mpu.get_pipeline_model_parallel_rank() == 0
        and mpu.get_context_parallel_rank() == 0
    )
    n = 0
    if is_query_rank:
        while True:
            partitions = run(actor.data_system_client.async_get_partition_list())
            if partitions:
                for p in partitions:
                    m = pat.match(p)
                    if m:
                        n = int(m.group(1))
                        break
            if n > 0:
                break
            time.sleep(1)
    n_t = torch.tensor([n], device=device_utils.make_current_torch_device(), dtype=torch.long)
    dist.broadcast(n_t, group=mpu.get_context_parallel_group(), group_src=0)
    dist.broadcast(n_t, group=mpu.get_tensor_model_parallel_group(), group_src=0)
    dist.broadcast(n_t, group=mpu.get_pipeline_model_parallel_group(), group_src=0)
    return int(n_t[0].item())


def _wait_for_eval_partition_present(actor, partition_id: str) -> None:
    """Wait until ``partition_id`` shows up in the TQ partition list.

    Only TP/PP/CP rank 0 polls; the synchronization across ranks happens
    implicitly inside the subsequent ``all_consumed`` broadcasts.
    """
    is_query_rank = (
        mpu.get_tensor_model_parallel_rank() == 0
        and mpu.get_pipeline_model_parallel_rank() == 0
        and mpu.get_context_parallel_rank() == 0
    )
    if not is_query_rank:
        return
    while True:
        partitions = run(actor.data_system_client.async_get_partition_list())
        if partitions and partition_id in partitions:
            return
        time.sleep(1)


def run_sft_eval(actor, rollout_id: int) -> None:
    """Consume eval partitions and log PPL or classification metrics.

    Producer chunks the eval set into ``global_batch_size``-sized pieces
    and pushes them serially to ``sft_eval_<rollout_id>_n<N>_<i>``. We
    discover N from any present partition's name, then drain each chunk
    in order with the existing ``all_consumed`` loop.

    Under CP > 1 each callback emits CP-local sufficient statistics, and the
    final all-reduce includes the CP group so each sample is counted once.
    """
    from relax.engine.sft.runtime import is_preference_mode

    if is_preference_mode(actor.args):
        _run_preference_eval(actor, rollout_id)
        return

    # Lazy imports: keep this module importable without Megatron initialized.
    from relax.backends.megatron.data import get_data_iterator
    from relax.backends.megatron.initialize import is_megatron_main_rank
    from relax.backends.megatron.model import forward_only

    is_classification = getattr(actor.args, "task_type", "causal_lm") == "seq_cls"
    if is_classification:
        from relax.engine.sft.eval.classification import (
            compute_classification_eval_step,
            compute_classification_metrics,
        )
    else:
        from relax.engine.sft.eval.ppl import compute_ppl_metrics, compute_sft_eval_step

    args = actor.args
    task_name = "sft_eval"
    # Eval consume chunk size mirrors the train micro-batching so that
    # `get_data_iterator` can build pipeline microbatches the same way.
    batch_size = args.global_batch_size // mpu.get_data_parallel_world_size(with_context_parallel=False)
    data_fields = ["tokens", "loss_masks", "total_lengths", "response_lengths"]
    if is_classification:
        data_fields.extend(["classification_labels", "sample_weights"])
    if args.multimodal_keys is not None:
        data_fields.append("multimodal_train_inputs")

    n_chunks = _wait_for_eval_chunk_count(actor, rollout_id)

    stat_keys = (
        ("loss_sum", "num_examples", "correct")
        if args.problem_type == "single_label_classification"
        else ("loss_sum", "num_examples", "tp", "fp", "fn", "exact_match")
    )
    local_stats = {key: 0.0 for key in stat_keys} if is_classification else {}
    local_neg_log_prob = 0.0
    local_num_tokens = 0
    _t_eval_start = time.monotonic()
    with timer("sft_eval"):
        for chunk_idx in range(n_chunks):
            partition_id = f"sft_eval_{rollout_id}_n{n_chunks}_{chunk_idx}"
            _wait_for_eval_partition_present(actor, partition_id)
            batch_index = 0
            while not actor.all_consumed(task_name, rollout_id, partition_id=partition_id):
                rollout_data, _batch_meta = actor._get_data_from_transfer_queue(
                    task_name, rollout_id, data_fields, batch_size, batch_index, partition_id=partition_id
                )
                if rollout_data is None:
                    continue
                batch_index += 1
                data_iterator, num_microbatches = get_data_iterator(args, actor.model, rollout_data)
                eval_step = compute_classification_eval_step if is_classification else compute_sft_eval_step
                per_mb = forward_only(
                    eval_step,
                    args,
                    actor.model,
                    data_iterator,
                    num_microbatches,
                    store_prefix="",
                    per_sample_output=False,
                )
                if mpu.is_pipeline_last_stage():
                    if is_classification:
                        for key in stat_keys:
                            for value in per_mb.get(key, []):
                                local_stats[key] += float(value.item())
                    else:
                        for sum_t in per_mb.get("sum_neg_log_prob", []):
                            local_neg_log_prob += float(sum_t.item())
                        for cnt_t in per_mb.get("num_tokens", []):
                            local_num_tokens += int(cnt_t.item())
            # Free this chunk so the producer's _wait_for_partition_drained
            # can return and push the next chunk. TQ never auto-cleans
            # consumed partitions.
            dist.barrier(group=get_gloo_group())
            if dist.get_rank() == 0:
                run(actor.data_system_client.async_clear_partition(partition_id=partition_id))

    device = device_utils.make_current_torch_device()
    if is_classification:
        agg = torch.tensor([local_stats[key] for key in stat_keys], device=device, dtype=torch.float64)
    else:
        agg = torch.tensor([local_neg_log_prob, float(local_num_tokens)], device=device, dtype=torch.float64)
    # Only the last PP stage holds non-zero values; SUM across PP propagates
    # them to every PP rank without needing a global src rank lookup.
    dist.all_reduce(agg, op=dist.ReduceOp.SUM, group=mpu.get_pipeline_model_parallel_group())
    # DP+CP all-reduce: each CP rank holds a partial sum over its zigzag
    # slice (callback uses the chunked loss mask for both numerator and
    # denominator), so we must include CP in the reduce to recover the
    # full-sequence totals; with_context_parallel=True is a no-op when
    # CP == 1.
    dist.all_reduce(agg, op=dist.ReduceOp.SUM, group=mpu.get_data_parallel_group(with_context_parallel=True))

    if is_classification:
        totals = {key: float(agg[i].item()) for i, key in enumerate(stat_keys)}
        metrics = compute_classification_metrics(args.problem_type, totals)
    else:
        total_neg_log_prob = float(agg[0].item())
        total_tokens = int(agg[1].item())
        metrics = compute_ppl_metrics(total_neg_log_prob, total_tokens)
    metrics["perf/sft_eval_time"] = time.monotonic() - _t_eval_start
    if is_megatron_main_rank():
        # Inject the step under `rollout/step` so non-wandb backends
        # (tensorboard / clearml / apprise / metrics-service) can index by
        # it — they read `metrics[step_key]` directly. Mirrors
        # `backends/megatron/data.py:368-370`.
        step = compute_rollout_step(args, rollout_id)
        metrics["rollout/step"] = step
        tracking_utils.log(args, metrics, step_key="rollout/step")
        # Eval finishes AFTER the train loop's own flush_metrics for this
        # step, so the just-buffered eval metrics would otherwise sit in
        # MetricsService until the next flush (which is keyed on a later
        # step) and never reach ClearML/W&B/TB.
        tracking_utils.flush_metrics(args, step)
        logger.info(f"SFT eval @ rollout_id={rollout_id}: {metrics}")


def _run_preference_eval(actor, rollout_id: int) -> None:
    """Evaluate DPO or RM on pair rows using the same TQ packing as
    training."""
    from relax.backends.megatron.data import expand_preference_rollout_data, get_data_iterator
    from relax.backends.megatron.initialize import is_megatron_main_rank
    from relax.backends.megatron.model import forward_only
    from relax.engine.sft.eval.acceptance import (
        PREFERENCE_PROBE_PAIR_COUNT,
        preference_eval_chunk_sizes,
        preference_eval_local_batch_sizes,
    )
    from relax.engine.sft.eval.preference import (
        compute_reward_model_eval_step,
        extract_preference_eval_pair_ids,
        finalize_pair_metrics,
        pair_metric_sums,
    )
    from relax.utils.training.preference_utils import dpo_pair_loss, reward_model_pair_loss

    args = actor.args
    task_name = "sft_eval"
    dp_size = mpu.get_data_parallel_world_size(with_context_parallel=False)
    data_fields = [
        "pair_ids",
        "chosen_tokens",
        "rejected_tokens",
        "chosen_loss_masks",
        "rejected_loss_masks",
        "chosen_total_lengths",
        "rejected_total_lengths",
        "chosen_score_positions",
        "rejected_score_positions",
    ]
    n_chunks = _wait_for_eval_chunk_count(actor, rollout_id)
    chunk_sizes = preference_eval_chunk_sizes(PREFERENCE_PROBE_PAIR_COUNT, args.global_batch_size)
    local_batch_sizes = preference_eval_local_batch_sizes(PREFERENCE_PROBE_PAIR_COUNT, args.global_batch_size, dp_size)
    if len(chunk_sizes) != n_chunks:
        raise RuntimeError(f"preference eval chunk plan mismatch: producer={n_chunks}, consumer={len(chunk_sizes)}")
    local = torch.zeros(7, device=device_utils.make_current_torch_device(), dtype=torch.float64)
    local_rows: list[dict] = []
    local_plan: list[dict] = []
    started = time.monotonic()
    with timer("preference_eval"):
        for chunk_idx, (global_chunk_size, batch_size) in enumerate(zip(chunk_sizes, local_batch_sizes, strict=True)):
            partition_id = f"sft_eval_{rollout_id}_n{n_chunks}_{chunk_idx}"
            _wait_for_eval_partition_present(actor, partition_id)
            batch_index = 0
            while not actor.all_consumed(task_name, rollout_id, partition_id=partition_id):
                pair_rows, _batch_meta = actor._get_data_from_transfer_queue(
                    task_name, rollout_id, data_fields, batch_size, batch_index, partition_id=partition_id
                )
                if pair_rows is None:
                    continue
                batch_index += 1
                rollout_data = expand_preference_rollout_data(pair_rows)
                rollout_data["dynamic_global_batch_size"] = global_chunk_size
                data_iterator, num_microbatches = get_data_iterator(args, actor.model, rollout_data)
                for microbatch_index, branch_indices in enumerate(data_iterator[0].micro_batch_indices):
                    encoded_ids = []
                    for branch_index in branch_indices:
                        pair_id = int(rollout_data["preference_branch_pair_ids"][branch_index])
                        if not encoded_ids or encoded_ids[-1] != pair_id:
                            encoded_ids.append(pair_id)
                    local_plan.append(
                        {
                            "rank": dist.get_rank(),
                            "chunk": chunk_idx,
                            "batch": batch_index - 1,
                            "microbatch": microbatch_index,
                            "encoded_pair_ids": encoded_ids,
                        }
                    )
                encoded_pair_ids = extract_preference_eval_pair_ids(rollout_data)
                if args.sft_objective == "dpo":
                    if args.dpo_reference_free:
                        reference_sums = None
                    else:
                        actor._switch_model("ref")
                        reference = actor.compute_log_prob(data_iterator, num_microbatches, store_prefix="ref_")[
                            "ref_log_probs"
                        ]
                        reference_sums = _masked_sequence_sums(reference, rollout_data["loss_masks"], local.device)
                    actor._switch_model("actor")
                    policy = actor.compute_log_prob(data_iterator, num_microbatches, store_prefix="")["log_probs"]
                    policy_sums = _masked_sequence_sums(policy, rollout_data["loss_masks"], local.device)
                    policy_chosen, policy_rejected = policy_sums[0::2], policy_sums[1::2]
                    if reference_sums is None:
                        reference_chosen = reference_rejected = None
                        chosen_values = args.dpo_beta * policy_chosen
                        rejected_values = args.dpo_beta * policy_rejected
                    else:
                        reference_chosen, reference_rejected = reference_sums[0::2], reference_sums[1::2]
                        chosen_values = args.dpo_beta * (policy_chosen - reference_chosen)
                        rejected_values = args.dpo_beta * (policy_rejected - reference_rejected)
                    losses = dpo_pair_loss(
                        policy_chosen,
                        policy_rejected,
                        reference_chosen=reference_chosen,
                        reference_rejected=reference_rejected,
                        beta=args.dpo_beta,
                        reference_free=args.dpo_reference_free,
                    )
                    local += pair_metric_sums(chosen_values, rejected_values, losses)
                    policy_chosen_values = policy_chosen.detach().cpu().tolist()
                    policy_rejected_values = policy_rejected.detach().cpu().tolist()
                    reference_chosen_values = (
                        [0.0] * len(encoded_pair_ids)
                        if reference_chosen is None
                        else reference_chosen.detach().cpu().tolist()
                    )
                    reference_rejected_values = (
                        [0.0] * len(encoded_pair_ids)
                        if reference_rejected is None
                        else reference_rejected.detach().cpu().tolist()
                    )
                    chosen_reward_values = chosen_values.detach().cpu().tolist()
                    rejected_reward_values = rejected_values.detach().cpu().tolist()
                    loss_values = losses.detach().cpu().tolist()
                    for index, encoded_pair_id in enumerate(encoded_pair_ids):
                        margin = chosen_reward_values[index] - rejected_reward_values[index]
                        local_rows.append(
                            {
                                "encoded_pair_id": encoded_pair_id,
                                "policy_chosen_logp": policy_chosen_values[index],
                                "policy_rejected_logp": policy_rejected_values[index],
                                "reference_chosen_logp": reference_chosen_values[index],
                                "reference_rejected_logp": reference_rejected_values[index],
                                "chosen_implicit_reward": chosen_reward_values[index],
                                "rejected_implicit_reward": rejected_reward_values[index],
                                "reward_margin": margin,
                                "pair_loss": loss_values[index],
                            }
                        )
                else:
                    outputs = forward_only(
                        compute_reward_model_eval_step,
                        args,
                        actor.model,
                        data_iterator,
                        num_microbatches,
                        store_prefix="",
                    )
                    if mpu.is_pipeline_last_stage():
                        scores = torch.stack(outputs["scores"]).to(local.device)
                        chosen_scores, rejected_scores = scores[0::2], scores[1::2]
                        losses = reward_model_pair_loss(chosen_scores, rejected_scores)
                        local += pair_metric_sums(chosen_scores, rejected_scores, losses, epsilon=0.0)
                        chosen_values = chosen_scores.detach().cpu().tolist()
                        rejected_values = rejected_scores.detach().cpu().tolist()
                        loss_values = losses.detach().cpu().tolist()
                        for index, encoded_pair_id in enumerate(encoded_pair_ids):
                            local_rows.append(
                                {
                                    "encoded_pair_id": encoded_pair_id,
                                    "chosen_score": chosen_values[index],
                                    "rejected_score": rejected_values[index],
                                    "pair_loss": loss_values[index],
                                }
                            )
            dist.barrier(group=get_gloo_group())
            if dist.get_rank() == 0:
                run(actor.data_system_client.async_clear_partition(partition_id=partition_id))

    dist.all_reduce(local, op=dist.ReduceOp.SUM, group=mpu.get_pipeline_model_parallel_group())
    dist.all_reduce(local, op=dist.ReduceOp.SUM, group=mpu.get_data_parallel_group(with_context_parallel=True))
    metrics = finalize_pair_metrics(local, prefix="dpo" if args.sft_objective == "dpo" else "rm")
    metrics["perf/preference_eval_time"] = time.monotonic() - started
    gloo_group = get_gloo_group()
    gathered_rows = [None] * dist.get_world_size(group=gloo_group)
    gathered_plans = [None] * dist.get_world_size(group=gloo_group)
    dist.all_gather_object(gathered_rows, local_rows, group=gloo_group)
    dist.all_gather_object(gathered_plans, local_plan, group=gloo_group)
    if is_megatron_main_rank():
        from relax.engine.sft.eval.acceptance import write_pair_artifacts

        summary = write_pair_artifacts(
            getattr(args, "save", None),
            args.sft_objective,
            rollout_id,
            [row for rank_rows in gathered_rows for row in rank_rows],
            [entry for rank_plan in gathered_plans for entry in rank_plan],
        )
        if summary is not None:
            metrics[f"eval/{'dpo' if args.sft_objective == 'dpo' else 'rm'}_bootstrap_lower_95"] = summary[
                "bootstrap"
            ]["lower_95"]
        step = compute_rollout_step(args, rollout_id)
        metrics["rollout/step"] = step
        tracking_utils.log(args, metrics, step_key="rollout/step")
        tracking_utils.flush_metrics(args, step)
        logger.info(f"Preference eval @ rollout_id={rollout_id}: {metrics}")


def _masked_sequence_sums(values, masks, device: torch.device) -> torch.Tensor:
    if len(values) != len(masks):
        raise ValueError("preference eval values/masks are not branch aligned")
    sums = []
    for value, mask in zip(values, masks, strict=True):
        value = torch.as_tensor(value, device=device)
        mask = torch.as_tensor(mask, device=device, dtype=value.dtype)
        if value.shape != mask.shape:
            raise ValueError(f"preference eval value/mask shape mismatch: {value.shape} vs {mask.shape}")
        sums.append((value * mask).sum())
    return torch.stack(sums)
