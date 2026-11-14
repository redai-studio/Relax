# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Versioned, task-scoped CPU plans and explicit multi-mini execution."""

from __future__ import annotations

import copy
import hashlib
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from relax.utils.data.flops_scheduling import row_weights
from relax.utils.data.global_flops_pack_planner import (
    ALGORITHM,
    COST_MODEL,
    SCHEMA,
    TRAIN_TASKS,
    ExecutionContract,
    GlobalPackPlan,
    digest,
    positive,
    scheduling_order,
)


PLAN_METADATA = "_no16_global_pack_v1"
STEP_PLANS = "_no16_global_pack_steps_v1"
REQUEST_CONTRACT = "no16_pack_execution_contract"
LOCAL_COUNTS = "rollout_mini_local_sample_counts"
GLOBAL_COUNTS = "rollout_mini_global_sample_counts"


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return _plain(vars(value)) if hasattr(value, "__dict__") else value


def cost_identity(estimator: Any) -> tuple[str, str]:
    """Hash P2's already validated snapshot, not HF wrapper default fields.

    This read-only coupling is pinned with the unchanged P2 source bytes; no
    formula, coefficients, or estimator API are reimplemented here.
    """
    model = digest(_plain(estimator._config))
    training = Path(__file__).resolve().parents[1] / "training"
    sources = {
        name: hashlib.sha256((training / name).read_bytes()).hexdigest()
        for name in (
            "flops_estimator.py",
            "flops_counter.py",
            "flops_estimator_qwen3.py",
        )
    }
    return model, digest(dict(version=COST_MODEL, scope=estimator.compute_scope, sources=sources))


def validate_pack_runtime(args: Any) -> None:
    if scheduling_order(args) != "pack-first":
        return
    if not getattr(args, "polling_mode", True):
        raise ValueError(
            "Pack-first requires polling_mode=True: the fixed TransferQueue nonpolling branch "
            "does not query cached assignments"
        )
    if getattr(args, "loss_type", None) == "sft" and getattr(args, "num_steps_per_rollout", None) not in (None, 1):
        raise ValueError(
            "Pack-first SFT supports one optimizer mini per sft_train partition; "
            "--num-steps-per-rollout must be omitted or 1"
        )
    unsupported = [
        name
        for name in (
            "dynamic_context_parallel",
            "hybrid",
            "debug_train_only",
            "debug_rollout_only",
            "load_debug_rollout_data",
            "use_routing_replay",
            "use_rollout_indexer_replay",
            "sft_train_data_prefetch",
        )
        if getattr(args, name, False)
    ]
    if unsupported:
        raise ValueError("Pack-first does not support these execution paths: " + ", ".join(unsupported))
    if getattr(args, "qkv_format", "thd") != "thd":
        raise ValueError("Pack-first currently requires THD physical packing")


def checkpoint_identity(args: Any, estimator: Any, dp_size: int) -> str:
    # Model construction discovers uses_unsplit_forward only on the Actor.
    # Its physical layout remains part of the complete ExecutionContract.
    model, cost = cost_identity(estimator)
    data = {
        name: _plain(getattr(args, name, None))
        for name in (
            "hf_checkpoint",
            "prompt_data",
            "input_key",
            "label_key",
            "n_samples_per_prompt",
            "seed",
            "sft_training_mode",
            "custom_dataset_class_path",
            "global_batch_size",
            "loss_type",
        )
    }
    geometry = {
        name: _plain(getattr(args, name, None))
        for name in (
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "context_parallel_size",
            "virtual_pipeline_model_parallel_size",
            "num_layers_per_virtual_pipeline_stage",
            "microbatch_group_size_per_vp_stage",
            "max_tokens_per_gpu",
            "data_pad_size_multiplier",
            "allgather_cp",
            "qkv_format",
        )
    }
    return digest(
        dict(
            schema=SCHEMA,
            algorithm=ALGORITHM,
            order="pack-first",
            mode="both",
            model=model,
            cost=cost,
            dp=dp_size,
            data=data,
            geometry=geometry,
        )
    )


def execution_contract(args: Any, mpu: Any, model_config: Any, estimator: Any) -> ExecutionContract:
    """Called only on an initialized Actor, never with guessed VPP geometry."""
    validate_pack_runtime(args)
    vpp = mpu.get_virtual_pipeline_model_parallel_world_size() or 1
    return ExecutionContract(
        dp=mpu.get_data_parallel_world_size(with_context_parallel=False),
        tp=mpu.get_tensor_model_parallel_world_size(),
        pp=mpu.get_pipeline_model_parallel_world_size(),
        cp=mpu.get_context_parallel_world_size(),
        vpp=vpp,
        mb_group=model_config.microbatch_group_size_per_vp_stage if vpp > 1 else 1,
        max_tokens=args.max_tokens_per_gpu,
        pad_multiplier=getattr(args, "data_pad_size_multiplier", 1),
        unsplit=bool(getattr(args, "is_vl_model", False) or getattr(args, "uses_unsplit_forward", False)),
        allgather_cp=bool(getattr(args, "allgather_cp", False)),
        checkpoint_signature=checkpoint_identity(
            args, estimator, mpu.get_data_parallel_world_size(with_context_parallel=False)
        ),
    )


def seal_header(header: dict) -> dict:
    value = copy.deepcopy(header)
    value.pop("rank_local_plan_digest", None)
    value["rank_local_plan_digest"] = digest(value)
    return value


def encode_plan(plan: GlobalPackPlan) -> tuple[list[list[int]], dict[int, dict]]:
    by_row = {row: i for i, row in enumerate(plan.rows)}
    assignments, markers = [], {}
    for rank in range(plan.contract.dp):
        rows = list(plan.rank_rows(rank))
        slots = [
            dict(
                mb_id=plan.microbatches[index].mb_id,
                rows=list(plan.microbatches[index].rows),
                dummy=dummy,
                physical_tokens=plan.microbatches[index].physical_tokens,
                estimated_flops=plan.microbatches[index].estimated_flops,
                oversized_singleton=plan.microbatches[index].oversized_singleton,
            )
            for index, dummy in plan.rank_slots(rank)
        ]
        header = seal_header(
            dict(
                schema_version=SCHEMA,
                algorithm=ALGORITHM,
                scheduling_order="pack-first",
                cost_model_version=COST_MODEL,
                cost_model_signature=plan.cost_signature,
                model_config_signature=plan.model_signature,
                checkpoint_signature=plan.checkpoint_signature,
                execution_contract=asdict(plan.contract),
                execution_geometry_signature=plan.contract.signature,
                identity=list(plan.identity),
                global_cohort_signature=plan.cohort_signature,
                global_logical_sample_count=plan.logical_count,
                global_physical_row_count=len(plan.rows),
                global_real_mb_count=len(plan.microbatches),
                target_dp_rank=rank,
                rows=rows,
                logical_lengths=[plan.lengths[by_row[row]] for row in rows],
                physical_lengths=[plan.physical_lengths[by_row[row]] for row in rows],
                weights=[plan.weights[by_row[row]] for row in rows],
                real_mb_count=len(plan.rank_mbs[rank]),
                K=plan.K,
                slots=slots,
                global_plan_digest=plan.plan_digest,
                fallback_reason=plan.fallback_reason,
            )
        )
        for row in rows:
            markers[row] = dict(
                schema_version=SCHEMA, identity=list(plan.identity), global_plan_digest=plan.plan_digest
            )
        markers[rows[0]]["header"] = header
        assignments.append(rows)
    return assignments, markers


def metadata_updates(partition: Any, markers: dict[int, dict], task: str) -> dict[int, dict]:
    current = partition.get_custom_meta(list(markers))
    updates = {}
    for row, marker in markers.items():
        value = copy.deepcopy(current[row])
        namespace = value.get(PLAN_METADATA, {})
        if not isinstance(namespace, dict):
            raise ValueError("Invalid task-scoped pack-first metadata")
        value[PLAN_METADATA] = {**namespace, task: copy.deepcopy(marker)}
        updates[row] = value
    return updates


def restore_plan(
    batch_meta: Any,
    payload: dict,
    *,
    task: str,
    partition: str,
    mini: int,
    rank: int,
    contract: ExecutionContract,
    estimator: Any,
) -> None:
    """Attach a one-element control list; normal concat preserves mini
    boundaries."""
    if task not in TRAIN_TASKS:
        return
    ids = list(batch_meta.global_indexes)
    entries = [row.get(PLAN_METADATA, {}).get(task) for row in batch_meta.get_all_custom_meta()]
    if not entries or not all(isinstance(entry, dict) for entry in entries):
        raise ValueError("Missing or partial task-scoped pack-first plan")
    headers = [entry["header"] for entry in entries if "header" in entry]
    if len(headers) != 1:
        raise ValueError("Pack-first requires exactly one rank header")
    header = copy.deepcopy(headers[0])
    identity = header.get("identity")
    if (
        not isinstance(identity, list)
        or len(identity) != 4
        or (identity[0], identity[2], identity[3]) != (partition, task, mini)
        or any(
            entry.get("identity") != identity
            or entry.get("schema_version") != SCHEMA
            or entry.get("global_plan_digest") != header.get("global_plan_digest")
            for entry in entries
        )
    ):
        raise ValueError("Stale, mixed, or wrong-task pack-first metadata")
    item = dict(header=header, payload_rows=ids)
    validate_local_plan(item, payload, contract, estimator, rank)
    payload[STEP_PLANS] = [item]
    payload[LOCAL_COUNTS] = [len(ids)]
    payload[GLOBAL_COUNTS] = [header["global_logical_sample_count"]]


def validate_local_plan(item: Any, payload: dict, contract: ExecutionContract, estimator: Any, rank: int) -> None:
    if not isinstance(item, dict) or set(item) != {"header", "payload_rows"}:
        raise ValueError("Invalid pack-first mini plan")
    h, ids = item["header"], item["payload_rows"]
    if not isinstance(h, dict) or h.get("schema_version") != SCHEMA or h.get("algorithm") != ALGORITHM:
        raise ValueError("Unsupported pack-first schema/algorithm")
    if seal_header(h) != h:
        raise ValueError("Pack-first local digest mismatch")
    identity = h.get("identity")
    if (
        not isinstance(identity, list)
        or len(identity) != 4
        or not isinstance(identity[0], str)
        or not identity[0]
        or type(identity[1]) is not int
        or identity[1] < 0
        or identity[2] not in TRAIN_TASKS
        or type(identity[3]) is not int
        or identity[3] < 0
    ):
        raise ValueError("Invalid pack-first partition/generation/task/mini identity")
    model, cost = cost_identity(estimator)
    if (h.get("model_config_signature"), h.get("cost_model_signature")) != (model, cost):
        raise ValueError("Pack-first P2 model/cost signature mismatch")
    if h.get("scheduling_order") != "pack-first" or h.get("cost_model_version") != COST_MODEL:
        raise ValueError("Pack-first order/cost version mismatch")
    if (
        ExecutionContract.from_dict(h.get("execution_contract")) != contract
        or h.get("execution_geometry_signature") != contract.signature
    ):
        raise ValueError("Pack-first execution contract mismatch")
    if contract.checkpoint_signature and h.get("checkpoint_signature") != contract.checkpoint_signature:
        raise ValueError("Pack-first checkpoint model/data/order configuration mismatch")
    if h.get("target_dp_rank") != rank or not 0 <= rank < contract.dp:
        raise ValueError("Pack-first DP owner mismatch")
    if (
        not isinstance(ids, list)
        or len(set(ids)) != len(ids)
        or set(ids) != set(h["rows"])
        or len(ids) != len(h["rows"])
    ):
        raise ValueError("Pack-first payload row mapping mismatch")
    if not ids or len(payload["total_lengths"]) != len(ids):
        raise ValueError("Pack-first payload row count mismatch")
    index = {row: i for i, row in enumerate(h["rows"])}
    actual = dict(zip(ids, payload["total_lengths"], strict=True))
    if [actual[row] for row in h["rows"]] != h["logical_lengths"]:
        raise ValueError("Pack-first logical lengths differ from actual payload")
    if "tokens" in payload and [len(tokens) for tokens in payload["tokens"]] != list(payload["total_lengths"]):
        raise ValueError("Pack-first token payload length mismatch")
    if tuple(h["physical_lengths"]) != contract.physical(h["logical_lengths"]):
        raise ValueError("Pack-first physical lengths mismatch")
    if row_weights(estimator, h["logical_lengths"]) != h["weights"]:
        raise ValueError("Pack-first cached costs differ from P2")
    K = positive(h["K"], "K")
    if K % contract.group or len(h["slots"]) != K:
        raise ValueError("Pack-first K/VPP mismatch")
    real = [slot for slot in h["slots"] if slot.get("dummy") is False]
    if len(real) != positive(h["real_mb_count"], "real_mb_count"):
        raise ValueError("Pack-first real MB count mismatch")
    if sorted(row for slot in real for row in slot["rows"]) != sorted(ids):
        raise ValueError("Pack-first real rows must be covered exactly once")
    for slot in h["slots"]:
        rows = slot["rows"]
        if not rows or any(row not in index for row in rows) or type(slot["dummy"]) is not bool:
            raise ValueError("Invalid pack-first MB membership")
        tokens = sum(h["physical_lengths"][index[row]] for row in rows)
        flops = math.fsum(h["weights"][index[row]] for row in rows)
        if slot["mb_id"] != digest([h["global_cohort_signature"], tuple(rows)]):
            raise ValueError("Pack-first MB identity mismatch")
        if tokens != slot["physical_tokens"] or flops != slot["estimated_flops"]:
            raise ValueError("Pack-first MB token/FLOPs accounting mismatch")
        if tokens > contract.capacity and len(rows) != 1:
            raise ValueError("Pack-first multiple-row MB exceeds physical capacity")
        if slot["oversized_singleton"] != (tokens > contract.capacity):
            raise ValueError("Pack-first oversized singleton flag mismatch")
        if slot["dummy"] and not any({**slot, "dummy": False} == other for other in real):
            raise ValueError("Pack-first dummy has no identical local real source")
    positive(h["global_logical_sample_count"], "global logical count")
    if (
        positive(h["global_physical_row_count"], "global physical rows") < len(ids)
        or positive(h["global_real_mb_count"], "global real MBs") < contract.dp
        or h["global_logical_sample_count"] > h["global_physical_row_count"]
    ):
        raise ValueError("Invalid pack-first global counts")
    for name in ("global_plan_digest", "global_cohort_signature"):
        value = h[name]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Invalid complete pack-first digest")


def local_schedule(
    payload: dict, contract: ExecutionContract, estimator: Any, rank: int
) -> tuple[list[list[int]], list[int], set[int]]:
    plans = payload.get(STEP_PLANS)
    if not isinstance(plans, list) or not plans:
        raise ValueError("Missing pack-first mini plans; local repacking is forbidden")
    if any(len(payload.get(name, [])) != len(plans) for name in (LOCAL_COUNTS, GLOBAL_COUNTS)):
        raise ValueError("Pack-first mini/count boundaries mismatch")
    indices, counts, dummy = [], [], set()
    row_offset = 0
    previous = None
    for step, item in enumerate(plans):
        count = len(item["payload_rows"])
        mini_payload = {"total_lengths": payload["total_lengths"][row_offset : row_offset + count]}
        if "tokens" in payload:
            mini_payload["tokens"] = payload["tokens"][row_offset : row_offset + count]
        validate_local_plan(item, mini_payload, contract, estimator, rank)
        h = item["header"]
        current = h["identity"]
        if previous is not None and (current[:3] != previous[:3] or current[3] != previous[3] + 1):
            raise ValueError("Pack-first mini identities are reordered or nonconsecutive")
        previous = current
        if payload[LOCAL_COUNTS][step] != count or payload[GLOBAL_COUNTS][step] != h["global_logical_sample_count"]:
            raise ValueError("Pack-first physical rows and logical counts must stay distinct")
        index = {row: row_offset + i for i, row in enumerate(item["payload_rows"])}
        for slot in h["slots"]:
            if slot["dummy"]:
                dummy.add(len(indices))
            indices.append([index[row] for row in slot["rows"]])
        counts.append(h["K"])
        row_offset += count
    if row_offset != len(payload["total_lengths"]):
        raise ValueError("Pack-first mini row offsets do not cover the payload")
    return indices, counts, dummy


def validate_cached_plan(plan: GlobalPackPlan, estimator: Any, key: tuple, signature: str, generation: int) -> None:
    """Validate all frozen rank plans before restoring any sampler state."""
    if not isinstance(plan, GlobalPackPlan) or plan.identity != (key[0], generation, key[1], key[2]):
        raise ValueError("Pack-first checkpoint plan identity/generation mismatch")
    if plan.checkpoint_signature != signature or plan.contract.checkpoint_signature != signature:
        raise ValueError("Pack-first checkpoint plan configuration mismatch")
    if len(plan.rank_mbs) != plan.contract.dp or any(not group for group in plan.rank_mbs):
        raise ValueError("Pack-first checkpoint must contain a real MB on each rank")
    if sorted(i for group in plan.rank_mbs for i in group) != list(range(len(plan.microbatches))):
        raise ValueError("Pack-first checkpoint repeats or omits a complete MB")
    if sorted(row for mb in plan.microbatches for row in mb.rows) != sorted(plan.rows):
        raise ValueError("Pack-first checkpoint repeats or omits real rows")
    group = plan.contract.group
    if plan.K != (max(map(len, plan.rank_mbs)) + group - 1) // group * group:
        raise ValueError("Pack-first checkpoint K differs from real MB counts/VPP")
    assignment, markers = encode_plan(plan)
    lengths = dict(zip(plan.rows, plan.lengths, strict=True))
    for rank, rows in enumerate(assignment):
        validate_local_plan(
            dict(header=markers[rows[0]]["header"], payload_rows=rows),
            dict(total_lengths=[lengths[row] for row in rows]),
            plan.contract,
            estimator,
            rank,
        )


def agree_cpu(values: list[int], group: Any) -> list[int]:
    if group is None:
        raise ValueError("Pack-first requires the explicit Actor CPU control group")
    import torch
    import torch.distributed as dist

    tensor = torch.tensor(values, dtype=torch.int64, device="cpu")
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=group)
    return tensor.tolist()


class PackFirstExhausted(Exception):
    """All Actor ranks agreed that no batch is held and production is
    drained."""


def checked_fetch(
    fetch: Callable[[], list],
    should_fetch: bool,
    agree: Callable[[list[int]], list[int]],
    *,
    timeout_seconds: float = 60.0,
    poll_seconds: float = 0.01,
    is_drained: Callable[[], bool] | None = None,
) -> list:
    """All Actor ranks agree before any model-parallel payload broadcast.

    Ready fetchers retain their exact batch while empty peers retry. External
    RPCs remain subject to their configured client/process-group timeouts.
    """
    started = time.monotonic()
    batch = [None, None]
    while True:
        error = None
        if should_fetch and batch[0] is None:
            try:
                batch = fetch()
                if not isinstance(batch, (list, tuple)) or len(batch) != 2:
                    raise ValueError("Invalid pack-first fetch response")
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        ready = should_fetch and batch[0] is not None
        values = agree(
            [
                int(error is not None),
                int(ready),
                int(should_fetch and not ready),
                int(time.monotonic() - started > timeout_seconds),
                int(should_fetch),
            ]
        )
        if values[0]:
            raise RuntimeError("Pack-first fetch failed on an Actor rank: " + (error or "peer fetch error"))
        if not values[4]:
            raise ValueError("Pack-first control group contains no fetching rank")
        # Select this branch from shared readiness, including non-fetching TP/PP/CP peers.
        # A held batch may already mark consumption complete: never discard it to recheck drained.
        if not values[1] and is_drained is not None and checked_drained(is_drained, agree):
            raise PackFirstExhausted("Pack-first producer drained before any Actor rank acquired a batch")
        if values[3]:
            raise TimeoutError("Pack-first fetch readiness timed out on an Actor rank")
        if not values[2]:
            return list(batch)
        if poll_seconds > 0:
            time.sleep(poll_seconds)


def agree_invocation(minis: int, batch_index: int, agree: Callable[[list[int]], list[int]]) -> None:
    bad = type(minis) is not int or minis <= 0 or type(batch_index) is not int or batch_index < 0
    values = agree([int(bad), minis, -minis, batch_index, -batch_index])
    if values[0] or values[1] != -values[2] or values[3] != -values[4]:
        raise ValueError("Pack-first Actor invocation mini count/index mismatch")


def checked_drained(check: Callable[[], bool], agree: Callable[[list[int]], list[int]]) -> bool:
    """Coordinate queue-status RPC errors before entering the fetch loop."""
    error = None
    drained = False
    try:
        drained = bool(check())
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    values = agree([int(error is not None), int(not drained)])
    if values[0]:
        raise RuntimeError("Pack-first queue status failed on an Actor rank: " + (error or "peer error"))
    return not values[1]


def checked_schedule(
    payload: dict,
    contract: ExecutionContract,
    estimator: Any,
    rank: int,
    agree: Callable[[list[int]], list[int]],
    *,
    contract_provider: Callable[[], ExecutionContract] | None = None,
) -> tuple[list[list[int]], list[int], set[int]]:
    error = None
    result = ([], [], set())
    try:
        if contract_provider is not None and contract_provider() != contract:
            raise ValueError("Pack-first cached contract differs from the current Actor execution geometry")
        result = local_schedule(payload, contract, estimator, rank)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    steps = len(result[1])
    control = agree([int(error is not None), steps, -steps])
    if control[0] or control[1] != -control[2]:
        raise ValueError("Pack-first plan rejected on an Actor rank: " + (error or "peer plan/mini error"))
    values = []
    for item in payload[STEP_PLANS]:
        header = item["header"]
        parts = [int(header["global_plan_digest"][i : i + 4], 16) for i in range(0, 64, 4)]
        row = [header["K"], header["global_logical_sample_count"], *parts]
        values.extend(row + [-v for v in row])
    control = agree(values)
    for start in range(0, len(control), 36):
        if control[start : start + 18] != [-v for v in control[start + 18 : start + 36]]:
            raise ValueError("Pack-first global digest/K/logical-count agreement failed")
    return result
