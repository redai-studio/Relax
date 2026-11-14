# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Compose existing samplers; commit plans before returning effective
assignments."""

from __future__ import annotations

import copy
import inspect
import json
import threading
from typing import Any

from transfer_queue.sampler import BaseSampler

from relax.utils.data.flops_scheduling import row_weights
from relax.utils.data.global_flops_pack_planner import (
    SCHEMA,
    TRAIN_TASKS,
    ExecutionContract,
    GlobalPackPlan,
    build_global_plan,
    positive,
)
from relax.utils.data.global_flops_pack_protocol import (
    PLAN_METADATA,
    REQUEST_CONTRACT,
    checkpoint_identity,
    cost_identity,
    encode_plan,
    metadata_updates,
    validate_cached_plan,
    validate_pack_runtime,
)
from relax.utils.data.seqlen_balancing import get_seqlen_balanced_partitions
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


class _Placement:
    def __init__(self, owner: GlobalPackSampler) -> None:
        self.owner = owner

    @property
    def signature(self) -> str:
        return self.owner.signature

    def __call__(self, rows: list[int], metadata: dict, dp_size: int) -> list[list[int]]:
        if self.owner._request is None:
            # Separate forward tasks retain ordinary Balanced row placement.
            parts = get_seqlen_balanced_partitions(
                [int(metadata[row]["total_lengths"]) for row in rows],
                dp_size,
                equal_size=len(rows) % dp_size == 0,
            )
            return [[rows[i] for i in group] for group in parts]
        return self.owner._prepare(rows, metadata)


class GlobalPackSampler(BaseSampler):
    """The wrapper owns plans, while its base owns selection and admission.

    Transactions snapshot the base's public checkpoint, never rewrite its
    admission logic. The lock is recreated after Ray/pickle transport.
    """

    def __init__(self, args: Any, context: Any, dp_size: int, original: Any) -> None:
        super().__init__()
        validate_pack_runtime(args)
        self.context = context
        self.dp_size = dp_size
        self.is_sft = getattr(args, "loss_type", None) == "sft"
        self.signature = checkpoint_identity(args, context.estimator, dp_size)
        self.model_signature, self.cost_signature = cost_identity(context.estimator)
        self.plans: dict[tuple[str, str, int], GlobalPackPlan] = {}
        self.generations: dict[str, int] = {}
        self.poisoned: set[tuple[str, str, int]] = set()
        self._unrecoverable: str | None = None
        self._request = None
        self._prepared = None
        self._lock = threading.RLock()
        if self.is_sft:
            if getattr(args, "n_samples_per_prompt", 1) != 1:
                raise ValueError("Pack-first SFT requires n_samples_per_prompt=1")
            if "assignment_transform" not in inspect.signature(original.sample).parameters:
                raise RuntimeError("Pack-first SFT requires the fixed TransferQueue assignment_transform patch")
            self.base = original
        else:
            if not getattr(args, "balance_data", False) or getattr(original, "placement", None) != "balanced":
                raise ValueError("Pack-first RL requires the original Balanced admission (--balance-data)")
            from relax.utils.data.identity_window_sampler import IdentityWindowSampler

            self.base = IdentityWindowSampler(dp_size=dp_size, placement="balanced", row_placement=_Placement(self))

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state.pop("_lock")
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._lock = threading.RLock()

    def _ensure_healthy(self) -> None:
        if self._unrecoverable is not None:
            raise RuntimeError(
                "Pack-first shared base state is untrusted; reconstruct the sampler or load a verified checkpoint: "
                + self._unrecoverable
            )

    def _restore_base(self, state: dict) -> None:
        self.base.load_checkpoint(copy.deepcopy(state))
        if self.base.save_checkpoint() != state:
            raise RuntimeError("Shared base rollback verification failed")

    def _prepare(self, rows: list[int], metadata: dict) -> list[list[int]]:
        request = self._request
        if request is None:
            raise RuntimeError("Global plan preparation requires an active training request")
        if any(metadata[row].get("multimodal_train_inputs") is not None for row in rows):
            raise ValueError("Pack-first requires ordinary text rows")
        lengths = [metadata[row]["total_lengths"] for row in rows]
        identities = None if self.is_sft else [int(metadata[row]["sample_index"]) for row in rows]
        contract = request["contract"]
        self._prepared = build_global_plan(
            rows,
            lengths,
            contract.physical(lengths),
            row_weights(self.context.estimator, lengths),
            contract,
            identity=request["identity"],
            logical_count=request["logical_count"],
            model_signature=self.model_signature,
            cost_signature=self.cost_signature,
            checkpoint_signature=self.signature,
            identities=identities,
        )
        return [list(self._prepared.rank_rows(rank)) for rank in range(self.dp_size)]

    def has_cached_result(self, partition_id: str, task_name: str, sampling_config: Any = None) -> bool:
        with self._lock:
            self._ensure_healthy()
            config = sampling_config or {}
            key = (partition_id, task_name, config.get("batch_index", 0))
            if key in self.poisoned:
                raise RuntimeError("Aborted pack-first metadata transaction requires partition clear/recovery")
            if task_name in TRAIN_TASKS and key not in self.plans:
                return False
            return self.base.has_cached_result(partition_id, task_name, sampling_config)

    def sample(
        self,
        ready_indexes: list[int],
        batch_size: int,
        task_name: str = "",
        partition_id: str = "",
        *args: Any,
        **kwargs: Any,
    ) -> tuple[list[int], list[int]]:
        with self._lock:
            self._ensure_healthy()
            if task_name not in TRAIN_TASKS:
                return self.base.sample(ready_indexes, batch_size, task_name, partition_id, *args, **kwargs)
            rank = kwargs.get("dp_rank")
            mini = kwargs.get("batch_index")
            if type(rank) is not int or not 0 <= rank < self.dp_size or type(mini) is not int or mini < 0:
                raise ValueError("Pack-first requires an explicit DP rank and mini index")
            if self.is_sft and mini != 0:
                raise ValueError("Pack-first SFT supports one optimizer mini per sft_train partition (batch_index=0)")
            contract = ExecutionContract.from_dict(kwargs.get(REQUEST_CONTRACT))
            if contract.dp != self.dp_size:
                raise ValueError("Pack-first sampler/Actor DP geometry mismatch")
            if contract.checkpoint_signature != self.signature:
                raise ValueError("Pack-first Actor/sampler model/data/order configuration mismatch")
            key = (partition_id, task_name, mini)
            if key in self.poisoned:
                raise RuntimeError("Aborted pack-first metadata transaction requires partition clear/recovery")
            partition = kwargs["partition"]
            cached = self.plans.get(key)
            if cached is not None:
                if cached.contract != contract:
                    raise ValueError("Pack-first replay execution geometry changed")
                if cached.logical_count != positive(batch_size, "per-DP logical quota") * self.dp_size:
                    raise ValueError("Pack-first replay logical batch quota changed")
                assignment, markers = encode_plan(cached)
                current = partition.get_custom_meta(list(markers))
                if any(
                    current[row].get(PLAN_METADATA, {}).get(task_name) != marker for row, marker in markers.items()
                ):
                    raise ValueError("Stale/missing committed pack-first metadata; cannot replay old row IDs")
                result = self.base.sample(ready_indexes, batch_size, task_name, partition_id, *args, **kwargs)
                if result != (assignment[rank], assignment[rank].copy()):
                    raise ValueError("Pack-first cache assignment differs from its immutable plan")
                return result
            if self.base.has_cached_result(partition_id, task_name, kwargs):
                raise ValueError("Pack-first assignment cache has no matching plan")
            saved = copy.deepcopy(self.base.save_checkpoint())
            old_metadata = None
            self._request = dict(
                contract=contract,
                identity=(partition_id, self.generations.get(partition_id, 0), task_name, mini),
                logical_count=positive(batch_size, "per-DP logical quota") * self.dp_size,
            )
            self._prepared = None
            try:
                if self.is_sft:

                    def transform(groups: list[list[int]]) -> list[list[int]]:
                        rows = sorted(row for group in groups for row in group)
                        return self._prepare(rows, partition.get_custom_meta(rows))

                    result = self.base.sample(
                        ready_indexes,
                        batch_size,
                        task_name,
                        partition_id,
                        *args,
                        assignment_transform=transform,
                        **kwargs,
                    )
                else:
                    result = self.base.sample(ready_indexes, batch_size, task_name, partition_id, *args, **kwargs)
                if not result[0]:
                    return result
                if self._prepared is None:
                    raise ValueError("Base returned assignments without preparing a global MB plan")
                assignment, markers = encode_plan(self._prepared)
                if result != (assignment[rank], assignment[rank].copy()):
                    raise ValueError("Base assignment does not match the prepared pack-first plan")
                old_metadata = copy.deepcopy(partition.get_custom_meta(list(markers)))
                updates = metadata_updates(partition, markers, task_name)
                partition.set_custom_meta(updates)
                installed = partition.get_custom_meta(list(markers))
                if any(
                    installed[row].get(PLAN_METADATA, {}).get(task_name) != marker for row, marker in markers.items()
                ):
                    raise ValueError("Incomplete pack-first metadata publication")
                self.plans[key] = self._prepared
                info = self._prepared.diagnostics()
                # Measure only our JSON protocol; producer custom_meta may contain opaque pickle values.
                info["metadata_serialization_bytes"] = len(json.dumps(markers, separators=(",", ":")).encode())
                info["rank_local_plan_digest"] = [
                    markers[rows[0]]["header"]["rank_local_plan_digest"] for rows in assignment
                ]
                logger.info("Pack-first committed plan: %s", info)
                return result
            except Exception:
                try:
                    self._restore_base(saved)
                except Exception as exc:
                    # The base owns every task/mini, so an unverified restore invalidates the shared instance.
                    self._unrecoverable = f"{type(exc).__name__}: {exc}"
                self.plans.pop(key, None)
                if old_metadata is not None:
                    try:
                        # Preserve current producer fields; replace only this protocol namespace.
                        rollback = copy.deepcopy(partition.get_custom_meta(list(old_metadata)))
                        for row, previous in old_metadata.items():
                            rollback[row][PLAN_METADATA] = copy.deepcopy(previous.get(PLAN_METADATA, {}))
                        partition.set_custom_meta(rollback)
                        restored = partition.get_custom_meta(list(old_metadata))
                        if any(
                            restored[row].get(PLAN_METADATA, {}) != previous.get(PLAN_METADATA, {})
                            for row, previous in old_metadata.items()
                        ):
                            raise RuntimeError("Metadata rollback verification failed")
                    except Exception:
                        self.poisoned.add(key)
                raise
            finally:
                self._request = self._prepared = None

    def clear_cache(self, partition_id: str) -> None:
        with self._lock:
            self._ensure_healthy()
            self.base.clear_cache(partition_id)
            self.plans = {key: value for key, value in self.plans.items() if key[0] != partition_id}
            self.poisoned = {key for key in self.poisoned if key[0] != partition_id}
            self.generations[partition_id] = self.generations.get(partition_id, 0) + 1

    def save_checkpoint(self) -> dict:
        with self._lock:
            self._ensure_healthy()
            return copy.deepcopy(
                dict(
                    schema=SCHEMA,
                    order="pack-first",
                    signature=self.signature,
                    base=self.base.save_checkpoint(),
                    plans=self.plans,
                    generations=self.generations,
                    poisoned=self.poisoned,
                )
            )

    def load_checkpoint(self, state: dict) -> None:
        with self._lock:
            try:
                if not isinstance(state, dict) or set(state) != {
                    "schema",
                    "order",
                    "signature",
                    "base",
                    "plans",
                    "generations",
                    "poisoned",
                }:
                    raise ValueError("Incomplete pack-first checkpoint")
                if (state["schema"], state["order"], state["signature"]) != (SCHEMA, "pack-first", self.signature):
                    raise ValueError("Pack-first checkpoint schema/order/model/data/DP signature mismatch")
                candidate = copy.deepcopy(state)
                if not isinstance(candidate["plans"], dict) or not isinstance(candidate["generations"], dict):
                    raise ValueError("Invalid pack-first checkpoint maps")
                if not isinstance(candidate["poisoned"], set):
                    raise ValueError("Invalid pack-first checkpoint transaction state")
                for partition, generation in candidate["generations"].items():
                    if not isinstance(partition, str) or type(generation) is not int or generation < 0:
                        raise ValueError("Invalid pack-first checkpoint generation")
                saved = copy.deepcopy(self.base.save_checkpoint())
                if not isinstance(candidate["base"], dict) or set(candidate["base"]) != set(saved):
                    raise ValueError("Incomplete base sampler checkpoint")
                cache = candidate["base"]["_states"]
                for key, plan in candidate["plans"].items():
                    validate_cached_plan(
                        plan, self.context.estimator, key, self.signature, candidate["generations"].get(key[0], 0)
                    )
                    ranks = cache[key[0]][key[1]]
                    if set(ranks) != set(range(self.dp_size)):
                        raise ValueError("Pack-first checkpoint is missing a DP rank cache")
                    for rank in range(self.dp_size):
                        expected = list(plan.rank_rows(rank))
                        if ranks[rank].get(key[2]) != (expected, expected.copy()):
                            raise ValueError("Pack-first checkpoint sampled/consumed/plan mismatch")
                    if self.is_sft and candidate["base"]["_balanced_cache"].get(key) != [
                        list(plan.rank_rows(rank)) for rank in range(self.dp_size)
                    ]:
                        raise ValueError("Pack-first SFT assignment replay cache differs from plan")
                expected = set()
                for partition, tasks in cache.items():
                    for task, ranks in tasks.items():
                        if task in TRAIN_TASKS:
                            for rank, minis in ranks.items():
                                for mini in minis:
                                    key = (partition, task, mini)
                                    if key not in candidate["plans"]:
                                        raise ValueError("Pack-first assignment cache has no plan")
                                    expected.add(key)
                if expected != set(candidate["plans"]):
                    raise ValueError("Pack-first checkpoint contains orphan plans")
                if (
                    self.is_sft
                    and {key for key in candidate["base"]["_balanced_cache"] if key[1] in TRAIN_TASKS} != expected
                ):
                    raise ValueError("Pack-first SFT checkpoint contains orphan replay assignments")
                # A disposable copy validates the base loader without touching live admission.
                probe = copy.deepcopy(self.base)
                probe.load_checkpoint(copy.deepcopy(candidate["base"]))
            except Exception as exc:
                raise ValueError("Invalid pack-first checkpoint") from exc
            try:
                self.base.load_checkpoint(candidate["base"])
                if self.base.save_checkpoint() != candidate["base"]:
                    raise RuntimeError("Shared base checkpoint load verification failed")
            except Exception:
                try:
                    self._restore_base(saved)
                except Exception as exc:
                    self._unrecoverable = f"{type(exc).__name__}: {exc}"
                raise
            self.plans = candidate["plans"]
            self.generations = candidate["generations"]
            self.poisoned = candidate["poisoned"]
            self._unrecoverable = None
