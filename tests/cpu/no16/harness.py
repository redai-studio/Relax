# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Fixture adapters and independent schedule invariants, confined to P1
tests."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from types import SimpleNamespace
from typing import Any

import numpy as np

from .source import HERE, ROOT, load, samplers, slime


MODEL = dict(
    hidden_size=8,
    num_attention_heads=2,
    num_query_groups=1,
    kv_channels=4,
    vocab_size=32,
    ffn_hidden_size=16,
    num_layers=2,
    num_experts=None,
)


class EvidenceLog:
    """Capture production fallback diagnostics without importing training
    logging."""

    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, message: str, *args: Any) -> None:
        self.warnings.append(message % args if args else message)

    def info(self, *args: Any, **kwargs: Any) -> None:
        pass


class Partition:
    """In-memory replacement of partition metadata I/O, not of sampler
    logic."""

    def __init__(self, samples: list[dict]) -> None:
        self.global_indexes = {sample["row"] for sample in samples}
        self.metadata = {
            sample["row"]: {
                "sample_index": sample["rollout_id"],
                "total_lengths": sample["length"],
                "producer_id": sample["id"],
            }
            for sample in samples
        }

    def get_custom_meta(self, indexes: list[int]) -> dict:
        return {index: self.metadata[index] for index in indexes}

    def set_custom_meta(self, values: dict) -> None:
        self.metadata.update(values)


def model_args(case: dict, *, flops: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        **{
            **MODEL,
            **case.get("model", {}),
            "use_dynamic_batch_size": case.get("dynamic", True),
            "max_tokens_per_gpu": case["capacity"],
            "micro_batch_size": case.get("micro_batch_size", 1),
            "balance_data": case.get("balance_data", True),
            "balance_by_flops": flops,
        }
    )


def fixture_steps(case: dict) -> tuple[list[list[dict]], list[str]]:
    samples = case["samples"]
    identities = list(dict.fromkeys(sample["rollout_id"] for sample in samples))
    size = case.get("global_batch_size", len(identities))
    groups = [identities[start : start + size] for start in range(0, len(identities) - size + 1, size)]
    steps = [
        [sample for identity in group for sample in samples if sample["rollout_id"] == identity] for group in groups
    ]
    kept = {sample["id"] for step in steps for sample in step}
    return steps, [sample["id"] for sample in samples if sample["id"] not in kept]


def microbatch(samples: list[dict], dummy: bool, costs: dict[str, float]) -> dict:
    return {
        "ids": [sample["id"] for sample in samples],
        "dummy": dummy,
        "tokens": sum(sample["length"] for sample in samples),
        "reference_fwd_flops": sum(costs[sample["id"]] for sample in samples),
    }


def costs_for(case: dict) -> dict[str, float]:
    # Exact pinned reference formula, only for reporting/Slime fixture execution.
    reference = slime()
    args = model_args(case)
    by_length = {
        sample["length"]: reference.calculate_fwd_flops([sample["length"]], args) for sample in case["samples"]
    }
    return {sample["id"]: by_length[sample["length"]] for sample in case["samples"]}


def run_slime(case: dict, *, flops: bool = True) -> dict:
    reference = slime()
    samples = case["samples"]
    args = model_args(case, flops=flops)
    config = dict(
        dp_size=case["dp"],
        cp_size=case.get("cp", 1),
        vpp_size=case.get("vpp", 1),
        microbatch_group_size_per_vp_stage=case.get("mb_group", 1),
    )
    groups, dropped = fixture_steps(case)
    gbs = case.get("global_batch_size", len({s["rollout_id"] for s in samples}))
    partitions, indices, counts, global_sizes = reference.build_dp_schedule(
        args,
        config,
        [s["length"] for s in samples],
        global_batch_size=gbs,
        rollout_indices=[s["rollout_id"] for s in samples],
    )
    costs = costs_for(case)
    steps = []
    offset = 0
    for step_index, (cohort, count) in enumerate(zip(groups, counts, strict=True)):
        initial = reference._pack_step_into_mbs(
            [s["length"] for s in cohort],
            args=args,
            use_dynamic_batch_size=args.use_dynamic_batch_size,
            max_per_bin=case["capacity"] * case.get("cp", 1),
            micro_batch_size=args.micro_batch_size,
            balance_by_flops=flops,
        )
        ranks = []
        for rank in range(case["dp"]):
            local_mbs = indices[rank][offset : offset + count]
            mbs = [microbatch([samples[partitions[rank][i]] for i in mb], False, costs) for mb in local_mbs]
            ranks.append({"rank": rank, "microbatches": mbs})
        steps.append(
            {
                "step": step_index,
                "logical_count": global_sizes[step_index],
                "ranks": ranks,
                "initial_pack_ids": [[cohort[i]["id"] for i in mb] for mb in initial],
            }
        )
        offset += count
    return {
        "engine": "slime_flops" if flops else "slime_tokens",
        "steps": steps,
        "dropped_ids": dropped,
        "raw_partitions": partitions,
        "raw_micro_batch_indices": indices,
        "raw_counts": counts,
    }


def production_partitioner() -> Any:
    namespace: dict[str, Any] = {"logger": EvidenceLog()}
    files = [
        ("relax/utils/data/seqlen_balancing.py", None),
        ("relax/utils/data/packing.py", None),
        ("relax/utils/data/data.py", {"get_first_fit_partitions", "get_minimum_num_micro_batch_size"}),
        (
            "relax/backends/megatron/data.py",
            {
                "_get_first_fit_partitions",
                "_partitions_fit_capacity_or_singleton_oversize",
                "_get_capacity_safe_balanced_partitions",
                "_get_seqlen_partitions_with_dummy_padding",
                "_round_up_to_microbatch_group",
                "_get_micro_batch_token_capacity",
                "DataIterator",
                "get_data_iterator",
            },
        ),
    ]
    for path, names in files:
        module = load(ROOT / path, names=names, injected=namespace, omitted_imports=("megatron",))
        namespace.update({key: value for key, value in vars(module).items() if not key.startswith("__")})
    # Every function must see the same dependency namespace, as in its real module.
    for value in list(namespace.values()):
        if callable(value) and hasattr(value, "__globals__"):
            value.__globals__.update(namespace)
    return SimpleNamespace(**namespace)


def iterator_plan(
    case: dict,
    assigned: list[list[dict]],
    costs: dict[str, float],
    local_counts: list[list[int]] | None = None,
) -> tuple[list[dict], list[str]]:
    """Execute actual get_data_iterator with serial NumPy MAX collective
    doubles."""
    dp = len(assigned)
    traces = []
    log = EvidenceLog()

    def one(rank: int, maxima: list[list[int]] | None) -> tuple[Any, list[int], list[list[int]]]:
        production = production_partitioner()
        reductions: list[list[int]] = []

        def all_reduce(values: np.ndarray, *, op: str, group: str) -> None:
            assert op == "MAX" and group == "fixture_dp"
            reductions.append(values.tolist())
            if maxima is not None:
                values[:] = maxima[len(reductions) - 1]

        args = SimpleNamespace(
            global_batch_size=case.get("global_batch_size", len(case["samples"])),
            use_dynamic_batch_size=case.get("dynamic", True),
            micro_batch_size=case.get("micro_batch_size", 1),
            max_tokens_per_gpu=case["capacity"],
            balance_data=case.get("balance_data", True),
            sft_packing_strategy=case.get("packing", "balanced"),
            data_pad_size_multiplier=case.get("pad_multiplier", 1),
            allgather_cp=case.get("allgather_cp", False),
            qkv_format="thd",
            dynamic_context_parallel=case.get("dynamic_cp", False),
            uses_unsplit_forward=case.get("unsplit", False),
        )
        dependencies = dict(
            np=np,
            logger=log,
            torch=SimpleNamespace(
                tensor=lambda values, **kwargs: np.array(values, dtype=int),
                int=int,
                clamp=lambda values, min: np.maximum(values, min),
            ),
            mpu=SimpleNamespace(
                get_data_parallel_world_size=lambda **kwargs: dp,
                get_data_parallel_group=lambda: "fixture_dp",
                get_virtual_pipeline_model_parallel_world_size=lambda: case.get("vpp", 1),
                get_context_parallel_world_size=lambda: case.get("cp", 1),
                get_tensor_model_parallel_world_size=lambda: case.get("tp", 1),
            ),
            dist=SimpleNamespace(all_reduce=all_reduce, ReduceOp=SimpleNamespace(MAX="MAX")),
            device_utils=SimpleNamespace(make_current_torch_device=lambda: "cpu"),
            get_model_config=lambda model: SimpleNamespace(microbatch_group_size_per_vp_stage=case.get("mb_group", 1)),
            is_preference_mode=lambda args: False,
            ROLLOUT_MINI_LOCAL_SAMPLE_COUNTS_KEY="rollout_mini_local_sample_counts",
            ROLLOUT_MINI_GLOBAL_SAMPLE_COUNTS_KEY="rollout_mini_global_sample_counts",
        )
        for value in vars(production).values():
            if callable(value) and hasattr(value, "__globals__"):
                value.__globals__.update(dependencies)
        cohort = assigned[rank]
        payload = {
            "total_lengths": [sample["length"] for sample in cohort],
            "ids": [s["id"] for s in cohort],
            "rollout_mini_local_sample_counts": [len(cohort)] if local_counts is None else local_counts[rank],
        }
        iterators, counts = production.get_data_iterator(args, [None], payload)
        return iterators, counts, reductions

    for rank in range(dp):
        traces.append(one(rank, None)[2])
    if len({len(trace) for trace in traces}) != 1:
        raise AssertionError("fixture ranks have different collective counts")
    maxima = [np.max(np.array([trace[index] for trace in traces]), axis=0).tolist() for index in range(len(traces[0]))]
    ranks = []
    samples_by_id = {sample["id"]: sample for sample in case["samples"]}
    for rank in range(dp):
        iterators, counts, _ = one(rank, maxima)
        # These are the actual production iterators, including dummy markers/reset.
        materialized = []
        layouts = []
        for iterator in iterators:
            batches = []
            emitted_offsets = []
            for _ in range(sum(counts)):
                batch = iterator.get_next(["ids"])
                emitted_offsets.append(batch["replay_micro_batch_index"])
                batches.append(
                    microbatch([samples_by_id[i] for i in batch["ids"]], batch.get("__is_dummy__", False), costs)
                )
            iterator.reset()
            assert iterator.get_next(["ids"])["ids"] == batches[0]["ids"]
            materialized.append(batches)
            if local_counts is not None:
                layouts.append(
                    {
                        "indices": copy.deepcopy(iterator.micro_batch_indices),
                        "dummy_offsets": sorted(iterator.dummy_micro_batch_offsets),
                        "emitted_offsets": emitted_offsets,
                    }
                )
        assert all(batches == materialized[0] for batches in materialized)
        ranks.append({"rank": rank, "microbatches": materialized[0], "iterator_copies": len(iterators)})
        if local_counts is not None:
            ranks[-1]["step_counts"] = counts
            ranks[-1]["collective_inputs"] = traces[rank]
            ranks[-1]["iterator_layouts"] = layouts
    return ranks, sorted(set(log.warnings))


def run_relax(case: dict) -> dict:
    classes = samplers()
    mode = case.get("placement", "balanced")
    sampler = (
        classes["sft"](dp_size=case["dp"]) if mode == "sft" else classes["rl"](dp_size=case["dp"], placement=mode)
    )
    partition = Partition(case["samples"])
    by_row = {sample["row"]: sample for sample in case["samples"]}
    costs = costs_for(case)
    cohorts, unadmitted = fixture_steps(case)
    steps, warnings = [], []
    for index, cohort in enumerate(cohorts):
        logical_count = len({sample["rollout_id"] for sample in cohort})
        if logical_count % case["dp"]:
            raise ValueError("fixture count admission must be divisible by DP")
        assigned = []
        for rank in range(case["dp"]):
            rows, consumed = sampler.sample(
                [sample["row"] for sample in cohort],
                logical_count // case["dp"],
                task_name="train",
                partition_id="fixture",
                dp_rank=rank,
                batch_index=index,
                partition=partition,
            )
            assert rows == consumed
            assigned.append([by_row[row] for row in rows])
        ranks, messages = iterator_plan(case, assigned, costs)
        steps.append({"step": index, "logical_count": logical_count, "ranks": ranks})
        warnings.extend(messages)
    return {
        "engine": "relax_legacy_" + mode,
        "steps": steps,
        "unadmitted_ids": unadmitted,
        "fallback_warnings": sorted(set(warnings)),
    }


def run_wholepack(case: dict) -> dict:
    """Exercise the existing recipe's real assignment, metadata and restore
    code."""
    classes = samplers()
    wp = classes["wholepack"]
    sampler = wp.WholepackSampler(dp_size=case["dp"], capacity=case["capacity"], pad_size=4096)
    partition = Partition(case["samples"])
    lookup = {sample["row"]: sample for sample in case["samples"]}
    costs = costs_for(case)
    ranks, hashes = [], set()
    for rank in range(case["dp"]):
        rows, consumed = sampler.sample(
            sorted(partition.global_indexes), 100, "sft_train", "p", dp_rank=rank, batch_index=0, partition=partition
        )
        assert rows == consumed
        payload = {"total_lengths": [lookup[row]["length"] for row in rows]}
        metadata = SimpleNamespace(
            global_indexes=rows, get_all_custom_meta=lambda rows=rows: [partition.metadata[row] for row in rows]
        )
        wp.restore_plan(metadata, payload)
        plan = payload[wp.PLAN_KEY]
        hashes.add(plan["sha256"])
        assert payload["rollout_mini_global_sample_counts"] == [1600]
        assert all(partition.metadata[row]["producer_id"] == lookup[row]["id"] for row in rows)
        ranks.append(
            {
                "rank": rank,
                "microbatches": [
                    microbatch([lookup[rows[i]] for i in indices], offset in plan["dummy"], costs)
                    for offset, indices in enumerate(plan["indices"])
                ],
            }
        )
    assert len(hashes) == 1
    return {
        "status": "ok",
        "engine": "relax_legacy_wholepack",
        "steps": [{"step": 0, "logical_count": 1600, "ranks": ranks}],
        "plan_sha256": hashes.pop(),
    }


def streaming_snapshot() -> dict:
    fixture = __import__("json").loads((HERE / "fixtures/streaming.json").read_text(encoding="utf-8"))
    partition = Partition(fixture["samples"])
    sampler = samplers()["rl"](dp_size=2, placement="streaming")
    trace = []
    for request in fixture["requests"]:
        rows, consumed = sampler.sample(
            request["ready"],
            0,
            "train",
            "p",
            dp_rank=request["rank"],
            batch_index=request["batch"],
            production_done=False,
            token_budget=10,
            dp_size=2,
            partition=partition,
            rollout_mini_index=0,
            window_quota=8,
        )
        trace.append(
            {
                "rank": request["rank"],
                "batch": request["batch"],
                "rows": rows,
                "consumed": consumed,
                "dummy": sampler.is_dummy_round("p", "train", request["rank"], request["batch"]),
            }
        )
    window = sampler._partition_task_states[("p", "train")].windows[0]
    return {
        "trace": trace,
        "admitted_identities": window.admitted_identities,
        "completed_identities": sorted(window.completed_identities),
        "terminal_batch_by_dp": {str(rank): index for rank, index in window.terminal_batch_by_dp.items()},
        "window_drained": sampler.is_window_drained("p", "train", 0, 8, False, False),
    }


def evaluate(case: dict, engine: str) -> dict:
    try:
        if engine == "relax":
            result = run_relax(case)
        else:
            result = run_slime(case, flops=engine == "slime_flops")
        return {"status": "ok", **result}
    except (AssertionError, ValueError, RuntimeError) as error:
        return {"status": "error", "error_type": type(error).__name__, "message": str(error)}


def raw_mapping_violations(case: dict, schedule: dict) -> list[str]:
    """Audit Slime global positions/local indices, not Relax queue-row indices.

    Slime's raw buffers span every admitted step. Counts slice the flattened
    per-rank microbatch list; successful Slime schedules contain no dummy.
    Relax/Wholepack have different contracts and do not expose these raw
    fields.
    """
    if schedule.get("engine") not in ("slime_tokens", "slime_flops"):
        return []
    errors = []
    parts = schedule.get("raw_partitions")
    indices = schedule.get("raw_micro_batch_indices")
    counts = schedule.get("raw_counts")
    steps = schedule["steps"]
    if (
        not isinstance(parts, list)
        or len(parts) != case["dp"]
        or not isinstance(indices, list)
        or len(indices) != case["dp"]
        or not isinstance(counts, list)
        or len(counts) != len(steps)
        or any(type(count) is not int or count < 0 for count in counts)
    ):
        return ["raw_geometry"]
    cohorts, _ = fixture_steps(case)
    admitted = Counter(s["id"] for cohort in cohorts for s in cohort)
    global_ids = []
    for rank, (buffer, local_mbs) in enumerate(zip(parts, indices, strict=True)):
        if not isinstance(buffer, list) or any(
            type(i) is not int or not 0 <= i < len(case["samples"]) for i in buffer
        ):
            errors.append("raw_global_index")
            continue
        global_ids.extend(case["samples"][i]["id"] for i in buffer)
        if not isinstance(local_mbs, list) or len(local_mbs) != sum(counts):
            errors.append("raw_microbatch_count")
            continue
        used = []
        base = 0
        for step, count in zip(steps, counts, strict=True):
            if len(step["ranks"]) <= rank or step["ranks"][rank]["rank"] != rank:
                errors.append("raw_rank_mapping")
                continue
            actual = step["ranks"][rank]["microbatches"]
            if len(actual) != count:
                errors.append("raw_microbatch_count")
            for local, mb in zip(local_mbs[base : base + count], actual):
                if (
                    not isinstance(local, list)
                    or not local
                    or any(type(i) is not int or not 0 <= i < len(buffer) for i in local)
                ):
                    errors.append("raw_local_index")
                    continue
                used.extend(local)
                if mb["dummy"]:
                    errors.append("raw_unexpected_dummy")
                if [case["samples"][buffer[i]]["id"] for i in local] != mb["ids"]:
                    errors.append("raw_sample_mapping")
            base += count
        if Counter(used) != Counter(range(len(buffer))):
            errors.append("raw_local_coverage")
    if Counter(global_ids) != admitted:
        errors.append("raw_global_coverage")
    return sorted(set(errors))


def _fixed_schedule_input(name: str) -> dict:
    """Read pre-remediation bytes, never derive expectations from run
    output."""
    files = ("fixtures/schedules.json", "golden/slime_tokens.json", "golden/slime_flops.json")
    # Public checksums in the same order as files.
    hashes = (
        "ad548005d9c556632ab4e97d738a664218c01b946d588e973b16ce4422a911eb",
        "eaebc02584c3e19535901547683d841481a84138d0dfe4a7702f995434eedf46",
        "8057167af39d676c42b6201454616e936cb5a602c830e1586561af00ef78231a",
    )
    data = (HERE / name).read_bytes()
    if hashlib.sha256(data).hexdigest() != hashes[files.index(name)]:
        raise ValueError(f"Fixed expected-schedule input changed: {name}")
    return json.loads(data)


def expected_slime_schedule(case: dict, engine: str) -> list[dict]:
    """Ordered IDs for fixed fixtures: hand proof plus historical regression.

    This function accepts inputs only, not raw buffers, Steps or
    initial_pack_ids. The small hand oracle covers mixed_dp2 and both
    multiple_steps cohorts. Other fixtures use the original, hash-pinned Golden
    AFTER splitting and DP assignment; initial packing alone cannot predict
    those final assignments. Unknown/modified fixtures fail explicitly rather
    than borrowing a namesake's expected output. This is deterministic
    algorithm regression, not a claim that every alternative valid MB ordering
    violates structural invariants.
    """
    if engine not in ("slime_tokens", "slime_flops"):
        raise ValueError("Expected-schedule oracle requires a Slime engine")
    fixtures = _fixed_schedule_input("fixtures/schedules.json")["cases"]
    if case not in fixtures:
        raise ValueError("No fixed expected-schedule oracle for this input fixture")
    if case["name"] in ("mixed_dp2", "multiple_steps"):
        # Hand derivation (see final remediation report): each 8-sample cohort
        # packs into (0,1),(2,3),(4,5),(6,7), all 10 tokens. No split: K=4,
        # DP=2, VPP=1. Token KK tie rules give ranks (0,3)/(1,2).
        # FLOPs weights 30784,30336,30016,29824 give the same ranks, with
        # 60608/60352 workloads. Final per-rank MB ordinals are sorted.
        result = []
        for start in range(0, len(case["samples"]), 8):
            cohort = case["samples"][start : start + 8]
            result.append(
                {
                    "step": start // 8,
                    "ranks": [
                        {"rank": rank, "microbatches": [[cohort[i]["id"] for i in mb] for mb in bins]}
                        for rank, bins in enumerate((((0, 1), (6, 7)), ((2, 3), (4, 5))))
                    ],
                }
            )
        return result
    golden = _fixed_schedule_input(f"golden/{engine}.json")[case["name"]]
    if golden["status"] != "ok":
        raise ValueError("Fixed fixture has no successful expected schedule")
    return _ordered_schedule_ids(golden)


def _ordered_schedule_ids(schedule: dict) -> list[dict]:
    return [
        {
            "step": step["step"],
            "ranks": [
                {"rank": rank["rank"], "microbatches": [mb["ids"] for mb in rank["microbatches"]]}
                for rank in step["ranks"]
            ],
        }
        for step in schedule["steps"]
    ]


def expected_schedule_violations(case: dict, schedule: dict) -> list[str]:
    """Separate deterministic regression from violations()'s invariants.

    Callers opt into the fixed-fixture contract; unsupported inputs raise, so
    absence of an oracle cannot silently be reported as a successful check.
    """
    expected = expected_slime_schedule(case, schedule["engine"])
    if schedule.get("status") != "ok" or _ordered_schedule_ids(schedule) != expected:
        return ["expected_schedule_mismatch"]
    return []


def mini_offset_violations(assigned: list[list[dict]], local_counts: list[list[int]], ranks: list[dict]) -> list[str]:
    """Recompute sample/MB bases independently for observed dynamic
    iterators."""
    errors = []
    if len(assigned) != len(local_counts) or len(ranks) != len(assigned):
        return ["mini_geometry"]
    for samples, counts, rank in zip(assigned, local_counts, ranks, strict=True):
        sizes = rank["step_counts"]
        if len(counts) != len(sizes) or sum(counts) != len(samples):
            errors.append("mini_geometry")
            continue
        for layout in rank["iterator_layouts"]:
            indices = layout["indices"]
            dummy = layout["dummy_offsets"]
            emitted = layout["emitted_offsets"]
            mbs = rank["microbatches"]
            if indices is None or len(indices) != sum(sizes) or len(mbs) != sum(sizes):
                errors.append("mini_geometry")
                continue
            if emitted != list(range(sum(sizes))):
                errors.append("mini_microbatch_offset")
            expected_dummy = [i for i, mb in enumerate(mbs) if mb["dummy"]]
            if dummy != expected_dummy:
                errors.append("mini_dummy_offset")
            sample_base, mb_base = 0, 0
            for n_samples, n_mbs in zip(counts, sizes, strict=True):
                wanted = Counter(range(sample_base, sample_base + n_samples))
                real = []
                real_bins = [indices[i] for i in range(mb_base, mb_base + n_mbs) if i not in dummy]
                for i in range(mb_base, mb_base + n_mbs):
                    local = indices[i]
                    if not local or any(type(j) is not int or j not in wanted for j in local):
                        errors.append("mini_sample_offset")
                        continue
                    if [samples[j]["id"] for j in local] != mbs[i]["ids"]:
                        errors.append("mini_sample_mapping")
                    if i in dummy:
                        if local not in real_bins:
                            errors.append("mini_dummy_source")
                    else:
                        real.extend(local)
                if Counter(real) != wanted:
                    errors.append("mini_sample_coverage")
                sample_base += n_samples
                mb_base += n_mbs
    return sorted(set(errors))


def violations(case: dict, schedule: dict) -> list[str]:
    """Independent structural oracle; reported Slime capacity risks are not
    repaired."""
    if schedule["status"] != "ok":
        return ["schedule_error"]
    errors = raw_mapping_violations(case, schedule)
    samples = {sample["id"]: sample for sample in case["samples"]}
    if len(samples) != len(case["samples"]):
        errors.append("fixture_duplicate_id")
    cohorts, _ = fixture_steps(case)
    if len(schedule["steps"]) != len(cohorts):
        errors.append("step_count")
    for index, step in enumerate(schedule["steps"]):
        counts, seen = [], []
        if index < len(cohorts) and step["logical_count"] != len({sample["rollout_id"] for sample in cohorts[index]}):
            errors.append("logical_count")
        if len(step["ranks"]) != case["dp"] or [r["rank"] for r in step["ranks"]] != list(range(case["dp"])):
            errors.append("rank_geometry")
        for rank in step["ranks"]:
            mbs = rank["microbatches"]
            counts.append(len(mbs))
            real = [mb["ids"] for mb in mbs if not mb["dummy"]]
            for mb in mbs:
                ids = mb["ids"]
                if not ids or any(identity not in samples for identity in ids):
                    errors.append("unknown_or_empty_microbatch")
                    continue
                token_count = sum(samples[identity]["length"] for identity in ids)
                if token_count != mb["tokens"]:
                    errors.append("token_accounting")
                if case.get("dynamic", True) and token_count > case["capacity"] * case.get("cp", 1) and len(ids) != 1:
                    errors.append("token_capacity")
                if case.get("dynamic", True) and schedule["engine"].startswith("relax"):
                    # Independent physical proxy oracle; Slime's contract uses raw
                    # lengths, so Relax-specific layout rules must not redefine it.
                    cp, tp = case.get("cp", 1), case.get("tp", 1)
                    capacity = case["capacity"] * cp
                    if case.get("allgather_cp", False):
                        alignment = tp * case.get("pad_multiplier", 1)
                        capacity = cp * (case["capacity"] // alignment) * alignment
                    lengths = [samples[identity]["length"] for identity in ids]
                    if case.get("unsplit", False) and cp > 1:
                        alignment = tp * cp * 2
                        lengths = [((length + alignment - 1) // alignment) * alignment for length in lengths]
                    if sum(lengths) > capacity and len(ids) != 1:
                        errors.append("physical_capacity_proxy")
                if not case.get("dynamic", True):
                    size = case.get("micro_batch_size", 1)
                    remainder = len(cohorts[index]) % size if index < len(cohorts) else 0
                    allowed = {size}
                    if schedule["engine"].startswith("slime") and remainder:
                        allowed.add(remainder)
                    if len(ids) not in allowed:
                        errors.append("static_microbatch_size")
                if mb["dummy"]:
                    if ids not in real:
                        errors.append("dummy_without_local_source")
                else:
                    seen.extend(ids)
            if case.get("vpp", 1) > 1 and len(mbs) % case.get("mb_group", 1):
                errors.append("vpp_alignment")
        if len(set(counts)) != 1:
            errors.append("rank_microbatch_count")
        if index < len(cohorts) and Counter(seen) != Counter(sample["id"] for sample in cohorts[index]):
            errors.append("real_sample_consumption")
    return sorted(set(errors))


def assert_schedule(case: dict, schedule: dict) -> None:
    errors = violations(case, schedule)
    if errors:
        raise AssertionError(", ".join(errors))


def compact(schedule: dict) -> dict:
    """Keep golden output readable without discarding rank or micro-batch
    order."""
    return copy.deepcopy(schedule)
