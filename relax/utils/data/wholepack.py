# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Recipe-scoped deterministic wholepack scheduling; no GPU imports."""

import hashlib
import inspect
import json

from transfer_queue.sampler import BaseSampler

from relax.utils.data.packing import get_binpack_partitions as _binpack_partitions
from relax.utils.data.wholepack_policy import (
    EXPERT_PARALLEL_SIZE,
    GLOBAL_BATCH_SIZE,
    WORLD_SIZE,
    config_contract,
    enabled,
    sampler_padding,
)


PLAN_KEY = "_sft_wholepack_v1"


def redistribute(old, capacity=131072):
    dp_size = len(old)
    if dp_size not in (16, 32) or not old[0]:
        raise ValueError("wholepack requires DP16/TP4 or DP32/TP2")
    tp_size = WORLD_SIZE // dp_size
    ep_dp_width = EXPERT_PARALLEL_SIZE // tp_size
    K = len(old[0])
    if any(len(g) != K for g in old):
        raise ValueError("wholepack source K mismatch")
    real_ids = []
    for g in old:
        for w in g:
            if not w["ids"] or w["t"] <= 0 or w["T"] < w["t"]:
                raise ValueError("invalid source pack")
            if w["t"] > capacity and len(w["ids"]) != 1:
                raise ValueError("invalid source oversize pack")
            if w["dummy"]:
                if not any(not r["dummy"] and r["ids"] == w["ids"] for r in g):
                    raise ValueError("source dummy has no local real source")
            else:
                real_ids.extend(w["ids"])
    if len(real_ids) != GLOBAL_BATCH_SIZE or len(set(real_ids)) != GLOBAL_BATCH_SIZE:
        raise ValueError("source does not contain1600 unique real identities")
    packs = [w for g in old for w in g]

    def fill(g):
        assert g
        g = list(g)
        while len(g) < K:
            g.append(dict(min(g, key=lambda w: w["t"]), dummy=True))
        return g

    def measure(gs):
        weights = [10 + int(max(g[w]["T"] for g in gs) > capacity) for w in range(K)]
        A = []
        B = []
        for e in range(EXPERT_PARALLEL_SIZE):
            group = gs[e * ep_dp_width : (e + 1) * ep_dp_width]
            A.append(sum(max(g[w]["Q"] for g in group) * weights[w] for w in range(K)))
            B.append(sum(max(g[w]["T"] for g in group) for w in range(K)))
        return dict(
            Amax=max(A),
            Bmax=max(B),
            A=A,
            B=B,
            long_waves=sum(x == 11 for x in weights),
            dummy_tokens=sum(w["t"] for g in gs for w in g if w["dummy"]),
            dummy_count=sum(w["dummy"] for g in gs for w in g),
            total_padded_tokens=sum(w["T"] for g in gs for w in g),
            samples_min=min(sum(len(w["ids"]) for w in g if not w["dummy"]) for g in gs),
            samples_max=max(sum(len(w["ids"]) for w in g if not w["dummy"]) for g in gs),
        )

    orig = measure(old)

    # Policy independent of step: global wholepack LPT by normalized hybrid cost, equal pack slots,oversize isolated firstwave.
    def cost(w):
        return 10 * w["Q"] / orig["Amax"] + w["T"] / orig["Bmax"]

    # Bundle each original dummy with its exact real source pack.
    sources = {tuple(w["ids"]): [w] for w in packs if not w["dummy"]}
    for w in packs:
        if w["dummy"]:
            sources[tuple(w["ids"])].append(w)
    bundles = list(sources.values())
    bins = [[] for _ in range(dp_size)]
    load = [0.0] * dp_size
    for bundle in sorted(bundles, key=lambda b: (-len(b), -sum(cost(w) for w in b), b[0]["ids"])):
        possible = [
            d for d, g in enumerate(bins) if len(g) + len(bundle) <= K and sum(x["oversize"] for x in g + bundle) <= 1
        ]
        if not possible:
            return old, {"original": orig, "candidate": orig, "fallback_reason": "bundle_greedy_infeasible"}
        d = min(possible, key=lambda d: (load[d], d))
        bins[d].extend(bundle)
        load[d] += sum(cost(w) for w in bundle)
    bins = [sorted(fill(g), key=lambda w: (-cost(w), w["ids"], w["dummy"])) for g in bins]
    lpt = measure(bins)
    # One deterministic first-improvement pass over whole DP-vector swaps: topology-aware with no sample/pack mutation.
    v = lpt
    perm = list(range(dp_size))
    tries = accept = 0
    for a in range(dp_size):
        for b in range(a + 1, dp_size):
            if a // ep_dp_width == b // ep_dp_width:
                continue
            bb = bins.copy()
            bb[a], bb[b] = bb[b], bb[a]
            vv = measure(bb)
            tries += 1

            def score(x):
                return max(A / orig["Amax"] + B / orig["Bmax"] for A, B in zip(x["A"], x["B"]))

            if (score(vv), sum(x * x for x in vv["A"])) < (score(v), sum(x * x for x in v["A"])):
                bins = bb
                v = vv
                perm[a], perm[b] = perm[b], perm[a]
                accept += 1
    assert all(
        all(any(not real["dummy"] and real["ids"] == w["ids"] for real in g) for w in g if w["dummy"]) for g in bins
    )
    ids = [i for g in bins for w in g if not w["dummy"] for i in w["ids"]]
    assert sorted(ids) == sorted(i for g in old for w in g if not w["dummy"] for i in w["ids"])
    assert all(len(g) == K for g in bins)
    assert len(packs) == sum(len(g) for g in bins)
    for g in bins:
        for w in g:
            if not w["dummy"]:
                assert w["T"] >= w["t"] and (w["t"] <= capacity or len(w["ids"]) == 1)
    # Keep original if the scored candidate regresses; stable floating point CPU only.
    fallback_reason = None
    if v["long_waves"] != orig["long_waves"]:
        bins, v = old, orig
        fallback_reason = "long_wave_count_change"
    elif score(v) > score(orig):
        bins, v = old, orig
        fallback_reason = "joint_proxy_regression"
    return bins, {"original": orig, "candidate": v, "fallback_reason": fallback_reason}


def encode_metadata(bins, total_lengths, capacity=131072, pad_size=2048):
    """Unique real IDs only; dummy slots reference real IDs on the same DP."""
    all_ids = [i for g in bins for w in g if not w["dummy"] for i in w["ids"]]
    digest = hashlib.sha256(json.dumps(bins, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assignments = []
    updates = {}
    for d, g in enumerate(bins):
        ids = [i for w in g if not w["dummy"] for i in w["ids"]]
        assert len(ids) == len(set(ids)) and ids
        slots = [dict(ids=w["ids"], dummy=w["dummy"]) for w in g]
        header = dict(
            version=1,
            dp_rank=d,
            K=len(g),
            global_count=len(all_ids),
            local_count=len(ids),
            capacity=capacity,
            pad_size=pad_size,
            lengths=[total_lengths[i] for i in ids],
            plan_sha256=digest,
            slots=slots,
        )
        for i in ids:
            updates[i] = {"total_lengths": total_lengths[i], PLAN_KEY: {"plan_sha256": digest}}
        updates[ids[0]][PLAN_KEY] = header
        assignments.append(ids)
    return assignments, updates


def restore_plan(batch_meta, rollout_data):
    """Called after TP metadata broadcast; preserve tensor payloads and sample
    order."""
    metadata = batch_meta.get_all_custom_meta()
    entries = [m.get(PLAN_KEY) for m in metadata]
    if not any(entries):
        return
    if not all(entries):
        raise ValueError("partial wholepack metadata")
    headers = [m for m in entries if "slots" in m]
    if len(headers) != 1:
        raise ValueError("wholepack requires exactly one local header")
    h = headers[0]
    if h["global_count"] != GLOBAL_BATCH_SIZE or h["version"] != 1:
        raise ValueError("wholepack GBS/version mismatch")
    if any(m["plan_sha256"] != h["plan_sha256"] for m in entries):
        raise ValueError("mixed wholepack plans")
    ids = list(batch_meta.global_indexes)
    if len(ids) != len(set(ids)) or len(ids) != h["local_count"]:
        raise ValueError("wholepack local identity mismatch")
    if list(rollout_data["total_lengths"]) != h["lengths"]:
        raise ValueError("wholepack actual length mismatch")
    if "tokens" in rollout_data and [len(t) for t in rollout_data["tokens"]] != h["lengths"]:
        raise ValueError("wholepack actual token payload length mismatch")
    index = {identity: i for i, identity in enumerate(ids)}
    indices = [[index[i] for i in w["ids"]] for w in h["slots"]]
    dummy = [j for j, w in enumerate(h["slots"]) if w["dummy"]]
    for j, group in enumerate(indices):
        if not group or (sum(h["lengths"][i] for i in group) > h["capacity"] and len(group) != 1):
            raise ValueError("wholepack violates capacity/oversize singleton")
        if j in dummy and not any(group == other for k, other in enumerate(indices) if k not in dummy):
            raise ValueError("wholepack dummy has no local real source")
    real = [i for j, group in enumerate(indices) if j not in dummy for i in group]
    if sorted(real) != list(range(len(ids))):
        raise ValueError("wholepack real samples must appear exactly once")
    if len(indices) != h["K"] or len(rollout_data["total_lengths"]) != len(ids):
        raise ValueError("wholepack payload size mismatch")
    rollout_data["rollout_mini_local_sample_counts"] = [len(ids)]
    rollout_data["rollout_mini_global_sample_counts"] = [h["global_count"]]
    rollout_data[PLAN_KEY] = dict(
        indices=indices, dummy=dummy, K=h["K"], capacity=h["capacity"], pad_size=h["pad_size"], sha256=h["plan_sha256"]
    )


class WholepackSampler(BaseSampler):
    """Composition preserves existing GRPO/TQ replay and checkpoint
    protocols."""

    def __init__(self, n_samples_per_prompt=1, dp_size=1, capacity=131072, pad_size=None):
        from transfer_queue import SeqlenBalancedSampler

        pad_size = sampler_padding(n_samples_per_prompt, dp_size, capacity, pad_size)
        self.base = SeqlenBalancedSampler(n_samples_per_prompt=n_samples_per_prompt, dp_size=dp_size)
        if "assignment_transform" not in inspect.signature(self.base.sample).parameters:
            raise RuntimeError("wholepack requires the image-built TransferQueue assignment_transform patch")
        self.capacity, self.pad_size = capacity, pad_size
        self.plan_metadata = {}

    def has_cached_result(self, partition_id, task_name, sampling_config=None):
        return self.base.has_cached_result(partition_id, task_name, sampling_config)

    def sample(self, ready_indexes, batch_size, task_name="", partition_id="", *args, **kwargs):
        if "eval" in task_name or "eval" in partition_id:
            return self.base.sample(ready_indexes, batch_size, task_name, partition_id, *args, **kwargs)
        key = (partition_id, task_name, kwargs.get("batch_index", 0))
        partition = kwargs.get("partition")
        if partition is None or batch_size * self.base.dp_size != GLOBAL_BATCH_SIZE:
            raise ValueError("wholepack requires partition metadata and GBS1600")

        def transform_assignment(original):
            metadata = partition.get_custom_meta([i for g in original for i in g])
            lengths = {i: int(metadata[i]["total_lengths"]) for g in original for i in g}
            real = [_binpack_partitions([lengths[i] for i in g], self.capacity) for g in original]
            if sum(map(len, original)) != GLOBAL_BATCH_SIZE or len(lengths) != GLOBAL_BATCH_SIZE:
                raise ValueError("wholepack requires exactly1600 unique identities")
            K = max(map(len, real))
            old = []
            for ids, groups in zip(original, real):
                rows = []
                for group in groups:
                    values = [lengths[ids[i]] for i in group]
                    t = sum(values)
                    padding = (-t) % self.pad_size
                    rows.append(
                        dict(
                            ids=[ids[i] for i in group],
                            t=t,
                            T=t + padding,
                            Q=sum(v * v for v in values) + padding * padding,
                            dummy=False,
                            oversize=t > self.capacity,
                        )
                    )
                while len(rows) < K:
                    rows.append(dict(min(rows, key=lambda w: w["t"]), dummy=True))
                old.append(rows)
            bins, evidence = redistribute(old, capacity=self.capacity)
            exchange_evidence = None
            if (self.base.dp_size, self.capacity) == (16, 262144):
                from relax.utils.data.wholepack_exchange import refine

                bins, exchange_evidence = refine(bins, self.capacity)
            from relax.utils.logging_utils import get_logger

            get_logger(__name__).info(
                "wholepack partition=%s batch=%s K=%s fallback=%s global_real=%s",
                partition_id,
                key[2],
                K,
                evidence.get("fallback_reason"),
                len(lengths),
            )
            if exchange_evidence is not None:
                get_logger(__name__).info("wholepack vector exchange evidence=%s", exchange_evidence)
            assignments, updates = encode_metadata(bins, lengths, self.capacity, self.pad_size)
            # Preserve any unrelated producer metadata.
            updates = {i: {**metadata[i], **update} for i, update in updates.items()}
            self.plan_metadata[key] = updates
            return assignments

        sampled, consumed = self.base.sample(
            ready_indexes,
            batch_size,
            task_name,
            partition_id,
            *args,
            assignment_transform=transform_assignment,
            **kwargs,
        )
        if not sampled:
            return sampled, consumed
        # Reapply after checkpoint recovery too, before TQ constructs BatchMeta.
        if key not in self.plan_metadata:
            raise RuntimeError("wholepack replay missing checkpoint plan metadata")
        partition.set_custom_meta(self.plan_metadata[key])
        return sampled, consumed

    def clear_cache(self, partition_id):
        self.base.clear_cache(partition_id)
        self.plan_metadata = {k: v for k, v in self.plan_metadata.items() if k[0] != partition_id}

    def save_checkpoint(self):
        return {
            "base": self.base.save_checkpoint(),
            "wholepack_metadata": self.plan_metadata,
            "geometry": {"dp": self.base.dp_size, "pad": self.pad_size, "capacity": self.capacity},
        }

    def load_checkpoint(self, state):
        if "wholepack_metadata" not in state:
            raise ValueError("wholepack requires a matching wholepack checkpoint")
        expected = {"dp": self.base.dp_size, "pad": self.pad_size, "capacity": self.capacity}
        if "geometry" in state and state["geometry"] != expected:
            raise ValueError("checkpoint wholepack geometry mismatch")
        for updates in state["wholepack_metadata"].values():
            for meta in updates.values():
                header = meta.get(PLAN_KEY, {})
                if "slots" in header and header["pad_size"] != self.pad_size:
                    raise ValueError("checkpoint padding geometry mismatch")
        self.base.load_checkpoint(state["base"])
        self.plan_metadata = state["wholepack_metadata"]


def restore_checked(args, batch_meta, rollout_data, agree):
    """Always agree across all training ranks before surfacing local plan
    errors.

    agree receives integer payload and returns elementwise MAX over WORLD.
    Production and tests use CPU Gloo int64 all_reduce.
    """
    if not enabled(args):
        return
    error = None
    K = 0
    digest = [0] * 16
    try:
        capacity, pad_size = config_contract(args)
        restore_plan(batch_meta, rollout_data)
        plan = rollout_data[PLAN_KEY]
        if (plan["capacity"], plan["pad_size"]) != (capacity, pad_size):
            raise ValueError("wholepack plan differs from active args")
        K = int(plan["K"])
        digest = [int(plan["sha256"][i : i + 4], 16) for i in range(0, 64, 4)]
        if len(digest) != 16 or K <= 0:
            raise ValueError("invalid wholepack K/hash")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    payload = [int(error is not None), K, -K] + digest + [-v for v in digest]
    result = agree(payload)
    if result[0] or result[1] != -result[2] or any(result[3 + i] != -result[19 + i] for i in range(16)):
        raise RuntimeError(
            "wholepack validation failed on one or more ranks: " + (error or "peer configuration/metadata error")
        )


def agree(payload):
    """CPU control collective over this train actor's explicit process
    group."""
    import torch
    import torch.distributed as dist

    from relax.utils.distributed_utils import get_gloo_group

    values = torch.tensor(payload, dtype=torch.int64, device="cpu")
    dist.all_reduce(values, op=dist.ReduceOp.MAX, group=get_gloo_group())
    return values.tolist()


def all_ready(value: bool) -> bool:
    return not agree([int(not value)])[0]


def await_batch(batch, fetch, agree, *, timeout_seconds: float, poll_seconds: float):
    """Keep ready DP slices while peers retry publication; never consume
    twice."""
    import time

    start = time.monotonic()
    while True:
        ready = batch[0] is not None
        any_ready, any_empty, expired = agree(
            [int(ready), int(not ready), int(time.monotonic() - start > timeout_seconds)]
        )
        if expired:
            raise TimeoutError("Timed out waiting for whole-pack data on peer ranks")
        if not any_ready or not any_empty:
            return batch
        if not ready:
            if poll_seconds > 0:
                time.sleep(poll_seconds)
            batch = fetch()
