# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Ordinary TP and expert adapters sharing one versioned IPC transaction."""

import hashlib
import json
from math import prod
from uuid import uuid4

import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu

from relax.backends.sglang.shard_p2p_weight_loader import expert_shapes
from relax.distributed.checkpoint_service.backends.shard_p2p.plan import fingerprint
from relax.utils.logging_utils import get_logger

from .tp_shard_source import OrdinaryTPSource


logger = get_logger(__name__)


class CombinedShardSource:
    def __init__(self, owner, *, process_group, expert=None) -> None:
        self.expert = expert
        self.group = process_group
        self.rank = dist.get_rank(self.group)
        self.ordinary = OrdinaryTPSource(owner._hf_weight_iterator, process_group=self.group)
        self._tp_size = owner.args.rollout_num_gpus_per_engine
        self._num_layers = owner.args.num_layers
        self.has_experts = expert is not None
        self.has_ordinary = True
        self._plans = None

    def describe(self) -> dict:
        base = self.expert.describe() if self.expert is not None else {}
        self.parallel_sizes = (mpu.get_context_parallel_world_size(), mpu.get_expert_tensor_parallel_world_size())
        return dict(
            base,
            tp_size=self._tp_size,
            num_layers=self._num_layers,
            source_world_size=dist.get_world_size(self.group),
            expert_enabled=self.has_experts,
            ordinary_targets=[{} for _ in range(self._tp_size)],
        )

    def validate_axes(self) -> None:
        if self.expert is not None:
            self.expert.validate_axes()

    def reset(self) -> None:
        if self.expert is not None:
            self.expert.reset()
        self.ordinary.selected = frozenset()
        self.native_skip_names = frozenset()
        self._plans = None

    @property
    def raw_recipe_available(self) -> bool:
        return self.expert is None or self.expert.raw_recipe_available

    def select_raw_recipe(self, enabled: bool) -> None:
        if self.expert is not None:
            self.expert.select_raw_recipe(enabled)

    def observe_native(self, values) -> None:
        # Packed expert slices use another loader and are not canonical HF inputs.
        self.ordinary.observe_native(
            [
                (name, value)
                for name, value in values
                if ".experts." not in name and not name.startswith("relax.expert_slice.v1:")
            ]
        )

    def begin_native_seed(self) -> None:
        self.ordinary.native_specs.clear()

    def replica_record(self, weights) -> dict:
        tensors, _, _ = self.ordinary._views(weights, selected_only=False)
        hashes = {}
        if len(self.ordinary.replica_ranks) > 1:
            for name, value in tensors.items():
                hashes[name] = hashlib.sha256(value.detach().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
        return dict(
            expert=self.expert.replica_record(weights) if self.expert is not None else None,
            global_rank=dist.get_global_rank(self.group, self.rank),
            replicas=self.ordinary.replica_ranks,
            hashes=hashes,
        )

    def validate_replicas(self, records) -> None:
        if self.expert is not None:
            self.expert.validate_replicas([record["expert"] for record in records])
        by_global = {record["global_rank"]: record for record in records}
        for record in records:
            if any(
                by_global[rank]["replicas"] != record["replicas"] or by_global[rank]["hashes"] != record["hashes"]
                for rank in record["replicas"]
            ):
                raise ValueError("Ordinary TP DP/CP replica weights differ")

    def prepare_targets(self, sync, weights, *, version: int) -> None:
        records = sync._gather(sync._phase(lambda: self.ordinary.catalogue(weights)))

        def commands():
            if self.rank != 0:
                return None
            specs = {}
            by_global = {record["global_rank"]: record for record in records}
            if len(by_global) != len(records):
                raise ValueError("Duplicate ordinary TP source owner")
            for rank, record in enumerate(records):
                if record["rank"] != rank or record["replica_ranks"][record["replica_rank"]] != record["global_rank"]:
                    raise ValueError("Ordinary TP source ownership disagrees")
                for position, peer in enumerate(record["replica_ranks"]):
                    other = by_global[peer]
                    if other["replica_ranks"] != record["replica_ranks"] or other["replica_rank"] != position:
                        raise ValueError("Ordinary TP replica groups disagree")
                for name, spec in record["native_specs"].items():
                    if name in specs and specs[name] != spec:
                        raise ValueError("Native HF schema differs across Actor stages")
                    specs[name] = spec
            description = dict(records=records, native_specs=specs, source_world_size=len(records))
            digest = fingerprint(description)
            return [
                [
                    dict(
                        op="describe_tp",
                        session=sync.session,
                        engine=engine,
                        tp_rank=tp,
                        version=version,
                        nonce=uuid4().hex,
                        geometry=sync.geometry,
                        description=description,
                        description_hash=digest,
                    )
                    for tp in range(self._tp_size)
                ]
                for engine in range(sync.engine_count)
            ]

        commands = sync._phase(commands)
        responses = sync._fanout(commands)

        def accept():
            if self.rank != 0:
                return None
            replies = []
            for pair, response in zip(commands, responses, strict=True):
                if not isinstance(response, dict) or response.get("success") is not True:
                    raise ValueError("Missing ordinary TP layout response")
                values = json.loads(response["message"])["shard_p2p_tp_targets"]
                if len(values) != self._tp_size:
                    raise ValueError("Incomplete ordinary TP target inventory")
                for command, reply in zip(pair, values, strict=True):
                    if (
                        reply["description_hash"] != command["description_hash"]
                        or reply["tp_rank"] != command["tp_rank"]
                    ):
                        raise ValueError("Misrouted ordinary TP layout reply")
                replies.append(values)
            if any(pair != replies[0] for pair in replies[1:]):
                raise ValueError("Inference engines disagree on ordinary TP geometry")
            selected = set.intersection(*(set(reply["selected_hf"]) for reply in replies[0]))
            groups = [group for reply in replies[0] for group in reply["components"]]
            groups += [names for record in records for names in record["groups"].values()]
            previous = None
            while previous != selected:
                previous = set(selected)
                for names in groups:
                    if not set(names) <= selected:
                        selected.difference_update(names)
            layouts, plans = [], [[] for _ in records]
            for tp, reply in enumerate(replies[0]):
                targets = {
                    name: shape
                    for name, shape in reply["target_shapes"].items()
                    if set(reply["target_hf"][name]) <= selected
                }
                layouts.append(targets)
                for rank in range(len(records)):
                    plans[rank].append(
                        [
                            write
                            for write in reply["writes"]
                            if write["source_rank"] == rank and write["param"] in targets and write["hf"] in selected
                        ]
                    )
            names = [[name for name, hfs in record["groups"].items() if set(hfs) <= selected] for record in records]
            source_bytes = sum(
                prod(commands[0][0]["description"]["native_specs"][name]["shape"]) * 2 for name in selected
            )
            logger.info(
                "SHARD_P2P_TP selected_hf=%d logical_bytes=%d receiver_bytes=%d effective=%s",
                len(selected),
                source_bytes,
                sum(prod(shape) * 2 for layout in layouts for shape in layout.values()) * len(replies),
                bool(selected),
            )
            return dict(
                layouts=layouts,
                plans=plans,
                names=names,
                selected_hf=sorted(selected),
                refresh=[bool(reply.get("native_refresh", False)) for reply in replies[0]],
            )

        result = [sync._phase(accept)]
        dist.broadcast_object_list(result, src=0, group=self.group)
        result = result[0]
        self.ordinary.selected = frozenset(result["names"][self.rank])
        self.native_skip_names = frozenset(name for names in result["names"] for name in names)
        self._plans = result["plans"][self.rank]
        self._target_shapes = result["layouts"]
        self._native_refresh = result["refresh"]
        self.empty = not result["selected_hf"] and not self.has_experts
        sync._phase(lambda: self.ordinary.bind(weights, seed=True))

    def local_plan(self, weights, *, sources_only: bool = False):
        sources = self.ordinary.bind(weights)
        plans = None if sources_only else [[dict(write) for write in pair] for pair in self._plans]
        if self.expert is not None:
            extra, expert_plans = self.expert.local_plan(weights, sources_only=sources_only)
            sources.update({"expert:" + name: value for name, value in extra.items()})
            if not sources_only:
                for plan, writes in zip(plans, expert_plans, strict=True):
                    plan.extend(dict(write, source="expert:" + write["source"]) for write in writes)
        return sources, plans

    def complete_geometry(self, plans, geometry: dict) -> dict:
        if self.expert is not None:
            geometry = self.expert.complete_geometry(plans, geometry)
        return dict(geometry, ordinary_targets=self._target_shapes)

    def geometries(self, geometry: dict) -> list[dict]:
        return [
            dict(geometry, ordinary_targets=targets, native_refresh=refresh)
            for targets, refresh in zip(self._target_shapes, self._native_refresh, strict=True)
        ]

    @staticmethod
    def target_layouts(geometry: dict) -> dict:
        shapes = expert_shapes(geometry) if geometry.get("expert_enabled", True) else {}
        shapes.update(geometry.get("ordinary_targets") or {})
        return {name: dict(nbytes=prod(shape) * 2) for name, shape in shapes.items()}
