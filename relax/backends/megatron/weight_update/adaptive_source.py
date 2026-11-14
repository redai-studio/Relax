# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Reuse or patch canonical Qwen tensors before the complete TP converter."""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from relax.distributed.weight_sync import traffic
from relax.distributed.weight_sync.checkpoint import snapshot_tensors
from relax.distributed.weight_sync.parallel_io import ordered_map
from relax.distributed.weight_sync.scratch import ScratchWorkspace
from relax.distributed.weight_sync.source_io import agree_errors, agree_io_state, coordinated_io
from relax.distributed.weight_sync.source_map import apply_replacements
from relax.distributed.weight_sync.source_patch import file_digest, plan_replacements, replacement_cost
from relax.distributed.weight_sync.store import DTYPES, digest
from relax.utils.logging_utils import get_logger

from .source_map import qwen_source_mapping


logger = get_logger(__name__)


class AdaptiveSourceExporter:
    """Keep one confirmed local-shard base and one private candidate per rank.

    Every route uses the same frozen snapshot. Unknown mappings and dense
    replacements use the existing complete converter. Cache loss selects FULL;
    replicas of the same source shard must agree before any route is selected.
    """

    def __init__(self, iterator, group, scratch: Path, *, stream="default", io_timeout=3600, allow_patch=True):
        import torch.distributed as dist

        self.iterator, self.group = iterator, group
        self.io_timeout = io_timeout
        self.allow_patch = allow_patch
        self.rank, self.world = dist.get_rank(group), dist.get_world_size(group)
        self.scratch = scratch / f"source-cache-{self.rank}"
        self.workspace = None
        self._io_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="weight-source-io")
        self._hash_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="weight-source-hash")

        def open_workspace():
            self.workspace = ScratchWorkspace(self.scratch, f"source-cache:{stream}:{self.rank}")

        try:
            self._io(open_workspace, "source cache recovery")
        except BaseException:
            if self.workspace:
                self.workspace.close()
            self._io_executor.shutdown(wait=True)
            self._hash_executor.shutdown(wait=True)
            raise
        self.base = None
        self.candidate = None
        self.statistics = {}

    def _agree(self, value):
        if self.world == 1:
            return [value]
        values = [None] * self.world
        traffic.all_gather_object(values, value, group=self.group)
        return values

    def _check(self, error):
        failures = agree_errors(error, self.group)
        if any(failures):
            raise RuntimeError(f"Source export failed: {failures}")

    def _io(self, operation, stage):
        return coordinated_io(
            operation,
            lambda state: agree_io_state(state, self.group),
            stage=stage,
            timeout=self.io_timeout,
            executor=self._io_executor,
        )

    def _snapshot(self, frozen, directory):
        from megatron.core import mpu

        hashes = {}
        table = snapshot_tensors(frozen.items(), directory, hashes=hashes)
        metadata = {}
        # Bind identity to the same frozen CPU bytes that were written. This
        # is not a disk-readback certificate: replacement planning validates
        # consumed candidate bytes, and future base reuse rehashes its file.
        for name, (dtype, shape, path) in table.items():
            expert = ".experts." in name
            shard = mpu.get_expert_tensor_parallel_rank() if expert else mpu.get_tensor_model_parallel_rank()
            metadata[name] = {"dtype": dtype, "shape": shape, "hash": hashes[name], "shard": shard}
        return table, metadata

    def prepare(self, frozen, *, force_full: bool = False):
        from megatron.core import mpu

        from .hf_weight_iterator_direct import _get_megatron_full_params

        self._io(self.discard, "discard previous source candidate")
        directory, output_dir = None, None
        error, local = None, {}
        try:
            directory = self.workspace.temporary()
            self.candidate = {"directory": directory, "source": {}, "outputs": {}}
            self.candidate.update(logical={}, logical_groups={}, output_hashes={}, logical_hashes={})
            output_dir = directory / "canonical"
            output_dir.mkdir()
        except Exception as caught:
            error = f"{type(caught).__name__}: {caught}"
        self._check(error)
        try:
            table, local = self._io(lambda: self._snapshot(frozen, directory / "shards"), "source shard snapshot/hash")
            self.candidate["source"] = table
            self.candidate["source_hashes"] = {name: item["hash"] for name, item in local.items()}
        except Exception as caught:
            error = f"{type(caught).__name__}: {caught}"
        self._check(error)
        metadata = self._agree(local)
        infos = [info for bucket in self.iterator.megatron_local_param_info_buckets for info in bucket]
        self.statistics = {"REUSE": 0, "PATCH": 0, "FULL": 0, "patch_transport_payload_bytes": 0, "fallbacks": []}
        self.statistics["patch_budgets"] = []
        used_outputs = set()
        # Base availability is a capture-wide fact. An anchor or cold capture
        # has no replacement work: do not execute thousands of empty file
        # futures and object collectives before taking the already-known FULL.
        may_plan = not force_full and all(self._agree(bool(self.base)))
        for info in infos:
            tp = (
                mpu.get_expert_tensor_parallel_world_size()
                if ".experts." in info.name
                else mpu.get_tensor_model_parallel_world_size()
            )
            sharded = info.attrs.get("tensor_model_parallel") and info.attrs.get("parallel_mode") != "duplicated"
            owners = []
            for shard in range(tp if sharded else 1):
                replicas = [
                    (rank, data[info.name])
                    for rank, data in enumerate(metadata)
                    if info.name in data and (not sharded or data[info.name]["shard"] == shard)
                ]
                if not replicas:
                    raise ValueError(f"Missing source shard: {info.name}, {shard}")
                content = {(item["hash"], item["dtype"], tuple(item["shape"])) for _, item in replicas}
                if len(content) != 1:
                    raise ValueError(f"SOURCE_MISMATCH: {info.name}, shard {shard}")
                owners.append(min(rank for rank, _ in replicas))
            try:
                mapping_fn = getattr(self.iterator, "source_mapping", None)
                mapping = (
                    mapping_fn(info, tp)
                    if mapping_fn
                    else qwen_source_mapping(self.iterator.args, self.iterator.model_name, info, tp)
                )
            except (ValueError, KeyError, AssertionError, NotImplementedError):
                mapping = None
            plans, error = {}, None

            def plan_local():
                if mapping is not None and not force_full and self.base:
                    for shard, owner in enumerate(owners):
                        if owner == self.rank:
                            old = self.base["source"].get(info.name)
                            changed = self.candidate["source_hashes"][info.name] != self.base["source_hashes"].get(
                                info.name
                            )
                            if old and changed and not self.allow_patch:
                                # D is FULL-or-REUSE. A changed frozen shard
                                # cannot reuse its base, so constructing sparse
                                # frames which D will discard is wasted work.
                                # FULL consumes the frozen tensor, not this file.
                                plans[shard] = {"route": "FULL", "frames": [], "payload_bytes": 0}
                                continue
                            if old and file_digest(old[2]) == self.base["source_hashes"][info.name]:
                                dtype, _, current = self.candidate["source"][info.name]
                                if self.candidate["source_hashes"][info.name] == self.base["source_hashes"][info.name]:
                                    # Both complete contents were just hashed,
                                    # including revalidation of the old shard.
                                    # Equal contents need no second pair of file
                                    # reads or elementwise replacement scan.
                                    plans[shard] = {"route": "REUSE", "frames": [], "payload_bytes": 0}
                                else:
                                    plans[shard] = plan_replacements(
                                        current,
                                        old[2],
                                        DTYPES[dtype],
                                        directory / "frames" / digest(info.name.encode()) / str(shard),
                                        expected_current_hash=self.candidate["source_hashes"][info.name],
                                    )

            if mapping is not None and may_plan:
                try:
                    self._io(plan_local, "source replacement planning")
                except Exception as caught:
                    error = f"{type(caught).__name__}: {caught}"
                proposals = self._agree({"plans": plans, "error": error, "has_base": bool(self.base)})
                selected = [proposals[owner]["plans"].get(shard) for shard, owner in enumerate(owners)]
            else:
                proposals, selected = [], []
            route = "FULL"
            if (
                mapping is not None
                and may_plan
                and all(item["has_base"] and item["error"] is None for item in proposals)
                and all(plan and plan["route"] != "FULL" for plan in selected)
            ):
                if not any(plan["frames"] for plan in selected):
                    route = "REUSE"
                else:
                    cost = replacement_cost(owners, selected, proposals)
                    # One TP/PP/EP group gives a lower bound on the original
                    # complete export. Replicated groups can add further sends.
                    full = info.size * (
                        (tp * (tp - 1) if sharded else 0)
                        + mpu.get_pipeline_model_parallel_world_size()
                        - 1
                        + (mpu.get_expert_model_parallel_world_size() - 1 if ".experts." in info.name else 0)
                    )
                    if self.allow_patch and sum(cost.values()) < full:
                        route = "PATCH"
                    self.statistics["patch_budgets"].append(
                        {"source": info.name, **cost, "full_payload_lower_bound": full, "selected": route}
                    )
            if route != "FULL":

                def link_base():
                    for name, shape in mapping.outputs:
                        dtype, old_shape, source = self.base["outputs"][name]
                        if tuple(old_shape) != shape:
                            raise ValueError("Canonical base schema changed")
                        if file_digest(source) != self.base["output_hashes"][name]:
                            raise ValueError("Canonical base content changed")
                        os.link(source, output_dir / (digest(name.encode()) + ".bin"))

                try:
                    self._io(link_base if self.rank == 0 else None, "canonical base validation")
                except (RuntimeError, TimeoutError):
                    route = "FULL"
            if route == "PATCH":
                error = self._patch(info, mapping, owners, selected, directory, output_dir)
                if error:
                    self.statistics["fallbacks"].append({"source": info.name, "reason": error})
                    route = "FULL"
            exported, error, host = {}, None, None
            if route == "FULL":
                # All ranks enter the original collectives in the same order.
                complete = _get_megatron_full_params([info], frozen, agreement=self._check)
                tensors = None
                try:
                    tensors = self.iterator._convert_to_hf_named_tensors(complete, [info])
                except Exception as caught:
                    error = f"{type(caught).__name__}: {caught}"
                self._check(error)
                host, error = None, None
                try:
                    if self.rank == 0:
                        host = [(name, tensor.detach().contiguous().cpu()) for name, tensor in tensors]
                except Exception as caught:
                    error = f"{type(caught).__name__}: {caught}"
                self._check(error)

                def write_complete():
                    for name, _ in host:
                        (output_dir / (digest(name.encode()) + ".bin")).unlink(missing_ok=True)
                    return snapshot_tensors(
                        host, output_dir, hashes=self.candidate["output_hashes"], hash_executor=self._hash_executor
                    )

                exported = self._io(write_complete if self.rank == 0 else None, "gathered source write") or {}
                del complete, tensors
            elif self.rank == 0:
                dtype = metadata[owners[0]][info.name]["dtype"]
                exported = {
                    name: (dtype, list(shape), output_dir / (digest(name.encode()) + ".bin"))
                    for name, shape in mapping.outputs
                }
                if route == "REUSE":
                    # link_base verified these exact files during this capture.
                    self.candidate["output_hashes"].update(
                        (name, self.base["output_hashes"][name]) for name in exported
                    )
            if used_outputs.intersection(exported):
                error = "Converter produced duplicate logical outputs"
            self._check(error)
            used_outputs.update(exported)
            self.candidate["outputs"].update(exported)
            finish = getattr(self.iterator, "finish_group", None)
            if finish is not None:
                error = None
                prepare_group = getattr(self.iterator, "prepare_group", None)
                logical, prepared, converted = None, None, None
                if prepare_group is not None:
                    resident = getattr(self.iterator, "prepare_resident_group", None)

                    def prepare_bridge():
                        if route == "FULL" and resident is not None:
                            return resident(info, host, directory)
                        return prepare_group(info, exported, directory, route, self.base)

                    prepared = self._io(
                        prepare_bridge if self.rank == 0 else None,
                        "Bridge resident input or cache validation",
                    )
                # Converters may touch device state; only frozen CPU input
                # reads and completed output writes run in file workers.
                try:
                    if self.rank == 0:
                        if prepare_group is None:
                            logical = finish(info, exported, directory, route, self.base)
                        else:
                            converted = self.iterator.convert_group(info, prepared)
                except Exception as caught:
                    error = f"{type(caught).__name__}: {caught}"
                self._check(error)
                if prepare_group is not None:

                    def write_bridge():
                        if getattr(self.iterator, "snapshot_hashes", False):
                            return self.iterator.write_group(
                                prepared,
                                converted,
                                hashes=self.candidate["logical_hashes"],
                                hash_executor=self._hash_executor,
                            )
                        return self.iterator.write_group(prepared, converted)

                    logical = self._io(write_bridge if self.rank == 0 else None, "Bridge logical output write")
                del prepared, converted
                if self.rank == 0:
                    if set(logical) & set(self.candidate["logical"]):
                        error = "Bridge produced duplicate logical outputs"
                    else:
                        self.candidate["logical"].update(logical)
                        self.candidate["logical_groups"][info.name] = list(logical)
                self._check(error)
            host = None
            self.statistics[route] += 1

        def hash_outputs():
            # New snapshots already carry the identity of their owned write
            # buffers. REUSE identities came from this capture's disk checks.
            # Only PATCH and legacy adapter outputs need a fresh file hash.
            tables = (
                (self.candidate["outputs"], self.candidate["output_hashes"]),
                (self.candidate["logical"], self.candidate["logical_hashes"]),
            )
            paths = list(
                dict.fromkeys(
                    item[2] for values, known in tables for name, item in values.items() if name not in known
                )
            )
            hashes = dict(zip(paths, ordered_map(file_digest, paths, 4), strict=True))
            for values, known in tables:
                known.update((name, hashes[item[2]]) for name, item in values.items() if name not in known)

        self._io(hash_outputs, "canonical output hashes")
        return self.candidate["logical"] if hasattr(self.iterator, "finish_group") else self.candidate["outputs"]

    def _patch(self, info, mapping, owners, plans, directory, output_dir):
        import torch
        import torch.distributed as dist

        error = None
        root = dist.get_global_rank(self.group, 0)
        for shard, (owner, plan) in enumerate(zip(owners, plans, strict=True)):
            for frame in plan["frames"]:

                def read_frame():
                    path = directory / "frames" / digest(info.name.encode()) / str(shard)
                    payload = (path / f"{frame['sequence']}.bin").read_bytes()
                    if len(payload) != frame["size"] or digest(payload) != frame["sha256"]:
                        raise ValueError("Source frame differs from its plan")
                    return payload

                try:
                    payload = self._io(read_frame if self.rank == owner else None, "source replacement read")
                except (RuntimeError, TimeoutError) as caught:
                    return str(caught)
                if owner != 0:
                    data, allocation_error = None, None
                    try:
                        if self.rank == owner:
                            data = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
                        elif self.rank == 0:
                            data = torch.empty(frame["size"], dtype=torch.uint8)
                    except Exception as caught:
                        allocation_error = str(caught)
                    failures = self._agree(allocation_error)
                    if any(failures):
                        return str(failures)
                    if self.rank == owner:
                        dist.send(data, dst=root, group=self.group)
                        traffic.emit("G", data.numel() * data.element_size(), f"source_patch:{self.rank}->0")
                    elif self.rank == 0:
                        dist.recv(data, src=dist.get_global_rank(self.group, owner), group=self.group)
                        payload = data.numpy().tobytes()
                    self.statistics["patch_transport_payload_bytes"] += frame["size"]

                def apply_frame():
                    if digest(payload) != frame["sha256"]:
                        raise ValueError("Transferred frame checksum differs")
                    apply_replacements(
                        mapping, shard, frame["offset"], payload, info.dtype.itemsize, output_dir, sync=False
                    )

                try:
                    self._io(apply_frame if self.rank == 0 and error is None else None, "source replacement apply")
                except (RuntimeError, TimeoutError) as caught:
                    # Finish the advertised schedule before choosing FULL.
                    error = str(caught)
        # These files are reconstructible private source scratch, never the
        # durable publication or training checkpoint. Closed shared mmap writes
        # are coherent for the following local read/hash without fsync. The
        # independent publication persists and verifies its own copy before
        # confirmation; a process loss reconstructs this cache from the model.
        failures = self._agree(error)
        return str(failures) if any(failures) else None

    def confirm(self):
        if self.candidate is None:
            return
        previous = self.base
        self.base, self.candidate = self.candidate, None
        if previous:
            self._remove(previous["directory"])

    def _remove(self, directory):
        self.workspace.remove(directory)

    def discard(self):
        if self.candidate:
            self._remove(self.candidate["directory"])
            self.candidate = None

    def close(self):
        try:
            self.discard()
            if self.base:
                self._remove(self.base["directory"])
                self.base = None
            self.workspace.close()
        finally:
            self._io_executor.shutdown(wait=True)
            self._hash_executor.shutdown(wait=True)
