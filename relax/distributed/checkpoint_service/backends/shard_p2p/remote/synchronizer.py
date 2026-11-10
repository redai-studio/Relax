# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Remote expert shards and native fallback through checked NIXL sessions."""

import json
import math
import shutil
import time
from pathlib import Path
from typing import Any, Callable

import requests
import torch
import torch.distributed as dist

from relax.utils.env import Envs
from relax.utils.logging_utils import get_logger

from .common import NATIVE_LEASE_PROTOCOL, atomic_json, digest, read_manifest, serialize_control
from .compat import build_replica
from .loader_probe import HfTensorSpec, ModelParamStager
from .transport import BufferedWriter


logger = get_logger(__name__)


class P2PSynchronizer:
    """Shared checked transfer lifecycle; adapters supply layout and
    exports."""

    def _phase(self, fn: Callable, *, result_rank: int | None = None) -> Any:
        profile = getattr(self, "_host_profile", None)
        started = time.monotonic() if profile is not None else 0.0
        value, error = None, None
        try:
            # result_rank is local to this group; src is a global rank. Resolve
            # it before error consensus, never while peers wait in broadcast.
            source_rank = dist.get_global_rank(self.group, result_rank) if result_rank is not None else None
            value = fn()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.exception("P2P phase failed on sender %d", self.rank)
        if profile is not None:
            mark = time.monotonic()
            profile["phase_fn_s"] += mark - started
            profile["phase_count"] += 1
        errors = [None] * self.world_size
        dist.all_gather_object(errors, error, group=self.group)
        if profile is not None:
            profile["phase_error_collective_s"] += time.monotonic() - mark
        if any(errors):
            raise RuntimeError(f"P2P update failed: {errors}")
        if result_rank is not None:
            values = [value if self.rank == result_rank else None]
            started = time.monotonic() if profile is not None else 0.0
            dist.broadcast_object_list(values, src=source_rank, group=self.group)
            if profile is not None:
                profile["phase_result_broadcast_s"] += time.monotonic() - started
                profile["phase_result_broadcast_count"] += 1
            return values[0]
        return value

    def _empty_cache(self) -> None:
        profile = getattr(self, "_host_profile", None)
        result = self._release_cache(profile is not None)
        if profile is not None:
            profile.update(result)

    def _release_cache(self, profiled: bool) -> dict:
        if not profiled:
            torch.cuda.empty_cache()
            return {}
        # Allocator counters and host clocks only: do not add a CUDA fence.
        device = self.writer.buffers[0].device
        profile = {}
        profile["allocator_device"] = str(device)
        profile["allocated_before_cache_bytes"] = torch.cuda.memory_allocated(device)
        profile["reserved_before_cache_bytes"] = torch.cuda.memory_reserved(device)
        started = time.monotonic()
        torch.cuda.empty_cache()
        profile["empty_cache_s"] = time.monotonic() - started
        profile["allocated_after_cache_bytes"] = torch.cuda.memory_allocated(device)
        profile["reserved_after_cache_bytes"] = torch.cuda.memory_reserved(device)
        return profile

    def _request(self, engine: str, endpoint: str, payload: dict | None = None) -> dict:
        response = requests.post(self.endpoints[engine] + endpoint, json=payload or {}, timeout=300)
        response.raise_for_status()
        result = response.json()
        if isinstance(result, dict) and result.get("success") is False:
            raise RuntimeError(f"{endpoint}: {result.get('message')}")
        return result

    def _fanout(self, endpoint: str, payload: dict | None = None) -> None:
        futures = [self.pool.submit(self._request, engine, endpoint, payload) for engine in self.engines]
        for future in futures:
            future.result()

    def _weight_session(self, endpoint: str, payload: dict | None = None) -> None:
        futures = [
            self.pool.submit(self._request, engine, endpoint, payload)
            for engine in self.engines
            if self.weight_update_sessions[engine]
        ]
        error = None
        for future in futures:
            try:
                future.result()
            except Exception as exc:
                error = exc
        if error is not None:
            raise error

    def _control(self, op: str, directory: Path, version: int, audit: bool) -> None:
        prepared = []
        # Serialization completes for the whole batch before any request starts.
        # Keep CPU storage until every receiver has consumed its payload.
        empty = torch.empty(0)
        tp_size = self.backend.args.rollout_num_gpus_per_engine
        for engine in self.engines:
            command = dict(op=op, directory=str(directory), engine=engine, version=version, audit=audit)
            command.update(
                native_weight_lease_protocol=NATIVE_LEASE_PROTOCOL,
                weight_update_session=self.weight_update_sessions[engine],
            )
            if op == "prepare":
                command["fallback_receipt_keys"] = [f"{peer}-{tp}" for peer in self.engines for tp in range(tp_size)]
            elif op == "commit" and version != self.direct.initial_version:
                if getattr(self, "_fallback_receipt_version", None) != version:
                    raise RuntimeError("Fallback completion gather has not finished this version")
                command["fallback_receipts"] = {
                    f"{engine}-{tp}": self._fallback_receipts[f"{engine}-{tp}"] for tp in range(tp_size)
                }
            if op == "plan" and self.direct.rpc_plans_active:
                command["plan_envelopes"] = self.direct.command_plans(engine, version)
            if op in ("commit", "expert_allgather") and getattr(self.direct, "rpc_receipts_active", False):
                command["source_receipts"] = self.direct.commit_receipts(engine, version)
            if op == "prepare" and self._collective_expert_export:
                command["collective_groups"] = self._expert_routing["collective_groups"]
            if op == "expert_allgather":
                command.update(self.direct.collective_command())
            blob = serialize_control(command, empty)
            prepared.append(
                (
                    engine,
                    dict(
                        serialized_named_tensors=[blob] * tp_size,
                        load_format=self.loader,
                        selector="target",
                        flush_cache=False,
                        abort_all_requests=False,
                    ),
                )
            )
        self._plan_requests.request_all(self.pool, self._request, prepared, empty)

    def _profile_stage(self, name: str, action: Callable) -> Any:
        profile = getattr(self, "_host_profile", None)
        if profile is None:
            return action()
        started = time.monotonic()
        try:
            return action()
        finally:
            profile[name] = profile.get(name, 0.0) + time.monotonic() - started

    def _prepare(self, directory: Path, version: int, audit: bool) -> None:
        if self.rank != 0:
            return
        self._profile_stage("prepare_mkdir_s", lambda: directory.mkdir(mode=0o700))
        self._profile_stage("prepare_pause_s", lambda: self._fanout("/pause_generation"))
        # Same cache flush as the existing DCS path, before acquiring the session.
        import ray

        refs = self._profile_stage(
            "prepare_flush_submit_s",
            lambda: [engine.flush_cache.remote() for engine in self.backend.rollout_engines.values()],
        )
        self._profile_stage("prepare_flush_wait_s", lambda: ray.get(refs, timeout=300))
        self._profile_stage(
            "prepare_begin_s", lambda: self._weight_session("/begin_weight_update", dict(selector="target"))
        )
        self._profile_stage("prepare_control_s", lambda: self._control("prepare", directory, version, audit))

    def _discover_hf_specs(self) -> dict[str, HfTensorSpec]:
        version = self.backend.weight_version
        self._phase(lambda: self.reference_catalog.validate(version))
        return self._phase(
            lambda: self.reference_catalog.resolve(version) if self.rank == 0 else None,
            result_rank=0,
        )

    def _initialize_replica(self, manifest: dict) -> None:
        from .model_replica import compute_param_group_nbytes

        buffer_device = torch.device("cuda", torch.cuda.current_device())
        started = time.monotonic()
        self.replica = build_replica(
            manifest["server_fields"], manifest["tp_rank"], transfer_buffer_device=buffer_device
        )
        self._cold_profile = dict(self.replica.cold_profile)
        self._cold_profile["replica_total_s"] = time.monotonic() - started
        started = time.monotonic()
        actual = self.replica.transfer_buffer_param_layouts
        if set(actual) != set(manifest["params"]):
            raise RuntimeError("Replica and rollout parameter names differ")
        for name, layout in actual.items():
            target = manifest["params"][name]
            if (
                list(layout.shape) != target["shape"]
                or list(layout.stride) != target["stride"]
                or str(layout.dtype) != target["dtype"]
                or layout.occupied_nbytes != target["nbytes"]
            ):
                raise RuntimeError(f"Replica and rollout layout differ: {name}")
        self._cold_profile["native_layout_validation_s"] = time.monotonic() - started
        started = time.monotonic()
        self.mapping = self.replica.map_hf_names(self.specs)
        self._cold_profile["loader_dependency_probe_s"] = time.monotonic() - started
        started = time.monotonic()
        if set(self.mapping.hf_names_by_param_name) != set(actual):
            raise RuntimeError("HF export does not cover every rollout parameter")
        self.replica.validate_reload_policy()
        stager = ModelParamStager(self.mapping)
        size = max(
            self.backend.args.update_weight_buffer_size,
            max(compute_param_group_nbytes(group, actual) for group in stager.param_groups),
        )
        self._cold_profile["buffer_budget_s"] = time.monotonic() - started
        started = time.monotonic()
        self.writer = BufferedWriter(size, device=buffer_device)
        self._cold_profile["writer_agent_registration_s"] = time.monotonic() - started
        logger.info(
            "P2P layout: sender=%d hf_bytes=%d native_bytes=%d buffer_bytes=%d device=%s",
            self.rank,
            sum(math.prod(spec.shape) * spec.dtype.itemsize for spec in self.specs.values()),
            sum(layout.occupied_nbytes for layout in actual.values()),
            size,
            buffer_device,
        )

    def update(self) -> None:
        from .model_replica import pack_into_buffers

        backend = self.backend
        started = time.monotonic()
        self._host_profile = (
            dict(
                phase_count=0,
                phase_fn_s=0.0,
                phase_error_collective_s=0.0,
                phase_result_broadcast_s=0.0,
                phase_result_broadcast_count=0,
                writer_submit_s=0.0,
            )
            if Envs.RELAX_P2P_HOST_PROFILE
            else None
        )
        profile = self._host_profile
        version = backend.weight_version + 1
        audit = self.replica is None
        phases = {}
        directory = self.root / str(version)
        self._phase(lambda: self._prepare(directory, version, audit))
        phases["prepare_s"] = time.monotonic() - started
        manifest = self._phase(lambda: read_manifest(directory / f"{self.key}.manifest.json", version, self.manifest))
        self._phase(lambda: self._validate_manifest(manifest, version))
        if self.specs is None:
            mark = time.monotonic()
            self.specs = self._discover_hf_specs()
            specs_seconds = time.monotonic() - mark
            self._phase(lambda: self._initialize_replica(manifest))
            self._cold_profile["hf_specs_export_s"] = specs_seconds
            logger.info(
                "Remote shard cold sender=%d version=%d phases=%s receiver_prepare=%s",
                self.rank,
                version,
                self._cold_profile,
                manifest.get("cold_profile"),
            )
            self.direct = self._phase(lambda: self._create_direct(manifest))
            self._phase(self.direct.agree)
        self._phase(lambda: self.writer.connect(manifest))
        mapping = self.direct.fallback_mapping
        stager = ModelParamStager(mapping)
        written, digests, payload_bytes = set(), {}, 0
        transfer_started = time.monotonic()
        if profile is not None:
            profile["setup_after_prepare_s"] = transfer_started - started - phases["prepare_s"]
        self.writer.profile.clear()
        for label in ("export_s", "buffer_wait_s", "load_s", "stage_s", "send_phase_s"):
            phases[label] = 0.0
        mark = time.monotonic()
        self._phase(lambda: self.direct.prepare(directory, version))
        phases["direct_prepare_s"] = time.monotonic() - mark
        mark = time.monotonic()
        self._phase(lambda: self._control("plan", directory, version, audit) if self.rank == 0 else None)
        phases["direct_control_s"] = time.monotonic() - mark
        mark = time.monotonic()
        payload_bytes += self._phase(self.direct.send)
        finish_started = time.monotonic()
        self._phase(lambda: self.direct.finish(directory, version))
        phases["direct_finish_s"] = time.monotonic() - finish_started
        if getattr(self.direct, "rpc_receipts_active", False):
            receipt_started = time.monotonic()
            self._phase(self.direct.gather_receipts)
            self._phase(self.direct.route_receipts)
            phases["direct_receipts_s"] = time.monotonic() - receipt_started
        phases["direct_send_s"] = time.monotonic() - mark
        if getattr(self.direct, "collective", False):
            mark = time.monotonic()
            self._phase(lambda: self.direct.replicate(directory, version, audit))
            phases["expert_allgather_s"] = time.monotonic() - mark
        fallback_started = time.monotonic()

        def send(bucket: list[tuple[str, torch.Tensor]]) -> None:
            nonlocal payload_bytes
            mark = time.monotonic()
            ready = stager.stage((name, tensor) for name, tensor in bucket if name in mapping.param_names_by_hf_name)
            phases["stage_s"] += time.monotonic() - mark
            for groups in pack_into_buffers(
                ready, self.replica.transfer_buffer_param_layouts, self.writer.buffers[0].numel()
            ):
                names = [name for group in groups for name in group]
                if written.intersection(names):
                    raise RuntimeError("A target parameter was emitted twice")
                hf_tensors = [pair for group in groups for pair in ready[group]]
                mark = time.monotonic()
                index, buffer = self.writer.acquire()
                phases["buffer_wait_s"] += time.monotonic() - mark
                mark = time.monotonic()
                tensors = self.replica.load_into(buffer, names, hf_tensors)
                phases["load_s"] += time.monotonic() - mark
                if audit:
                    digests.update({name: digest(tensor) for name, tensor in tensors.items()})
                mark = time.monotonic() if profile is not None else 0.0
                self.writer.submit(index, tensors, manifest)
                if profile is not None:
                    profile["writer_submit_s"] += time.monotonic() - mark
                written.update(names)
                payload_bytes += sum(tensor.numel() for tensor in tensors.values())

        skip = self.direct.fallback_skip_sources
        buckets = iter(self._hf_buckets(skip))
        while True:
            mark = time.monotonic()
            try:
                bucket = next(buckets)
            except StopIteration:
                phases["export_s"] += time.monotonic() - mark
                break
            phases["export_s"] += time.monotonic() - mark
            # Synchronize errors between buckets, before the next TP collective.
            mark = time.monotonic()
            self._phase(lambda: send(bucket))
            phases["send_phase_s"] += time.monotonic() - mark
        mark = time.monotonic()
        self._phase(self.writer.finish)
        phases["write_tail_s"] = time.monotonic() - mark
        phases.update({"writer_" + key: value for key, value in self.writer.profile.items()})
        self._phase(stager.assert_all_done)

        def validate_coverage() -> None:
            if written != set(mapping.hf_names_by_param_name):
                raise RuntimeError("Incomplete P2P parameter coverage")

        self._phase(validate_coverage)
        self._publish_fallback_receipt(directory, version, audit, manifest, written, digests)
        transfer_seconds = time.monotonic() - transfer_started
        phases["fallback_s"] = time.monotonic() - fallback_started
        snapshot_audit = getattr(self, "_snapshot_audit", None)
        if snapshot_audit is not None:
            self._phase(lambda: snapshot_audit.verify_source(version))
        mark = time.monotonic()
        self._phase(lambda: self._commit(directory, version, audit))
        phases["commit_s"] = time.monotonic() - mark
        backend.weight_version = version
        self.completed_version = version
        self.manifest = manifest
        elapsed = time.monotonic() - started
        logger.info("Weight-sync phases: version=%d sender=%d %s", version, self.rank, json.dumps(phases))
        logger.info(
            "ShardP2P sync complete: version=%d sender=%d params=%d payload_bytes=%d transfer_s=%.3f total_s=%.3f audit=%s",
            version,
            self.rank,
            len(written),
            payload_bytes,
            transfer_seconds,
            elapsed,
            audit,
        )
        self._empty_cache()
        if profile is not None:
            profile["body_s"] = elapsed
            profile["host_total_s"] = time.monotonic() - started
            # phase_fn_s includes nested work; these fields are not additive
            # GPU timings. The total ends after cache release, before this log.
            logger.info("Weight-sync host profile: version=%d sender=%d %s", version, self.rank, json.dumps(profile))
        self._host_profile = None

    def _publish_fallback_receipt(
        self, directory: Path, version: int, audit: bool, manifest: dict, written: set[str], digests: dict
    ) -> None:
        def original() -> dict:
            return dict(
                generation=manifest["generation"],
                version=version,
                params=sorted(written),
                digests=digests,
                loader_buffers=sorted(self.replica.loader_written_buffer_names),
            )

        if version == self.direct.initial_version:
            self._phase(lambda: atomic_json(directory / f"{self.key}.done.json", original()))
            return
        from .fallback_receipts import RPC_FALLBACK_DELIVERY, checked_names, route_receipts, serialize_receipt

        def serialize() -> tuple:
            if self.completed_version != version - 1 or not self.direct.rpc_plans_active:
                raise RuntimeError("Fallback RPC requires the preceding completed version")
            keys = [
                f"{engine}-{tp}"
                for engine in self.engines
                for tp in range(self.backend.args.rollout_num_gpus_per_engine)
            ]
            if len(keys) != self.world_size or keys[self.rank] != self.key:
                raise RuntimeError("Fallback owner keys differ from Actor group slots")
            # Schemas come from native receiver manifests, not incoming receipts.
            schemas = {}
            for rank, key in enumerate(keys):
                target = self.direct.manifests[key]
                if (
                    target["version"] != version
                    or target.get("fallback_receipt_delivery") != RPC_FALLBACK_DELIVERY
                    or target.get("fallback_receipt_owner") != rank
                ):
                    raise RuntimeError("Fallback receiver capability or version changed")
                direct = set(self.direct.target_names)
                schemas[key] = dict(
                    generation=target["generation"],
                    params=set(target["params"]) - direct,
                    loader_buffers=checked_names(target["fallback_loader_buffers"]),
                )
            return (
                serialize_receipt(self.rank, self.key, original()),
                [None] * self.world_size if self.rank == 0 else None,
                dist.get_global_rank(self.group, 0),
                keys,
                schemas,
            )

        payload, gathered, root_rank, keys, schemas = self._phase(serialize)
        self._phase(lambda: dist.gather_object(payload, gathered, dst=root_rank, group=self.group))

        def accept() -> None:
            if self.rank == 0:
                self._fallback_receipts = route_receipts(gathered, keys, schemas, version, audit)
                self._fallback_receipt_version = version

        self._phase(accept)
        if getattr(self, "_host_profile", None) is not None:
            self._host_profile["rpc_fallback_receipts"] = self.world_size if self.rank == 0 else 0

    def _commit(self, directory: Path, version: int, audit: bool) -> None:
        if self.rank != 0:
            return
        self._profile_stage("commit_receivers_s", lambda: self._control("commit", directory, version, audit))
        self._complete_commit(directory, version, audit)

    def _complete_commit(self, directory: Path, version: int, audit: bool) -> None:
        if self.rank != 0:
            return
        profile = getattr(self, "_host_profile", None)

        def stage(name: str, action: Callable) -> None:
            if profile is None:
                action()
                return
            mark = time.monotonic()
            try:
                action()
            finally:
                profile[name] = time.monotonic() - mark

        stage("commit_end_session_s", lambda: self._weight_session("/end_weight_update"))
        snapshot_audit = getattr(self, "_snapshot_audit", None)
        if snapshot_audit is not None:
            stage("commit_snapshot_audit_s", lambda: snapshot_audit.control("compare"))
        stage(
            "commit_publish_version_s",
            lambda: self._fanout("/update_weight_version", dict(new_version=str(version), abort_all_requests=False)),
        )
        stage("commit_resume_generation_s", lambda: self._fanout("/continue_generation"))
        stage("commit_metadata_cleanup_s", lambda: self._cleanup_expired_metadata(version))

    def _cleanup_expired_metadata(self, version: int) -> None:
        # Keep the initial audit and two most recent versions, not every step's
        # registered-address directory during long utilization runs.
        if version > 4:
            expired = self.root / str(version - 2)
            if expired.is_dir() and not (expired / ".retain-audit").exists():
                try:
                    shutil.rmtree(expired)
                except OSError as exc:
                    logger.warning("Could not remove expired P2P metadata: %s", exc)
