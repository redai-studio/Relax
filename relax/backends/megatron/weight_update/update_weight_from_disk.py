# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Synchronous full HF export, durable publication, and fixed-group disk
install."""

import copy
import json
from dataclasses import asdict
from pathlib import Path
from time import monotonic, sleep

import ray
import torch.distributed as dist

from relax.distributed.weight_sync.__main__ import checkpoint_tensors
from relax.distributed.weight_sync.config import WeightSyncConfig
from relax.distributed.weight_sync.publisher import Publisher
from relax.distributed.weight_sync.store import SharedFSStore, atomic_write
from relax.utils.distributed_utils import get_gloo_group
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


class UpdateWeightFromDisk:
    """The source commits before discovering a consumer; no cross-role
    collective.

    This first implementation deliberately waits for installation at the normal
    synchronous update boundary. Export, encoding and prepare are independent
    of rollout liveness. Only the rank-zero coordinator accesses the fixed
    group.
    """

    def __init__(self, args, model, weights_getter, *, model_name, quantization_config) -> None:
        if quantization_config:
            raise ValueError("Disk weight sync currently requires unquantized weights")
        self.args, self.model = args, model
        self.config_path = Path(args.disk_weight_sync_config).resolve()
        self.config = WeightSyncConfig.load(self.config_path)
        self.root = self.config_path.parent
        self.weight_version = 0
        from relax.distributed.weight_sync.provenance import runtime_identity

        identity = runtime_identity()
        identity["config"] = json.loads(self.config_path.read_text())
        atomic_write(
            self.root / "ranks" / f"train-{dist.get_rank(get_gloo_group())}.json",
            json.dumps(identity, indent=2).encode(),
        )

    def update_weights(self, rollout_manager) -> None:
        from relax.backends.megatron.model import save_hf_model

        started = monotonic()
        group = get_gloo_group()
        rank = dist.get_rank(group)
        version = self.weight_version
        export = self.root / "exports" / f"update-{version:06d}"
        args = copy.copy(self.args)
        # Keep the exporter's rollout_id template: a concrete path would make
        # save_hf_model append an additional iter_XXXXXXX directory.
        args.save_hf = str(self.root / "exports" / "update-{rollout_id:06d}")
        args.save_hf_post_hook_path = None
        error = None
        try:
            # Every training rank participates, on the caller thread, while the
            # optimizer is stopped. Bridge writes the independent FULL reference.
            save_hf_model(args, version, self.model, force_sync=True)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        errors = [None] * dist.get_world_size(group)
        dist.all_gather_object(errors, error, group=group)
        if any(errors):
            raise RuntimeError(f"Disk HF export failed: {errors}")
        export_s = monotonic() - started
        result = [None]
        if rank == 0:
            try:
                store = SharedFSStore(
                    self.config.store_root, self.config.disk_budget_bytes, canonical_root=self.config.canonical_root
                )
                pub = Publisher(
                    store,
                    self.config.stream,
                    {
                        "model_config": json.loads((export / "config.json").read_text()),
                        "exporter": "hf-safetensors-v1",
                    },
                    **self.config.publisher_kwargs(),
                )
                try:
                    published = pub.finalize_pending(
                        pub.submit(f"optimizer-boundary-{version}", checkpoint_tensors(export)),
                        self.config.sync_timeout_s,
                    )
                    metrics = {"export_s": export_s, "publication": dict(pub.last_metrics)}
                finally:
                    pub.close()
                if published.version != version:
                    raise RuntimeError(
                        "Use a fresh stream/run directory; source version does not match update boundary"
                    )
                # A durable publication is independently consumable even if the
                # following rollout lookup/install fails.
                receipt_path = self.root / "versions" / f"{version:06d}.json"
                receipt = asdict(published) | {"export": str(export), "metrics": metrics, "status": "PUBLISHED"}
                atomic_write(receipt_path, json.dumps(receipt, indent=2).encode())
                engines, lock, *_ = ray.get(
                    rollout_manager.get_rollout_engines_and_lock.remote(), timeout=self.config.sync_timeout_s
                )
                metrics["installation"] = self._install(tuple(engines), lock, asdict(published), str(export))
                metrics["training_wait_s"] = monotonic() - started
                receipt.update(status="ACTIVE", metrics=metrics)
                atomic_write(receipt_path, json.dumps(receipt, indent=2).encode())
                result[0] = {"version": version}
                logger.info("Disk weight sync ACTIVE: %s/%s root=%s", published.stream, version, published.root)
            except Exception as exc:
                result[0] = {"error": f"{type(exc).__name__}: {exc}"}
        dist.broadcast_object_list(result, src=0, group=group)
        if "error" in result[0]:
            raise RuntimeError(f"Disk weight sync failed, generation remains paused: {result[0]['error']}")
        self.weight_version += 1

    def _install(self, engines: tuple, lock, published: dict, reference: str) -> dict:
        if not engines:
            raise RuntimeError("Cannot install into an empty inference group")
        timeout = self.config.sync_timeout_s
        deadline = monotonic() + timeout

        def remaining() -> float:
            value = deadline - monotonic()
            if value <= 0:
                raise TimeoutError("Fixed-group disk installation deadline exceeded")
            return value

        # Preparation does not touch live GPU parameters. The receiver cache and
        # reconstruction directories are distinct from the producer store.
        identities = ray.get([engine.get_pid_and_node_id.remote() for engine in engines], timeout=remaining())
        nodes = [identity.get("node_id") for identity in identities]
        if any(not isinstance(node, str) or not node for node in nodes):
            raise RuntimeError("Cannot prepare a fixed inference group without every member's node identity")
        representatives = {}
        for node, engine in zip(nodes, engines):
            representatives.setdefault(node, engine)
        node_targets = dict(
            zip(
                representatives,
                ray.get(
                    [
                        engine.prepare_disk_checkpoint.remote(str(self.config_path), published, reference)
                        for engine in representatives.values()
                    ],
                    timeout=remaining(),
                ),
            )
        )
        prepared = [node_targets[node] for node in nodes]
        while not ray.get(lock.acquire.remote(), timeout=remaining()):
            sleep(min(0.05, remaining()))
        paused_at = monotonic()
        try:
            ray.get([engine.pause_generation.remote(timeout=remaining()) for engine in engines], timeout=remaining())
            checked = ray.get(
                [
                    engine.install_disk_checkpoint.remote(target, published, remaining())
                    for engine, target in zip(engines, prepared)
                ],
                timeout=remaining(),
            )
            if any(item.get("status") != "VERIFIED" or item.get("root") != published["root"] for item in checked):
                raise RuntimeError("Inference member did not verify the requested disk target")
            resume_started = monotonic()
            ray.get(
                [engine.continue_generation.remote(timeout=remaining()) for engine in engines], timeout=remaining()
            )
            # This mode is restricted to isolated synchronous RL, with no
            # concurrent external callers. Probe every member before returning.
            ray.get(
                [engine.disk_weight_generation_probe.remote(remaining()) for engine in engines], timeout=remaining()
            )
            return {
                "members": checked,
                "generation_pause_s": monotonic() - paused_at,
                "resume_and_probe_s": monotonic() - resume_started,
            }
        except Exception:
            # Also covers partial continue/probe success. A member that cannot
            # acknowledge pause must not continue serving a possibly bad version.
            for engine in engines:
                try:
                    ray.get(engine.pause_generation.remote(timeout=5), timeout=10)
                except Exception:
                    ray.kill(engine, no_restart=True)
            raise
        finally:
            ray.get(lock.release.remote(), timeout=10)
