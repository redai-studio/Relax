# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Opt-in measurement through the existing custom Megatron init hook.

Both arms keep the ordinary updater. The NCCL arm receives the same explicit
one-token readiness probe as the memory arm, outside its update-return timer.
No parameter values are copied or changed by this hook.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
from time import monotonic
from typing import Any


def install(args: Any) -> None:
    import torch.distributed as dist

    from relax.distributed.weight_sync.provenance import runtime_identity
    from relax.distributed.weight_sync.store import atomic_write
    from relax.utils.distributed_utils import get_gloo_group
    from relax.utils.logging_utils import get_logger

    if getattr(args, "disk_weight_sync_config", None):
        raise ValueError("Measure the ordinary NCCL or TCP memory path")
    memory = bool(getattr(args, "memory_weight_sync_config", None))
    if memory:
        from relax.backends.megatron.weight_update.update_weight_from_memory import UpdateWeightFromMemory

        cls = UpdateWeightFromMemory
    else:
        from relax.backends.megatron.weight_update.update_weight_from_distributed import UpdateWeightFromDistributed

        cls = UpdateWeightFromDistributed
    if getattr(cls, "_dws_measured", False):
        return
    group = get_gloo_group()
    rank = dist.get_rank(group)
    logger = get_logger(__name__)
    error = None
    try:
        root = Path(os.environ["TENSORBOARD_DIR"]).resolve().parents[1] / "sync-measurements"
        identity = runtime_identity()
        source = Path(inspect.getfile(cls)).resolve()
        identity.update(updater_file=str(source), updater_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        atomic_write(root / "ranks" / f"{rank}.json", json.dumps(identity, indent=2).encode())
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    errors = [None] * dist.get_world_size(group)
    dist.all_gather_object(errors, error, group=group)
    if any(errors):
        raise RuntimeError(f"Measurement initialization failed: {errors}")

    if memory:
        original_install = cls._install

        def measured_install(self: Any, manager: Any, buckets: list, started: float) -> dict:
            capture_s = monotonic() - started
            value = original_install(self, manager, buckets, started)
            self._dws_details = value | {
                "capture_encode_s": capture_s,
                "version": self.encoder.version,
                "elements": sum(b.elements for b in buckets),
                "changed": sum(b.changed for b in buckets),
                "payload_bytes": sum(len(b.payload) for b in buckets),
                "members": len(self.members),
                "full": all(b.encoding == "full" for b in buckets),
            }
            return value

        cls._install = measured_install
    else:
        original_bucket = cls._update_bucket_weights_from_distributed

        def measured_bucket(self: Any, named: list, *call_args: Any, **call_kwargs: Any) -> Any:
            # The ordinary method clears this list after broadcasting it.
            self._dws_payload += sum(t.numel() * t.element_size() for _, t in named)
            self._dws_elements += sum(t.numel() for _, t in named)
            return original_bucket(self, named, *call_args, **call_kwargs)

        cls._update_bucket_weights_from_distributed = measured_bucket

    original_update = cls.update_weights

    def measured_update(self: Any, *call_args: Any, **call_kwargs: Any) -> Any:
        import ray

        self._dws_payload = self._dws_elements = 0
        started = monotonic()
        value = original_update(self, *call_args, **call_kwargs)
        update_returned = monotonic()
        report = [None]
        if rank == 0:
            try:
                if memory:
                    metrics = dict(self._dws_details)
                    metrics["arm"] = "memory"
                    metrics["ready_boundary"] = "all group continue calls returned; before the final result broadcast"
                    metrics["payload_scope"] = "encoded bucket bytes per replica, including sparse positions"
                else:
                    self._dws_version = getattr(self, "_dws_version", -1) + 1
                    engines = self.rollout_engines
                    probe_started = monotonic()
                    ray.get([e.disk_weight_generation_probe.remote(120) for e in engines], timeout=150)
                    metrics = {
                        "arm": "nccl",
                        "version": self._dws_version,
                        "sync_to_ready_s": update_returned - started,
                        "ready_boundary": "ordinary updater return including its final training barrier",
                        "probe_s": monotonic() - probe_started,
                        "members": len(engines),
                        "elements": self._dws_elements,
                        "payload_bytes": self._dws_payload,
                        "full": True,
                        "payload_scope": "broadcast tensor bytes per replica",
                    }
                metrics["update_return_s"] = update_returned - started
                metrics["sync_with_probe_s"] = monotonic() - started
                metrics["payload_excludes"] = "protocol headers; not aggregate replica traffic or measured NIC bytes"
                metrics["initialization"] = metrics["version"] == 0
                atomic_write(root / f"{metrics['version']:06d}.json", json.dumps(metrics, indent=2).encode())
                logger.info("Weight sync measurement: %s", metrics)
                report[0] = {"success": True}
            except Exception as exc:
                report[0] = {"error": f"{type(exc).__name__}: {exc}"}
        dist.broadcast_object_list(report, src=0, group=group)
        if "error" in report[0]:
            raise RuntimeError(f"Weight sync measurement failed: {report[0]['error']}")
        return value

    cls.update_weights = measured_update
    cls._dws_measured = True
