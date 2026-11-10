# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Separate, opt-in audit after measured memory updates; never a benchmark arm.

Capture every delivered GPU model before independently exporting the frozen
training model. A later ordinary SGLang FULL load supplies the comparison.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from time import monotonic, sleep
from typing import Any


def file_sha256(path: Path) -> str:
    """Bounded-memory hashing, including Python 3.10 benchmark environments."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_engine(engine: Any, destination: str, expected_update: str) -> dict:
    import ray
    import requests
    import torch
    from sglang.srt.utils import MultiprocessingSerializer

    from relax.backends.sglang.disk_weight_dump import DUMP_LOADER
    from relax.distributed.weight_sync.provenance import runtime_identity

    url = f"http://{engine.server_host}:{engine.server_port}/model_info"

    def info() -> dict:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        value = response.json()
        if not expected_update or value.get("weight_version") != expected_update:
            raise RuntimeError(f"Unexpected live SGLang version: {value}")
        return value

    before = info()
    path = Path(destination) / f"engine-{engine.rank}"
    path.mkdir(parents=True, exist_ok=False)
    control = json.dumps(dict(directory=str(path), packed=True))
    serialized = MultiprocessingSerializer.serialize([(control, torch.empty(0))], output_str=True)
    # The ordinary memory session has already ended. Do not start/end a new
    # session: those APIs can restore/postprocess model storage before a dump.
    response = engine._make_request(
        "update_weights_from_tensor",
        dict(
            serialized_named_tensors=[serialized],
            load_format=DUMP_LOADER,
            flush_cache=False,
            selector="target",
            abort_all_requests=False,
            weight_version=expected_update,
        ),
        timeout=240,
    )
    if not isinstance(response, dict) or response.get("success") is not True:
        raise RuntimeError(f"Read-only model dump failed: {response}")
    after = info()
    return dict(
        directory=str(path / "rank-0"),
        engine_rank=engine.rank,
        actor_id=str(ray.get_runtime_context().get_actor_id()),
        node_id=ray.get_runtime_context().get_node_id(),
        before=before,
        after=after,
        runtime=runtime_identity(),
    )


def install(args: Any) -> None:
    import ray
    import torch.distributed as dist

    from relax.backends.megatron.model import save_hf_model
    from relax.backends.megatron.weight_update.update_weight_from_memory import UpdateWeightFromMemory
    from relax.distributed.weight_sync.provenance import runtime_identity
    from relax.distributed.weight_sync.store import atomic_write
    from relax.utils.distributed_utils import get_gloo_group
    from task_delta_weight_sync.measure_sync import install as install_measurement

    if not getattr(args, "memory_weight_sync_config", None):
        raise ValueError("GPU audit requires the TCP memory updater")
    install_measurement(args)
    cls = UpdateWeightFromMemory
    if getattr(cls, "_dws_audited", False):
        return
    group = get_gloo_group()
    rank = dist.get_rank(group)

    def errors(local: str | None) -> list:
        values = [None] * dist.get_world_size(group)
        dist.all_gather_object(values, local, group=group)
        return values

    original_init = cls.__init__

    def audited_init(self: Any, actual_args: Any, model: Any, *pos: Any, **kw: Any) -> None:
        original_init(self, actual_args, model, *pos, **kw)
        self._audit_args, self._audit_model = actual_args, model

    original_update = cls.update_weights

    def quarantine(self: Any, manager: Any, error: str) -> dict:
        self.failed = True
        closure_errors = []
        try:
            closure_errors.extend(self._close_generation(manager))
        except Exception as exc:
            closure_errors.append(f"Cannot quarantine inference group: {type(exc).__name__}: {exc}")
        if self.source is not None:
            try:
                self.source.close()
            except Exception as exc:
                closure_errors.append(f"Cannot close memory source: {type(exc).__name__}: {exc}")
            self.source = None
        return dict(error=error, closure_errors=closure_errors)

    def audited_update(self: Any, manager: Any) -> Any:
        update_error, value = None, None
        try:
            value = original_update(self, manager)
        except Exception as exc:
            update_error = f"{type(exc).__name__}: {exc}"
        update_errors = errors(update_error)
        if any(update_errors):
            failure = [quarantine(self, manager, f"Measured update failed: {update_errors}") if rank == 0 else None]
            dist.broadcast_object_list(failure, src=0, group=group)
            self.failed = True
            raise RuntimeError(f"GPU memory audit update failed: {failure[0]}")
        # Both sync timers and readiness probes are complete. The optimizer
        # cannot advance while any rank remains in this synchronous hook.
        started = monotonic()
        local_error, export_args, root = None, None, None
        try:
            root = Path(os.environ["TENSORBOARD_DIR"]).resolve().parents[1] / "gpu-audit"
            run = Path(self._audit_args.memory_weight_sync_config).resolve().parent
            if root.parent != run:
                raise ValueError("Audit and memory config must belong to the same run directory")
            export_args = copy.copy(self._audit_args)
            export_args.save_hf = str(root / "hf" / "update-{rollout_id:06d}")
            export_args.save_hf_post_hook_path = None
            export_args.save_hf_dtype = "bf16"
            identity = runtime_identity()
            atomic_write(root / "ranks" / f"{rank}.json", json.dumps(identity, indent=2).encode())
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        preflight = errors(local_error)
        lock, acquired, engines = None, False, ()
        report = [None]
        if rank == 0:
            try:
                if any(preflight):
                    raise RuntimeError(f"Audit preflight failed: {preflight}")
                version = self.encoder.version
                receipt_path = root / "receipts" / f"{version:06d}.json"
                if (root / "hf" / f"update-{version:06d}").exists():
                    raise RuntimeError("Independent HF export directory already exists")
                if receipt_path.exists():
                    raise RuntimeError("Audit receipts must never be overwritten")
                engines, lock, *_ = ray.get(manager.get_rollout_engines_and_lock.remote(), timeout=30)
                engines = tuple(engines)
                if {e._actor_id.hex() for e in engines} != self.members:
                    raise RuntimeError("Audit inference membership changed")
                deadline = monotonic() + 30
                while not ray.get(lock.acquire.remote(), timeout=10):
                    if monotonic() >= deadline:
                        raise TimeoutError("Audit group lock timed out")
                    sleep(0.05)
                acquired = True
                ray.get([e.pause_generation.remote(timeout=30) for e in engines], timeout=40)
                updates = {r["update"] for r in self._dws_details["receivers"]}
                if len(updates) != 1:
                    raise RuntimeError("Inference members disagree about their committed update")
                update = updates.pop()
                captured = ray.get(
                    [
                        e.__ray_call__.remote(snapshot_engine, str(root / "delivered" / f"{version:06d}"), update)
                        for e in engines
                    ],
                    timeout=270,
                )
                report[0] = dict(version=version, update=update, captured=captured, dump_s=monotonic() - started)
            except Exception as exc:
                report[0] = dict(error=f"{type(exc).__name__}: {exc}")
        dist.broadcast_object_list(report, src=0, group=group)
        local_error = None
        if "error" not in report[0]:
            try:
                # A separate exporter reads the still-frozen LIVE model; never
                # use the delta encoder's retained baseline as the reference.
                save_hf_model(export_args, report[0]["version"], self._audit_model, force_sync=True)
            except Exception as exc:
                local_error = f"{type(exc).__name__}: {exc}"
        export_errors = errors(local_error)
        if rank == 0:
            try:
                if "error" in report[0]:
                    raise RuntimeError(report[0]["error"])
                if any(export_errors):
                    raise RuntimeError(f"Independent HF export failed: {export_errors}")
                export = root / "hf" / f"update-{report[0]['version']:06d}"
                files = {}
                for path in sorted(export.rglob("*")):
                    if path.is_file():
                        files[str(path.relative_to(export))] = dict(
                            bytes=path.stat().st_size, sha256=file_sha256(path)
                        )
                if not any(name.endswith(".safetensors") for name in files):
                    raise RuntimeError("Independent HF export has no weight shards")
                receipt = report[0] | dict(
                    status="CAPTURED_NOT_COMPARED",
                    stream=self.config.stream,
                    schema=self.encoder.schema,
                    source_runtime=identity,
                    independent_hf=str(export),
                    hf_files=files,
                    audit_s=monotonic() - started,
                    timing_scope="audit only; after frozen sync measurements",
                )
                atomic_write(receipt_path, json.dumps(receipt, indent=2).encode())
                ray.get([e.continue_generation.remote(timeout=30) for e in engines], timeout=40)
                ray.get(lock.release.remote(), timeout=10)
                acquired = False
            except Exception as exc:
                report[0] = quarantine(self, manager, f"{type(exc).__name__}: {exc}")
            finally:
                if acquired:
                    try:
                        ray.get(lock.release.remote(), timeout=10)
                    except Exception:
                        pass
        dist.broadcast_object_list(report, src=0, group=group)
        if "error" in report[0]:
            self.failed = True
            raise RuntimeError(f"GPU memory audit failed: {report[0]}")
        return value

    cls.__init__ = audited_init
    cls.update_weights = audited_update
    cls._dws_audited = True
