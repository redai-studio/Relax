# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Synchronous FULL export at a completed training-step boundary."""

import hashlib
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import torch.distributed as dist

from relax.distributed.weight_sync import traffic
from relax.distributed.weight_sync.checkpoint import snapshot_tensors
from relax.distributed.weight_sync.metrics import SyncMetrics
from relax.distributed.weight_sync.source_io import agree_errors, agree_io_state, coordinated_io
from relax.distributed.weight_sync.store import VersionStore, canonical
from relax.utils.logging_utils import get_logger


if TYPE_CHECKING:
    from ..actor import MegatronTrainRayActor


logger = get_logger(__name__)


def _converter_revision() -> str:
    """Fingerprint the code that defines the Megatron-to-HF conversion path."""
    revision = hashlib.sha256()
    source_root = Path(__file__).resolve().parent.parent
    for directory in (source_root / "weight_conversion", source_root / "weight_update"):
        for path in sorted(directory.rglob("*.py")):
            revision.update(path.relative_to(source_root).as_posix().encode())
            revision.update(path.read_bytes())
    return revision.hexdigest()


def export_full_checkpoint(
    actor: "MegatronTrainRayActor",
    process_group: dist.ProcessGroup,
    *,
    store_path: str | Path,
    stream: str,
    version: int,
    scratch: str | Path,
    disk_budget: int,
    max_weight_bytes: int,
    update_id: str | None = None,
    metrics_dir: str | Path | None = None,
    io_timeout: float = 3600,
) -> dict:
    """All training ranks call this while optimizer updates are quiescent.

    The existing HF iterator gathers and converts the training layout. Rank
    zero persists logical tensors and commits a FULL package. Consumers can
    rebuild it after the training processes exit. LoRA adapter mode is rejected
    on all ranks before source conversion or publication because a standalone
    FULL checkpoint cannot omit the adapter's contribution.
    """
    rank = dist.get_rank(process_group)
    source_rank = dist.get_global_rank(process_group, 0)
    world = dist.get_world_size(process_group)
    if dist.get_backend(process_group) != "gloo":
        raise ValueError("FULL export requires the training Gloo control group")
    metrics = SyncMetrics(str(version) if update_id is None else str(update_id))
    metrics.source_participant = {"rank": rank, "world_size": world}

    def check(error):
        failures = agree_errors(error, process_group)
        if any(failures):
            raise RuntimeError(f"FULL export failed: {failures}")

    def io(operation, stage):
        return coordinated_io(
            operation if rank == 0 else None,
            lambda state: agree_io_state(state, process_group),
            stage=stage,
            timeout=io_timeout,
            committing=True,
        )

    temporary = None
    store = None
    table = {}
    try:
        with traffic.capture(metrics):
            named_params_and_buffers, HfWeightIteratorBase, error = None, None, None
            try:
                from relax.utils.megatron_peft_utils import is_lora_adapter_mode, is_lora_enabled

                if is_lora_enabled(actor.args) and is_lora_adapter_mode(actor.args):
                    raise ValueError("FULL checkpoint export does not support LoRA adapter mode")

                from .common import named_params_and_buffers
                from .hf_weight_iterator_base import HfWeightIteratorBase
            except Exception as caught:
                error = f"{type(caught).__name__}: {caught}"
            check(error)

            model_config, error = None, None
            try:
                model_config = actor.hf_config.to_dict()
                model_config_id = hashlib.sha256(canonical(model_config)).hexdigest()
            except Exception as caught:
                error = f"{type(caught).__name__}: {caught}"
            check(error)

            identity, error = None, None
            try:
                model_name = actor.args.model_name or type(actor.hf_config).__name__.lower()
                identity = (
                    str(store_path),
                    stream,
                    version,
                    update_id,
                    disk_budget,
                    max_weight_bytes,
                    io_timeout,
                    actor.args.megatron_to_hf_mode,
                    model_name,
                    model_config_id,
                )
            except Exception as caught:
                error = f"{type(caught).__name__}: {caught}"
            check(error)
            identities = [None] * world
            traffic.all_gather_object(identities, identity, group=process_group)
            if any(item != identity for item in identities):
                raise ValueError("Training ranks requested different FULL exports")

            def setup():
                nonlocal temporary, store
                Path(scratch).mkdir(parents=True, exist_ok=True)
                temporary = tempfile.TemporaryDirectory(prefix="full-export-", dir=scratch)
                store = VersionStore(store_path, disk_budget=disk_budget, max_weight_bytes=max_weight_bytes)

            io(setup, "FULL export setup")
            weights, error = None, None
            try:
                weights = dict(
                    named_params_and_buffers(
                        actor.args,
                        actor.model,
                        convert_to_global_name=actor.args.megatron_to_hf_mode == "raw",
                        include_persistent_buffers=actor.args.megatron_to_hf_mode == "bridge",
                    )
                )
            except Exception as caught:
                error = f"{type(caught).__name__}: {caught}"
            check(error)
            # Both iterator constructors may perform rank-local setup before
            # their next collective; they must share the same control agreement.
            # Verify that every rank will execute the same converter code before
            # any iterator setup or conversion can enter model-parallel collectives.
            revision_id, error = None, None
            try:
                revision_id = _converter_revision()
            except Exception as caught:
                error = f"{type(caught).__name__}: {caught}"
            check(error)
            revisions = [None] * world
            traffic.all_gather_object(revisions, revision_id, group=process_group)
            if any(item != revision_id for item in revisions):
                raise ValueError("Training ranks are running different weight converters")

            extra = {"publication_agreement": check}
            iterator, error = None, None
            try:
                iterator = HfWeightIteratorBase.create(
                    actor.args,
                    actor.model,
                    model_name=model_name,
                    quantization_config=None,
                    **extra,
                )
            except Exception as caught:
                error = f"{type(caught).__name__}: {caught}"
            # Iterator construction inspects rank-local model/config state.
            # Agree before any rank enters the iterator's collectives.
            check(error)
            with metrics.stage("source_conversion", path=scratch):
                for bucket in iterator.get_hf_weight_chunks_checked(weights, check):
                    host, error = None, None
                    try:
                        if rank == 0:
                            host = [(name, tensor.detach().contiguous().cpu()) for name, tensor in bucket]
                    except Exception as caught:
                        error = f"{type(caught).__name__}: {caught}"
                    check(error)

                    def persist():
                        names = [name for name, _ in host]
                        if len(names) != len(set(names)) or set(table).intersection(names):
                            raise ValueError("Converter produced duplicate logical tensors")
                        table.update(snapshot_tensors(host, temporary.name))

                    io(persist, "FULL tensor export")

            def publish():
                manifest = store.publish(
                    stream,
                    version,
                    table,
                    full_only=True,
                    model_config=model_config,
                    update_id=update_id,
                    converter_revision=f"{type(iterator).__name__}:{revision_id}",
                )
                return {key: manifest[key] for key in ("stream", "version", "root", "artifact_id", "update_id")}

            with metrics.stage("full_package", path=store_path):
                receipt = io(publish, "FULL package commit")
            receipts = [receipt]
            traffic.broadcast_object_list(receipts, src=source_rank, group=process_group)
            return receipts[0]
    finally:
        if temporary is not None:
            try:
                temporary.cleanup()
            except OSError:
                logger.warning("FULL export scratch cleanup pending: %s", temporary.name, exc_info=True)
        if metrics_dir is not None:
            try:
                metrics.save(Path(metrics_dir) / f"full-{version}-rank-{rank}.json")
            except OSError:
                logger.warning("FULL export metrics could not be saved", exc_info=True)
