# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
import os
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _receiver_agreement_worker(rank, rendezvous, directory, fault):
    from relax.backends.megatron.weight_update import update_weight_from_tensor as module

    for name in ("OVERLAP_WEIGHT_BACKUP",):
        os.environ["RELAX_COLOCATE_" + name] = "1" if name == "OVERLAP_WEIGHT_BACKUP" else "0"
    if fault in ("disabled", "serial") or (fault == "switch" and rank == 1):
        del os.environ["RELAX_COLOCATE_OVERLAP_WEIGHT_BACKUP"]
    dist.init_process_group(
        "gloo", init_method="file://" + rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=15)
    )
    try:
        module.get_gloo_group = lambda: dist.group.WORLD
        updater = object.__new__(module.UpdateWeightFromTensor)
        updater.actor_weight_retention_requested = fault != "disabled"
        updater.actor_weight_retention_fallback_reason = (
            "unsupported offloader" if fault == "requested" and rank == 1 else None
        )
        updater.actor_weight_retention_enabled = fault == "receiver" and rank == 0
        updater.actor_weight_retention_active = False
        updater.quantization_config = None
        updater.lora_enabled = False
        updater._route_experts = False
        updater.args = SimpleNamespace(rollout_num_gpus_per_engine=2)
        queries = []
        gathers = []
        original_gather = dist.all_gather_object

        def gather(output, local, **kwargs):
            gathers.append(local)
            return original_gather(output, local, **kwargs)

        module.dist.all_gather_object = gather

        from relax.backends.sglang.weight_loader import NATIVE_SYNC_FENCE_LOADER

        class NativeQueryReached(Exception):
            pass

        def query():
            queries.append(rank)
            if fault in ("receiver", "serial", "cold_reject"):
                return {"node_id": "local-node"}
            raise NativeQueryReached

        module.ray.get = lambda refs: refs
        module.ray.get_runtime_context = lambda: SimpleNamespace(get_node_id=lambda: "local-node")
        engine = SimpleNamespace(
            get_pid_and_node_id=SimpleNamespace(remote=query),
            get_expert_weight_sync_layout=SimpleNamespace(
                remote=lambda: dict(
                    device="cuda",
                    dtype="bfloat16",
                    tp_size=2,
                    custom_weight_loader=[NATIVE_SYNC_FENCE_LOADER] if fault == "serial" else [],
                )
            ),
        )

        def qualified():
            assert updater.actor_weight_retention_enabled == (fault == "serial")
            if fault == "serial":
                assert not module.Envs.RELAX_COLOCATE_OVERLAP_WEIGHT_BACKUP
            raise NativeQueryReached

        updater._ipc_placement = None
        updater._configure_sync_process_groups = qualified
        error = None
        try:
            updater.connect_rollout_engines([engine], None)
        except NativeQueryReached:
            assert fault in ("requested", "disabled", "serial", "cold_reject")
        except RuntimeError as exception:
            error = str(exception)
        if fault in ("requested", "disabled", "serial", "cold_reject"):
            assert (
                error is None and updater.actor_weight_retention_enabled == (fault == "serial") and len(queries) == 1
            )
            assert updater.actor_weight_retention_requested == (fault != "disabled")
        else:
            assert error is not None and len(queries) == int(fault == "receiver")
            assert ("lost its capability" if fault == "receiver" else "disagree") in error
        assert len(gathers) == 1 + int(
            fault in ("receiver", "serial", "cold_reject")
        )  # Node gather, plus opt-in receiver qualification.
        if fault == "disabled":
            assert gathers == ["local-node"]  # Native payload; no fusion capability RPCs.
        if fault == "receiver":
            from relax.backends.megatron.weight_update.utils import run_with_distributed_error_check

            def finish():
                if rank == 0:
                    raise RuntimeError("D2H completion failed")

            with pytest.raises(RuntimeError, match="D2H completion failed"):
                run_with_distributed_error_check(finish, dist.group.WORLD)
            # A later submission failure must drain every earlier RPC, even if
            # those RPCs also failed, before both ranks leave the resume phase.
            import ray

            refs, drained = [object(), object()], []

            def get_failed_ref(ref):
                drained.append(ref)
                raise RuntimeError("secondary receiver failure")

            updater.rollout_engines = [
                SimpleNamespace(continue_generation=SimpleNamespace(remote=lambda ref=ref: ref)) for ref in refs
            ] + [SimpleNamespace()]  # Missing method fails inside the rank-0 guard.
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(ray, "get", get_failed_ref)
                with pytest.raises(RuntimeError, match="continue_generation"):
                    updater.resume_generation_after_sync()
            assert drained == (refs if rank == 0 else [])
        # Both ranks must remain able to enter the same collective after rejection.
        marker = torch.ones(1, dtype=torch.int32)
        dist.all_reduce(marker, group=dist.group.WORLD)
        assert marker.item() == 2 and not torch.cuda.is_initialized()
        Path(directory, f"rank-{rank}.json").write_text(
            json.dumps({"fault": fault, "queries": len(queries), "error": error})
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("fault", ["disabled", "switch", "requested", "receiver", "serial", "cold_reject"])
def test_receiver_request_is_agreed_before_queries(tmp_path, fault):
    context = mp.spawn(
        _receiver_agreement_worker, args=(str(tmp_path / "rendezvous"), str(tmp_path), fault), nprocs=2, join=False
    )
    try:
        deadline = time.monotonic() + 45
        while not context.join(timeout=1):
            assert time.monotonic() < deadline, "Receiver qualification did not converge"
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
    records = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(2)]
    assert all(record["fault"] == fault for record in records)
