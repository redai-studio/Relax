# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Two CPU ranks verify that shared FSDP gradients reach the optimizer.

Real Gloo reductions catch replica divergence hidden by single-rank tests. Both
legacy and precision-aware gradients share one bounded process group.
"""

import importlib.util
import sys
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate


# Load the CPU-only patch without the backend package's GPU initialization.
_SPEC = importlib.util.spec_from_file_location(
    "fsdp_pp_patch_under_test",
    Path(__file__).resolve().parents[4] / "relax/backends/megatron/fsdp_pp_patch.py",
)
fsdp_pp_patch = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fsdp_pp_patch)


def _shared_gradient_worker(rank, rendezvous):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=20))
    try:
        group = dist.group.WORLD
        mesh = DeviceMesh("cpu", [0, 1])

        def wrapped(model, name, return_model_obj=False):
            return model if return_model_obj else getattr(model, name)

        # Only MCore's import surface is stubbed; tensors and collectives are real.
        fmg = SimpleNamespace(
            _flatten_dense_tensors=torch._utils._flatten_dense_tensors,
            _unflatten_dense_tensors=torch._utils._unflatten_dense_tensors,
            _get_main_grad_attr=lambda param: "grad",
            _unshard_if_dtensor=lambda tensor: tensor,
            _reshard_if_dtensor=lambda tensor, reference: tensor,
            get_attr_wrapped_model=wrapped,
            get_pg_size=lambda pg: dist.get_world_size(pg),
            is_pp_first_stage=lambda pg: dist.get_rank(pg) == 0,
            is_pp_last_stage=lambda pg: dist.get_rank(pg) == 1,
            _allreduce_embedding_grad=lambda *args, **kwargs: None,
        )
        sys.modules["megatron.core"] = SimpleNamespace(parallel_state=SimpleNamespace())
        sys.modules["megatron.core.distributed.finalize_model_grads"] = fmg
        source = Path(__file__).resolve().parents[4] / "relax/models/common/chunked_grad_coalesce.py"
        spec = importlib.util.spec_from_file_location("chunked_grad_coalesce", source)
        coalesce = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(coalesce)
        fsdp_pp_patch.patch_embedding_grad_reshard_for_pp()

        for gradient_field in ("grad", "decoupled_grad"):
            for reduction in ("tp", "pp"):
                param = torch.nn.Parameter(DTensor.from_local(torch.ones(4), mesh, [Replicate()]))
                # One replica has no local contribution, as with short SP inputs.
                local_grad = torch.full((4,), 2.0 if rank == 0 else 0.0)
                setattr(param, gradient_field, DTensor.from_local(local_grad, mesh, [Replicate()]))
                param.sequence_parallel = True
                model = SimpleNamespace(
                    ddp_config=SimpleNamespace(use_megatron_fsdp=True),
                    named_parameters=lambda: [("mtp.enorm.weight", param)],
                )
                config = SimpleNamespace(sequence_parallel=True, qk_layernorm=False, mtp_num_layers=1)
                if reduction == "tp":
                    coalesce._allreduce_non_tensor_model_parallel_grads([model], config, group)
                else:
                    fmg._allreduce_embedding_grad([model], group, group, lambda model: param, config=config)
                gradient = getattr(param, gradient_field)
                assert isinstance(gradient, DTensor)
                torch.testing.assert_close(gradient.to_local(), torch.full((4,), 2.0), rtol=0, atol=0)
                # The synchronized gradients must produce equal Adam moments.
                weight = torch.nn.Parameter(torch.ones(4))
                optimizer = torch.optim.AdamW([weight])
                weight.grad = gradient.to_local()
                optimizer.step()
                for moment in ("exp_avg", "exp_avg_sq"):
                    values = [torch.empty_like(weight) for _ in range(2)]
                    dist.all_gather(values, optimizer.state[weight][moment], group=group)
                    torch.testing.assert_close(values[0], values[1], rtol=0, atol=0)
                if gradient_field == "decoupled_grad":
                    assert param.grad is None
    finally:
        dist.destroy_process_group()


def test_fsdp_tp_and_pp_sync_legacy_and_decoupled_gradients(tmp_path):
    context = mp.start_processes(
        _shared_gradient_worker,
        args=(f"file://{tmp_path / 'rendezvous'}",),
        nprocs=2,
        join=False,
        start_method="spawn",
    )
    deadline = time.monotonic() + 30
    try:
        while time.monotonic() < deadline:
            if context.join(timeout=max(0, deadline - time.monotonic())):
                return
        raise TimeoutError("Shared-gradient regression exceeded 30 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
