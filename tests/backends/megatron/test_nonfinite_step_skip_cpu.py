# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Exercise the trainer's actual update/cleanup block with CPU Adam and Gloo.

Extracting the block avoids importing GPU-only Megatron/TE dependencies. This
covers trainer state transitions, not CUDA kernels or distributed param gather.
"""

import ast
import copy
import importlib.util
import math
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from relax.utils.training.nonfinite_grad import configure_nonfinite_guard, install_nonfinite_guard


ROOT = Path(__file__).resolve().parents[3]


def guard_module():
    spec = importlib.util.spec_from_file_location("norm_guard_test", ROOT / "relax/models/qwen3_5/grad_norm.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def update_block(guard):
    tree = ast.parse((ROOT / "relax/backends/megatron/model.py").read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train_one_step")
    begin = next(
        i
        for i, n in enumerate(function.body)
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "valid_step" for t in n.targets)
    )
    end = next(
        i
        for i, n in enumerate(function.body[begin:], begin)
        if isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call)
        and ast.unparse(n.value.func) == "optimizer.zero_grad"
    )
    code = compile(ast.Module(body=function.body[begin : end + 1], type_ignores=[]), "trainer-update-block", "exec")

    def run(optimizer, scheduler, chunks):
        errors = []
        context = dict(
            optimizer=optimizer,
            opt_param_scheduler=scheduler,
            model=chunks,
            args=SimpleNamespace(check_for_nan_in_loss_and_grad=False),
            losses_reduced=[],
            _is_global_zero_token_step=lambda _: False,
            step_id=128,
            rollout_id=128,
            step_global_batch_size=1600,
            critic_value_head_snapshot=None,
            maybe_verify_critic_value_head_movement=lambda *a: None,
            NonFiniteGradientNormError=guard.NonFiniteGradientNormError,
            torch=torch,
            math=math,
            logger=SimpleNamespace(warning=lambda *a: None, error=lambda *a, **kw: errors.append((a, kw))),
            logged_errors=errors,
        )
        exec(code, context)
        return context

    return run


class Optimizer:
    _relax_nonfinite_guard_installed = True

    def __init__(self, guard, distributed=False):
        self.param = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        self.adam = torch.optim.AdamW([self.param], lr=0.01, weight_decay=0.1)
        self.guard = guard
        self.distributed = distributed
        self.error = None
        self.clears = 0
        self.chained_optimizers = [SimpleNamespace(config=SimpleNamespace(), get_parameters=lambda: [self.param])]

    def get_grad_norm(self):
        return self.param.grad.square().sum().sqrt()

    def step(self):
        if self.error:
            raise self.error
        norm = self.guard.finite_norm(self.get_grad_norm())
        self.adam.step()
        return True, norm, None

    def zero_grad(self):
        self.clears += 1
        self.adam.zero_grad(set_to_none=True)


class Chunk:
    def __init__(self):
        self.buffer = torch.zeros(2)
        self.clears = 0

    def zero_grad_buffer(self):
        self.clears += 1
        self.buffer.zero_()


def install_cpu_guard(optimizer):
    """Real installer and real Gloo; stub only the GPU-heavy import
    surfaces."""
    import sys

    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(
            sys.modules,
            "megatron.core.optimizer.optimizer",
            SimpleNamespace(
                ChainedOptimizer=Optimizer,
                _get_param_grad_norm_group=lambda p: None,
                _is_separate_grad_norm_group=lambda group: False,
            ),
        )
        patch.setitem(sys.modules, "relax.utils.device", SimpleNamespace(make_current_torch_device=lambda: "cpu"))
        patch.setitem(
            sys.modules,
            "relax.utils.logging_utils",
            SimpleNamespace(get_logger=lambda name: SimpleNamespace(warning=lambda *a: None)),
        )
        model = [SimpleNamespace(ddp_config=SimpleNamespace(check_for_nan_in_grad=False))]
        install_nonfinite_guard(optimizer, model, process_group=torch.distributed.group.WORLD)
        # A checkpoint load must not cause the wrapper to be installed twice.
        wrapped = optimizer.get_grad_norm
        install_nonfinite_guard(optimizer, model, process_group=torch.distributed.group.WORLD)
        assert optimizer.get_grad_norm is wrapped


def exercise_skip(distributed=False, rank=0, bad=float("nan")):
    guard = guard_module()
    run = update_block(guard)
    opt = Optimizer(guard, distributed)
    if distributed:
        opt._relax_nonfinite_guard_installed = False
        # No pre-existing WORLD norm: rank 1's local norm stays finite while
        # rank 0's norm becomes NaN. The installed guard alone synchronizes it.
        install_cpu_guard(opt)
    chunk = Chunk()
    scheduler_calls = []
    scheduler = SimpleNamespace(step=lambda increment: scheduler_calls.append(increment))
    opt.param.grad = torch.tensor([0.3, -0.2])
    run(opt, scheduler, [chunk])
    before_param = opt.param.detach().clone()
    before_state = copy.deepcopy(opt.adam.state_dict())
    opt.param.grad = torch.tensor([bad if rank == 0 else 0.4, 0.5])
    chunk.buffer.fill_(float("nan"))
    result = run(opt, scheduler, [chunk])
    assert result["skipped_nonfinite_grad"] and not result["update_successful"]
    assert math.isnan(result["grad_norm"])
    assert opt._relax_nonfinite_skip_count == 1
    assert len(result["logged_errors"]) == 1
    log_args, log_kwargs = result["logged_errors"][0]
    assert "NONFINITE_GRAD_STEP_SKIPPED" in log_args[0]
    assert "PARAMETERS/ADAM/LR/WD NOT UPDATED" in log_args[0]
    assert "TRAINING CONTINUES" in log_args[0]
    assert log_kwargs["exc_info"] is True
    assert scheduler_calls == [1600]
    torch.testing.assert_close(opt.param, before_param, rtol=0, atol=0)
    for key, value in before_state["state"][0].items():
        torch.testing.assert_close(opt.adam.state_dict()["state"][0][key], value, rtol=0, atol=0)
    assert opt.param.grad is None and chunk.buffer.count_nonzero() == 0
    assert opt.clears == chunk.clears == 2
    # A reference Adam sees just the two healthy gradients; skip must be invisible
    # to parameters, momentum, variance, Adam step and LR/WD schedule.
    reference = Optimizer(guard)
    for grad in ([0.3, -0.2], [-0.1, 0.4]):
        reference.param.grad = torch.tensor(grad)
        reference.step()
        reference.zero_grad()
    opt.param.grad = torch.tensor([-0.1, 0.4])
    result = run(opt, scheduler, [chunk])
    assert result["update_successful"] and not result["skipped_nonfinite_grad"]
    assert scheduler_calls == [1600, 1600]
    torch.testing.assert_close(opt.param, reference.param, rtol=0, atol=0)
    for key, value in reference.adam.state_dict()["state"][0].items():
        torch.testing.assert_close(opt.adam.state_dict()["state"][0][key], value, rtol=0, atol=0)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_skip_preserves_adam_and_next_update(bad):
    exercise_skip(bad=bad)


def test_skip_requires_guard_and_opt_in():
    guard = guard_module()
    opt = Optimizer(guard)
    opt._relax_nonfinite_guard_installed = False
    opt.param.grad = torch.tensor([float("nan"), 0.0])
    with pytest.raises(guard.NonFiniteGradientNormError):
        update_block(guard)(opt, None, [Chunk()])


@pytest.mark.parametrize("error", [FloatingPointError("unrelated"), RuntimeError("CUDA or NCCL failure")])
def test_unrelated_errors_propagate(error):
    guard = guard_module()
    opt = Optimizer(guard)
    opt.error = error
    with pytest.raises(type(error), match=str(error)):
        update_block(guard)(opt, None, [Chunk()])


def distributed_worker(rank, init_method):
    torch.distributed.init_process_group(
        "gloo", init_method=init_method, rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:
        exercise_skip(distributed=True, rank=rank)
    finally:
        torch.distributed.destroy_process_group()


def test_one_bad_rank_causes_all_ranks_to_skip_and_continue(tmp_path):
    torch.multiprocessing.spawn(distributed_worker, args=(f"file://{tmp_path / 'init'}",), nprocs=2, join=True)


def test_generic_guard_defaults_off_without_changing_bucket_checks(monkeypatch):
    monkeypatch.delenv("RELAX_SKIP_NONFINITE_GRAD", raising=False)
    monkeypatch.delenv("RELAX_QWEN35_SKIP_NONFINITE_GRAD", raising=False)
    args = SimpleNamespace(check_for_nan_in_loss_and_grad=True)
    assert configure_nonfinite_guard(args) is False
    assert args.check_for_nan_in_loss_and_grad is True


@pytest.mark.parametrize("tp,pp,cp,ep", [(1, 1, 4, 8), (4, 1, 1, 8), (2, 4, 1, 8), (2, 4, 4, 8)])
def test_generic_guard_configuration_does_not_restrict_topology(monkeypatch, tp, pp, cp, ep):
    monkeypatch.setenv("RELAX_SKIP_NONFINITE_GRAD", "1")
    args = SimpleNamespace(
        bf16=True,
        fp16=False,
        check_for_nan_in_loss_and_grad=True,
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        context_parallel_size=cp,
        expert_model_parallel_size=ep,
    )
    assert configure_nonfinite_guard(args) is True
    assert args.check_for_nan_in_loss_and_grad is False


@pytest.mark.parametrize("option", ["fp8", "use_megatron_fsdp", "chunked_optimizer_state_offload"])
def test_unvalidated_storage_modes_fail_explicitly(monkeypatch, option):
    monkeypatch.setenv("RELAX_SKIP_NONFINITE_GRAD", "1")
    args = SimpleNamespace(bf16=True, fp16=False, check_for_nan_in_loss_and_grad=True)
    setattr(args, option, True)
    with pytest.raises(ValueError, match=option):
        configure_nonfinite_guard(args)


def test_real_mcore_chain_guard_survives_checkpoint_and_skips_all_children(tmp_path):
    """Actual MCore step/get_norm/load_state_dict, using CPU Adam children.

    Different child norm groups exercise MCore's composite-norm path. This does
    not emulate CUDA DistributedOptimizer/HDO or claim their runtime
    validation.
    """
    mcore = pytest.importorskip("megatron.core.optimizer.optimizer")
    config = SimpleNamespace(
        clip_grad=0.0,
        grad_norm_skip_threshold=float("inf"),
        log_num_zeros_in_grad=False,
        reuse_grad_buf_for_mxfp8_param_ag=False,
        use_precision_aware_optimizer=False,
        use_precision_aware_optimizer_no_fp8_or_ds_fp8=False,
        overlap_param_gather_with_optimizer_step=False,
    )

    class Child:
        def __init__(self):
            self.config = config
            self.param = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
            self.optimizer = torch.optim.AdamW([self.param], lr=0.01)
            self.group = object()

        def prepare_grads(self):
            return False

        def get_grad_norm(self):
            return self.param.grad.norm().item()

        def get_grad_stats_parallel_group(self):
            return self.group

        def get_parameters(self):
            return [self.param]

        def step_with_ready_grads(self):
            self.optimizer.step()
            return True

        def zero_grad(self, set_to_none=True):
            self.optimizer.zero_grad(set_to_none=set_to_none)

        def state_dict(self):
            return self.optimizer.state_dict()

        def load_state_dict(self, state):
            self.optimizer.load_state_dict(state)

    torch.distributed.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'mcore-init'}", rank=0, world_size=1, timeout=timedelta(seconds=30)
    )
    try:
        children = [Child(), Child()]
        chain = mcore.ChainedOptimizer(children)
        model = [SimpleNamespace(ddp_config=SimpleNamespace(check_for_nan_in_grad=False))]
        install_nonfinite_guard(chain, model, process_group=torch.distributed.group.WORLD)
        method = chain.get_grad_norm
        for child in children:
            child.param.grad = torch.tensor([0.1, -0.2])
        assert chain.step()[0]
        state = copy.deepcopy(chain.state_dict())
        chain.load_state_dict(state)
        install_nonfinite_guard(chain, model, process_group=torch.distributed.group.WORLD)
        assert chain.get_grad_norm is method
        before = [child.param.detach().clone() for child in children]
        children[1].param.grad[0] = float("nan")
        # A bad second child must prevent the first child from updating too.
        with pytest.raises(guard_module().NonFiniteGradientNormError):
            chain.step()
        for i, child in enumerate(children):
            torch.testing.assert_close(child.param, before[i], rtol=0, atol=0)
            for key, value in state[i]["state"][0].items():
                torch.testing.assert_close(child.state_dict()["state"][0][key], value, rtol=0, atol=0)
        chain.zero_grad()
        for child in children:
            assert child.param.grad is None
            child.param.grad = torch.tensor([0.3, 0.4])
        assert chain.step()[0]
        for child in children:
            assert child.state_dict()["state"][0]["step"] == 2
    finally:
        torch.distributed.destroy_process_group()
