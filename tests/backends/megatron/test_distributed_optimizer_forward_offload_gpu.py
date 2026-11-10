# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""World-one NCCL regression for forward offload through real
DistributedOptimizer.

Run this file in a dedicated pytest process. It checks real DDP gradient
buffers, optimizer shards, prepare_grads, clipping and parameter gather; it
does not claim coverage of multi-rank reductions, QSA, MoE routing, or long-
sequence kernels.
"""

from argparse import Namespace
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist


@pytest.mark.skipif(not torch.cuda.is_available(), reason="DistributedOptimizer and TE Adam require CUDA")
@pytest.mark.parametrize("overlap", [False, True], ids=["synchronous_gather", "overlapped_gather"])
def test_distributed_optimizer_forward_offload_matches_resident_master(
    monkeypatch: pytest.MonkeyPatch, tmp_path, record_property, overlap: bool
) -> None:
    pytest.importorskip("megatron.core", exc_type=ImportError)
    from megatron.core import parallel_state
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.distributed import param_and_grad_buffer as buffer_module
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
    from megatron.core.tensor_parallel.layers import set_defaults_if_not_set_tensor_model_parallel_attributes
    from megatron.core.transformer.module import MegatronModule
    from megatron.core.transformer.transformer_config import TransformerConfig
    from transformer_engine.pytorch.optimizers import FusedAdam

    from relax.backends.megatron import model as module

    assert not dist.is_initialized(), "Run this regression in a dedicated pytest process"
    torch.cuda.set_device(0)
    dist.init_process_group(
        "nccl",
        init_method=(tmp_path / "rendezvous").as_uri(),
        rank=0,
        world_size=1,
        timeout=timedelta(seconds=90),
    )
    try:
        parallel_state.initialize_model_parallel(create_gloo_process_groups=False)
        config = TransformerConfig(
            num_layers=1, hidden_size=32, num_attention_heads=4, bf16=True, params_dtype=torch.bfloat16
        )

        class TinyMLP(MegatronModule):
            def __init__(self):
                super().__init__(config)
                self.up = torch.nn.Linear(32, 64)
                self.down = torch.nn.Linear(64, 32)

            def forward(self, inputs):
                return self.down(torch.nn.functional.gelu(self.up(inputs)))

        torch.manual_seed(137)
        source = TinyMLP().to(device="cuda", dtype=torch.bfloat16)
        model_bytes = sum(parameter.numel() * parameter.element_size() for parameter in source.parameters())
        assert model_bytes < 1024**2
        models, optimizers = [], []
        for _ in range(2):
            model = TinyMLP().to(device="cuda", dtype=torch.bfloat16)
            model.load_state_dict(source.state_dict())
            for parameter in model.parameters():
                set_defaults_if_not_set_tensor_model_parallel_attributes(parameter)
            ddp = DistributedDataParallel(
                config,
                DistributedDataParallelConfig(
                    use_distributed_optimizer=True,
                    grad_reduce_in_fp32=True,
                    overlap_grad_reduce=overlap,
                    overlap_param_gather=overlap,
                    bucket_size=1024,
                ),
                model,
            )
            optimizer = get_megatron_optimizer(
                OptimizerConfig(
                    optimizer="adam",
                    lr=0.003,
                    min_lr=0.0001,
                    weight_decay=0.1,
                    adam_beta1=0.9,
                    adam_beta2=0.98,
                    adam_eps=1e-8,
                    clip_grad=0.05,
                    bf16=True,
                    params_dtype=torch.bfloat16,
                    use_distributed_optimizer=True,
                    use_precision_aware_optimizer=True,
                    store_param_remainders=False,
                    main_params_dtype=torch.float32,
                    main_grads_dtype=torch.float32,
                    exp_avg_dtype=torch.float32,
                    exp_avg_sq_dtype=torch.float32,
                    chunked_optimizer_state_offload=True,
                    optimizer_state_offload_fraction=1.0,
                    optimizer_state_offload_chunk_size_mb=0,
                ),
                [ddp],
                use_gloo_process_groups=False,
            )
            assert len(optimizer.chained_optimizers) == 1
            child = optimizer.chained_optimizers[0]
            assert isinstance(child, DistributedOptimizer)
            assert isinstance(child.optimizer, FusedAdam)
            assert child.optimizer.master_weights and child._optimizer_state_offloader is not None
            models.append(ddp)
            optimizers.append(optimizer)

        # Both paths retain chunked state execution. Only the baseline omits the
        # newly connected optimizer-to-forward master offload lifecycle call.
        monkeypatch.setattr(optimizers[0], "offload_optimizer_state_for_forward", Mock())
        args = Namespace(
            empty_unused_memory_level=0,
            custom_megatron_before_train_step_hook_path=None,
            ci_test=False,
            seq_length=8,
            micro_batch_size=1,
            decoder_seq_length=None,
            reuse_grad_buf_for_mxfp8_param_ag=False,
            overlap_param_gather=overlap,
        )
        monkeypatch.setattr(module, "get_args", lambda: args)
        monkeypatch.setattr(module.capture_hooks, "begin_step_for", Mock())
        monkeypatch.setattr(module.capture_hooks, "end_step_for", Mock())
        monkeypatch.setattr(module, "maybe_verify_critic_value_head_movement", Mock())
        monkeypatch.setattr(module.mpu, "is_pipeline_last_stage", lambda **_: False)
        monkeypatch.setattr(module, "_is_global_zero_token_step", lambda _: False)
        gather = Mock(wraps=buffer_module.dist_all_gather_func)
        monkeypatch.setattr(buffer_module, "dist_all_gather_func", gather)
        context = {}
        losses = [[], []]

        def forward_backward(**_):
            variant, step = context["variant"], context["step"]
            optimizer = optimizers[variant].chained_optimizers[0]
            if step:
                states = optimizer.optimizer.state.values()
                assert states
                assert all(state["master_param"].is_cuda == (variant == 0) for state in states)
            prediction = models[variant](context["inputs"])
            loss = (prediction.float() - context["targets"]).square().mean()
            losses[variant].append(loss.detach())
            loss.backward()
            models[variant].finish_grad_sync()
            return []

        monkeypatch.setattr(module, "get_forward_backward_func", lambda: forward_backward)
        schedulers = []
        for optimizer in optimizers:

            def advance_lr(optimizer=optimizer, **_):
                for group in optimizer.param_groups:
                    group["lr"] *= 0.9

            schedulers.append(SimpleNamespace(step=advance_lr))

        gradient_norms = []
        for step in range(4):
            context.update(
                step=step,
                inputs=torch.randn(8, 32, device="cuda", dtype=torch.bfloat16),
                targets=torch.randn(8, 32, device="cuda", dtype=torch.float32),
            )
            norms = []
            for variant in range(2):
                context["variant"] = variant
                _, norm = module.train_one_step(
                    args, 0, step, [], [models[variant]], optimizers[variant], schedulers[variant], 1, 8
                )
                norms.append(norm)
                # Complete the actual DistOpt gather before comparing published
                # weights; next forward still exercises DDP's normal pre-hooks.
                models[variant].start_param_sync(force_sync=True)
            torch.cuda.synchronize()
            assert norms[0] == norms[1]
            gradient_norms.append(norms[0])
            torch.testing.assert_close(losses[0][-1], losses[1][-1], rtol=0, atol=0)
            for left, right in zip(models[0].parameters(), models[1].parameters(), strict=True):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
            left_opt, right_opt = (optimizer.chained_optimizers[0].optimizer for optimizer in optimizers)
            for left_group, right_group in zip(left_opt.param_groups, right_opt.param_groups, strict=True):
                assert left_group["step"] == right_group["step"] == step + 1
                assert left_group["lr"] == right_group["lr"] > 0
                for left, right in zip(left_group["params"], right_group["params"], strict=True):
                    for key in ("master_param", "exp_avg", "exp_avg_sq"):
                        torch.testing.assert_close(
                            left_opt.state[left][key].detach().cpu(),
                            right_opt.state[right][key].detach().cpu(),
                            rtol=0,
                            atol=0,
                        )
        assert gather.call_count >= 8
        assert any(
            not torch.equal(parameter, initial)
            for parameter, initial in zip(models[0].module.parameters(), source.parameters(), strict=True)
        )
        record_property("model_parameter_bytes", model_bytes)
        record_property("gradient_norms", gradient_norms)
        record_property("param_gather_calls", gather.call_count)
    finally:
        torch.cuda.synchronize()
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()
