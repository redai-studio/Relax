# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Real TE MoE and native scheduled activation offload on CUDA."""

import os
from contextlib import nullcontext

import pytest
import torch
import torch.distributed as dist


@pytest.fixture(scope="module")
def model_parallel(tmp_path_factory):
    from megatron.core import parallel_state as mpu

    path = tmp_path_factory.mktemp("moe-process-group")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group(
        "nccl", **({} if "RANK" in os.environ else dict(init_method=f"file://{path}/pg", rank=0, world_size=1))
    )
    mpu.initialize_model_parallel(expert_model_parallel_size=dist.get_world_size(group=dist.group.WORLD))
    try:
        yield
    finally:
        mpu.destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available() or os.environ.get("NVTE_CPU_OFFLOAD_V1") != "1",
    reason="requires CUDA and NVTE_CPU_OFFLOAD_V1=1 before importing Transformer Engine",
)
@pytest.mark.parametrize("mode", ["recompute", "moe-recompute", "offload", "recompute-offload"])
@pytest.mark.parametrize("empty_experts", [False, True])
def test_real_moe_schedule_memory_and_gradients(model_parallel, mode, empty_experts, record_property):
    # Must be set before importing TE in a fresh test process.
    assert os.environ.get("NVTE_CPU_OFFLOAD_V1") == "1"
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
        FineGrainedActivationOffloadingInterface as Offload,
    )
    from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
        PipelineOffloadManager,
    )
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.transformer_config import MLATransformerConfig as TransformerConfig

    ep = dist.get_world_size(group=dist.group.WORLD)
    model_parallel_cuda_manual_seed(47)
    pg = ProcessGroupCollection.use_mpu_process_groups()
    base = dict(
        num_layers=2,
        multi_latent_attention=True,
        q_lora_rank=128,
        kv_lora_rank=128,
        qk_head_dim=64,
        qk_pos_emb_head_dim=32,
        v_head_dim=64,
        normalization="RMSNorm",
        apply_rope_fusion=False,
        hidden_size=256,
        num_attention_heads=4,
        ffn_hidden_size=512,
        num_moe_experts=4,
        moe_ffn_hidden_size=512,
        moe_shared_expert_intermediate_size=512,
        moe_router_topk=2,
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        moe_permute_fusion=False,
        moe_router_load_balancing_type="none",
        moe_router_score_function="sigmoid",
        moe_router_pre_softmax=True,
        moe_router_enable_expert_bias=True,
        moe_router_dtype="fp32",
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        add_bias_linear=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        gradient_accumulation_fusion=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        expert_model_parallel_size=ep,
    )

    def create(active):
        kwargs = dict(base)
        if active and "recompute" in mode:
            kwargs.update(
                recompute_granularity="selective", recompute_modules=["moe" if mode == "moe-recompute" else "moe_act"]
            )
        if active and "offload" in mode:
            kwargs.update(
                fine_grained_activation_offloading=True,
                offload_modules=["expert_fc1", "moe_act"],
                min_offloaded_tensor_size=0,
            )
        config = TransformerConfig(**kwargs)
        model = GPTModel(
            config=config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(
                num_experts=4, moe_grouped_gemm=True, multi_latent_attention=True
            ),
            vocab_size=128,
            max_sequence_length=512,
            position_embedding_type="rope",
            share_embeddings_and_output_weights=False,
            pg_collection=pg,
        ).cuda()
        return model

    try:
        reference = create(False)
        actual = create(True)
        actual.load_state_dict(reference.state_dict())
        if empty_experts:
            with torch.no_grad():
                for model in (reference, actual):
                    for module in model.modules():
                        bias = getattr(module, "expert_bias", None)
                        if bias is not None:
                            bias.fill_(-100)
                            bias[:2].zero_()
        schedule = get_forward_backward_func(pp_size=1)
        outcomes = []
        for model in (reference, actual):
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            peaks, elapsed = [], []
            for iteration in range(3):
                model.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                size = 128 if iteration < 2 else 192
                tokens = torch.arange(size, device="cuda")[None, :] % 128
                positions = torch.arange(size, device="cuda")[None, :]
                losses = []

                def forward_step(iterator, mdl):
                    next(iterator)
                    output = mdl(
                        input_ids=tokens, position_ids=positions, attention_mask=None, labels=(tokens + 1) % 128
                    )
                    losses.append(output.detach().clone())
                    return output, lambda tensor: (tensor.float().mean(), {"loss": tensor.detach().float().mean()})

                profile = (
                    torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
                    )
                    if model is actual and "offload" in mode and iteration == 1
                    else nullcontext()
                )
                start.record()
                with profile as trace:
                    schedule(
                        forward_step_func=forward_step,
                        data_iterator=iter(range(2)),
                        model=model,
                        num_microbatches=2,
                        seq_length=size,
                        micro_batch_size=1,
                        forward_only=False,
                        pg_collection=pg,
                    )
                end.record()
                torch.cuda.synchronize()
                peaks.append(torch.cuda.max_memory_allocated())
                elapsed.append(start.elapsed_time(end))
                gradients = {
                    name: param.grad.detach().cpu().clone()
                    for name, param in model.named_parameters()
                    if param.grad is not None
                }
                if model is reference:
                    outcomes.append(
                        (
                            [loss.cpu() for loss in losses],
                            gradients,
                            {name: buf.detach().cpu().clone() for name, buf in model.named_buffers()},
                        )
                    )
                else:
                    expected_loss, expected_gradients, expected_buffers = outcomes[iteration]
                    for a, b in zip(losses, expected_loss, strict=True):
                        torch.testing.assert_close(a.cpu(), b, rtol=0, atol=0)
                    assert gradients.keys() == expected_gradients.keys()
                    for name, grad in gradients.items():
                        torch.testing.assert_close(grad, expected_gradients[name], rtol=0, atol=0)
                    for name, buf in model.named_buffers():
                        torch.testing.assert_close(buf.cpu(), expected_buffers[name], rtol=0, atol=0)
                optimizer.step()
                if trace is not None:
                    copies = [event.name for event in trace.events() if "memcpy" in event.name.lower()]
                    record_property("profiled_copy_events", sorted(set(copies)))
                    if not empty_experts or dist.get_rank(group=dist.group.WORLD) == 0:
                        assert any("DtoH" in name for name in copies)
                        assert any("HtoD" in name for name in copies)
            record_property("actual" if model is actual else "reference", dict(peak_bytes=peaks, elapsed_ms=elapsed))
            if model is reference:
                model.zero_grad(set_to_none=True)
            if model is actual and "offload" in mode:
                summary = PipelineOffloadManager.get_instance().offload_summary_bytes
                record_property("offload_bytes", summary)
                total = torch.tensor(summary.get("expert_fc1", 0), device="cuda", dtype=torch.long)
                dist.all_reduce(total, group=dist.group.WORLD)
                assert total.item() > 0
        for (name, param), (_, expected) in zip(actual.named_parameters(), reference.named_parameters(), strict=True):
            torch.testing.assert_close(param, expected, rtol=0, atol=0)
    finally:
        Offload.reset_instance()
