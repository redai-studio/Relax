# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CUDA acceptance tests. Run under torchrun for real TP/SP collectives.

Forward tolerance: 3e-5 absolute / 3e-6 relative (FP32 GEMM/reduction).
Backward: BF16 quantization and chunk/TP reduction ordering permit 2% per
coordinate with a 0.02 absolute floor; also require <1% relative L2 error.
Expected values come from FP64 math or a test-only dense FP32 head.
"""

import importlib.util
import os
import tempfile
from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist


pytest.importorskip("megatron.core")
from megatron.core import parallel_state as mpu
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig

from relax.backends.megatron.fused_linear_ce import linear_log_probs
from relax.utils.training.ppo_utils import calculate_log_probs_and_entropy


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA BF16")


@pytest.fixture(scope="module", autouse=True)
def groups():
    torch.cuda.set_device(int(os.getenv("LOCAL_RANK", "0")))
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
    else:
        with tempfile.NamedTemporaryFile(delete=False) as file:
            path = file.name
        dist.init_process_group("nccl", init_method=f"file://{path}", rank=0, world_size=1)
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=int(os.getenv("RELAX_TEST_TP", str(dist.get_world_size()))),
        context_parallel_size=int(os.getenv("RELAX_TEST_CP", "1")),
        pipeline_model_parallel_size=int(os.getenv("RELAX_TEST_PP", "1")),
    )
    model_parallel_cuda_manual_seed(123)
    yield
    mpu.destroy_model_parallel()
    dist.destroy_process_group()


def error(actual, expected):
    delta = (actual.detach().float() - expected.detach().float()).abs()
    return {
        "max_abs": delta.max().item(),
        "mean_abs": delta.mean().item(),
        "relative_l2": (delta.norm() / expected.detach().float().norm().clamp_min(1e-9)).item(),
    }


def check_grad(actual, expected):
    metrics = error(actual, expected)
    assert metrics["relative_l2"] < 0.01, metrics
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
    return metrics


def configure_fp32_reference(model):
    """Test-only dense FP32 GEMM with native low-precision MCore backward.

    The production path never installs or imports this oracle.
    """
    from types import MethodType

    from megatron.core.tensor_parallel import layers

    class Dense(torch.autograd.Function):
        @staticmethod
        @layers.custom_fwd
        def forward(ctx, x, weight, bias, fusion, reduce, sp, buffer, limit, group):
            ctx.save_for_backward(x, weight)
            ctx.main_grad = getattr(weight, "main_grad", None) if fusion else None
            ctx.use_bias = bias is not None
            ctx.gradient_accumulation_fusion = fusion
            ctx.allreduce_dgrad = reduce
            ctx.sequence_parallel = sp
            ctx.grad_output_buffer = buffer
            ctx.wgrad_deferral_limit = limit
            ctx.tp_group = group
            ctx.gtp_remat_size = 1
            total = x
            if sp:
                total = torch.empty((x.shape[0] * group.size(), *x.shape[1:]), device=x.device, dtype=x.dtype)
                layers.dist_all_gather_func(total, x, group=group)
            result = torch.mm(total.reshape(-1, total.shape[-1]), weight.T, out_dtype=torch.float32)
            if bias is not None:
                result = result + bias.float()
            return result.reshape(*total.shape[:-1], weight.shape[0])

        @staticmethod
        @layers.custom_bwd
        def backward(ctx, grad):
            return layers.LinearWithGradAccumulationAndAsyncCommunication.backward(
                ctx, grad.to(ctx.saved_tensors[0].dtype)
            )[:10]

    class Frozen(torch.autograd.Function):
        @staticmethod
        @layers.custom_fwd
        def forward(ctx, x, weight, bias, reduce, group):
            ctx.save_for_backward(weight)
            ctx.allreduce_dgrad, ctx.tp_group = reduce, group
            result = torch.mm(x.reshape(-1, x.shape[-1]), weight.T, out_dtype=torch.float32)
            if bias is not None:
                result = result + bias.float()
            return result.reshape(*x.shape[:-1], weight.shape[0])

        @staticmethod
        @layers.custom_bwd
        def backward(ctx, grad):
            return layers.LinearWithFrozenWeight.backward(ctx, grad.to(ctx.saved_tensors[0].dtype))

    def project(
        self,
        input,
        weight,
        bias,
        gradient_accumulation_fusion,
        allreduce_dgrad,
        sequence_parallel,
        grad_output_buffer=None,
        wgrad_deferral_limit=None,
        tp_group=None,
        gtp_remat_size=1,
    ):
        assert gtp_remat_size == 1
        x = input
        if weight.requires_grad:
            return Dense.apply(
                x,
                weight,
                bias,
                gradient_accumulation_fusion,
                allreduce_dgrad,
                sequence_parallel,
                grad_output_buffer,
                wgrad_deferral_limit,
                tp_group,
            )
        if sequence_parallel:
            x = gather_from_sequence_parallel_region(x, tensor_parallel_output_grad=True, group=tp_group)
        return Frozen.apply(x, weight, bias, allreduce_dgrad, tp_group)

    model.output_layer._forward_impl = MethodType(project, model.output_layer)


def restore_native_head(head):
    head.__dict__.pop("_forward_impl", None)


def make_head(sp=False, tied=False, fusion=False):
    config = TransformerConfig(
        num_layers=1,
        hidden_size=48,
        num_attention_heads=4,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=False,
        sequence_parallel=sp,
        tensor_model_parallel_size=mpu.get_tensor_model_parallel_world_size(),
        gradient_accumulation_fusion=fusion,
    )
    head = ColumnParallelLinear(
        48,
        1034,
        config=config,
        init_method=torch.nn.init.normal_,
        bias=False,
        skip_weight_param_allocation=tied,
        tp_group=mpu.get_tensor_model_parallel_group(),
    )
    model = torch.nn.Module()
    model.output_layer = head
    configure_fp32_reference(model)
    return model, head


@pytest.mark.parametrize("temperature", [1.0, 0.7, 1.3])
@pytest.mark.parametrize("mode", ["joint", "logprob", "entropy", "zero", "metric"])
def test_math_reference_and_arbitrary_backward(temperature, mode):
    torch.manual_seed(13)
    x = (torch.randn(137, 48, device="cuda", dtype=torch.bfloat16) * 0.5).requires_grad_()
    w = (torch.randn(517, 48, device="cuda", dtype=torch.bfloat16) * 0.25).requires_grad_()
    y = torch.randint(517, (137,), device="cuda")
    a = torch.randn(137, device="cuda") if mode in ("joint", "logprob", "metric") else None
    b = torch.randn(137, device="cuda") if mode in ("joint", "entropy") else None
    lp, e = linear_log_probs(x, w, y, None, temperature=temperature, entropy_grad=mode != "metric")
    z = (x.detach().double() @ w.detach().double().T).requires_grad_()
    p = (z / temperature).log_softmax(-1)
    ref_lp = p.gather(-1, y[:, None]).squeeze(-1)
    ref_e = -(p.exp() * p).sum(-1)
    torch.testing.assert_close(lp.double(), ref_lp, rtol=3e-6, atol=3e-5)
    torch.testing.assert_close(e.double(), ref_e, rtol=3e-6, atol=3e-5)
    assert e.requires_grad == (mode != "metric")
    objective = (
        lp.sum() * 0
        if a is None and b is None
        else sum(
            value
            for value in (None if a is None else (a * lp).sum(), None if b is None else (b * e).sum())
            if value is not None
        )
    )
    ref_objective = (
        ref_lp.sum() * 0
        if a is None and b is None
        else sum(
            value
            for value in (None if a is None else (a * ref_lp).sum(), None if b is None else (b * ref_e).sum())
            if value is not None
        )
    )
    objective.backward()
    ref_objective.backward()
    # Match the single BF16 cast at the combined head gradient boundary.
    dz = z.grad.bfloat16()
    dx = dz @ w.detach()
    dw = dz.T @ x.detach()
    metrics = {
        "lp": error(lp, ref_lp),
        "entropy": error(e, ref_e),
        "dx": check_grad(x.grad, dx),
        "dw": check_grad(w.grad, dw),
    }
    print("MATH", temperature, mode, metrics, flush=True)


def test_fp32_statistics_match_independent_reference_with_common_offset():
    """A shared logit offset must not invalidate the entropy oracle.

    Integer multiples of 1/16 make these BF16 products and their FP32
    accumulator sums exact, including the 2048 common offset. The independent
    FP64 log_softmax therefore isolates normalization/cancellation error.
    """
    torch.manual_seed(63)
    x = torch.randint(-8, 8, (17, 48), device="cuda").to(torch.bfloat16) / 16
    weight = torch.randint(-8, 8, (517, 48), device="cuda").to(torch.bfloat16) / 16
    x[:, 0] = 32
    labels = torch.randint(517, (17,), device="cuda")
    previous = None
    for offset in (0, 2048):
        weight[:, 0] = offset / 32
        logits64 = x.double() @ weight.double().T
        normalized64 = logits64.log_softmax(-1)
        expected_lp = normalized64.gather(-1, labels[:, None]).squeeze(-1)
        expected_entropy = -(normalized64.exp() * normalized64).sum(-1)
        lp, entropy = linear_log_probs(x, weight, labels, None)
        torch.testing.assert_close(lp.double(), expected_lp, rtol=3e-6, atol=3e-5)
        torch.testing.assert_close(entropy.double(), expected_entropy, rtol=3e-6, atol=3e-5)
        if previous is not None:
            for actual, before in zip((lp, entropy), previous, strict=True):
                torch.testing.assert_close(actual, before, rtol=3e-6, atol=3e-5)
        previous = lp, entropy


@pytest.mark.parametrize("sp", [False, True])
@pytest.mark.parametrize("frozen,tied", [(False, False), (False, True), (True, True)])
@pytest.mark.parametrize("temperature", [1.0, 0.7, 1.3])
def test_native_head_tp_sp_and_shared_microbatches(sp, frozen, tied, temperature):
    if sp and dist.get_world_size() == 1:
        pytest.skip("SP requires TP2")
    torch.manual_seed(29)
    _, head = make_head(sp, tied)
    weight = (
        torch.nn.Parameter(torch.randn(1034 // dist.get_world_size(), 48, device="cuda", dtype=torch.bfloat16))
        if tied
        else head.weight
    )
    weight.requires_grad_(not frozen)
    group = head.tp_group
    xs = [torch.randn(rows, 1, 48, device="cuda", dtype=torch.bfloat16, requires_grad=True) for rows in (68, 132)]
    ys = [torch.randint(1034, (len(x) * (dist.get_world_size() if sp else 1),), device="cuda") for x in xs]
    coeffs = [(torch.randn(len(y), device="cuda"), torch.randn(len(y), device="cuda")) for y in ys]
    references = []
    for x, y, (a, b) in zip(xs, ys, coeffs, strict=True):
        z, _ = head(x, weight=weight)
        lp, e = calculate_log_probs_and_entropy(z.squeeze(1) / temperature, y, group, with_entropy=True)
        references.append((lp.detach().squeeze(-1), e.detach()))
        (lp.squeeze(-1).mul(a).sum() + e.mul(b).sum()).backward()
    dxs = [x.grad.clone() for x in xs]
    dw = None if frozen else weight.grad.clone()
    weight.grad = None
    for x in xs:
        x.grad = None
    saved = []
    forward_errors = []

    def pack(tensor):
        saved.append((tuple(tensor.shape), tensor.dtype))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        objectives = []
        for x, y, (a, b), (ref_lp, ref_e) in zip(xs, ys, coeffs, references, strict=True):
            h = gather_from_sequence_parallel_region(x, tensor_parallel_output_grad=False, group=group) if sp else x
            lp, e = linear_log_probs(h, weight, y[:, None], group, temperature=temperature)
            torch.testing.assert_close(lp.squeeze(-1), ref_lp, rtol=3e-6, atol=3e-5)
            torch.testing.assert_close(e.squeeze(-1), ref_e, rtol=3e-6, atol=3e-5)
            forward_errors.append({"log_probs": error(lp.squeeze(-1), ref_lp), "entropy": error(e.squeeze(-1), ref_e)})
            objectives.append(lp.squeeze(-1).mul(a).sum() + e.squeeze(-1).mul(b).sum())
        sum(objectives).backward()
    # Parameters/hidden/O(tokens) stats only, never a token×vocab activation.
    assert not any(shape == (len(y), weight.shape[0]) for shape, _ in saved for y in ys)
    metrics = {
        "forward": forward_errors,
        "dx": [check_grad(x.grad, expected) for x, expected in zip(xs, dxs, strict=True)],
    }
    if not frozen:
        metrics["dw"] = check_grad(weight.grad, dw)
        # Next optimizer update on the same original Parameter.
        before = weight.detach().clone()
        torch.optim.SGD([weight], lr=0.001).step()
        assert not torch.equal(weight, before)
    print("NATIVE", dist.get_rank(group), sp, tied, frozen, temperature, metrics, flush=True)


def test_empty_extreme_metric_and_invalid_temperature():
    for rows, scale in ((0, 1.0), (17, 25.0)):
        x = torch.randn(rows, 48, device="cuda", dtype=torch.bfloat16).mul_(scale).requires_grad_()
        w = torch.randn(259, 48, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        y = torch.zeros(rows, device="cuda", dtype=torch.long)
        lp, e = linear_log_probs(x, w, y, None, temperature=0.7)
        assert torch.isfinite(lp).all() and torch.isfinite(e).all()
        (lp.sum() + e.sum()).backward()
        assert torch.isfinite(x.grad).all() and torch.isfinite(w.grad).all()
    for temperature in (0, -1, float("nan")):
        with pytest.raises(ValueError, match="positive temperature"):
            linear_log_probs(x, w, y, None, temperature=temperature)


def test_main_grad_accumulation_and_dummy_hook_protocol():
    torch.manual_seed(21)
    x = torch.randn(267, 48, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.nn.Parameter(torch.randn(517, 48, device="cuda", dtype=torch.bfloat16))
    w.main_grad = torch.zeros_like(w, dtype=torch.float32)
    w.grad_added_to_main_grad = False
    w.zero_out_wgrad = True
    y = torch.randint(517, (267,), device="cuda")
    a, b = torch.randn(267, device="cuda"), torch.randn(267, device="cuda")
    # Independent FP32-head projection + old logprob, with FP32 dWeight
    # accumulation from the BF16 combined upstream gradient.
    z = torch.empty((267, 517), device="cuda", dtype=torch.float32)
    torch.mm(x.detach(), w.detach().T, out=z, out_dtype=torch.float32)
    z.requires_grad_()
    # This is a local mathematical oracle even under a TP torchrun. The old
    # helper interprets None as WORLD, so pass the explicit singleton DP group.
    local_group = mpu.get_data_parallel_group()
    lp, e = calculate_log_probs_and_entropy(z / 0.7, y, local_group, with_entropy=True)
    (lp.squeeze(-1).mul(a).sum() + e.mul(b).sum()).backward()
    expected = torch.mm(z.grad.bfloat16().T, x.detach(), out_dtype=torch.float32)
    for _ in range(3):
        lp, e = linear_log_probs(x, w, y, local_group, temperature=0.7, accumulate=True)
        (lp.mul(a).sum() + e.mul(b).sum()).backward()
    torch.testing.assert_close(w.main_grad, expected * 3, rtol=0.02, atol=0.02)
    assert w.grad_added_to_main_grad and torch.count_nonzero(w.grad) == 0
    assert torch.isfinite(x.grad).all()


def make_gpt(sp=False, tied=False, mtp=False):
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_local_spec,
        get_gpt_layer_with_transformer_engine_spec,
    )
    from megatron.core.models.gpt.gpt_model import GPTModel

    checkpoint = os.getenv("RELAX_TEST_HF_CHECKPOINT")
    if checkpoint:
        from megatron.bridge import AutoBridge
        from megatron.core.process_groups_config import ProcessGroupCollection

        bridge = AutoBridge.from_hf_pretrained(checkpoint, trust_remote_code=True)
        provider = bridge.to_megatron_provider(load_weights=False)
        provider.tensor_model_parallel_size = mpu.get_tensor_model_parallel_world_size()
        provider.sequence_parallel = sp
        provider.gradient_accumulation_fusion = False
        provider.cross_entropy_loss_fusion = False
        provider.hidden_dropout = provider.attention_dropout = 0.0
        provider.transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(qk_layernorm=True)
        provider.finalize()
        provider._pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        model = provider.provide().cuda().bfloat16()
        bridge.load_hf_weights([model])
        model.role = "actor"
        return provider, model

    config = TransformerConfig(
        num_layers=mpu.get_pipeline_model_parallel_world_size(),
        hidden_size=64,
        num_attention_heads=4,
        ffn_hidden_size=96,
        params_dtype=torch.bfloat16,
        bf16=True,
        gradient_accumulation_fusion=not mtp
        and importlib.util.find_spec("fused_weight_gradient_mlp_cuda") is not None,
        mtp_num_layers=1 if mtp else None,
        tensor_model_parallel_size=mpu.get_tensor_model_parallel_world_size(),
        pipeline_model_parallel_size=mpu.get_pipeline_model_parallel_world_size(),
        context_parallel_size=mpu.get_context_parallel_world_size(),
        calculate_per_token_loss=mpu.get_context_parallel_world_size() > 1,
        pipeline_dtype=torch.bfloat16,
        variable_seq_lengths=True,
        sequence_parallel=sp,
        hidden_dropout=0,
        attention_dropout=0,
        masked_softmax_fusion=False,
        bias_activation_fusion=False,
        bias_dropout_fusion=False,
        use_cpu_initialization=False,
    )
    use_te = sp or mpu.get_context_parallel_world_size() > 1
    if use_te:
        pytest.importorskip("transformer_engine", reason="full GPT SP/CP needs native TE")
    spec = get_gpt_layer_with_transformer_engine_spec() if use_te else get_gpt_layer_local_spec()
    kwargs = {}
    if mtp:
        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec

        kwargs["mtp_block_spec"] = get_gpt_mtp_block_spec(config, spec, use_transformer_engine=sp)
    model = (
        GPTModel(
            config,
            spec,
            vocab_size=1034,
            max_sequence_length=128,
            share_embeddings_and_output_weights=tied,
            position_embedding_type="rope",
            pre_process=mpu.is_pipeline_first_stage(),
            post_process=mpu.is_pipeline_last_stage(),
            **kwargs,
        )
        .cuda()
        .bfloat16()
    )
    from megatron.core.enums import ModelType

    model.model_type = ModelType.encoder_or_decoder
    model.role = "actor"
    return config, model


def actor_args():
    return SimpleNamespace(
        fused_linear_ce=False,
        loss_type="policy_loss",
        advantage_estimator="grpo",
        rollout_temperature=0.7,
        entropy_coef=0.01,
        qkv_format=os.getenv("RELAX_TEST_LAYOUT", "bshd"),
        allgather_cp=False,
        use_opd=False,
        opd_token_selection="all",
        use_opsm=False,
        use_rollout_logprobs=False,
        bf16=True,
        fp16=False,
        custom_pg_loss_reducer_function_path=None,
        get_mismatch_metrics=True,
        use_tis=True,
        custom_tis_function_path=None,
        tis_clip=2.0,
        tis_clip_low=0.5,
        eps_clip=0.2,
        eps_clip_high=0.28,
        use_kl_loss=True,
        kl_loss_type="low_var_kl",
        use_unbiased_kl=False,
        kl_loss_coef=0.01,
        calculate_per_token_loss=mpu.get_context_parallel_world_size() > 1,
        global_batch_size=6 * mpu.get_data_parallel_world_size(),
        recompute_loss_function=False,
        log_probs_chunk_size=-1,
        use_rollout_entropy=True,
        data_pad_size_multiplier=8,
        ci_test=False,
        enable_mtp_training=False,
        custom_megatron_before_train_step_hook_path=None,
        custom_megatron_before_log_prob_hook_path=None,
        seq_length=32,
        micro_batch_size=2,
        decoder_seq_length=None,
        use_routing_replay=False,
        hidden_size=64,
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_actor_reference_forward_only_preserves_initial_ci_equality(monkeypatch, enabled):
    if mpu.get_context_parallel_world_size() > 1 or mpu.get_pipeline_model_parallel_world_size() > 1:
        pytest.skip("greedy CI fixture requires a complete local sequence and model")
    from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region
    from megatron.core.transformer.module import Float16Module
    from megatron.training import global_vars

    from relax.backends.megatron import data as data_module
    from relax.backends.megatron import fused_linear_ce as fused_module
    from relax.backends.megatron import model as runtime
    from relax.backends.megatron.data import DataIterator
    from relax.backends.megatron.loss import get_log_probs_and_entropy

    args = actor_args()
    args.fused_linear_ce = enabled
    args.ci_test = True
    args.use_rollout_entropy = False
    args.micro_batch_size = args.rollout_batch_size = args.n_samples_per_prompt = 1
    args.log_multi_turn = False
    args.qkv_format = "bshd"
    monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", args)
    torch.manual_seed(538)
    config, bare = make_gpt()
    bare.output_layer.gradient_accumulation_fusion = False
    model = Float16Module(config, bare).eval()
    model.role = "actor"
    # Native BF16 head, including its actual output rounding. Confident greedy
    # targets also satisfy CI's existing log-probability range check.
    with torch.no_grad():
        bare.output_layer.weight.mul_(100)
        tokens = torch.tensor([[42]], device="cuda")
        for _ in range(23):
            logits = model(input_ids=tokens, position_ids=None, attention_mask=None)[:, -1]
            logits = gather_from_tensor_model_parallel_region(logits, group=bare.output_layer.tp_group)
            tokens = torch.cat((tokens, logits.argmax(-1, keepdim=True)), dim=1)
    data = {
        "tokens": [tokens[0]],
        "total_lengths": [24],
        "response_lengths": [10],
        "max_seq_lens": [32],
        "loss_masks": [torch.ones(10, device="cuda", dtype=torch.int)],
    }
    hits = []
    original_processor = fused_module.output_processor

    def record_processor(**kwargs):
        output = original_processor(**kwargs)
        hits.append(kwargs["context"].fused)
        return output

    monkeypatch.setattr(fused_module, "output_processor", record_processor)

    def forward(prefix, consumer=get_log_probs_and_entropy):
        return runtime.forward_only(consumer, args, [model], [DataIterator(data, 1)], [1], store_prefix=prefix)

    native = forward("teacher_")
    actor = forward("")
    reference = forward("ref_")
    torch.testing.assert_close(actor["log_probs"], reference["ref_log_probs"], rtol=0, atol=1e-8)
    topk = forward("ref_", partial(get_log_probs_and_entropy, with_topk=True, topk_k=3))
    torch.testing.assert_close(topk["ref_log_probs"], native["teacher_log_probs"], rtol=0, atol=0)
    if not enabled:
        torch.testing.assert_close(actor["log_probs"], native["teacher_log_probs"], rtol=0, atol=0)
    assert hits == ([True, True] if enabled else [])
    # Replace only the metrics transport; run the real reducer and CI assertions.
    monkeypatch.setattr(
        data_module,
        "gather_log_data",
        lambda prefix, _args, _rollout_id, values, *_rest: {f"{prefix}/{key}": val for key, val in values.items()},
    )
    data_module.log_rollout_data(0, args, {**data, **actor, **reference})


@pytest.mark.parametrize("sp,tied", [(False, False), (False, True), (True, True)])
@pytest.mark.parametrize("legacy_sft_env", [False, True])
def test_real_rl_actor_multiple_updates_and_forward_only(monkeypatch, sp, tied, legacy_sft_env):
    monkeypatch.setenv("RELAX_SFT_FUSED_LINEAR_CE", str(int(legacy_sft_env)))
    if sp and mpu.get_tensor_model_parallel_world_size() == 1:
        pytest.skip("SP requires TP2")
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.distributed.finalize_model_grads import finalize_model_grads
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
    from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
    from megatron.training import global_vars

    from relax.backends.megatron import fused_linear_ce as fused_module
    from relax.backends.megatron import model as runtime
    from relax.backends.megatron.data import DataIterator
    from relax.backends.megatron.loss import get_log_probs_and_entropy

    args = actor_args()
    hits = []
    original_processor = fused_module.output_processor

    def record_processor(**kwargs):
        output = original_processor(**kwargs)
        hits.append(kwargs["context"].fused)
        return output

    monkeypatch.setattr(fused_module, "output_processor", record_processor)
    monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", args)
    # DP replicas consume different samples; TP peers still share token data.
    torch.manual_seed(47 + mpu.get_data_parallel_rank())
    lengths = [24, 17, 23, 27, 13, 21]
    responses = [10, 6, 8, 11, 0, 7]
    data = {
        "tokens": [torch.randint(1034, (n,), device="cuda") for n in lengths],
        "total_lengths": lengths,
        "response_lengths": responses,
        "max_seq_lens": [32] * 6,
        "loss_masks": [(torch.arange(n, device="cuda") % 3 != 0).int() for n in responses],
        "advantages": [torch.randn(n, device="cuda") for n in responses],
    }
    if os.getenv("RELAX_TEST_PROMPT_DATA"):
        import json

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(os.environ["RELAX_TEST_HF_CHECKPOINT"])
        with open(os.environ["RELAX_TEST_PROMPT_DATA"]) as file:
            rows = [json.loads(next(file)) for _ in range(6)]
        # Fixed replay: retain the prompt prefix and the actual dataset answer.
        # Signed advantages are controlled inputs; no reward-curve claim.
        prompts = [
            tokenizer(
                tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True),
                add_special_tokens=False,
            )["input_ids"][:16]
            for row in rows
        ]
        answers = [
            tokenizer("Answer: " + str(row["label"]), add_special_tokens=False)["input_ids"][:12] for row in rows
        ]
        responses = [len(answer) for answer in answers]
        data.update(
            tokens=[
                torch.tensor(prompt + answer, device="cuda") for prompt, answer in zip(prompts, answers, strict=True)
            ],
            total_lengths=[len(prompt) + len(answer) for prompt, answer in zip(prompts, answers, strict=True)],
            response_lengths=responses,
            loss_masks=[(torch.arange(n, device="cuda") % 3 != 0).int() for n in responses],
            advantages=[torch.randn(n, device="cuda") for n in responses],
        )
    if mpu.get_context_parallel_world_size() > 1:
        from relax.backends.megatron.cp_utils import slice_log_prob_with_cp

        data["advantages"] = [
            slice_log_prob_with_cp(value, total, response, args.qkv_format, 32)
            for value, total, response in zip(
                data["advantages"], data["total_lengths"], data["response_lengths"], strict=True
            )
        ]
    initial = None
    references = []
    master_references = []
    diagnose_master = os.getenv("RL_FUSED_DIAGNOSE_MASTER") == "1"
    # Use the repository's pretrained RL learning rate; the tiny random GPT
    # fixture uses a larger rate so BF16 parameter updates are observable.
    learning_rate = 1e-6 if os.getenv("RELAX_TEST_HF_CHECKPOINT") else 1e-4
    baseline_control = os.getenv("RL_FUSED_BASELINE_CONTROL") == "1"
    for fused in (False, True):
        args.fused_linear_ce = fused and not baseline_control
        print(
            "RL_EXECUTION", "baseline-repeat" if baseline_control else "fused-pair", fused, learning_rate, flush=True
        )
        torch.manual_seed(31)
        config, bare = make_gpt(sp, tied)
        config.finalize_model_grads_func = finalize_model_grads
        if not args.fused_linear_ce and bare.post_process:
            configure_fp32_reference(bare)
        if initial is None:
            initial = {
                key: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
                for key, value in bare.state_dict().items()
            }
        else:
            bare.load_state_dict(initial)
        distributed_optimizer = os.getenv("RELAX_TEST_DISTRIBUTED_OPTIMIZER") == "1"
        ddp = DistributedDataParallel(
            config,
            DistributedDataParallelConfig(
                grad_reduce_in_fp32=True,
                use_distributed_optimizer=distributed_optimizer,
            ),
            bare,
        )
        ddp.role = "actor"
        opt = get_megatron_optimizer(
            OptimizerConfig(
                lr=learning_rate,
                min_lr=learning_rate,
                bf16=True,
                params_dtype=torch.bfloat16,
                use_distributed_optimizer=distributed_optimizer,
            ),
            [ddp],
        )
        config.grad_scale_func = opt.scale_loss
        if diagnose_master:
            master_initial = [param.detach().float().cpu().clone() for param in opt.get_parameters()]
        recorded_grads = []
        original_step = opt.step

        def record_step(_module=bare, _step=original_step, _grads=recorded_grads):
            _grads.append(
                {
                    name: param.main_grad.detach().float().cpu().clone()
                    for name, param in _module.named_parameters()
                    if param.requires_grad
                }
            )
            if tied and mpu.get_pipeline_model_parallel_world_size() > 1:
                group = mpu.get_embedding_group()
                grad = _module.shared_embedding_or_output_weight().main_grad
                reference = grad.clone()
                dist.broadcast(reference, src=dist.get_process_group_ranks(group)[0], group=group)
                torch.testing.assert_close(grad, reference, rtol=0, atol=0)
            result = _step()
            if tied and mpu.get_pipeline_model_parallel_world_size() > 1:
                weight = _module.shared_embedding_or_output_weight()
                reference = weight.detach().clone()
                dist.broadcast(reference, src=dist.get_process_group_ranks(group)[0], group=group)
                torch.testing.assert_close(weight, reference, rtol=0, atol=0)
            return result

        opt.step = record_step
        scheduler = OptimizerParamScheduler(
            opt,
            init_lr=learning_rate,
            max_lr=learning_rate,
            min_lr=learning_rate,
            lr_warmup_steps=0,
            lr_decay_steps=100,
            lr_decay_style="constant",
            start_wd=0.0,
            end_wd=0.0,
            wd_incr_steps=100,
            wd_incr_style="constant",
        )
        no_grad = runtime.forward_only(get_log_probs_and_entropy, args, [ddp], [DataIterator(data, 2)], [3])
        if not fused and bare.post_process:
            # Fixed actor old log-probs and reference/sampling probabilities in
            # BOTH runs. Never replace the target with fused outputs.
            data["log_probs"] = [value.detach() for value in no_grad["log_probs"]]
            data["ref_log_probs"] = [value.detach() - 0.02 for value in no_grad["log_probs"]]
            data["rollout_log_probs"] = [value.detach() - 0.01 for value in no_grad["log_probs"]]
            expected_no_grad = no_grad
        elif bare.post_process:
            for actual, expected in zip(no_grad["log_probs"], expected_no_grad["log_probs"], strict=True):
                torch.testing.assert_close(actual, expected, rtol=3e-6, atol=3e-5)
        ddp.train()
        for step in range(3):
            metrics, grad_norm = runtime.train_one_step(
                args, 0, step, [DataIterator(data, 2)], [ddp], opt, scheduler, 3, args.global_batch_size
            )
            grad_norm = float(grad_norm)
            assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values()), metrics
            assert torch.isfinite(torch.tensor(grad_norm)) and grad_norm > 0
            state = {key: value.detach().cpu().clone() for key, value in bare.named_parameters()}
            if diagnose_master:
                masters = [param.detach().float().cpu().clone() for param in opt.get_parameters()]
                if not fused:
                    master_references.append(masters)
                else:
                    expected_masters = master_references[step]
                    master_delta = sum(
                        (value - expected).square().sum().double()
                        for value, expected in zip(masters, expected_masters, strict=True)
                    )
                    master_scale = sum(
                        (value - initial).square().sum().double()
                        for value, initial in zip(expected_masters, master_initial, strict=True)
                    )
                    ulp_max = 0.0
                    for key, value in state.items():
                        expected = references[step][2][key]
                        upper = torch.nextafter(expected, torch.full_like(expected, float("inf")))
                        lower = torch.nextafter(expected, torch.full_like(expected, -float("inf")))
                        ulp = torch.maximum(
                            (upper.float() - expected.float()).abs(), (lower.float() - expected.float()).abs()
                        )
                        ulp_max = max(
                            ulp_max, ((value.float() - expected.float()).abs() / ulp.clamp_min(1e-38)).max().item()
                        )
                    print(
                        "MASTER_DIAGNOSTIC",
                        step,
                        (master_delta / master_scale.clamp_min(1e-20)).sqrt().item(),
                        ulp_max,
                        [str(param.dtype) for param in opt.get_parameters()][:3],
                        flush=True,
                    )
            if not fused:
                references.append((metrics, grad_norm, state, recorded_grads[-1]))
            else:
                expected_metrics, expected_norm, expected_state, expected_grads = references[step]
                for key, value in metrics.items():
                    assert value == pytest.approx(expected_metrics[key], rel=0.02, abs=2e-4), (
                        key,
                        value,
                        expected_metrics[key],
                    )
                assert grad_norm == pytest.approx(expected_norm, rel=0.02)
                for key, value in state.items():
                    torch.testing.assert_close(value, expected_state[key], rtol=0.02, atol=2e-3)
                grad_delta = sum(
                    (value - expected_grads[key]).square().sum().double() for key, value in recorded_grads[-1].items()
                )
                grad_scale = sum(value.square().sum().double() for value in expected_grads.values())
                grad_error = (grad_delta / grad_scale.clamp_min(1e-20)).sqrt().item()
                update_delta = sum(
                    (value.float() - expected_state[key].float()).square().sum().double()
                    for key, value in state.items()
                )
                update_scale = sum(
                    (value.float() - initial[key].float()).square().sum().double()
                    for key, value in expected_state.items()
                )
                update_error = (update_delta / update_scale.clamp_min(1e-20)).sqrt().item()
                # Whole-transformer BF16 propagation and Adam threshold rounding
                # permit 2% gradient / 5% update-relative L2 error.
                print(
                    f"RL_ERROR tp={mpu.get_tensor_model_parallel_rank()} "
                    f"pp={mpu.get_pipeline_model_parallel_rank()} cp={mpu.get_context_parallel_rank()} "
                    f"sp={sp} tied={tied} step={step} gradient={grad_error} update={update_error}",
                    flush=True,
                )
                assert grad_error < 0.02, grad_error
                assert update_error < 0.05, update_error
            print(
                "RL_STEP",
                dist.get_rank(mpu.get_tensor_model_parallel_group()),
                fused,
                sp,
                tied,
                step,
                grad_norm,
                metrics,
                flush=True,
            )
        assert any(not torch.equal(value, initial[key]) for key, value in state.items())
        assert (len(hits) == 12 and all(hits)) if fused and not baseline_control and bare.post_process else not hits
        opt.step = original_step
        del record_step, original_step, no_grad, opt, scheduler, ddp, bare
        torch.cuda.empty_cache()


@pytest.mark.parametrize(
    "case",
    [
        "off",
        "actor_fwd",
        "reference",
        "unknown_role",
        "value",
        "opd",
        "topk",
        "lora",
        "mtp",
        "dcp",
        "scaling",
        "head_hook",
        "head_impl",
        "mask",
        "score_centering",
        "native_ce",
    ],
)
def test_unsupported_microbatch_keeps_original_model_forward(monkeypatch, case):
    from types import MethodType

    from relax.backends.megatron.fused_linear_ce import prepare_output_processor

    torch.manual_seed(15)
    _, model = make_gpt()
    args = actor_args()
    batch = {"total_lengths": [17], "response_lengths": [9]}
    args.fused_linear_ce = case != "off"
    consumer = "policy_loss"
    if case in ("actor_fwd", "reference", "unknown_role"):
        model.role = None if case == "unknown_role" else case
    elif case == "value":
        consumer = "value_loss"
    elif case == "opd":
        args.use_opd = True
    elif case == "topk":
        batch["rollout_topk_token_ids"] = []
    elif case == "lora":
        args.lora_rank = 8
    elif case == "mtp":
        args.enable_mtp_training = True
    elif case == "dcp":
        batch["dynamic_cp_size"] = 1
    elif case == "scaling":
        model._scale_logits = MethodType(lambda self, z: z * 2, model)
    elif case == "head_hook":
        # Only one TP rank has the unsupported hook: all ranks must fall back.
        if dist.get_rank(model.output_layer.tp_group) == 0:
            model.output_layer.register_forward_hook(lambda *args: None)
    elif case == "head_impl":
        impl = model.output_layer._forward_impl
        model.output_layer._forward_impl = lambda *args, **kwargs: impl(*args, **kwargs)
    elif case == "mask":
        batch["log_prob_keep_mask"] = True
    elif case == "score_centering":
        args.advantage_estimator = "score_centering"
    elif case == "native_ce":
        model.output_layer.config.cross_entropy_loss_fusion = True
    tokens = torch.randint(1034, (1, 17), device="cuda")
    kwargs = {"input_ids": tokens, "position_ids": None, "attention_mask": None}
    with torch.no_grad():
        expected = model(**kwargs)
        hooks = prepare_output_processor(model, args, batch, consumer=consumer)
        assert not hooks, case
        actual = model(**kwargs, **hooks)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("layout", ["thd", "bshd"])
def test_packed_response_coordinates_and_masks(monkeypatch, layout):
    from relax.backends.megatron.fused_linear_ce import OutputContext, output_processor
    from relax.backends.megatron.loss import get_log_probs_and_entropy

    _, head = make_head()
    args = actor_args()
    args.qkv_format = layout
    lengths, responses = [17, 9, 13], [8, 0, 6]
    tokens = [torch.randint(1034, (n,), device="cuda") for n in lengths]
    if layout == "thd":
        input_ids = torch.cat(tokens)[None]
    else:
        input_ids = torch.stack([torch.nn.functional.pad(t, (0, 24 - len(t))) for t in tokens])
    h = torch.randn(
        input_ids.shape[1], input_ids.shape[0], 48, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    if mpu.get_tensor_model_parallel_world_size() > 1:
        # This test is about response coordinates, without SP sharding.
        dist.broadcast(h.detach(), src=dist.get_process_group_ranks(head.tp_group)[0], group=head.tp_group)
    baseline, _ = head(h)
    common = dict(
        args=args,
        unconcat_tokens=tokens,
        total_lengths=lengths,
        response_lengths=responses,
        max_seq_lens=[24] * 3 if layout == "bshd" else None,
        with_entropy=True,
    )
    _, expected = get_log_probs_and_entropy(baseline.transpose(0, 1).contiguous(), **common)
    restore_native_head(head)
    context = OutputContext(0.7, True)
    actual = output_processor(
        hidden_states=h,
        output_layer=head,
        output_weight=None,
        input_ids=input_ids,
        context=context,
        runtime_gather_output=None,
        scale_logits=lambda z: z,
    )
    _, result = get_log_probs_and_entropy(actual, **common, rl_fused=context.fused)
    for key in ("log_probs", "entropy"):
        for left, right in zip(result[key], expected[key], strict=True):
            torch.testing.assert_close(left, right, rtol=3e-6, atol=3e-5)


@pytest.mark.parametrize("layout", ["bshd", "thd"])
@pytest.mark.parametrize("empty", [False, True])
def test_static_cp_coordinates_and_signed_gradients(monkeypatch, layout, empty):
    if mpu.get_context_parallel_world_size() == 1:
        pytest.skip("requires a real CP group")
    from megatron.training import global_vars

    from relax.backends.megatron.data import DataIterator, get_batch
    from relax.backends.megatron.fused_linear_ce import OutputContext, _cp_labels, output_processor
    from relax.backends.megatron.loss import get_log_probs_and_entropy, get_responses

    args = actor_args()
    args.qkv_format = layout
    monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", args)
    lengths, responses = [17, 9, 13], [0, 0, 0] if empty else [8, 0, 1]
    # Unique global token identities make the cross-shard next-token boundary
    # check independent of the label partition implementation.
    tokens = [torch.arange(n, device="cuda") + 30 * i for i, n in enumerate(lengths)]
    data = {
        "tokens": tokens,
        "total_lengths": lengths,
        "response_lengths": responses,
        "max_seq_lens": [24] * 3,
        "loss_masks": [torch.ones(n, device="cuda", dtype=torch.int) for n in responses],
    }
    batch = get_batch(DataIterator(data, 3), list(data), pad_multiplier=8, qkv_format=layout)
    common = dict(
        args=args,
        unconcat_tokens=tokens,
        total_lengths=lengths,
        response_lengths=responses,
        max_seq_lens=batch["max_seq_lens"],
    )
    labels = _cp_labels(args, batch)
    for selected, target in get_responses(labels.unsqueeze(-1).float(), **common, apply_temperature=False):
        torch.testing.assert_close(selected[:, 0].long(), target, rtol=0, atol=0)
    torch.manual_seed(82)
    _, head = make_head()
    h = torch.randn(
        batch["tokens"].shape[1], batch["tokens"].shape[0], 48, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    baseline, _ = head(h)
    _, expected = get_log_probs_and_entropy(baseline.transpose(0, 1).contiguous(), **common, with_entropy=True)
    coefficients = [(torch.randn_like(lp), torch.randn_like(lp)) for lp in expected["log_probs"]]

    def objective(values):
        return sum(
            (lp * a).sum() + (entropy * b).sum()
            for lp, entropy, (a, b) in zip(values["log_probs"], values["entropy"], coefficients, strict=True)
        )

    objective(expected).backward()
    expected_dx, expected_dw = h.grad.clone(), head.weight.grad.clone()
    h.grad, head.weight.grad = None, None
    restore_native_head(head)
    context = OutputContext(0.7, True, labels=labels)
    stats = output_processor(
        hidden_states=h,
        output_layer=head,
        output_weight=None,
        input_ids=batch["tokens"],
        context=context,
        runtime_gather_output=None,
        scale_logits=lambda z: z,
    )
    assert context.fused
    _, actual = get_log_probs_and_entropy(stats, **common, with_entropy=True, rl_fused=True)
    for key in ("log_probs", "entropy"):
        for left, right in zip(actual[key], expected[key], strict=True):
            torch.testing.assert_close(left, right, rtol=3e-6, atol=3e-5)
    objective(actual).backward()
    check_grad(h.grad, expected_dx)
    check_grad(head.weight.grad, expected_dw)


def test_tp_control_groups_allow_independent_dp_forward_counts():
    if mpu.get_tensor_model_parallel_world_size() == 1 or mpu.get_data_parallel_world_size() == 1:
        pytest.skip("requires distinct TP groups in a TP2/DP2 torchrun")
    from relax.backends.megatron.fused_linear_ce import _tp_fallback_reason

    group = mpu.get_tensor_model_parallel_group()
    # Forward-only data partitions can have different microbatch counts. Only
    # members of this TP group may enter its lazy creation and reductions.
    for step in range(1 + mpu.get_data_parallel_rank()):
        reason = "unsupported local shard" if step % 2 == 0 and dist.get_rank(group) == 0 else None
        result = _tp_fallback_reason(reason, group)
        assert (result is not None) is (step % 2 == 0)
        if reason is not None:
            assert result == reason


@pytest.mark.parametrize("invalid", ["bf16_main_grad", "cp_labels"])
def test_unsupported_output_operands_fall_back_before_projection(monkeypatch, invalid):
    pytest.importorskip("fused_weight_gradient_mlp_cuda", reason="native fused main_grad oracle needs MCore extension")
    from relax.backends.megatron.fused_linear_ce import OutputContext, output_processor

    _, head = make_head(fusion=True)
    head.weight.main_grad = torch.zeros_like(head.weight, dtype=torch.float32)
    restore_native_head(head)
    h = torch.randn(17, 1, 48, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    tokens = torch.randint(1034, (1, 17), device="cuda")

    def forbidden_objects(*_args, **_kwargs):
        raise AssertionError("TP eligibility must not serialize Python objects")

    original_item = torch.Tensor.item

    def cpu_item(tensor, *args, **kwargs):
        if tensor.is_cuda:
            raise AssertionError("TP eligibility must not synchronize a CUDA scalar to the host")
        return original_item(tensor, *args, **kwargs)

    processor_kwargs = {
        "hidden_states": h,
        "output_layer": head,
        "output_weight": None,
        "input_ids": tokens,
        "runtime_gather_output": None,
        "scale_logits": lambda z: z,
    }
    # Warm up the negotiated path, then change eligibility on only one TP rank.
    # Caching a local decision would make peers take incompatible branches.
    with monkeypatch.context() as guard:
        guard.setattr(dist, "all_gather_object", forbidden_objects)
        guard.setattr(torch.Tensor, "item", cpu_item)
        context = OutputContext(0.7, True)
        with torch.no_grad():
            output_processor(context=context, **processor_kwargs)
        assert context.fused
    rank = dist.get_rank(head.tp_group)
    if invalid == "bf16_main_grad" and rank == 0:
        head.weight.main_grad = head.weight.main_grad.bfloat16()
    context = OutputContext(0.7, True, labels=tokens[:, :-1] if invalid == "cp_labels" and rank == 0 else None)
    expected, _ = head(h)
    actual = output_processor(
        hidden_states=h,
        output_layer=head,
        output_weight=None,
        input_ids=tokens,
        context=context,
        runtime_gather_output=None,
        scale_logits=lambda z: z,
    )
    assert not context.fused
    torch.testing.assert_close(actual, expected.transpose(0, 1).contiguous(), rtol=0, atol=0)
    actual.sum().backward()
    assert torch.isfinite(h.grad).all()
    head.weight.main_grad = torch.zeros_like(head.weight, dtype=torch.float32)
    context = OutputContext(0.7, True)
    with monkeypatch.context() as guard, torch.no_grad():
        guard.setattr(dist, "all_gather_object", forbidden_objects)
        guard.setattr(torch.Tensor, "item", cpu_item)
        output_processor(context=context, **processor_kwargs)
    assert context.fused


@pytest.mark.parametrize("tied", [False, True])
def test_native_fp32_main_grad_shared_accumulation(tied):
    pytest.importorskip("fused_weight_gradient_mlp_cuda", reason="native fused wgrad extension required")
    torch.manual_seed(23)
    _, head = make_head(tied=tied, fusion=True)
    weight = (
        torch.nn.Parameter(torch.randn(1034 // dist.get_world_size(), 48, device="cuda", dtype=torch.bfloat16))
        if tied
        else head.weight
    )
    weight.main_grad = torch.zeros_like(weight, dtype=torch.float32)
    weight.grad_added_to_main_grad = False
    weight.zero_out_wgrad = tied
    inputs = [torch.randn(n, 1, 48, device="cuda", dtype=torch.bfloat16, requires_grad=True) for n in (67, 267, 19)]
    targets = [torch.randint(1034, (len(x),), device="cuda") for x in inputs]
    gradients = [(torch.randn(len(x), device="cuda"), torch.randn(len(x), device="cuda")) for x in inputs]
    for x, y, (a, b) in zip(inputs, targets, gradients, strict=True):
        z, _ = head(x, weight=weight)
        lp, e = calculate_log_probs_and_entropy(z.squeeze(1) / 0.7, y, head.tp_group, with_entropy=True)
        (lp.squeeze(-1).mul(a).sum() + e.mul(b).sum()).backward()
    expected = weight.main_grad.clone()
    dx = [x.grad.clone() for x in inputs]
    weight.main_grad.zero_()
    weight.grad_added_to_main_grad = False
    weight.grad = None
    losses = []
    for x, y, (a, b) in zip(inputs, targets, gradients, strict=True):
        x.grad = None
        lp, e = linear_log_probs(x, weight, y[:, None], head.tp_group, temperature=0.7, accumulate=True)
        losses.append(lp.squeeze(-1).mul(a).sum() + e.squeeze(-1).mul(b).sum())
    sum(losses).backward()
    check_grad(weight.main_grad, expected)
    assert weight.grad_added_to_main_grad
    for x, expected_dx in zip(inputs, dx, strict=True):
        check_grad(x.grad, expected_dx)


@pytest.mark.parametrize("allgather", [False, True])
@pytest.mark.parametrize("layout", ["bshd", "thd"])
def test_real_parallel_topology_prepares_only_output_stage(allgather, layout):
    """Real topology selection; complete schedule coverage lives in real_rl."""
    cp = mpu.get_context_parallel_world_size()
    pp = mpu.get_pipeline_model_parallel_world_size()
    if cp == pp == 1:
        pytest.skip("requires a CP/PP torchrun")
    from relax.backends.megatron.cp_utils import slice_with_cp
    from relax.backends.megatron.fused_linear_ce import prepare_output_processor

    _, model = make_gpt()
    args = actor_args()
    args.fused_linear_ce = True
    args.qkv_format = layout
    args.allgather_cp = allgather
    tokens = torch.arange(24, device="cuda")
    local = slice_with_cp(tokens, 0, "bshd", 24)
    batch = {
        "total_lengths": [24],
        "response_lengths": [8],
        "unconcat_tokens": [tokens],
        "tokens": local[None],
        "max_seq_lens": [24],
    }
    hooks = prepare_output_processor(model, args, batch, consumer="policy_loss")
    unsupported_layout = cp > 1 and allgather
    assert bool(hooks) is (model.post_process and not unsupported_layout)


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("cli", [False, True])
def test_sft_and_mtp_cli_hopper_fallback_matches_native(monkeypatch, cli, legacy):
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("tests the real SM90 unsupported-hardware fallback")
    from megatron.training import global_vars

    from relax.backends.megatron import fused_linear_ce as fused
    from relax.backends.megatron.loss import get_log_probs_and_entropy
    from relax.backends.megatron.model import _bypass_output_layer

    try:
        from relax.models.common.chunked_mtp_loss import _chunked_head_ce
    except (ImportError, AttributeError) as exc:
        pytest.skip(f"requires the pinned MTP runtime: {exc}")

    monkeypatch.setenv("RELAX_SFT_FUSED_LINEAR_CE", str(int(legacy)))
    args = actor_args()
    args.entropy_coef = 0
    args.loss_type = "sft"
    args.sft_chunked_logits = True
    args.sft_logits_chunk_size = 7
    monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", args)
    model, head = make_head()
    restore_native_head(head)
    torch.manual_seed(81)
    tokens = torch.randint(1034, (24,), device="cuda")
    x = torch.randn(24, 1, 48, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    results = []

    def forbidden_kernel(*_args, **_kwargs):
        raise AssertionError("SM90 must retain the native head, not execute SM100 CE")

    monkeypatch.setattr(fused, "linear_cross_entropy", forbidden_kernel)

    def native_ce(labels, logits):
        lp, _ = calculate_log_probs_and_entropy(
            logits.squeeze(1).float(), labels[0], head.tp_group, with_entropy=False
        )
        return -lp.T

    for enabled in (False, cli):
        args.fused_linear_ce = enabled
        x.grad = head.weight.grad = None
        with _bypass_output_layer(model, fused_linear_ce=enabled) as projected:
            hidden, _ = head(x)
            _, output = get_log_probs_and_entropy(
                hidden.transpose(0, 1).contiguous(),
                args=args,
                unconcat_tokens=[tokens],
                total_lengths=[24],
                response_lengths=[8],
                max_seq_lens=[24],
                with_entropy=False,
                lm_head_forward=projected,
            )
            sft = output["log_probs"][0]
        mtp, _, _ = _chunked_head_ce(x, tokens[None], head, None, None, native_ce, None, 7, tp_group=head.tp_group)
        objective = sft.sum() + mtp.sum()
        objective.backward()
        results.append((sft.detach(), mtp.detach(), x.grad.clone(), head.weight.grad.clone()))
    torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)


def test_real_mtp_gpt_preserves_native_forward_and_backward(monkeypatch):
    from megatron.training import global_vars

    from relax.backends.megatron.fused_linear_ce import prepare_output_processor

    args = actor_args()
    args.enable_mtp_training = True
    args.sft_chunked_logits = False
    monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", args)
    _, model = make_gpt(mtp=True)
    assert model.config.mtp_num_layers == 1 and model.mtp_process
    tokens = torch.randint(1034, (1, 24), device="cuda")
    results = []
    for enabled in (False, True):
        args.fused_linear_ce = enabled
        model.zero_grad(set_to_none=True)
        hooks = prepare_output_processor(
            model, args, {"total_lengths": [24], "response_lengths": [8]}, consumer="policy_loss"
        )
        assert not hooks
        output = model(
            input_ids=tokens, position_ids=None, attention_mask=None, labels=tokens.roll(-1, dims=-1), **hooks
        )
        output.float().square().mean().backward()
        grads = {name: param.grad.clone() for name, param in model.named_parameters() if param.grad is not None}
        assert any("mtp" in name for name in grads)
        results.append((output.detach(), grads))
    torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)
