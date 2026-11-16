import os
from datetime import timedelta

import torch
import torch.distributed as dist
from megatron.core import parallel_state, tensor_parallel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_submodules
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig


rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl", device_id=torch.device("cuda", rank), timeout=timedelta(seconds=180))
parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=2, expert_model_parallel_size=8, expert_tensor_parallel_size=1
)
tensor_parallel.model_parallel_cuda_manual_seed(1234)
common = dict(
    num_layers=1,
    hidden_size=2560,
    num_attention_heads=20,
    num_moe_experts=512,
    moe_ffn_hidden_size=640,
    moe_shared_expert_intermediate_size=640,
    moe_router_topk=10,
    moe_grouped_gemm=True,
    gated_linear_unit=True,
    activation_func=torch.nn.functional.silu,
    gradient_accumulation_fusion=True,
    add_bias_linear=False,
    bf16=True,
    params_dtype=torch.bfloat16,
    tensor_model_parallel_size=2,
    expert_model_parallel_size=8,
    expert_tensor_parallel_size=1,
    sequence_parallel=True,
    moe_token_dispatcher_type="alltoall",
    moe_router_load_balancing_type="none",
    moe_aux_loss_coeff=0.0,
    moe_router_dtype="fp32",
    moe_shared_expert_overlap=False,
)
spec = get_submodules(get_gpt_layer_with_transformer_engine_submodules(num_experts=512, moe_grouped_gemm=True).mlp)
base = MoELayer(TransformerConfig(**common), spec).cuda()
mok = MoELayer(
    TransformerConfig(
        **common,
        moe_megakernel_backend="mok",
        moe_megakernel_backend_config={"shared_expert_mode": "native", "fwd_num_comm_sms": 2, "bwd_num_comm_sms": 2},
    ),
    spec,
).cuda()
mok.load_state_dict(base.state_dict(), strict=True)
assert not mok.megakernel_fuses_shared_expert
for model in (base, mok):
    model.train()
    for p in model.parameters():
        if p.requires_grad:
            p.main_grad = torch.zeros_like(p, dtype=torch.float32)
            p.grad_added_to_main_grad = False
# Equal native/MoK input per rank; SP shards carry distinct token rows.
gen = torch.Generator(device="cuda").manual_seed(998 + rank)
x = torch.randn(512, 1, 2560, device="cuda", dtype=torch.bfloat16, generator=gen) * 0.1
dy = torch.randn(x.shape, device="cuda", dtype=x.dtype, generator=gen) * 0.01
mask = torch.zeros(1, 1024, device="cuda", dtype=torch.bool)
mask[:, -16:] = True
for recompute in (False, True):
    outputs = []
    inputs = []
    grads = []
    for model in (base, mok):
        for p in model.parameters():
            p.grad = None
            if hasattr(p, "main_grad"):
                p.main_grad.zero_()
                p.grad_added_to_main_grad = False
        xx = x.detach().clone().requires_grad_(True)

        def forward(z):
            return model(z, padding_mask=mask)[0]

        yy = tensor_parallel.checkpoint(forward, False, xx) if recompute else forward(xx)
        yy.backward(dy)
        torch.cuda.synchronize()
        outputs.append(yy.detach())
        inputs.append(xx.grad.detach())
        captured = {}
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            gradient = parameter.main_grad if getattr(parameter, "grad_added_to_main_grad", False) else parameter.grad
            assert gradient is not None, ("missing gradient", name)
            assert torch.isfinite(gradient).all(), ("nonfinite gradient", name)
            assert gradient.abs().max() > 0, ("zero gradient", name)
            captured[name] = gradient.detach().clone()
        assert any(n.startswith("router.") for n in captured)
        assert any(n.startswith("experts.") for n in captured)
        assert any(n.startswith("shared_experts.") for n in captured)
        grads.append(captured)

    def check(name, a, b):
        assert torch.isfinite(a).all() and torch.isfinite(b).all(), name
        rel = (a.float() - b.float()).norm() / a.float().norm().clamp_min(1e-12)
        maxerr = (a.float() - b.float()).abs().max()
        print(
            f"rank={rank} recompute={recompute} {name} rel_l2={rel.item():.6g} max_abs={maxerr.item():.6g}", flush=True
        )
        assert rel < 0.025, (name, rel.item())

    check("output", outputs[0], outputs[1])
    check("input_grad", inputs[0], inputs[1])
    assert grads[0].keys() == grads[1].keys(), (grads[0].keys(), grads[1].keys())
    for name in grads[0]:
        check(name, grads[0][name], grads[1][name])
    dist.barrier()
if rank == 0:
    print("PASS native-vs-MoK TP2/SP/ETP1/EP8 H2560 I640 E512 K10 shared-native padding recompute", flush=True)
dist.destroy_process_group()
