# Copyright (c) 2026 Relax Authors. All Rights Reserved.
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path


if os.environ.get("MEGATRON_ROOT"):
    sys.path.insert(0, os.environ["MEGATRON_ROOT"])
import torch
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net import GatedDeltaNet, GatedDeltaNetSubmodules
from megatron.core.transformer import TransformerConfig


spec = importlib.util.spec_from_file_location(
    "qwen35_prefix_state", str(Path(__file__).resolve().parents[2] / "relax/utils/training/qwen35_prefix_state.py")
)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


class Linear(torch.nn.Module):
    def __init__(self, a, b, config, **kw):
        super().__init__()
        self.linear = torch.nn.Linear(a, b, bias=False, device="cuda", dtype=config.params_dtype)

    def forward(self, x):
        return self.linear(x), None


class Norm(torch.nn.Module):
    def __init__(self, config, hidden_size, eps):
        super().__init__()
        self.norm = torch.nn.RMSNorm(hidden_size, eps=eps, device="cuda", dtype=config.params_dtype)

    def forward(self, x):
        return self.norm(x)


rendezvous = tempfile.TemporaryDirectory()
torch.distributed.init_process_group("nccl", init_method="file://" + rendezvous.name + "/init", rank=0, world_size=1)
pg = torch.distributed.group.WORLD
cfg = TransformerConfig(
    num_layers=1,
    hidden_size=64,
    num_attention_heads=2,
    params_dtype=torch.bfloat16,
    perform_initialization=False,
    linear_conv_kernel_dim=4,
    linear_key_head_dim=32,
    linear_value_head_dim=32,
    linear_num_key_heads=2,
    linear_num_value_heads=2,
    activation_func=torch.nn.functional.silu,
)
layer = GatedDeltaNet(
    cfg,
    GatedDeltaNetSubmodules(in_proj=Linear, out_norm=Norm, out_proj=Linear),
    layer_number=1,
    pg_collection=ProcessGroupCollection(tp=pg, cp=pg),
)
with torch.no_grad():
    layer.A_log.fill_(0)
    layer.dt_bias.fill_(1)
x = torch.randn(128, 1, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
y, _ = layer(x, None)
with torch.no_grad():
    _, _, state = m.gdn_forward(layer, x[:64])
z, _, _ = m.gdn_forward(layer, x[64:], state)
error = (y[64:].float() - z.float()).abs().max().item()
torch.testing.assert_close(y[64:], z, atol=0.03, rtol=0.04)
z.float().square().mean().backward()
assert x.grad[:64].count_nonzero() == 0 and x.grad[64:].count_nonzero() > 0
print(
    json.dumps(
        {
            "scope": "actual Megatron GDN class; local torch projections TP1; not actor",
            "max_abs_error": error,
            "prefix_input_grad_zero": True,
            "suffix_grad_nonzero": True,
        }
    )
)
initial = {k: v.detach().clone() for k, v in layer.state_dict().items()}


def accumulate(reuse):
    layer.load_state_dict(initial)
    layer.zero_grad(set_to_none=True)
    optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-4)
    previous = None
    previous_end = 0
    outputs = []
    for end in (32, 64):
        with torch.no_grad():
            start = previous_end if reuse else 0
            _, _, prefix = m.gdn_forward(layer, x.detach()[start:end], previous if reuse else None)
        output, _, _ = m.gdn_forward(layer, x.detach()[end : end + 32], prefix)
        outputs.append(output.detach())
        output.float().square().mean().backward()
        previous, previous_end = prefix, end
    grads = [p.grad.detach().clone() for p in layer.parameters()]
    optimizer.step()
    return outputs, grads, [p.detach().clone() for p in layer.parameters()]


detached, reused = accumulate(False), accumulate(True)
for left, right in zip(detached, reused):
    for a, b in zip(left, right):
        torch.testing.assert_close(a, b, atol=0.03, rtol=0.04)
print(
    json.dumps(
        {
            "scope": "actual GDN TP1 reduced model",
            "detached_vs_reuse_outputs_gradients_adamw": "passed",
            "atol": 0.03,
            "rtol": 0.04,
        }
    )
)
# Check the decoder executor's checkpoint closure across two pending windows.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
decoder_adapter = importlib.import_module("relax.utils.training.qwen35_prefix_decoder")
decoder_chunk = decoder_adapter.decoder_chunk
validate_decoder = decoder_adapter.validate_decoder


class ResidualLayer(torch.nn.Module):
    def __init__(self, mixer):
        super().__init__()
        self.self_attention = mixer

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        out, _ = self.self_attention(hidden_states, attention_mask, **kwargs)
        return hidden_states + out, None


class ProbeDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = cfg
        self.layers = torch.nn.ModuleList([ResidualLayer(layer)])
        self.final_layernorm = None


decoder = ProbeDecoder()
cfg.hidden_dropout = 0.0
cfg.attention_dropout = 0.0
validate_decoder(decoder)
layer.load_state_dict(initial)
layer.zero_grad(set_to_none=True)
cfg.recompute_granularity = "full"
losses = []
for end in (32, 64):
    with torch.no_grad():
        _, state = decoder_chunk(decoder, x.detach()[:end], None, build_state=True)
    hidden = x.detach()[end : end + 32].requires_grad_()
    out, _ = decoder_chunk(decoder, hidden, None, state)
    losses.append(out.float().square().mean())
# Backward after a second distinct state was used: replay must retain state 1.
sum(losses).backward()
checkpoint_grads = [p.grad.detach().clone() for p in layer.parameters()]
layer.zero_grad(set_to_none=True)
cfg.recompute_granularity = None
for end in (32, 64):
    with torch.no_grad():
        _, state = decoder_chunk(decoder, x.detach()[:end], None, build_state=True)
    out, _ = decoder_chunk(decoder, x.detach()[end : end + 32], None, state)
    out.float().square().mean().backward()
for expected, parameter in zip(checkpoint_grads, layer.parameters()):
    torch.testing.assert_close(expected, parameter.grad, atol=0.003, rtol=0.01)
print(
    json.dumps(
        {
            "scope": "decoder dispatcher with real GDN and residual probe layer",
            "checkpoint_two_pending_windows": "passed",
        }
    )
)

torch.distributed.destroy_process_group()
