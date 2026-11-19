# Copyright (c) 2026 Relax Authors. All Rights Reserved.
import json

import torch
from fla.modules.convolution import causal_conv1d
from fla.ops.gated_delta_rule import chunk_gated_delta_rule


torch.manual_seed(12)
device = "cuda:0"
x = torch.randn(1, 128, 96, device=device, dtype=torch.bfloat16, requires_grad=True)
w = torch.randn(96, 4, device=device, dtype=torch.bfloat16, requires_grad=True)


def conv(a, s=None):
    return causal_conv1d(a, w, activation="silu", initial_state=s, output_final_state=True)


y, _ = conv(x)
with torch.no_grad():
    p, st = conv(x[:, :64])
st_copy = st.clone()
s, _ = conv(x[:, 64:], st.detach().clone())
conv_error = (y[:, 64:].float() - s.float()).abs().max().item()
torch.testing.assert_close(y[:, 64:], s, atol=0.04, rtol=0.02)
s.float().square().mean().backward()
assert x.grad[:, :64].count_nonzero() == 0 and x.grad[:, 64:].count_nonzero() > 0
assert torch.equal(st, st_copy)
q = torch.nn.functional.normalize(
    torch.randn(1, 128, 2, 32, device=device, dtype=torch.bfloat16), dim=-1
).requires_grad_()
k = torch.nn.functional.normalize(torch.randn_like(q), dim=-1).requires_grad_()
v = torch.randn_like(q, requires_grad=True)
g = (-torch.rand(1, 128, 2, device=device)).requires_grad_()
b = torch.rand(1, 128, 2, device=device, dtype=torch.bfloat16, requires_grad=True)


def scan(lo, hi, st=None):
    return chunk_gated_delta_rule(
        q[:, lo:hi],
        k[:, lo:hi],
        v[:, lo:hi],
        g=g[:, lo:hi],
        beta=b[:, lo:hi],
        initial_state=st,
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
    )


full, _ = scan(0, 128)
with torch.no_grad():
    _, st = scan(0, 64)
snapshot = st.clone()
suffix, _ = scan(64, 128, st.detach().clone())
scan_error = (full[:, 64:].float() - suffix.float()).abs().max().item()
torch.testing.assert_close(full[:, 64:], suffix, atol=0.02, rtol=0.03)
suffix.float().square().mean().backward()
assert q.grad[:, :64].count_nonzero() == 0 and q.grad[:, 64:].count_nonzero() > 0
assert torch.equal(st, snapshot)
with torch.no_grad():
    _, a = scan(0, 32)
    _, a = scan(32, 64, a.clone())
torch.testing.assert_close(st, a, atol=0.02, rtol=0.03)
print(
    json.dumps(
        {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "dtype": "bfloat16",
            "conv_max_abs_error": conv_error,
            "gdn_max_abs_error": scan_error,
            "suffix_grad_nonzero": True,
            "prefix_input_grad_zero": True,
            "state_unmodified": True,
            "scope": "installed FLA kernels only; not actor/model validation",
        }
    )
)
