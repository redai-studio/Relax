# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise the TE deployment patch without depending on the installed TE
version or invoking distributed kernels."""

import os
from pathlib import Path
from textwrap import dedent

import pytest


torch = pytest.importorskip("torch")
PATCH = Path(__file__).resolve().parents[3] / "docker/patch/transformer_engine/2.18.0-cp-padding-no-host-sync.patch"


@pytest.fixture(scope="module")
def padding_code():
    # This single-hunk patch replaces only the padding body. Test those exact
    # added statements: CI may use an older TE than the target Docker image.
    # Reading the installed module would test stale code (or find no block).
    lines = PATCH.read_text().splitlines(keepends=True)
    assert sum(line.startswith("@@ ") for line in lines) == 1
    added_code = dedent("".join(line[1:] for line in lines if line.startswith("+") and not line.startswith("+++")))
    assert added_code.strip(), "TE padding patch has no replacement code"
    return compile(added_code, str(PATCH), "exec")


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                os.environ.get("RELAX_TEST_K3_PARALLEL_GPU") != "1" or not torch.cuda.is_available(),
                reason="Requires explicit RELAX_TEST_K3_PARALLEL_GPU=1 and an idle GPU",
            ),
        ),
    ],
)
@pytest.mark.parametrize("q_end,kv_end", [(0, 0), (7, 9), (3, 5), (7, 0)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_te_cp_padding_zeros_tail_without_host_scalar_reads(padding_code, monkeypatch, device, q_end, kv_end, dtype):
    gradients = [torch.randn(2, n, 3, device=device, dtype=dtype).transpose(0, 1) for n in (7, 9, 9)]
    for grad, end in zip(gradients, (q_end, kv_end, kv_end)):
        grad[end:] = float("nan")
        grad[end::2] = float("inf")
    expected = [grad.clone() for grad in gradients]
    for grad, end in zip(expected, (q_end, kv_end, kv_end)):
        grad[end:] = 0
    namespace = dict(
        torch=torch,
        dq=gradients[0],
        dk=gradients[1],
        dv=gradients[2],
        cu_seqlens_q_padded=torch.tensor([0, q_end], dtype=torch.int32, device=device),
        cu_seqlens_kv_padded=torch.tensor([0, kv_end], dtype=torch.int32, device=device),
    )

    def reject_scalar_read(*args, **kwargs):
        raise AssertionError("Padding must not convert a tensor boundary to a host scalar")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "__index__", reject_scalar_read)
        patch.setattr(torch.Tensor, "item", reject_scalar_read)
        exec(padding_code, namespace)
    for grad, reference in zip(gradients, expected):
        torch.testing.assert_close(grad, reference, rtol=0, atol=0)
