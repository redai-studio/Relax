# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise the actual patched DSA method and FA3 on one Hopper GPU.

Existing backend tests do not cover NaNs in physical V padding outside varlen
metadata. These small attention calls require one GPU, without a model load or
process group. SGLANG_TEST_SOURCE_ROOT can select a patched SGLang source tree.
"""

import importlib.util
import sys
from collections.abc import Callable, Iterator
from types import SimpleNamespace

import pytest
import torch

from tests.backends.sglang.source_utils import sglang_source_path


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a Hopper GPU and SGLang FA3")


@pytest.fixture
def dense_mha() -> Iterator[Callable]:
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("Regression targets the Hopper FA3 dense MHA path")
    path = sglang_source_path("srt/layers/attention/dsa_backend.py")
    name = "_relax_test_dsa_padding_backend"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module.DeepseekSparseAttnBackend._forward_standard_mha
    finally:
        sys.modules.pop(name, None)


@pytest.mark.parametrize("q_lengths,k_lengths,q_rows,kv_rows", [([1], [1], 4, 4), ([2, 3], [5, 7], 8, 16)])
@pytest.mark.parametrize("poison", ["v_nan", "qkv_nan", "v_inf"])
def test_dense_mha_padding_preserves_valid_output(
    dense_mha: Callable, q_lengths: list[int], k_lengths: list[int], q_rows: int, kv_rows: int, poison: str
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(42)
    q = torch.randn(q_rows, 16, 256, device="cuda", dtype=torch.bfloat16, generator=generator)
    k = torch.randn(kv_rows, 16, 256, device="cuda", dtype=torch.bfloat16, generator=generator)
    backing = torch.randn(kv_rows, 16, 448, device="cuda", dtype=torch.bfloat16, generator=generator)
    v = backing[..., 192:]
    nq, nk = sum(q_lengths), sum(k_lengths)
    q[nq:] = 0
    k[nk:] = 0
    v[nk:] = 0
    layer = SimpleNamespace(
        tp_q_head_num=16, tp_k_head_num=16, tp_v_head_num=16, head_dim=256, v_head_dim=256, scaling=256**-0.5
    )
    metadata = SimpleNamespace(
        cu_seqlens_q=torch.tensor([0, *torch.tensor(q_lengths).cumsum(0).tolist()], device="cuda", dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, *torch.tensor(k_lengths).cumsum(0).tolist()], device="cuda", dtype=torch.int32),
        max_seq_len_q=max(q_lengths),
        max_seq_len_k=max(k_lengths),
    )
    backend, batch = SimpleNamespace(device_sm_major=9), SimpleNamespace(batch_size=len(q_lengths))
    expected = dense_mha(backend, q, k, v, layer, batch, metadata)[:nq].clone()
    assert torch.isfinite(expected).all()
    v[nk:] = float("inf") if poison == "v_inf" else float("nan")
    if poison == "qkv_nan":
        q[nq:] = float("nan")
        k[nk:] = float("nan")
    before = backing.clone()
    actual = dense_mha(backend, q, k, v, layer, batch, metadata)[:nq]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(backing[:nk], before[:nk], rtol=0, atol=0)
    torch.testing.assert_close(backing[..., :192], before[..., :192], rtol=0, atol=0)
    # A real NaN in a valid value must remain visible, including with no padding.
    v[0, 0, 0] = float("nan")
    actual = dense_mha(backend, q[:nq], k[:nk], v[:nk], layer, batch, metadata)
    assert torch.isnan(actual).any()
    assert torch.isnan(v[0, 0, 0])
