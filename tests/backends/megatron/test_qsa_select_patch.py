# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import math
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture(autouse=True)
def _selector_environment(monkeypatch):
    # Exercise the static fused dispatcher and a genuinely native reference,
    # regardless of the optimization settings exported by the training shell.
    monkeypatch.setenv("RELAX_QWEN38_QSA_CPU_LAYOUT", "0")
    monkeypatch.setenv("RELAX_QWEN38_QSA_FIXED_POOL", "0")
    monkeypatch.setenv("RELAX_QWEN38_QSA_FUSED_SELECTOR", "1")


def _selector_module():
    from megatron.core.transformer.experimental_attention_variant import qsa

    module = SimpleNamespace(compress_ratio=4, block_topk=512)
    module._relax_native_pool_keys = qsa.QSAIndexer._relax_native_pool_keys.__get__(module)
    module._relax_native_select_blocks = qsa.QSAIndexer._relax_native_select_blocks.__get__(module)
    module._select_blocks = qsa.QSAIndexer._select_blocks.__get__(module)
    return module


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Fused QSA bitset packing requires CUDA")
@pytest.mark.parametrize(
    "columns,k", [(7, 1), (8, 8), (16, 16), (24, 24), (32, 32), (33, 33), (4096, 511), (32768, 512)]
)
@pytest.mark.parametrize("kind", ["contended", "nonfinite", "all_invalid"])
def test_topk_packing_preserves_rows_tail_and_finite_filter(columns, k, kind):
    from relax.models.qwen3_8 import qsa_select as opt

    rows = 7
    # Adjacent IDs make different warps update the same words; the final row
    # reaches the highest legal block and the partial last byte/word.
    ids = torch.arange(k, device="cuda").expand(rows, -1).clone()
    ids[-1] = columns - k + ids[-1]
    values = torch.ones(rows, k, device="cuda")
    if kind == "nonfinite":
        values[:, ::4] = float("nan")
        values[:, 1::4] = float("inf")
        values[:, 2::4] = float("-inf")
        values[:, 3::4] = -0.0
    if kind == "all_invalid":
        values.fill_(float("-inf"))
    nbytes = (columns + 8) // 8
    expected = torch.zeros(rows, nbytes, device="cuda", dtype=torch.int32)
    valid = torch.isfinite(values)
    expected.scatter_add_(1, ids >> 3, ((1 << (ids & 7)) * valid).int())
    buffer = torch.full((rows + 2, nbytes), 173, device="cuda", dtype=torch.uint8)
    for _ in range(5):
        buffer[1:-1].fill_(255)
        opt.pack_topk(ids, values, buffer[1:-1])
        assert torch.equal(buffer[1:-1], expected.byte())
        assert (buffer[[0, -1]] == 173).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Fused QSA bitset packing requires CUDA")
def test_topk_packing_supports_empty_selection_and_independent_streams():
    from relax.models.qwen3_8 import qsa_select as opt

    output = torch.full((3, 5), 255, device="cuda", dtype=torch.uint8)
    opt.pack_topk(torch.empty(3, 0, device="cuda", dtype=torch.int64), torch.empty(3, 0, device="cuda"), output)
    assert not output.any()
    ids = torch.arange(64, device="cuda").expand(17, -1).contiguous()
    values = torch.ones_like(ids, dtype=torch.float32)
    ids_per_stream = [ids, ids + 64]
    outputs = [torch.empty(17, 17, device="cuda", dtype=torch.uint8) for _ in range(2)]
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for stream, stream_ids, out in zip(streams, ids_per_stream, outputs):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            opt.pack_topk(stream_ids, values, out)
    for stream in streams:
        torch.cuda.current_stream().wait_stream(stream)
    for i, out in enumerate(outputs):
        expected = torch.zeros_like(out)
        expected[:, i * 8 : (i + 1) * 8] = 255
        assert torch.equal(out, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Fused QSA candidate packing requires CUDA")
@pytest.mark.parametrize("rows,columns,nbytes", [(7, 1, 1), (17, 33, 9), (489, 512, 1025), (3, 511, 4097)])
@pytest.mark.parametrize("kind", ["random", "negative_zero", "nonfinite", "overflow"])
def test_all_candidate_packing_preserves_finite_causal_scores_and_padding(rows, columns, nbytes, kind):
    from relax.models.qwen3_8 import qsa_select as opt

    torch.manual_seed(83)
    scores = torch.randn(rows, 4, columns, device="cuda")
    if kind == "negative_zero":
        scores.fill_(-0.0)
    if kind == "nonfinite":
        scores.flatten()[::7] = float("nan")
        scores.flatten()[1::7] = float("inf")
        scores.flatten()[2::7] = float("-inf")
    if kind == "overflow":
        # Finite heads can sum to infinity before multiplication by the scale.
        scores.fill_(torch.finfo(torch.float32).max / 2)
    positions = torch.arange(rows, device="cuda", dtype=torch.int32) * 79
    processed = opt.score_postprocess(scores, positions, 4, 128)
    top = torch.topk(processed, columns, dim=-1, sorted=False)
    expected = torch.empty(rows, nbytes, device="cuda", dtype=torch.uint8)
    opt.pack_topk(top.indices, top.values, expected)
    buffer = torch.full((rows + 2, nbytes), 173, device="cuda", dtype=torch.uint8)
    buffer[1:-1].fill_(255)
    opt.pack_all_candidates(scores, positions, 4, 128, buffer[1:-1])
    assert torch.equal(buffer[1:-1], expected)
    assert (buffer[[0, -1]] == 173).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Fused QSA selector requires CUDA")
def test_all_candidate_shortcut_only_applies_when_topk_retains_every_column(monkeypatch):
    from relax.models.qwen3_8 import qsa_select as opt

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (0, 0))
    lengths = (2048, 2052, 4096)
    total, columns = sum(lengths), max(lengths) // 4
    torch.manual_seed(91)
    q = torch.randn(total, 4, 128, device="cuda", dtype=torch.bfloat16)
    keys = torch.randn(len(lengths), columns, 128, device="cuda", dtype=torch.bfloat16)
    docs = torch.repeat_interleave(
        torch.arange(len(lengths), device="cuda", dtype=torch.int32), torch.tensor(lengths, device="cuda")
    )
    positions = torch.cat([torch.arange(n, device="cuda", dtype=torch.int32) for n in lengths])
    valid = torch.arange(columns, device="cuda")[None, :] < torch.tensor(lengths, device="cuda")[:, None] // 4
    module = SimpleNamespace(compress_ratio=4, block_topk=512)
    original_pack = opt.pack_all_candidates
    shortcut_widths = []
    topk_widths = []
    original_topk = torch.topk

    def reference_pack(scores, positions, ratio, dim, output):
        processed = opt.score_postprocess(scores, positions, ratio, dim)
        top = original_topk(processed, scores.shape[-1], dim=-1, sorted=False)
        opt.pack_topk(top.indices, top.values, output)

    def tracked_pack(scores, *args):
        shortcut_widths.append(scores.shape[-1])
        return original_pack(scores, *args)

    def tracked_topk(scores, *args, **kwargs):
        topk_widths.append(scores.shape[-1])
        return original_topk(scores, *args, **kwargs)

    token = opt._layout.set((0, 2048, 4100, total))
    try:
        monkeypatch.setattr(opt, "pack_all_candidates", reference_pack)
        expected, expected_flag = opt.select_blocks(module, q, keys, valid, docs, positions)
        monkeypatch.setattr(opt, "pack_all_candidates", tracked_pack)
        monkeypatch.setattr(torch, "topk", tracked_topk)
        actual, flag = opt.select_blocks(module, q, keys, valid, docs, positions)
    finally:
        opt._layout.reset(token)
    assert shortcut_widths == [512]
    assert topk_widths == [513, 1024]
    assert flag == expected_flag and torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Fused QSA TopK requires CUDA")
@pytest.mark.parametrize("columns", [513, 1025, 4097, 8193, 16384, 16385, 32768])
@pytest.mark.parametrize(
    "kind",
    [
        "random",
        "zeros",
        "ties",
        "negative_zero",
        "nan511",
        "nan512",
        "nan513",
        "inf511",
        "inf512",
        "inf513",
        "overflow",
        "masked_nan",
        "subnormal",
    ],
)
def test_fused_topk_preserves_nonfinite_slots_causal_rows_and_padding(columns, kind):
    from relax.models.qwen3_8 import qsa_select as opt

    torch.manual_seed(553)
    positions = torch.tensor(
        [0, 2, 3, 2047, 2048, 2050, 2051, 2052, 4 * columns - 1, 4 * columns], device="cuda", dtype=torch.int32
    )
    rows = len(positions)
    scores = torch.randn(rows, 4, columns, device="cuda")
    if kind == "zeros":
        scores.zero_()
    if kind == "ties":
        scores = scores.sign()
    if kind == "negative_zero":
        scores.fill_(-0.0)
    if kind.startswith(("nan", "inf")):
        scores.abs_()
        scores[:, 0, : int(kind[3:])] = float("nan") if kind.startswith("nan") else float("inf")
    if kind == "overflow":
        scores.fill_(torch.finfo(torch.float32).max / 2)
    if kind == "masked_nan":
        mask = torch.arange(columns, device="cuda")[None, :] >= ((positions.long() + 1) // 4)[:, None]
        scores[:, 0, :].masked_fill_(mask, float("nan"))
    if kind == "subnormal":
        scores *= torch.finfo(torch.float32).tiny
    processed = opt.score_postprocess(scores, positions, 4, 128)
    top = torch.topk(processed, 512, dim=-1, sorted=False)
    expected = torch.empty(rows, 4097, device="cuda", dtype=torch.uint8)
    opt.pack_topk(top.indices, top.values, expected)
    buffer = torch.full((rows + 2, 4097), 173, device="cuda", dtype=torch.uint8)
    buffer[1:-1].fill_(255)
    opt.score_topk_pack(scores, positions, 4, 128, 512, buffer[1:-1])
    assert torch.equal(buffer[1:-1], expected)
    assert (buffer[[0, -1]] == 173).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA radix TopK comparison requires CUDA")
@pytest.mark.parametrize("columns,k", [(513, 1), (513, 512), (2048, 512), (8193, 512), (16385, 512), (32768, 512)])
@pytest.mark.parametrize("kind", ["random", "ties", "signed_zero", "nan_payloads"])
def test_topk_mask_matches_chosen_ids_before_nonfinite_filter(columns, k, kind):
    import triton
    import triton.language as tl

    from relax.models.qwen3_8 import qsa_select as opt

    @triton.jit
    def chosen_kernel(scores, output, columns: tl.constexpr, k: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        col = tl.arange(0, BLOCK)
        valid = col < columns
        value = tl.load(scores + row * columns + col, valid, other=0)
        raw = value.to(tl.uint32, bitcast=True)
        keys = raw ^ tl.where((raw & 0x80000000) != 0, 0xFFFFFFFF, 0x80000000).to(tl.uint32)
        keys = tl.where(value == value, keys, 0xFFFFFFFF).to(tl.uint32)
        chosen = opt._topk_mask(keys, valid, k)
        tl.store(output + row * columns + col, chosen, valid)

    torch.manual_seed(203)
    scores = torch.randn(7, columns, device="cuda")
    if kind == "ties":
        scores = scores.sign()
    if kind == "signed_zero":
        scores.zero_()
        scores[:, ::2] = -0.0
    if kind == "nan_payloads":
        # Positive/negative NaNs with different payloads must share one key.
        scores.view(torch.int32)[:, ::3] = 0x7FC00001
        scores.view(torch.int32)[:, 1::3] = -0x003FFFFE
    top = torch.topk(scores, k, dim=-1, sorted=False)
    expected = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, top.indices, True)
    actual = torch.empty_like(expected)
    chosen_kernel[(7,)](
        scores,
        actual,
        columns,
        k,
        triton.next_power_of_2(columns),
        num_warps=4 if columns <= 8192 else (8 if columns <= 16384 else 16),
    )
    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Fused QSA selector dispatch requires CUDA")
@pytest.mark.parametrize("capability,limit", [((9, 0), 16384), ((10, 3), 32768), ((10, 0), 0)])
def test_selector_limits_fused_topk_to_validated_shapes_and_architecture(monkeypatch, capability, limit):
    from relax.models.qwen3_8 import qsa_select as opt

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)
    monkeypatch.setattr(opt, "_TILE_BYTES", 32 << 20)
    lengths = (4 * (max(16384, limit) + 1),)
    total, columns = lengths[0], lengths[0] // 4
    q = torch.zeros(total, 4, 128, device="cuda", dtype=torch.bfloat16)
    keys = torch.zeros(1, columns, 128, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(total, device="cuda", dtype=torch.int32)
    docs = torch.zeros_like(positions)
    valid = torch.ones(1, columns, device="cuda", dtype=torch.bool)
    fused_widths, aten_widths = [], []

    def record_fused(scores, positions, ratio, dim, k, output):
        fused_widths.append(scores.shape[-1])
        output.zero_()

    def record_topk(scores, k, **kwargs):
        aten_widths.append(scores.shape[-1])
        return SimpleNamespace(indices=torch.empty(0, device="cuda"), values=torch.empty(0, device="cuda"))

    monkeypatch.setattr(opt, "score_topk_pack", record_fused)
    monkeypatch.setattr(torch, "topk", record_topk)
    monkeypatch.setattr(opt, "pack_topk", lambda ids, values, output: output.zero_())
    opt.select_blocks(SimpleNamespace(compress_ratio=4, block_topk=512), q, keys, valid, docs, positions)
    if limit:
        assert fused_widths and all(512 < n <= limit for n in fused_widths)
        assert aten_widths == [limit + 1]
    else:
        assert not fused_widths and min(aten_widths) > 512 and max(aten_widths) == 16385


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Fused QSA selector requires CUDA")
@pytest.mark.parametrize("rows,columns", [(1, 257), (17, 8192), (489, 8192), (7, 33)])
@pytest.mark.parametrize("kind", ["random", "ties", "special", "negative_zero"])
def test_fused_score_and_bitset_match_aten_bitwise(rows, columns, kind):
    from relax.models.qwen3_8 import qsa_select as opt

    torch.manual_seed(1234)
    scores = torch.randn(rows, 4, columns, device="cuda", dtype=torch.float32)
    if kind == "ties":
        scores.round_()
    if kind == "negative_zero":
        scores.fill_(-0.0)
    if kind == "special":
        scores.reshape(-1)[:4] = torch.tensor([float("nan"), float("inf"), float("-inf"), -0.0], device="cuda")
    pos = torch.arange(rows, device="cuda", dtype=torch.int32) * 997 + 7
    ref = scores.relu().sum(1) / math.sqrt(128)
    ref.masked_fill_(torch.arange(columns, device="cuda")[None, :] >= ((pos.long() + 1) // 4)[:, None], float("-inf"))
    out = opt.score_postprocess(scores, pos, 4, 128)
    finite = torch.isfinite(ref)
    assert torch.equal(out[finite].view(torch.int32), ref[finite].view(torch.int32)), (
        rows,
        columns,
        kind,
        (out[finite] - ref[finite]).abs().max(),
    )
    assert torch.equal(torch.isnan(ref), torch.isnan(out)) and torch.equal(torch.isinf(ref), torch.isinf(out))
    top = torch.topk(ref, min(512, columns), dim=-1)
    nbytes = (columns + 8) // 8
    expected = torch.zeros(rows, nbytes, device="cuda", dtype=torch.int32)
    valid = torch.isfinite(top.values)
    ids = top.indices.masked_fill(~valid, 0)
    expected.scatter_add_(1, ids >> 3, ((1 << (ids & 7)) * valid.to(ids.dtype)).int())
    packed = torch.empty(rows, nbytes, device="cuda", dtype=torch.uint8)
    opt.pack_topk(top.indices, top.values, packed)
    assert torch.equal(packed, expected.byte())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Fused QSA selector requires CUDA")
@pytest.mark.parametrize("lengths", [(2052, 4096, 1028), (8192,), (2053, 4103, 1031)])
@pytest.mark.parametrize("kind", ["random", "zeros", "ties"])
def test_fused_selector_matches_route_bytes(lengths, kind):
    from relax.models.qwen3_8 import qsa_select as opt

    torch.manual_seed(1234)
    total = sum(lengths)
    columns = max(lengths) // 4
    q = torch.randn(total, 4, 128, device="cuda", dtype=torch.bfloat16)
    keys = torch.randn(len(lengths), columns, 128, device="cuda", dtype=torch.bfloat16)
    if kind == "zeros":
        q.zero_()
    if kind == "ties":
        q = q.sign()
        keys = keys.sign()
    docs = torch.repeat_interleave(
        torch.arange(len(lengths), device="cuda", dtype=torch.int32), torch.tensor(lengths, device="cuda")
    )
    pos = torch.cat([torch.arange(n, device="cuda", dtype=torch.int32) for n in lengths])
    valid = torch.arange(columns, device="cuda")[None, :] < torch.tensor(lengths, device="cuda")[:, None] // 4
    module = _selector_module()
    expected, flag = module._relax_native_select_blocks(q, keys, valid, docs, pos)
    offsets = [0]
    for n in lengths:
        offsets.append(offsets[-1] + n)
    token = opt._layout.set(tuple(offsets))
    try:
        actual, actual_flag = module._select_blocks(q, keys, valid, docs, pos)
    finally:
        opt._layout.reset(token)
    assert flag == actual_flag and torch.equal(expected, actual), (lengths, kind, (expected != actual).sum().item())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Fused QSA selector requires CUDA")
@pytest.mark.parametrize("lengths", [(2052, 4096, 1028), (12288,), (3, 4099, 1031), (8192, 2049)])
@pytest.mark.parametrize("tile_bytes", [1 << 20, 1 << 22])
def test_fused_selector_narrows_tiles_without_changing_route_bytes(lengths, tile_bytes, monkeypatch):
    """A small tile budget forces many tiles per document, so each one narrows
    its GEMM and TopK to a different visible width."""
    from megatron.core.transformer.experimental_attention_variant import qsa

    from relax.models.qwen3_8 import qsa_select as opt

    monkeypatch.setattr(qsa, "_QSA_SELECT_TILE_BYTES", tile_bytes)
    monkeypatch.setattr(opt, "_TILE_BYTES", tile_bytes)
    torch.manual_seed(99)
    total = sum(lengths)
    columns = max(lengths) // 4
    q = torch.randn(total, 4, 128, device="cuda", dtype=torch.bfloat16)
    keys = torch.randn(len(lengths), columns, 128, device="cuda", dtype=torch.bfloat16)
    docs = torch.repeat_interleave(
        torch.arange(len(lengths), device="cuda", dtype=torch.int32), torch.tensor(lengths, device="cuda")
    )
    pos = torch.cat([torch.arange(n, device="cuda", dtype=torch.int32) for n in lengths])
    valid = torch.arange(columns, device="cuda")[None, :] < torch.tensor(lengths, device="cuda")[:, None] // 4
    module = _selector_module()
    expected, flag = module._relax_native_select_blocks(q, keys, valid, docs, pos)
    offsets = [0]
    for n in lengths:
        offsets.append(offsets[-1] + n)
    token = opt._layout.set(tuple(offsets))
    try:
        actual, actual_flag = module._select_blocks(q, keys, valid, docs, pos)
    finally:
        opt._layout.reset(token)
    assert flag == actual_flag and torch.equal(expected, actual), (
        lengths,
        tile_bytes,
        (expected != actual).sum().item(),
    )


def test_selector_metadata_context_is_restored_after_exception(monkeypatch):
    from relax.models.qwen3_8 import qsa_select as opt

    offsets = (0, 12, 20)

    def fail(*args):
        assert opt._layout.get() == offsets
        raise ValueError("expected")

    monkeypatch.setattr(opt.qsa_dispatch, "replay_forward", fail)
    with pytest.raises(ValueError, match="expected"):
        opt.indexer_forward(None, None, None, SimpleNamespace(qsa_select_offsets_cpu=offsets))
    assert opt._layout.get() is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA TopK and fused bitsets require CUDA")
@pytest.mark.parametrize("columns,k", [(33, 32), (513, 512), (8192, 512), (32768, 512)])
@pytest.mark.parametrize("kind", ["random", "zeros", "cutoff_ties", "nonfinite"])
def test_unsorted_topk_preserves_selected_set_including_ties(columns, k, kind):
    from relax.models.qwen3_8 import qsa_select as opt

    torch.manual_seed(321)
    scores = torch.randn(17, columns, device="cuda")
    if kind == "zeros":
        scores.zero_()
    if kind == "cutoff_ties":
        scores = scores.sign()
    if kind == "nonfinite":
        scores[:, ::5] = float("nan")
        scores[:, 1::5] = float("inf")
        scores[:, 2::5] = float("-inf")
    sorted_top = torch.topk(scores, k, dim=-1)
    unsorted_top = torch.topk(scores, k, dim=-1, sorted=False)
    # Test the chosen set itself, including the tied cutoff and nonfinite IDs.
    assert torch.equal(sorted_top.indices.sort().values, unsorted_top.indices.sort().values)
    valid = torch.isfinite(sorted_top.values)
    nbytes = (columns + 8) // 8
    expected = torch.zeros(17, nbytes, device="cuda", dtype=torch.int32)
    expected.scatter_add_(1, sorted_top.indices >> 3, ((1 << (sorted_top.indices & 7)) * valid).int())
    actual = torch.empty(17, nbytes, device="cuda", dtype=torch.uint8)
    opt.pack_topk(unsorted_top.indices, unsorted_top.values, actual)
    assert torch.equal(actual, expected.byte())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="QSA routing and fused attention require CUDA")
@pytest.mark.parametrize("kind", ["random", "zeros", "ties"])
def test_selector_preserves_attention_output_and_backbone_gradients(kind):
    from megatron.core.transformer.experimental_attention_variant import qsa

    from relax.models.qwen3_8.ops import qsa_fused_attention as fused

    torch.manual_seed(17)
    total, columns = 4099, 1024
    query = torch.randn(total, 4, 128, device="cuda", dtype=torch.bfloat16)
    keys = torch.randn(1, columns, 128, device="cuda", dtype=torch.bfloat16)
    if kind == "zeros":
        query.zero_()
    if kind == "ties":
        query = query.sign()
        keys = keys.sign()
    docs = torch.zeros(total, device="cuda", dtype=torch.int32)
    positions = torch.arange(total, device="cuda", dtype=torch.int32)
    valid = torch.ones(1, columns, device="cuda", dtype=torch.bool)
    module = _selector_module()
    old_bits, old_flag = module._relax_native_select_blocks(query, keys, valid, docs, positions)
    new_bits, new_flag = module._select_blocks(query, keys, valid, docs, positions)
    assert old_flag == new_flag and torch.equal(old_bits, new_bits)

    query_ids = torch.tensor([0, 3, 2047, 2048, 4095, 4098], device="cuda", dtype=torch.int64)
    hidden = torch.randn(total, 32, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(4 * 64, 32, device="cuda", dtype=torch.bfloat16) / 32
    upstream = torch.randn(len(query_ids), 2, 64, device="cuda", dtype=torch.bfloat16)
    results = []
    for bits in (old_bits, new_bits):
        state = hidden.clone().requires_grad_()
        projection = weights.clone().requires_grad_()
        projected = torch.nn.functional.linear(state, projection).view(total, 4, 64)
        q = projected[:, :2].index_select(0, query_ids).contiguous()
        k = projected[:, 2:3].contiguous()
        v = projected[:, 3:4].contiguous()
        for x in (q, k, v):
            x.retain_grad()
        selection = qsa.QSASelection(
            doc_ids=docs[None, :],
            positions=positions[None, :],
            selected_bits=bits.flatten(),
            bits_per_row=bits.shape[1],
            bits_per_row_t=torch.tensor(bits.shape[1], device="cuda", dtype=torch.int64),
            compress_ratio=4,
            all_selected=False,
        )
        plan = fused.build_plan(selection, total, query_ids=query_ids)
        output = fused.qsa_fused_attention(q, k, v, selection, plan, query_ids, 64**-0.5)
        output.backward(upstream)
        results.append([output.detach(), q.grad, k.grad, v.grad, state.grad, projection.grad])
    for name, old, new in zip(("output", "dq", "dk", "dv", "hidden", "projection"), *results):
        assert torch.equal(old, new), (name, kind, (old.float() - new.float()).abs().max().item())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="QSA selector layout comparison requires CUDA")
@pytest.mark.parametrize(
    "lengths,uniform",
    [((0, 3, 4099, 0, 1031), False), ((0, 3, 0, 7, 0), False), ((2053, 2053), True), ((0, 128, 0), False)],
)
@pytest.mark.parametrize("output_format", ["bits", "ids"])
def test_selector_preserves_empty_short_and_uniform_layout_fallbacks(lengths, uniform, output_format):
    from relax.models.qwen3_8 import qsa_select as opt

    torch.manual_seed(29)
    total, columns = sum(lengths), max(lengths) // 4
    q = torch.randn(total, 4, 128, device="cuda", dtype=torch.bfloat16)
    keys = torch.randn(len(lengths), columns, 128, device="cuda", dtype=torch.bfloat16)
    docs = torch.repeat_interleave(
        torch.arange(len(lengths), device="cuda", dtype=torch.int32), torch.tensor(lengths, device="cuda")
    )
    positions = torch.cat([torch.arange(n, device="cuda", dtype=torch.int32) for n in lengths])
    valid = torch.arange(columns, device="cuda")[None, :] < torch.tensor(lengths, device="cuda")[:, None] // 4
    offsets = [0]
    for n in lengths:
        offsets.append(offsets[-1] + n)
    module = _selector_module()
    token = opt._layout.set(tuple(offsets))
    try:
        args = (q, keys, valid, docs, positions, lengths[0] if uniform else None, output_format)
        expected, expected_flag = module._relax_native_select_blocks(*args)
        actual, actual_flag = module._select_blocks(*args)
    finally:
        opt._layout.reset(token)
    assert expected_flag == actual_flag and torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="QSA pooled-tail routing requires CUDA")
@pytest.mark.parametrize("use_fused", [True, False])
@pytest.mark.parametrize("poison", [1e6, float("inf"), float("nan")])
def test_selector_excludes_invalid_pooled_tails_without_extra_mask(monkeypatch, use_fused, poison):
    from megatron.core.transformer.experimental_attention_variant import qsa

    from relax.models.qwen3_8 import qsa_select as opt

    lengths = (35, 3, 73)
    module = _selector_module()
    module.block_topk = 8
    positions = torch.cat([torch.arange(n, device="cuda", dtype=torch.int32) for n in lengths])
    docs = torch.repeat_interleave(
        torch.arange(len(lengths), device="cuda", dtype=torch.int32), torch.tensor(lengths, device="cuda")
    )
    # Increasing complete-block means make the last K visible blocks the exact
    # expected selection, without depending on a TopK tie-breaking convention.
    raw_keys = (positions + 1).to(torch.bfloat16)[:, None].expand(-1, 128).contiguous()
    query = torch.ones(sum(lengths), 4, 128, device="cuda", dtype=torch.bfloat16)
    keys, _, block_valid = qsa.QSAIndexer._pool_keys(module, raw_keys, docs, positions, max(lengths))
    complete = torch.tensor(lengths, device="cuda") // module.compress_ratio
    assert torch.equal(block_valid, torch.arange(keys.shape[1], device="cuda")[None, :] < complete[:, None])
    # Poison every invalid column, including all columns of the short document.
    keys[~block_valid] = poison
    monkeypatch.setattr(qsa, "_QSA_SELECT_TILE_BYTES", 1)  # Exercise the shared-key path.
    monkeypatch.setattr(opt, "_TILE_BYTES", 32 << 10)  # Exercise both <=K and >K tiles.
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0) if use_fused else (10, 0))
    token = opt._layout.set((0, 35, 38, 111))
    try:
        old, _ = opt.qsa_dispatch.cpu_select(module, query, keys, block_valid, docs, positions)
        new, _ = opt.select_blocks(module, query, keys, block_valid, docs, positions)
    finally:
        opt._layout.reset(token)
    assert torch.equal(new, old)
    columns = torch.arange(new.shape[1] * 8, device="cuda")
    chosen = (new[:, columns // 8].int() & (1 << (columns % 8))) != 0
    visible = (positions.long() + 1) // module.compress_ratio
    expected = (columns[None, :] < visible[:, None]) & (
        columns[None, :] >= (visible - module.block_topk).clamp_min(0)[:, None]
    )
    assert torch.equal(chosen, expected)
    assert not (chosen & (columns[None, :] >= complete[docs.long(), None])).any()
