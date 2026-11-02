# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU interpreter and offline CUDA compilation checks for dynamic QSA
kernels."""

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _extract_kernels(tmp_path):
    pytest.importorskip("triton")
    pytest.importorskip("torch")
    root = Path(__file__).resolve().parents[3] / "relax/models/qwen3_8/ops"
    for name, source in (("qsa_select_patch", "qsa_selector"), ("qsa_block_mask_patch", "qsa_block_mask")):
        tree = ast.parse((root / f"{source}.py").read_text())
        kernels = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and any("triton.jit" in ast.unparse(decorator) for decorator in node.decorator_list)
        ]
        (tmp_path / f"{name}_kernels.py").write_text(
            "import triton\nimport triton.language as tl\n" + "\n\n".join(ast.unparse(node) for node in kernels)
        )


def _run(tmp_path, code, interpret):
    _extract_kernels(tmp_path)
    runner = tmp_path / "check.py"
    runner.write_text(code)
    env = dict(os.environ, TRITON_INTERPRET=str(int(interpret)), CUDA_VISIBLE_DEVICES="")
    env["TRITON_CACHE_DIR"] = str(tmp_path / "cache")
    subprocess.run([sys.executable, str(runner)], env=env, check=True, timeout=180)


def test_dynamic_qsa_kernels_match_dense_reference(tmp_path):
    _run(tmp_path, _REFERENCE_CHECK, interpret=True)


def test_dynamic_qsa_kernels_compile_for_sm90_without_gpu(tmp_path):
    _run(tmp_path, _COMPILE_CHECK, interpret=False)


_REFERENCE_CHECK = r"""
import math
import torch
import triton
from qsa_select_patch_kernels import _score_kernel, _pack_kernel, _pack_all_candidates_kernel, _score_topk_pack_kernel
from qsa_block_mask_patch_kernels import _classify_blocks, _pack_block_indices

torch.manual_seed(1234)
for rows, columns in [(1, 7), (7, 33), (17, 65), (33, 129), (65, 257)]:
    x = torch.randn(rows, 4, columns)
    pos = torch.arange(rows, dtype=torch.int32) * 31 + 7
    y = torch.empty(rows, columns)
    _score_kernel[(triton.cdiv(rows * columns, 256),)](x, pos, y, columns, rows, 4, 1 / math.sqrt(128), 256, enable_fp_fusion=False)
    v = torch.where(x <= 0, 0., x)
    ref = ((v[:, 0] + v[:, 1]) + v[:, 2]) + v[:, 3]
    ref *= 1 / math.sqrt(128)
    ref.masked_fill_(torch.arange(columns)[None, :] >= ((pos + 1) // 4)[:, None], float('-inf'))
    assert torch.equal(y, ref), ('score', rows, columns)
    k = min(19, columns)
    top = torch.topk(ref, k, dim=-1)
    nbytes = triton.cdiv(columns, 8)
    wc = triton.cdiv(nbytes, 4)
    words = torch.empty(rows, wc, dtype=torch.int32)
    out = torch.empty(rows, nbytes, dtype=torch.uint8)
    _pack_kernel[(rows,)](top.indices, top.values, words, out, k, nbytes, wc,
        triton.next_power_of_2(k), triton.next_power_of_2(wc), triton.next_power_of_2(nbytes))
    expected = torch.zeros_like(out)
    for r in range(rows):
        for j in range(k):
            if torch.isfinite(top.values[r, j]):
                idx = int(top.indices[r, j]); expected[r, idx // 8] |= 1 << (idx % 8)
    assert torch.equal(out, expected), ('bits', rows, columns)
    fused = torch.full((rows, nbytes + 65), 255, dtype=torch.uint8)
    _score_topk_pack_kernel[(rows,)](x, pos, fused, columns, nbytes + 65, 4, 1 / math.sqrt(128), k,
        max(8, triton.next_power_of_2(columns)), triton.next_power_of_2(nbytes + 65), enable_fp_fusion=False)
    assert torch.equal(fused[:, :nbytes], expected), ('fused topk', rows, columns)
    assert not fused[:, nbytes:].any(), ('topk padding', rows, columns)
    all_out = torch.full((rows, nbytes + 65), 255, dtype=torch.uint8)
    _pack_all_candidates_kernel[(rows, triton.cdiv(nbytes + 65, 64))](x, pos, all_out, columns, nbytes + 65, 4, 1 / math.sqrt(128), 64, enable_fp_fusion=False)
    all_expected = torch.zeros_like(all_out)
    for r in range(rows):
        for col in range(columns):
            if torch.isfinite(ref[r, col]):
                all_expected[r, col // 8] |= 1 << (col % 8)
    assert torch.equal(all_out, all_expected), ('all candidates', rows, columns)
print('selector: 5 dynamic shapes, exact scores and route bytes passed')

for n in (129, 257, 384, 513):
    for local in (False, True):
        docs = torch.zeros(n, dtype=torch.int32)
        pos = torch.arange(n, dtype=torch.int32)
        nb = triton.cdiv(n, 128)
        width = triton.cdiv(triton.cdiv(n, 4), 8)
        bits = torch.randint(0, 256, (n, width), dtype=torch.uint8)
        ids = torch.arange(n, dtype=torch.int64) if not local else torch.cat((torch.arange(min(128, n)), torch.arange(min(256, n), n)))
        qn = len(ids); qb = triton.cdiv(qn, 128)
        flags = torch.empty(qb, nb, dtype=torch.uint8)
        _classify_blocks[(qb * nb,)](docs, pos, bits, flags, n, nb, width, 128, 4, ids if local else None, qn, local, num_warps=8)
        keys = torch.arange(n)
        selected = ((bits[ids[:, None], (keys // 4 // 8)[None, :]].int() >> (keys // 4 % 8)[None, :]) & 1).bool()
        dense = (keys[None, :] <= ids[:, None]) & (selected | (keys[None, :] >= ((ids[:, None] + 1) // 4) * 4))
        ref = torch.empty_like(flags)
        for q in range(qb):
            for k in range(nb):
                block = dense[q*128:(q+1)*128, k*128:(k+1)*128]
                ref[q,k] = 0 if not block.any() else 2 if block.shape == (128,128) and block.all() else 1
        assert torch.equal(flags, ref), ('classify', n, local, flags, ref)
        pc = torch.empty(qb, dtype=torch.int32); fc = torch.empty_like(pc)
        pi = torch.full((qb,nb),-1,dtype=torch.int32); fi = torch.full_like(pi,-1)
        _pack_block_indices[(qb,)](flags,pc,pi,fc,fi,nb,triton.next_power_of_2(nb))
        for q in range(qb):
            assert torch.equal(pi[q,:pc[q]].long(),torch.where(ref[q]==1)[0])
            assert torch.equal(fi[q,:fc[q]].long(),torch.where(ref[q]==2)[0])
print('blockmask: 8 dynamic global/local shapes matched dense causal bitset reference')
"""

_COMPILE_CHECK = r"""
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from qsa_select_patch_kernels import _score_kernel, _pack_kernel, _pack_all_candidates_kernel, _score_topk_pack_kernel
from qsa_block_mask_patch_kernels import _classify_blocks, _pack_block_indices
cases = [
 (_score_topk_pack_kernel, {'scores':'*fp32','positions':'*i32','output':'*u8','columns':'i32','nbytes':'i32'}, {'ratio':4,'scale':0.08838834764831843,'k':512,'BLOCK':1024,'BYTES':2048}),
 (_pack_all_candidates_kernel, {'scores':'*fp32','positions':'*i32','output':'*u8','columns':'i32','nbytes':'i32'}, {'ratio':4,'scale':0.08838834764831843,'BLOCK':64}),
 (_score_kernel, {'scores':'*fp32','positions':'*i32','output':'*fp32','columns':'i32','rows':'i32'}, {'ratio':4,'scale':0.08838834764831843,'BLOCK':256}),
 (_pack_kernel, {'ids':'*i64','values':'*fp32','words':'*i32','output':'*u8','k':'i32','nbytes':'i32','word_count':'i32'}, {'BLOCK':512,'WORDS':256,'BYTES':1024}),
 (_pack_block_indices, {'flags':'*u8','partial_counts':'*i32','partial_indices':'*i32','full_counts':'*i32','full_indices':'*i32','blocks_per_row':'i32'}, {'scan_width':1024}),
]
for local in (False,True):
 cases.append((_classify_blocks, {'doc_ids':'*i32','positions':'*i32','selected_bits':'*u8','flags':'*u8','sequence_length':'i32','blocks_per_row':'i32','bytes_per_row':'i32','query_ids':'*i64','query_length':'i32'}, {'block_size':128,'ratio':4,'local_query':local}))
for fn,sig,const in cases:
 src=ASTSource(fn,dict(sig,**{k:'constexpr' for k in const}),constexprs=const)
 compiled=triton.compile(src,target=GPUTarget('cuda',90,32),options={'num_warps':8 if fn is _classify_blocks else 4,'enable_fp_fusion':False})
 print(fn.__name__,const,'compiled',compiled.metadata.name)
"""
