# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""PR7553 selection, packed coordinates and sparse metadata without CUDA."""

from __future__ import annotations

import pytest
import torch


@pytest.fixture(autouse=True)
def operator_imports(monkeypatch):
    # The optional model-registration package eagerly imports CUDA-only TE.
    # Isolate that external boundary; load and execute the real operator files.
    import importlib
    import sys
    from pathlib import Path
    from types import ModuleType, SimpleNamespace

    package = ModuleType("relax.models")
    package.__path__ = [str(Path(__file__).resolve().parents[4] / "relax/models")]
    monkeypatch.setitem(sys.modules, "relax.models", package)
    backend = ModuleType("relax.backends.megatron")
    backend.__path__ = [str(Path(__file__).resolve().parents[4] / "relax/backends/megatron")]
    monkeypatch.setitem(sys.modules, "relax.backends.megatron", backend)
    existing = set(sys.modules)
    exports = {
        "cp": ("rotary_rows", "selector_query_plan", "restore_selected_shards"),
        "layout": ("block_metadata", "validate_bounds"),
        "selector": ("select_ids",),
        "triton_qsa": ("build_qsa_kv_tile_csr", "build_qsa_supersets"),
        "triton_qsa_select": ("qsa_cp_layout_static",),
    }
    loaded = {}
    for suffix, names in exports.items():
        module = importlib.import_module("relax.models.qwen3_8.ops.qsa_cute." + suffix)
        for name in names:
            loaded[name] = getattr(module, name)
    yield SimpleNamespace(**loaded)
    for name in set(sys.modules) - existing:
        if name.startswith(("relax.models.", "relax.backends.megatron.")):
            sys.modules.pop(name, None)


@pytest.mark.parametrize("layout,bounds", [("per_document", (0, 8, 8, 32)), ("per_sequence", (0, 3, 3, 19, 32))])
@pytest.mark.parametrize("size", [2, 4])
def test_qsa_cp_layout_reassembles_global_tokens(operator_imports, layout, bounds, size):
    operator_imports.validate_bounds(bounds, bounds[-1], size, layout)
    cu = torch.tensor(bounds)
    plans = [operator_imports.qsa_cp_layout_static(cu, bounds[-1] // size, size, rank, layout) for rank in range(size)]
    rank_major = torch.cat([plan[0] for plan in plans])
    for positions, reorder in plans:
        torch.testing.assert_close(rank_major[reorder], torch.arange(bounds[-1]))
        assert positions.unique().numel() == bounds[-1] // size
    if layout == "per_document":
        for start, end in zip(bounds, bounds[1:]):
            for rank, (positions, _) in enumerate(plans):
                half = (end - start) // (2 * size)
                owned = positions[(positions >= start) & (positions < end)]
                expected = torch.cat(
                    (
                        torch.arange(start + rank * half, start + (rank + 1) * half),
                        torch.arange(end - (rank + 1) * half, end - rank * half),
                    )
                )
                torch.testing.assert_close(owned, expected)


def test_qsa_cp_layout_rejects_misaligned_physical_documents(operator_imports):
    with pytest.raises(ValueError, match="divisible"):
        operator_imports.validate_bounds((0, 7, 16), 16, 2)
    with pytest.raises(ValueError, match="boundaries"):
        operator_imports.validate_bounds((0, 10, 8), 8)


@pytest.mark.parametrize("bounds,tp_size", [((0, 0, 16, 2072, 6192), 2), ((0, 16, 56), 4)])
def test_cp_selector_tiles_restore_ids_with_empty_owners(operator_imports, bounds, tp_size):
    torch.manual_seed(97)
    cp_size, budget = 2, 16
    rows, _ = operator_imports.qsa_cp_layout_static(
        torch.tensor(bounds), bounds[-1] // cp_size, cp_size, 1, "per_document"
    )
    plans, counts = operator_imports.selector_query_plan(bounds, cp_size, tp_size, budget)
    query = torch.randn(rows.numel(), 2, 8)
    # Tied scores exercise deterministic selection while tail tiles stay intact.
    query[::3] = 0
    expected = torch.full((rows.numel(), budget // 4), -7, dtype=torch.int32)
    shards = torch.full((tp_size, max(counts), budget // 4), -1, dtype=torch.int32)
    offset = 0
    for document, (start, end) in enumerate(zip(bounds, bounds[1:])):
        size = (end - start) // cp_size
        if plans[document]:
            keys = torch.randn((end - start) // 4, 8)
            expected[offset : offset + size] = operator_imports.select_ids(
                query[offset : offset + size], keys, rows[offset : offset + size] - start, budget // 4
            )
            for first, last, owner, target in plans[document]:
                shards[owner, target : target + last - first] = operator_imports.select_ids(
                    query[first:last], keys, rows[first:last] - start, budget // 4
                )
        offset += size
    actual = torch.full_like(expected, -7)
    operator_imports.restore_selected_shards(actual, shards, plans)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert sum(counts) == sum(
        (end - start) // cp_size for start, end in zip(bounds, bounds[1:]) if end - start > budget + 3
    )
    if tp_size == 4:
        assert counts.count(0) == 3


@pytest.mark.parametrize("local", [False, True])
def test_qsa_cute_metadata_preserves_tails_documents_and_inverse_csr(operator_imports, local):
    bounds = (0, 0, 1, 8, 12, 23)
    positions = torch.tensor([0, 1, 4, 7, 8, 11, 12, 18, 22]) if local else torch.arange(23)
    cu = torch.tensor(bounds)
    docs = torch.searchsorted(cu[1:], positions, right=True)
    doc_pos = positions - cu[docs]
    # Deliberately leave a hole in the complete-block IDs to test compaction.
    ids = torch.stack(
        (torch.where(doc_pos >= 3, 0, -1), torch.full_like(doc_pos, -1), torch.where(doc_pos >= 7, 1, -1)), -1
    )[None].int()
    meta = operator_imports.block_metadata(ids, bounds, positions)
    superset = operator_imports.build_qsa_supersets(
        meta.indices, meta.counts, meta.bases, meta.ends, meta.bases.numel(), 3
    )
    csr = operator_imports.build_qsa_kv_tile_csr(meta.indices, meta.counts, meta.bases.numel(), 4)
    for row, (absolute, relative, doc) in enumerate(zip(positions.tolist(), doc_pos.tolist(), docs.tolist())):
        expected = set()
        for block in ids[0, row].tolist():
            if block >= 0:
                expected.update(range(bounds[doc] + block * 4, bounds[doc] + block * 4 + 4))
        expected.update(range(bounds[doc] + (relative + 1) // 4 * 4, absolute + 1))
        actual = set()
        tile = row // 3
        for entry in range(int(superset.tile_offsets[tile]), int(superset.tile_offsets[tile + 1])):
            if superset.membership[entry, row % 3]:
                actual.update(
                    range(int(superset.entry_base[entry]), min(int(superset.entry_end[entry]), absolute + 1))
                )
        assert actual == expected
        selected_blocks = meta.indices[0, row, : meta.counts[0, row]].tolist()
        inverse_blocks = []
        for tile in range(csr.n_tiles):
            for entry in range(int(csr.tile_offsets[tile]), int(csr.tile_offsets[tile + 1])):
                if int(csr.entry_query[entry]) == row:
                    inverse_blocks.extend(
                        tile * 4 + bit for bit in range(4) if int(csr.entry_bits[entry]) & (1 << bit)
                    )
        assert set(inverse_blocks) == set(selected_blocks)


def test_qsa_radix_contract_ties_causality_and_empty_history(operator_imports):
    query = torch.zeros(5, 4, 128)
    keys = torch.zeros(7, 128)
    positions = torch.tensor([0, 2, 3, 7, 23])
    selected = operator_imports.select_ids(query, keys, positions, 3)
    torch.testing.assert_close(
        selected, torch.tensor([[-1, -1, -1], [-1, -1, -1], [0, -1, -1], [0, 1, -1], [0, 1, 2]], dtype=torch.int32)
    )
    assert (operator_imports.select_ids(query, keys[:0], positions, 3) == -1).all()


def test_qsa_rotary_coordinates_distinguish_text_reset_and_absolute_multimodal(operator_imports):
    freqs = torch.arange(32, dtype=torch.float32).view(32, 1, 1, 1)
    physical = torch.tensor([0, 3, 12, 15, 16, 19, 28, 31])
    relative = physical % 16
    torch.testing.assert_close(
        operator_imports.rotary_rows(freqs, physical, relative, absolute=True).flatten(), physical.float()
    )
    torch.testing.assert_close(
        operator_imports.rotary_rows(freqs, physical, relative, absolute=False).flatten(), relative.float()
    )


@pytest.mark.parametrize("absolute", [False, True])
def test_qsa_cp_selector_projects_pools_and_rotates_in_global_document_order(operator_imports, monkeypatch, absolute):
    import ast
    import importlib
    import sys
    from dataclasses import dataclass
    from pathlib import Path
    from types import ModuleType, SimpleNamespace

    cp = importlib.import_module("relax.models.qwen3_8.ops.qsa_cute.cp")
    # Execute the image's actual selection dataclass; the rest of Megatron is
    # an external projection/rotary service for this CPU adapter contract test.
    patch = (Path(__file__).resolve().parents[4] / "docker/patch/megatron/20260915-4a4de8657.patch").read_text()
    section = patch.split("+++ b/megatron/core/transformer/experimental_attention_variant/qsa.py\n", 1)[1]
    source = "\n".join(line[1:] for line in section.split("\ndiff --git", 1)[0].splitlines() if line.startswith("+"))
    node = next(
        node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == "QSASelection"
    )
    qsa = ModuleType("megatron.core.transformer.experimental_attention_variant.qsa")
    qsa.__dict__.update(dataclass=dataclass, torch=torch)
    monkeypatch.setitem(sys.modules, qsa.__name__, qsa)
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<image-selection>", "exec"), qsa.__dict__)
    # A coordinate-sensitive rotary boundary exposes local/global table mistakes.
    rope = lambda tensor, freqs, **kwargs: tensor + freqs
    monkeypatch.setitem(
        sys.modules,
        "megatron.core.models.common.embeddings.rope_utils",
        SimpleNamespace(_apply_rotary_pos_emb_bshd=rope),
    )
    monkeypatch.setitem(
        sys.modules,
        "megatron.core.tensor_parallel.mappings",
        SimpleNamespace(gather_from_sequence_parallel_region=None),
    )
    monkeypatch.setenv("RELAX_QWEN38_QSA_SELECTOR_BACKEND", "radix")
    torch.manual_seed(17)
    hidden = torch.randn(32, 1, 12)
    freqs = torch.arange(32).float().view(32, 1, 1, 1) * 0.01
    cp_group, tp_group = object(), object()
    monkeypatch.setattr(cp.dist, "get_world_size", lambda group: 2 if group is cp_group else 1)
    owner = SimpleNamespace(
        config=SimpleNamespace(
            qsa_indexer_loss_coeff=0, sequence_parallel=False, mrope_section=[1] if absolute else None
        ),
        parameters=lambda: [],
        compress_ratio=4,
        kv_heads=1,
        token_budget=4,
        block_topk=1,
        pg_collection=SimpleNamespace(cp=cp_group, tp=tp_group),
        n_heads=2,
        head_dim=4,
        rotary_interleaved=False,
        index_qk_proj=lambda value: (value, None),
        q_layernorm=torch.nn.LayerNorm(4),
        k_layernorm=torch.nn.LayerNorm(4),
    )
    packed = SimpleNamespace(qkv_format="thd", cp_partition_mode="zigzag", qsa_select_offsets_cpu=(0, 8, 32))
    expected = []
    for start, end in ((0, 8), (8, 32)):
        q = owner.q_layernorm(hidden[start:end, 0, :8].reshape(-1, 2, 4))
        k = owner.k_layernorm(hidden[start:end, 0, 8:].reshape(-1, 4, 4).mean(1))
        q = q + freqs[(start if absolute else 0) : (end if absolute else end - start), 0]
        k = k + freqs[torch.arange(0, end - start, 4) + (start if absolute else 0), 0, 0]
        scores = torch.einsum("qhd,kd->qhk", q, k).relu().sum(1)
        valid = torch.arange(k.shape[0])[None] < (torch.arange(end - start)[:, None] + 1) // 4
        ids = scores.masked_fill(~valid, -torch.inf).argsort(dim=-1, descending=True, stable=True)[:, :1]
        expected.append(torch.where(valid.gather(1, ids), ids, -1).int())
    expected = torch.cat(expected)
    for rank in (0, 1):
        rows, _ = operator_imports.qsa_cp_layout_static(torch.tensor([0, 8, 32]), 16, 2, rank, "per_document")
        monkeypatch.setattr(cp.dist, "get_rank", lambda group, rank=rank: rank)
        monkeypatch.setattr(
            cp, "gather_sequence", lambda value, group, reorder: freqs if value.ndim == 4 else hidden[:, 0, 8:]
        )
        selected = cp.forward(owner, hidden[rows], freqs[rows] if absolute else freqs, packed)
        torch.testing.assert_close(selected.selected_ids, expected[rows])
        torch.testing.assert_close(selected._relax_qsa_query_positions, rows)


def test_qsa_cp_rejects_dynamic_groups_before_collectives(operator_imports):
    import importlib
    from types import SimpleNamespace

    cp = importlib.import_module("relax.models.qwen3_8.ops.qsa_cute.cp")
    with pytest.raises(NotImplementedError, match="static CP"):
        cp.producer_bounds(SimpleNamespace(local_cp_size=1), 16, 2)


@pytest.mark.parametrize("unsupported_group", [3, 129])
def test_qsa_cute_rejects_unsupported_group_before_kernel_import(operator_imports, unsupported_group):
    from relax.models.qwen3_8.ops.qsa_mixed_ops.geometry import validate_geometry

    validate_geometry("cute", 12, 1, 256, 256)
    validate_geometry("cute", 6, 1, 256, 256)
    with pytest.raises(NotImplementedError, match="membership"):
        validate_geometry("cute", unsupported_group, 1, 256, 256)


@pytest.mark.parametrize("bound_layout", [False, True])
def test_mtp_physical_offsets_preserve_qsa_cp_producer_contract(operator_imports, bound_layout):
    """Execute Bridge helpers without importing unrelated CUDA model classes.

    Earlier packing/head tests never crossed MTP's packed-parameter copy. Both
    producer contracts must survive that copy; this runs only CPU tensors.
    """
    import __future__

    import ast
    import importlib.util
    from copy import copy
    from dataclasses import dataclass, replace
    from pathlib import Path

    from relax.backends.megatron.qsa_layout_metadata import bind_physical_layout, valid_physical_layout
    from relax.models.qwen3_8.ops.qsa_cute.cp import producer_bounds

    spec = importlib.util.find_spec("megatron")
    if spec is None:
        pytest.skip("requires the patched Megatron Bridge source")
    roots = [Path(path) for path in spec.submodule_search_locations]
    bridge = next((path / "bridge" for path in roots if (path / "bridge").is_dir()), None)
    if bridge is None:
        pytest.skip("requires the patched Megatron Bridge source")
    namespace = {"copy": copy, "replace": replace}
    for path, name in (
        (bridge / "training/utils/packed_seq_utils.py", "get_packed_seq_q_cu_seqlens"),
        (bridge / "models/qwen_vl/modelling_qwen3_vl/text_model.py", "_get_mtp_packed_seq_params"),
    ):
        # Execute the installed production function bodies, not an AST/text assertion
        # or a rewritten implementation. The omitted classes import CUDA-only TE.
        tree = ast.parse(path.read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
        code = ast.Module(body=[function], type_ignores=[])
        exec(compile(code, str(path), "exec", flags=__future__.annotations.compiler_flag), namespace)

    @dataclass
    class PackedFixture:
        cu_seqlens_q: torch.Tensor
        cu_seqlens_kv: torch.Tensor
        cu_seqlens_q_padded: torch.Tensor
        cu_seqlens_kv_padded: torch.Tensor
        qkv_format: str = "thd"

    logical = torch.tensor([0, 5, 24], dtype=torch.int32)
    physical = torch.tensor([0, 8, 32], dtype=torch.int32)
    original = PackedFixture(logical, logical, physical, physical)
    original.cp_partition_mode = "zigzag"
    original.qsa_select_offsets_cpu = (0, 8, 32)
    if bound_layout:
        bind_physical_layout(original, (0, 8, 32))
    result = namespace["_get_mtp_packed_seq_params"](original)
    assert result is not original
    assert original.cu_seqlens_q is logical and original.cu_seqlens_kv is logical
    assert result.cu_seqlens_q is physical and result.cu_seqlens_kv is physical
    assert producer_bounds(result, 32, 2) == (0, 8, 32)
    assert result.cp_partition_mode == "zigzag"
    if bound_layout:
        assert valid_physical_layout(result) == valid_physical_layout(original)
