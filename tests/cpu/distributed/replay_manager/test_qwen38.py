# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""QSA capture representation and replay cursors without model/GPU imports."""

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from relax.distributed.replay_manager.backend.megatron import IndexerReplay, consumer_batch_layout
from relax.distributed.replay_manager.backend.qwen38 import (
    install_capture_patch,
    replay_selection,
    token_ids_to_blocks,
)
from relax.distributed.replay_manager.config import ReplayConfig
from relax.distributed.replay_manager.consumer import ConsumerLayerReader
from relax.distributed.replay_manager.core import ReplayInventory
from relax.distributed.replay_manager.storage import TQReplayStorage


def test_qsa_capture_removes_compacted_partial_tail_and_preserves_document_ids():
    # Prefill rows from different documents and a decode row with a shuffled
    # selection. Positions and token IDs are document-local in both paths.
    tokens = torch.tensor(
        [
            [0, 1, 2, -1, -1, -1, -1, -1, -1, -1, -1],
            [0, 1, 2, 3, -1, -1, -1, -1, -1, -1, -1],
            [0, 1, 2, 3, 8, -1, -1, -1, -1, -1, -1],
            [8, 9, 10, 11, 0, 1, 2, 3, 12, -1, -1],
        ],
        dtype=torch.int32,
    )
    expected = torch.tensor([[-1, -1], [0, -1], [0, -1], [2, 0]], dtype=torch.int32)
    assert torch.equal(token_ids_to_blocks(tokens, torch.tensor([2, 3, 8, 12]), 8), expected)


@pytest.fixture
def mixed_modules(monkeypatch):
    # Only MCore's external data container is substituted; execute the actual
    # project's MixedQSASelection, metadata binding, and token route expansion.
    @dataclass
    class QSASelection:
        doc_ids: torch.Tensor
        positions: torch.Tensor
        selected_ids: torch.Tensor
        selected_bits: torch.Tensor | None = None
        all_selected: bool = True
        compress_ratio: int = 4

    monkeypatch.setitem(
        sys.modules,
        "megatron.core.transformer.experimental_attention_variant.qsa",
        SimpleNamespace(QSASelection=QSASelection),
    )
    root = Path(__file__).resolve().parents[4] / "relax/models/qwen3_8/ops/qsa_mixed_ops"
    loaded = []
    for name in ("selection", "routes"):
        spec = importlib.util.spec_from_file_location(
            f"relax.models.qwen3_8.ops.qsa_mixed_ops.{name}", root / f"{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        loaded.append(module)
    return QSASelection, *loaded


@pytest.fixture(autouse=True)
def clear_replay():
    IndexerReplay.clear()
    yield
    IndexerReplay.clear()


def test_qsa_reference_packs_two_documents_and_replays_mixed_routes(mixed_modules, monkeypatch):
    base, mixed, routes = mixed_modules
    mpu = SimpleNamespace(
        get_tensor_model_parallel_world_size=lambda: 4,
        get_context_parallel_world_size=lambda: 1,
        get_context_parallel_rank=lambda: 0,
    )
    monkeypatch.setitem(sys.modules, "megatron.core", SimpleNamespace(mpu=mpu))
    # Load the real packing implementation without the backend package's
    # optional model registration/CUDA imports.
    name = "relax.backends.megatron.cp_utils"
    path = Path(__file__).resolve().parents[4] / "relax/backends/megatron/cp_utils.py"
    spec = importlib.util.spec_from_file_location(name, path)
    cp_utils = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, cp_utils)
    spec.loader.exec_module(cp_utils)
    config = ReplayConfig(delivery="reference", run_id="a", chunk_rows=2)
    bodies = {}
    backend = SimpleNamespace(
        write_many=lambda keys, values: bodies.update(zip(keys, (value.copy() for value in values), strict=True)),
        read=lambda keys: [bodies[key].copy() for key in keys],
        delete=lambda keys: [bodies.pop(key, None) for key in keys],
    )
    store = TQReplayStorage(config, inventory=ReplayInventory(config, backend.delete))
    store._backend = backend
    refs = []
    try:
        # Captured rows exclude the final input token of each document. Replay
        # packing must restore both holes without shifting the second document.
        for length in (6, 9):
            values = np.full((length - 1, 2), -1, dtype=np.int32)
            for row in range(3, length - 1):
                values[row, 0] = min((row + 1) // 4 - 1, 1)
            publication = store.begin_publication(kind="indexer", layers=1, topk=2)
            for start in range(0, length - 1, 2):
                stop = min(start + 2, length - 1)
                publication.append(stop - start, lambda layer, start=start, stop=stop: values[start:stop].copy())
            refs.append(publication.finish(list(range(length))))
        args = SimpleNamespace(qkv_format="thd", is_vl_model=False, data_pad_size_multiplier=1, allgather_cp=False)
        layout = consumer_batch_layout(args, refs, [6, 9], sequence_parallel=False)
        store.attach(refs, "train")
        store.commit(refs)
        with store.reader(refs, partition="train") as reader:
            source = ConsumerLayerReader(reader, layout, layer=0, kind="indexer", device="cpu")
        captured, valid = source.read()
    finally:
        store._writer.close()
    assert captured.shape == (16, 2)
    assert valid.tolist() == [True] * 5 + [False] + [True] * 8 + [False, False]
    positions = torch.tensor([list(range(6)) + list(range(9)) + [0]], dtype=torch.int32)
    computed = torch.full((16, 2), -1, dtype=torch.int32)
    computed[positions[0] >= 3, 0] = 0
    selection = mixed.bind_document_bounds(
        base(torch.tensor([[0] * 6 + [1] * 9 + [2]]), positions, computed), (0, 6, 15, 16)
    )
    indexer = SimpleNamespace(
        config=SimpleNamespace(context_parallel_size=1, qsa_indexer_loss_coeff=0),
        layer_number=4,
        parameters=lambda: (),
    )
    assert replay_selection(indexer, selection) is selection
    IndexerReplay.begin_step([4])
    IndexerReplay.record(4, captured, valid)
    assert replay_selection(indexer, selection) is selection
    expected = torch.where(valid[:, None], captured, computed)
    for stage in ("replay_forward", "replay_backward"):
        IndexerReplay.set_stage(stage)
        replayed = replay_selection(indexer, selection)
        assert torch.equal(replayed.selected_ids, expected)
        assert isinstance(replayed, mixed.MixedQSASelection)
        assert replayed._relax_qsa_mixed_bounds == (0, 6, 15, 16)
        assert replayed.selected_bits is None
        assert replayed.all_selected is False
        physical = routes.token_routes(replayed, 1, 16)
        assert physical[0, 4].tolist() == [0, 1, 2, 3, -1, -1, -1, -1, 4, -1, -1]
        assert physical[0, 13].tolist() == [10, 11, 12, 13, -1, -1, -1, -1, -1, -1, -1]
        assert physical[0, 14].tolist() == [6, 7, 8, 9, -1, -1, -1, -1, 14, -1, -1]
    IndexerReplay.finish_step(expect_backward=True)
    source.release()


def test_qsa_scheduler_capture_factory_maps_sparse_layers_and_both_token_paths(monkeypatch):
    # Replace SGLang's allocation/runtime boundary; execute the real installation,
    # factory, prefill/decode wrappers and block conversion on CPU tensors.
    def module(name, **values):
        result = ModuleType(name)
        result.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, result)
        return result

    class Capturer:
        def __init__(self, config, *args):
            self.layer_slots = {3: 0}
            self.values = torch.full((8, 1, 2), -9, dtype=torch.int32)

        def capture_tokens(self, layer, tokens, positions):
            self.values[: len(tokens), self.layer_slots[layer]] = token_ids_to_blocks(tokens, positions, 8)

    monkeypatch.setattr("relax.distributed.replay_manager.backend.qwen38.QSABitmapCapturer", Capturer)

    class Indexer:
        token_topk = 8
        compress_ratio = 4
        layer_id = 3

        def select_prefill_tokens(self, q, keys, starts, ends, query_positions, sequence_lengths):
            return q

        def select_decode_tokens(self, q, cache, table, lengths, maximum, query_positions, sequence_lengths):
            return q

    holder = SimpleNamespace(capturer=None)
    capture = module(
        "sglang.srt.state_capturer.indexer_topk",
        create_indexer_capturer=lambda **kwargs: "native",
        get_global_indexer_capturer=lambda: holder.capturer,
    )
    module("sglang.srt.state_capturer", indexer_topk=capture)
    module("sglang.srt.layers.attention.qsa.qsa_indexer", QSAIndexer=Indexer)
    module(
        "sglang.srt.runtime_context",
        get_exec=lambda: SimpleNamespace(features=SimpleNamespace(enable_return_indexer_topk=True)),
        get_schedule=lambda: SimpleNamespace(chunked_prefill_size=8),
        get_parallel=lambda: SimpleNamespace(attn_tp_rank=0),
    )
    runner = module("sglang.srt.model_executor.model_runner", create_indexer_capturer=capture.create_indexer_capturer)
    install_capture_patch()
    install_capture_patch()  # Repeated kind/worker registration must not double-capture.
    config = SimpleNamespace(
        model_type="qwen4_exp_text",
        num_hidden_layers=4,
        layer_types=["linear_attention"] * 3 + ["qwen_sparse_attention"],
        indexer_budget=8,
        indexer_compress_ratio=4,
    )
    holder.capturer = runner.create_indexer_capturer(
        model_config=SimpleNamespace(hf_text_config=config, context_len=32),
        num_tokens=8,
        max_running_requests=2,
        device="cuda",
    )
    indexer = Indexer()
    tokens = torch.tensor([[0, 1, 2, 3, 8, -1, -1, -1, -1, -1, -1]], dtype=torch.int32)
    positions = torch.tensor([8])
    assert indexer.select_prefill_tokens(tokens, None, None, None, positions, None) is tokens
    assert torch.equal(holder.capturer.values[0, 0], torch.tensor([0, -1], dtype=torch.int32))
    tokens = torch.tensor([[4, 5, 6, 7, 0, 1, 2, 3, -1, -1, -1]], dtype=torch.int32)
    assert indexer.select_decode_tokens(tokens, None, None, None, None, positions, None) is tokens
    assert torch.equal(holder.capturer.values[0, 0], torch.tensor([1, 0], dtype=torch.int32))
    assert (
        runner.create_indexer_capturer(
            model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type="other")),
            num_tokens=8,
            max_running_requests=2,
            device="cuda",
        )
        == "native"
    )
