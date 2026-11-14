# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Reuse JSONL offsets while preserving independent SFT sampling state."""

import ast
import copy
import json
import threading
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import ray
import yaml

from relax.engine.sft.bootstrap import resolve_sft_num_rollout
from relax.engine.sft.dataset.streaming import SFTStreamingDataset, _build_reader
from relax.utils.data.streaming_dataset import StreamingReader


@pytest.mark.parametrize("workers,expected", [(None, 16), ("3", 3), ("0", None), ("-1", None)])
def test_index_workers_default_override_and_validation(tmp_path, monkeypatch, workers, expected):
    from concurrent.futures import ThreadPoolExecutor

    monkeypatch.delenv("RELAX_JSONL_INDEX_WORKERS", raising=False)
    if workers is not None:
        monkeypatch.setenv("RELAX_JSONL_INDEX_WORKERS", workers)
    for i in range(20):
        (tmp_path / f"{i}.jsonl").write_text("{}\n")
    pools = []

    def pool(*args, **kwargs):
        pools.append(kwargs["max_workers"])
        return ThreadPoolExecutor(*args, **kwargs)

    monkeypatch.setattr("relax.utils.data.streaming_dataset.ThreadPoolExecutor", pool)
    if expected is None:
        with pytest.raises(ValueError, match="positive integer"):
            _build_reader(str(tmp_path))
    else:
        assert len(_build_reader(str(tmp_path))) == 20
        assert pools == [expected]


def _args(path):
    return Namespace(
        loss_type="sft", prompt_data=path, rollout_batch_size=2, num_epoch=1, num_rollout=None, eval_size=None
    )


@pytest.fixture
def published(monkeypatch):
    values = []

    def put(offsets):
        values.append(tuple(offsets))
        return values[-1]

    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    monkeypatch.setattr(ray, "put", put)
    monkeypatch.setattr(ray, "get", lambda ref: list(ref))
    return values


@pytest.mark.parametrize("mode", ["file", "directory", "list_slice", "outer_slice", "mixture"])
def test_driver_offsets_skip_second_scan_and_preserve_rows(tmp_path, monkeypatch, published, mode):
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    first.write_text("\n".join(json.dumps({"id": i}) for i in range(4)) + "\n")
    second.write_text("\n".join(json.dumps({"id": i}) for i in range(4, 8)) + "\n")
    if mode == "file":
        path = str(first)
    elif mode == "directory":
        path = str(tmp_path)
    elif mode == "list_slice":
        path = [str(first) + "@[1:3]", str(second)]
    elif mode == "outer_slice":
        path = repr([str(first), str(second)]) + "@[-4:]"
    else:
        mixture = tmp_path / "mix.yaml"
        mixture.write_text(yaml.safe_dump({"datasets": {"a": "a.jsonl", "b": "b.jsonl"}}))
        path = str(mixture)
    expected = _build_reader(path)
    rows = [expected[i] for i in range(len(expected))]
    args = _args(path)
    cache = resolve_sft_num_rollout(args)
    assert args._sft_dataset_names == (("a", "b") if mode == "mixture" else ())
    assert cache and len(cache) == len(published)
    assert not hasattr(args, "_sft_line_offsets_cache")
    assert args.num_rollout == len(rows) // 2
    monkeypatch.setattr(StreamingReader, "_build_index", Mock(side_effect=AssertionError("SFT scanned again")))
    dataset = SFTStreamingDataset(path=path, line_offsets_cache=cache, prefetch_max_cached=0)
    assert [dataset.reader[i] for i in range(len(dataset))] == rows
    dataset.stop()


@pytest.mark.parametrize(
    "content", [b"", b"\r\n \t\r\n", b"{}\r\n\r\n{}", '{"text":"中文"}\n \n{}'.encode(), b"invalid-json\n{}"]
)
def test_cached_offsets_preserve_blank_line_and_encoding_rules(tmp_path, monkeypatch, published, content):
    path = tmp_path / "rows.jsonl"
    path.write_bytes(content)
    expected = len(StreamingReader(str(path)))
    cache = {}
    assert len(_build_reader(str(path), pre_make_index=True, line_offsets_cache=cache)) == expected
    monkeypatch.setattr(StreamingReader, "_build_index", Mock(side_effect=AssertionError("SFT scanned again")))
    reader = StreamingReader(str(path), line_offsets_cache=cache)
    assert len(reader) == expected


def test_cached_offsets_keep_shuffle_and_position_independent(tmp_path, monkeypatch, published):
    path = tmp_path / "rows.jsonl"
    path.write_text("{}\n" * 8)
    cache = resolve_sft_num_rollout(_args(str(path)))
    monkeypatch.setattr(StreamingReader, "_build_index", Mock(side_effect=AssertionError("SFT scanned again")))
    first = SFTStreamingDataset(path=str(path), line_offsets_cache=cache, prefetch_max_cached=0, seed=42)
    second = SFTStreamingDataset(path=str(path), line_offsets_cache=cache, prefetch_max_cached=0, seed=42)
    first.index_manager.shuffle(0)
    second.index_manager.shuffle(0)
    assert first.index_manager.indices == second.index_manager.indices
    first.index_manager.get_next_indices(2)
    assert first.index_manager.position == 2
    assert second.index_manager.position == 0
    assert first.reader is not second.reader
    first.stop()
    second.stop()


def test_parquet_sizing_does_not_publish_rows(tmp_path, published):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "rows.parquet"
    pq.write_table(pa.table({"id": list(range(8))}), path)
    args = _args(str(path))
    assert resolve_sft_num_rollout(args) is None
    assert not published
    assert args.num_rollout == 4


@pytest.mark.parametrize("role", ["sft", "actor"])
def test_controller_limits_offsets_references_to_sft_config(role):
    path = Path(__file__).resolve().parents[4] / "relax/core/controller.py"
    tree = ast.parse(path.read_text())
    controller = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Controller")
    method = next(
        node for node in controller.body if isinstance(node, ast.FunctionDef) and node.name == "_create_service_task"
    )
    seen = []

    def service(cls, **kwargs):
        seen.append(kwargs["config"])
        return SimpleNamespace()

    ns = {
        "threading": threading,
        "copy": copy,
        "Service": service,
        "ACTOR_ROLLOUT_PG_ROLES": ["actor"],
        "logger": Mock(),
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), ns)
    config = Namespace(prompt_data="train.jsonl")
    cache = {"train.jsonl": object()}
    owner = SimpleNamespace(
        config=config, _sft_line_offsets_cache=cache, _health_manager=SimpleNamespace(status=None), runtime_env=None
    )
    result = ns["_create_service_task"](owner, role, object, 0, None, None, defer_deploy=True)
    assert result[2] is None
    assert not hasattr(config, "_sft_line_offsets_cache")
    if role == "sft":
        assert seen[0] is not config
        assert seen[0]._sft_line_offsets_cache is cache
    else:
        assert seen[0] is config
        assert not hasattr(seen[0], "_sft_line_offsets_cache")


def test_sft_factory_reuses_offsets_without_scanning(tmp_path, monkeypatch, published):
    from relax.components.sft import _create_sft_train_dataset

    path = tmp_path / "rows.jsonl"
    path.write_text("{}\n" * 8)
    args = _args(str(path))
    args._sft_line_offsets_cache = resolve_sft_num_rollout(args)
    args.input_key = "messages"
    args.label_key = None
    args.multimodal_keys = None
    args.metadata_key = "metadata"
    args.tool_key = None
    args.system_prompt = None
    monkeypatch.setattr(StreamingReader, "_build_index", Mock(side_effect=AssertionError("SFT scanned again")))
    dataset = _create_sft_train_dataset(
        args,
        tokenizer=None,
        processor_pool=None,
        capacity=128,
        prefetch_buffer_size=0,
        prefetch_chunk_size=2,
        prefetch_num_workers=1,
        pad_token_ids=frozenset(),
        oversize_strategy="keep",
        oversize_custom_fn=None,
        invalid_multimodal_strategy="error",
        task_type="causal_lm",
        classification_sentinel_token_id=None,
    )
    assert len(dataset) == 8
    dataset.stop()


def test_sizing_without_ray_does_not_start_runtime(tmp_path, monkeypatch):
    path = tmp_path / "rows.jsonl"
    path.write_text("{}\n" * 8)
    monkeypatch.setattr(ray, "is_initialized", lambda: False)
    publish = Mock(side_effect=AssertionError("Sizing must not auto-start Ray"))
    monkeypatch.setattr(ray, "put", publish)
    args = _args(str(path))
    assert resolve_sft_num_rollout(args) is None
    assert args.num_rollout == 4
    publish.assert_not_called()


def test_repeated_bootstrap_rebuilds_offsets_for_new_runtime(tmp_path, published):
    path = tmp_path / "rows.jsonl"
    path.write_text("{}\n" * 4)
    first = resolve_sft_num_rollout(_args(str(path)))
    path.write_text("{}\n" * 8)
    second = resolve_sft_num_rollout(_args(str(path)))
    assert first is not second
    assert len(first[str(path)]) == 4
    assert len(second[str(path)]) == 8
    assert len(published) == 2


def test_sft_and_preference_parquet_column_projection(tmp_path, monkeypatch):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    from relax.engine.sft.dataset.preference import PreferenceStreamingDataset

    recorded_columns: list = []
    orig_read_row_group = pq.ParquetFile.read_row_group

    def spy_read_row_group(pf_self, i, columns=None, *args, **kwargs):
        recorded_columns.append(columns)
        return orig_read_row_group(pf_self, i, columns=columns, *args, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "read_row_group", spy_read_row_group)

    sft_path = tmp_path / "sft.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "messages": [
                        {"role": "user", "content": "hi"},
                        {"role": "assistant", "content": "hello"},
                    ],
                    "message_loss_mask": [0, 1],
                    "unused_blob": "drop_me",
                }
            ]
        ),
        sft_path,
    )
    sft_ds = SFTStreamingDataset(str(sft_path), prompt_key="messages", prefetch_max_cached=0)
    try:
        canonical = sft_ds.get_canonical_sample(0)
        assert recorded_columns == [["messages", "message_loss_mask"]]
        assert [m.learn for m in canonical.messages] == [False, True]
    finally:
        sft_ds.stop()

    recorded_columns.clear()
    bad_ds = SFTStreamingDataset(str(sft_path), prompt_key="wrong_key", prefetch_max_cached=0)
    try:
        with pytest.raises(ValueError, match="unused_blob"):
            bad_ds.get_canonical_sample(0)
        assert recorded_columns == [None]
    finally:
        bad_ds.stop()

    recorded_columns.clear()
    cpt_path = tmp_path / "cpt.parquet"
    pq.write_table(pa.Table.from_pylist([{"text": "corpus doc", "unused_blob": "drop_me"}]), cpt_path)
    cpt_ds = SFTStreamingDataset(
        str(cpt_path),
        prompt_key="auto",
        training_mode="cpt",
        require_response=False,
        prefetch_max_cached=0,
    )
    try:
        assert cpt_ds.get_canonical_sample(0).messages[0].content == "corpus doc"
        assert recorded_columns == [["text"]]
    finally:
        cpt_ds.stop()

    recorded_columns.clear()
    bad_cpt_path = tmp_path / "bad_cpt.parquet"
    pq.write_table(
        pa.Table.from_pylist([{"text": "corpus doc", "images": ["img.png"], "unused_blob": "drop_me"}]),
        bad_cpt_path,
    )
    bad_cpt_ds = SFTStreamingDataset(
        str(bad_cpt_path),
        prompt_key="auto",
        training_mode="cpt",
        require_response=False,
        prefetch_max_cached=0,
    )
    try:
        with pytest.raises(ValueError, match="text-only CPT does not support 'images'"):
            bad_cpt_ds.get_canonical_sample(0)
        assert recorded_columns == [["text", "images"]]
    finally:
        bad_cpt_ds.stop()

    recorded_columns.clear()
    pref_path = tmp_path / "pref.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "chosen": [
                        {"role": "user", "content": "q"},
                        {"role": "assistant", "content": "good"},
                    ],
                    "rejected": [
                        {"role": "user", "content": "q"},
                        {"role": "assistant", "content": "bad"},
                    ],
                    "unused_blob": "drop_me",
                }
            ]
        ),
        pref_path,
    )
    pref_ds = PreferenceStreamingDataset(str(pref_path), prefetch_max_cached=0)
    try:
        pair = pref_ds.get_canonical_pair(0)
        assert pair.chosen.content == "good"
        assert pair.rejected.content == "bad"
        assert recorded_columns == [["chosen", "rejected"]]
    finally:
        pref_ds.stop()
