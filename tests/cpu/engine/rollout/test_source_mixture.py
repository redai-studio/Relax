# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import copy
import json
import os
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from relax.engine.rollout.data_source import RolloutDataSourceWithBuffer
from relax.engine.rollout.source_mixture import (
    BATCH_ID_KEY,
    SOURCE_MIXTURE_PATH,
    SourceMixtureDataSource,
    validate_source_mixture_args,
    validate_source_mixture_batch,
)
from relax.utils.types import Sample


@pytest.fixture
def make_args(tmp_path):
    def make(*, sizes=None, weights=None, batch_size=40, prefetch=0, rows=None):
        sizes = sizes or {"A": 50, "B": 37}
        weights = weights or {"A": 30, "B": 10}
        sources = {}
        for name, size in sizes.items():
            path = tmp_path / f"{name}.jsonl"
            source_rows = (
                rows[name]
                if rows and name in rows
                else [{"text": f"{name}{index}", "metadata": {"row_id": index}} for index in range(size)]
            )
            path.write_text("".join(json.dumps(row) + "\n" for row in source_rows))
            sources[name] = {"path": path.name, "weight": weights[name]}
        config = tmp_path / "mixture.json"
        config.write_text(json.dumps({"sources": sources}))
        return SimpleNamespace(
            rollout_source_mixture=str(config),
            colocate=True,
            use_opd=True,
            opd_type="sglang",
            opd_teacher_key="data_source",
            opd_teacher_routes=json.dumps({name: f"teacher-{name}" for name in sizes}),
            use_streaming_dataset=True,
            rollout_global_dataset=True,
            n_samples_per_prompt=1,
            rollout_function_path="relax.engine.rollout.sglang_rollout.generate_rollout",
            data_source_path="relax.engine.rollout.data_source.RolloutDataSourceWithBuffer",
            prompt_data=None,
            num_epoch=None,
            num_rollout=20,
            global_batch_size=batch_size,
            rollout_batch_size=batch_size,
            over_sampling_batch_size=batch_size,
            hf_checkpoint=None,
            dump_details=None,
            input_key="text",
            multimodal_keys=None,
            label_key=None,
            tool_key=None,
            metadata_key="metadata",
            system_prompt=None,
            apply_chat_template=False,
            apply_chat_template_kwargs={},
            use_audio_in_video=False,
            rollout_seed=17,
            rollout_max_prompt_len=None,
            rollout_shuffle=True,
            streaming_buffer_size=7,
            prefetch_chunk_size=3,
            prefetch_max_cached=prefetch,
            prefetch_num_workers=2,
            buffer_filter_path=None,
            save=str(tmp_path / "checkpoint"),
            load=None,
        )

    return make


@pytest.fixture
def make_source():
    sources = []

    def make(args):
        source = SourceMixtureDataSource(args)
        sources.append(source)
        return source

    yield make
    for source in sources:
        source.close()


def _signature(groups):
    return [
        (s.metadata["data_source"], s.metadata["row_id"], s.index, s.group_index, s.metadata[BATCH_ID_KEY])
        for (s,) in groups
    ]


def test_source_mixture_exact_quotas_and_without_replacement_epochs(make_args, make_source):
    source = make_source(make_args())
    sequences = {"A": [], "B": []}
    for rollout_id in range(15):
        groups = source.get_samples(40, rollout_id)
        assert Counter(group[0].metadata["data_source"] for group in groups) == {"A": 30, "B": 10}
        assert [group[0].index for group in groups] == list(range(40 * rollout_id, 40 * (rollout_id + 1)))
        for (sample,) in groups:
            sequences[sample.metadata["data_source"]].append(sample.metadata["row_id"])
        if rollout_id == 0:
            assert source.index_managers["A"].get_state() == {"epoch_id": 0, "position": 30}
        if rollout_id == 1:
            assert len(set(sequences["A"][:50])) == 50  # 30, then remaining 20 before new epoch's 10
            assert source.index_managers["A"].get_state() == {"epoch_id": 1, "position": 10}
    for name, size in (("A", 50), ("B", 37)):
        sequence = sequences[name]
        for offset in range(0, len(sequence) - size + 1, size):
            assert sorted(sequence[offset : offset + size]) == list(range(size))
        assert sequence[:size] != sequence[size : 2 * size]


def test_source_mixture_quota_larger_than_source_keeps_occurrence_ids_unique(make_args, make_source):
    source = make_source(make_args(sizes={"A": 2, "B": 1}, weights={"A": 3, "B": 1}, batch_size=8))
    groups = source.get_samples(8, 0)
    assert Counter(s.metadata["row_id"] for (s,) in groups if s.metadata["data_source"] == "A") == {0: 3, 1: 3}
    assert len({s.index for (s,) in groups}) == 8
    assert len({s.group_index for (s,) in groups}) == 8
    assert source.index_managers["A"].get_state() == {"epoch_id": 2, "position": 2}


def test_source_mixture_yaml_paths_decimal_weights_and_idempotent_validation(make_args, monkeypatch):
    args = make_args(batch_size=10)
    config = Path(args.rollout_source_mixture)
    monkeypatch.setenv("MIXTURE_TEST_DATA_DIR", str(config.parent))
    config.write_text(
        "sources:\n  A: {path: A.jsonl, weight: 0.10000000000000000001}\n"
        "  B: {path: '${MIXTURE_TEST_DATA_DIR}/B.jsonl', weight: 0.40000000000000000004}\n"
    )
    validate_source_mixture_args(args)
    assert args.source_mixture_quotas == {"A": 2, "B": 8}
    assert args.data_source_path == SOURCE_MIXTURE_PATH
    assert [source["path"] for source in args.source_mixture_spec] == [
        str(config.parent / "A.jsonl"),
        str(config.parent / "B.jsonl"),
    ]
    before = copy.deepcopy(args.source_mixture_spec)
    validate_source_mixture_args(args)
    assert args.source_mixture_spec == before
    assert args.source_mixture_quotas == {"A": 2, "B": 8}


@pytest.mark.parametrize("weight", [0, -1, "NaN", "Infinity", "bad", True])
def test_source_mixture_invalid_weight_fails(make_args, weight):
    args = make_args(weights={"A": weight, "B": 1})
    with pytest.raises(ValueError, match="finite and positive"):
        validate_source_mixture_args(args)


@pytest.mark.parametrize(
    "batch_size,weights,expected",
    [
        (100, {"A": 30, "B": 40, "C": 10, "D": 10, "E": 10}, {"A": 30, "B": 40, "C": 10, "D": 10, "E": 10}),
        (32, {"A": 3, "B": 2}, {"A": 19, "B": 13}),
        (10, {"C": 1, "A": 1, "B": 1}, {"C": 4, "A": 3, "B": 3}),
        (2, {"C": 1, "A": 1, "B": 1}, {"C": 1, "A": 1, "B": 0}),
        (1, {"A": "1", "B": "1.00000000000000000001"}, {"A": 0, "B": 1}),
    ],
)
def test_source_mixture_largest_remainder_quotas(make_args, make_source, caplog, batch_size, weights, expected):
    args = make_args(sizes={name: 3 for name in weights}, weights=weights, batch_size=batch_size)
    validate_source_mixture_args(args)
    assert args.source_mixture_quotas == expected
    if 0 in expected.values():
        assert any(
            record.levelname == "WARNING" and "will not be sampled in any batch" in record.getMessage()
            for record in caplog.records
        )
    source = make_source(args)
    groups = source.get_samples(batch_size, 0)
    assert validate_source_mixture_batch(args, groups, 0) == expected
    for name, quota in expected.items():
        if quota == 0:
            assert source.index_managers[name].get_state() == {"epoch_id": 0, "position": 0}


def test_source_mixture_teacher_route_mismatch_fails(make_args):
    args = make_args(weights={"A": 1, "B": 2}, batch_size=10)
    args.opd_teacher_routes = '{"A":"teacher-A","C":"teacher-C"}'
    with pytest.raises(ValueError, match="exactly match"):
        validate_source_mixture_args(args)


@pytest.mark.parametrize(
    "name,value",
    [
        ("colocate", False),
        ("use_opd", False),
        ("opd_type", "megatron"),
        ("opd_teacher_key", "other"),
        ("use_streaming_dataset", False),
        ("rollout_global_dataset", False),
        ("n_samples_per_prompt", 2),
        ("rollout_function_path", "custom.generate"),
        ("data_source_path", "custom.Source"),
        ("debug_train_only", True),
        ("load_debug_rollout_data", "replay.pt"),
        ("fully_async", True),
        ("hybrid", True),
        ("partial_rollout", True),
        ("use_dynamic_global_batch_size", True),
        ("dynamic_sampling_filter_path", "custom.filter"),
        ("buffer_filter_path", "custom.buffer"),
        ("custom_convert_samples_to_train_data_path", "custom.convert"),
        ("custom_reward_post_process_path", "custom.reward"),
        ("rollout_sample_filter_path", "custom.filter"),
        ("prompt_data", "other.jsonl"),
        ("num_epoch", 1),
        ("num_rollout", None),
        ("num_rollout", True),
        ("num_steps_per_rollout", 2),
        ("num_iters_per_train_update", 2),
        ("global_batch_size", 80),
        ("over_sampling_batch_size", 80),
    ],
)
def test_source_mixture_unsupported_contract_fails(make_args, name, value):
    args = make_args()
    setattr(args, name, value)
    with pytest.raises(ValueError):
        validate_source_mixture_args(args)


def test_source_mixture_empty_source_fails(make_args, make_source):
    args = make_args(sizes={"A": 0, "B": 3})
    with pytest.raises(ValueError, match="Source 'A' is empty"):
        make_source(args)


@pytest.mark.parametrize("prefetch", [0, 4])
def test_source_mixture_invalid_json_fails(make_args, make_source, monkeypatch, prefetch):
    args = make_args(prefetch=prefetch)
    args.multimodal_keys = {"image": "images"}
    (Path(args.rollout_source_mixture).parent / "A.jsonl").write_text("malformed json\n")
    source = make_source(args)
    if prefetch:
        buffer = source.dataset._prefetch_buffer
        set_index_order = buffer.set_index_order

        def wait_for_first(indices):
            set_index_order(indices)
            assert buffer.wait_for(indices[0], timeout=2)

        monkeypatch.setattr(buffer, "set_index_order", wait_for_first)
    error = ValueError if prefetch else json.JSONDecodeError
    match = "Source 'A' has no usable records" if prefetch else None
    with pytest.raises(error, match=match):
        source.get_samples(40, 0)
    if prefetch:
        assert buffer._prefetch_hits >= 1


@pytest.mark.parametrize("prefetch", [0, 4])
@pytest.mark.parametrize("stage", ["processing", "length_check", "after_resume"])
def test_source_mixture_processing_errors_follow_tolerant_policy(
    make_args, make_source, monkeypatch, caplog, prefetch, stage
):
    args = make_args(sizes={"A": 3, "B": 2}, batch_size=4, prefetch=prefetch)
    args.multimodal_keys = {"image": "images"}
    if stage == "length_check":
        args.rollout_max_prompt_len = 5
    source = make_source(args)
    first_rollout = 0
    if stage == "after_resume":
        source.get_samples(4, 0)
        source.save(0)
        args.load = args.save
        source.load(0)
        first_rollout = 1
    bad_index = copy.copy(source.index_managers["A"]).get_next_indices(1)[0][0]
    process = source.dataset._process_data

    def failing_processor(raw, **kwargs):
        if stage == "length_check" or raw["text"] == f"A{bad_index}":
            raise RuntimeError("sample processing failed")
        return process(raw)

    if stage == "length_check":
        source.dataset.tokenizer = SimpleNamespace(apply_chat_template=failing_processor)
    else:
        monkeypatch.setattr(source.dataset, "_process_data", failing_processor)
    if prefetch:
        buffer = source.dataset._prefetch_buffer
        set_index_order = buffer.set_index_order

        def wait_for_first(indices):
            set_index_order(indices)
            assert buffer.wait_for(indices[0], timeout=2)

        monkeypatch.setattr(buffer, "set_index_order", wait_for_first)
    for rollout_id in range(first_rollout, first_rollout + 2):
        groups = source.get_samples(4, rollout_id)
        assert Counter(s.metadata["data_source"] for (s,) in groups) == {"A": 3, "B": 1}
        expected_rows = {0, 1, 2} if stage == "length_check" else {0, 1, 2} - {bad_index}
        assert {s.metadata["row_id"] for (s,) in groups if s.metadata["data_source"] == "A"} == expected_rows
    warning = "Error checking length" if stage == "length_check" else "Error processing data"
    assert any(warning in record.getMessage() for record in caplog.records)
    if prefetch:
        assert buffer._prefetch_hits >= 1


@pytest.mark.parametrize("prefetch", [0, 4])
def test_source_mixture_overlong_rows_only_refill_their_source(make_args, make_source, prefetch):
    args = make_args(
        sizes={"A": 3, "B": 2},
        weights={"A": 3, "B": 1},
        batch_size=4,
        prefetch=prefetch,
        rows={
            "A": [
                {"text": "yes", "metadata": {"row_id": 0}},
                {"text": "too long to pass", "metadata": {"row_id": 1}},
                {"text": "ok", "metadata": {"row_id": 2}},
            ]
        },
    )
    args.multimodal_keys = {"image": "images"}
    args.rollout_max_prompt_len = 5
    source = make_source(args)
    source.dataset.tokenizer = lambda text, **kwargs: {"input_ids": list(text)}
    source.dataset.tokenizer.apply_chat_template = lambda prompt, **kwargs: prompt[0]["content"][0]["text"]
    for rollout_id in range(4):
        groups = source.get_samples(4, rollout_id)
        assert Counter(s.metadata["data_source"] for (s,) in groups) == {"A": 3, "B": 1}
        assert {s.metadata["row_id"] for (s,) in groups if s.metadata["data_source"] == "A"} == {0, 2}


@pytest.mark.parametrize("prefetch", [0, 4])
@pytest.mark.parametrize("failure", ["overlong", "processing"])
def test_source_mixture_all_unusable_source_fails_with_bounded_reads(
    make_args, make_source, monkeypatch, prefetch, failure
):
    args = make_args(sizes={"A": 2, "B": 1}, batch_size=4, prefetch=prefetch)
    args.multimodal_keys = {"image": "images"}
    args.rollout_max_prompt_len = 1
    source = make_source(args)
    source.dataset.tokenizer = lambda text, **kwargs: {"input_ids": list(text)}
    source.dataset.tokenizer.apply_chat_template = lambda prompt, **kwargs: prompt[0]["content"][0]["text"]
    if failure == "processing":

        def fail(raw):
            raise RuntimeError("sample processing failed")

        monkeypatch.setattr(source.dataset, "_process_data", fail)
    get_sample = source._get_sample
    reads = []

    def record_read(index):
        reads.append(index)
        return get_sample(index)

    monkeypatch.setattr(source, "_get_sample", record_read)
    with pytest.raises(ValueError, match="Source 'A' has no usable records"):
        source.get_samples(4, 0)
    assert len(reads) == 4


@pytest.mark.parametrize("location", ["top", "metadata", "both"])
@pytest.mark.parametrize("prefetch", [0, 4])
def test_source_mixture_conflicting_source_fails(make_args, make_source, location, prefetch):
    record = {"text": "a", "metadata": {"row_id": 0}}
    if location in ("top", "both"):
        record["data_source"] = "wrong"
    if location in ("metadata", "both"):
        record["metadata"]["data_source"] = "A" if location == "both" else "wrong"
    args = make_args(sizes={"A": 1, "B": 1}, rows={"A": [record]}, prefetch=prefetch)
    args.multimodal_keys = {"image": "images"}
    source = make_source(args)
    with pytest.raises(ValueError, match="Source 'A' has no usable records" if prefetch else "conflicts"):
        source.get_samples(40, 0)


def test_source_mixture_copies_cached_sample_and_shares_heavy_payload(make_args, make_source):
    args = make_args(sizes={"A": 1, "B": 1}, batch_size=4, weights={"A": 3, "B": 1})
    source = make_source(args)
    heavy_payload = {"image": torch.ones(2, 2)}
    process = source.dataset._process_raw_data

    def attach_payload(raw, **kwargs):
        sample = process(raw, **kwargs)
        sample.multimodal_inputs = heavy_payload
        return sample

    source.dataset._process_raw_data = attach_payload
    groups = source.get_samples(4, 0)
    first, second = groups[0][0], groups[1][0]
    assert first.multimodal_inputs is second.multimodal_inputs is heavy_payload
    first.metadata["changed"] = True
    first.tokens.append(9)
    cached = source.dataset.buffer.get(0)
    assert cached.metadata == {"row_id": 0}
    assert second.metadata == {"row_id": 0, "data_source": "A", BATCH_ID_KEY: 0}
    assert cached.tokens == second.tokens == []
    assert source.dataset.reader[0]["metadata"] == {"row_id": 0}


def test_source_mixture_checkpoint_uses_committed_batch_after_next_batch_prefetch(make_args, make_source):
    batch_size = 32
    args = make_args(batch_size=batch_size, weights={"A": 3, "B": 2})
    source = make_source(args)
    source.get_samples(batch_size, 0)
    expected_next = _signature(source.get_samples(batch_size, 1))
    source.save(0)  # batch 1 has already been fetched, but save must select batch 0's cursor
    expected_after = _signature(source.get_samples(batch_size, 2))
    state = torch.load(Path(args.save) / "dataset/global_dataset_state_dict_0.pt", weights_only=True)
    assert state["next_rollout_id"] == 1
    assert state["sources"]["A"]["position"] == 19
    # Only plain checkpoint state is saved; no image/tensor cache or permutation lists.
    json.dumps(state)
    resume_args = copy.copy(args)
    resume_args.load = args.save
    resumed = make_source(resume_args)
    resumed.load(0)
    assert _signature(resumed.get_samples(batch_size, 1)) == expected_next
    assert _signature(resumed.get_samples(batch_size, 2)) == expected_after
    # Restoring on a live source also discards previously issued/prefetched state.
    source.args.load = args.save
    source.load(0)
    assert _signature(source.get_samples(batch_size, 1)) == expected_next


@pytest.mark.parametrize("change", ["relocation", "weights", "equivalent_slice"])
def test_source_mixture_resume_accepts_equivalent_data_and_config(make_args, make_source, change):
    args = make_args(prefetch=4)
    source = make_source(args)
    source.get_samples(40, 0)
    source.save(0)
    resume_args = copy.copy(args)
    resume_args.load = args.save
    config = Path(args.rollout_source_mixture)
    if change == "relocation":
        moved = config.parent / "relocated"
        moved.mkdir()
        for path in config.parent.glob("*.jsonl"):
            target = moved / path.name
            target.write_bytes(path.read_bytes())
            stat = path.stat()
            os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        new_config = moved / "new-run-config.yaml"
        new_config.write_text(config.read_text())
        resume_args.rollout_source_mixture = str(new_config)
    else:
        contents = json.loads(config.read_text())
        if change == "weights":
            for spec in contents["sources"].values():
                spec["weight"] *= 2
        else:
            contents["sources"]["A"]["path"] += "@[0:50]"
        config.write_text(json.dumps(contents))
    validate_source_mixture_args(resume_args)
    resumed = make_source(resume_args)
    resumed.load(0)
    assert _signature(resumed.get_samples(40, 1)) == _signature(source.get_samples(40, 1))


@pytest.mark.parametrize(
    "change",
    [
        "seed",
        "route",
        "file",
        "tail",
        "row_order",
        "slice",
        "source_order",
        "quotas",
        "batch_size",
        "multimodal",
        "missing",
    ],
)
def test_source_mixture_resume_rejects_missing_or_changed_state(make_args, make_source, change):
    args = make_args()
    if change == "tail":
        path = Path(args.rollout_source_mixture).parent / "A.jsonl"
        path.write_bytes(path.read_bytes().replace(b'"A0"', b'"' + b"a" * (1024 * 1024) + b'"'))
    source = make_source(args)
    source.get_samples(40, 0)
    source.save(0)
    resume_args = copy.copy(args)
    resume_args.load = args.save
    if change == "seed":
        resume_args.rollout_seed += 1
    elif change == "route":
        resume_args.opd_teacher_routes = '{"A":"different-teacher","B":"teacher-B"}'
    elif change == "file":
        with (Path(args.rollout_source_mixture).parent / "A.jsonl").open("a") as stream:
            stream.write('{"text":"new"}\n')
    elif change in ("tail", "row_order"):
        path = Path(args.rollout_source_mixture).parent / "A.jsonl"
        stat = path.stat()
        original = path.read_bytes()
        if change == "tail":
            changed = original.replace(b'"A49"', b'"Z49"')
            assert len(original) > 1024 * 1024
            assert changed[: 1024 * 1024] == original[: 1024 * 1024]
        else:
            changed = b"".join(reversed(original.splitlines(keepends=True)))
        assert changed != original and len(changed) == len(original)
        path.write_bytes(changed)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    elif change in ("slice", "source_order", "quotas"):
        path = Path(args.rollout_source_mixture)
        contents = json.loads(path.read_text())
        if change == "slice":
            contents["sources"]["A"]["path"] += "@[1:50]"
        elif change == "source_order":
            contents["sources"] = dict(reversed(list(contents["sources"].items())))
        else:
            contents["sources"]["A"]["weight"] = 10
        path.write_text(json.dumps(contents))
    elif change == "batch_size":
        resume_args.global_batch_size = resume_args.rollout_batch_size = resume_args.over_sampling_batch_size = 32
    elif change == "multimodal":
        resume_args.image_max_token_num = 64
    elif change == "missing":
        (Path(args.save) / "dataset/global_dataset_state_dict_0.pt").unlink()
    validate_source_mixture_args(resume_args)
    resumed = make_source(resume_args)
    with pytest.raises((ValueError, FileNotFoundError)):
        resumed.load(0)


@pytest.mark.parametrize(
    "batch_size,weights",
    [(8, {"A": 3, "B": 2}), (1, {"A": 1, "B": 3})],
)
def test_source_mixture_prefetch_toggle_preserves_order_and_resume(
    make_args, make_source, monkeypatch, batch_size, weights
):
    args = make_args(sizes={"A": 5, "B": 3}, weights=weights, batch_size=batch_size)
    args.multimodal_keys = {"image": "images"}
    reads = []
    read_sample = SourceMixtureDataSource._read_sample

    def record_read(source, index):
        reads.append(index)
        return read_sample(source, index)

    monkeypatch.setattr(SourceMixtureDataSource, "_read_sample", record_read)
    synchronous = make_source(args)
    prefetch_args = copy.copy(args)
    prefetch_args.prefetch_max_cached = 4
    prefetched = make_source(prefetch_args)
    for rollout_id in range(4):
        actual = prefetched.get_samples(batch_size, rollout_id)
        assert _signature(actual) == _signature(synchronous.get_samples(batch_size, rollout_id))
        assert validate_source_mixture_batch(args, actual, rollout_id) == args.source_mixture_quotas
        if rollout_id == 1:
            prefetched.save(0)
    prefetched.args.load = args.save
    old_buffer = prefetched.dataset._prefetch_buffer
    prefetched.load(0)
    assert prefetched.dataset._prefetch_buffer is not old_buffer
    assert not old_buffer.is_alive
    synchronous.args.load = args.save
    synchronous.load(0)
    assert _signature(prefetched.get_samples(batch_size, 1)) == _signature(synchronous.get_samples(batch_size, 1))
    if args.source_mixture_quotas["A"] == 0:
        for source in (synchronous, prefetched):
            source.close()
            assert source.index_managers["A"].get_state() == {"epoch_id": 0, "position": 0}
        assert reads
        assert all(index >= 5 for index in reads)  # Source A owns indices 0 through 4.


def test_source_mixture_batch_validation_rejects_invalid_output(make_args, make_source):
    batch_size = 4
    expected = {"A": 3, "B": 1}
    args = make_args(batch_size=batch_size, weights=expected)
    source = make_source(args)
    groups = source.get_samples(batch_size, 0)
    assert validate_source_mixture_batch(args, groups, 0) == expected
    for (sample,) in groups:
        sample.status = Sample.Status.COMPLETED
    groups[0][0].status = Sample.Status.TRUNCATED
    assert validate_source_mixture_batch(args, groups, 0, completed=True) == expected
    for mutation in (
        "short",
        "group",
        "duplicate",
        "wrong_batch",
        "wrong_source",
        "unknown_source",
        "incomplete",
        "removed",
    ):
        bad = copy.deepcopy(groups)
        if mutation == "short":
            bad.pop()
        elif mutation == "group":
            bad[0].append(copy.deepcopy(bad[0][0]))
        elif mutation == "duplicate":
            bad[0][0].index = bad[1][0].index
        elif mutation == "wrong_batch":
            bad[0][0].metadata[BATCH_ID_KEY] = 1
        elif mutation == "wrong_source":
            bad[0][0].metadata["data_source"] = "B"
        elif mutation == "unknown_source":
            bad[0][0].metadata["data_source"] = "unknown"
        elif mutation == "incomplete":
            bad[0][0].status = Sample.Status.ABORTED
        elif mutation == "removed":
            bad[0][0].remove_sample = True
        with pytest.raises(ValueError):
            validate_source_mixture_batch(args, bad, 0, completed=True)
    with pytest.raises(ValueError, match="Expected source mixture rollout"):
        source.get_samples(batch_size, 0)
    with pytest.raises(ValueError, match="complete global batch"):
        source.get_samples(batch_size - 1, 1)
    with pytest.raises(ValueError, match="returning samples"):
        source.add_samples(groups)
    source.add_samples([])


def test_source_mixture_slice_and_parquet_share_one_dataset(make_args, make_source):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    args = make_args(sizes={"A": 5, "B": 3}, weights={"A": 1, "B": 1}, batch_size=4)
    config_path = Path(args.rollout_source_mixture)
    parquet_path = config_path.parent / "B.parquet"
    pq.write_table(
        pa.Table.from_pylist([{"text": f"B{i}", "metadata": {"row_id": i}} for i in range(3)]), parquet_path
    )
    config = json.loads(config_path.read_text())
    config["sources"]["A"]["path"] = "A.jsonl@[1:3]"
    config["sources"]["B"]["path"] = "B.parquet"
    config_path.write_text(json.dumps(config))
    source = make_source(args)
    assert len(source) == source.lengths() == 5
    groups = source.get_samples(4, 0)
    assert {s.metadata["row_id"] for (s,) in groups if s.metadata["data_source"] == "A"} == {1, 2}
    assert len(source.dataset.reader.readers) == 2


def test_source_mixture_flag_absent_preserves_default_source(make_args):
    args = make_args()
    args.rollout_source_mixture = None
    args.prompt_data = str(Path(args.save).parent / "A.jsonl")
    before = vars(args).copy()
    validate_source_mixture_args(args)
    assert vars(args) == before
    ordinary = RolloutDataSourceWithBuffer(args)
    assert len(ordinary.get_samples(3)) == 3
    assert validate_source_mixture_batch(args, [], 100, completed=True) == {}
