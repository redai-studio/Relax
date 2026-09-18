# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Tests for full-text repetition detection and its offline diagnostic."""

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

# Import the detection core from its dependency-free home so these tests run
# on a plain CPU box; metric_utils re-exports the same objects for callers that
# already go through the metrics package.
from relax.utils import repetition as metric_utils
from relax.utils.repetition import (
    REPETITION_WINDOW_SIZE_CHARS,
    REPETITION_WINDOW_STRIDE_CHARS,
    has_repetition,
    repetition_window_bounds,
    scan_repetition,
)
from relax.utils.repetition_diagnose import diagnose_samples


WINDOW = REPETITION_WINDOW_SIZE_CHARS
STRIDE = REPETITION_WINDOW_STRIDE_CHARS


def _incompressible(n: int, salt: str = "") -> str:
    """Deterministic low-compressibility filler of exactly ``n`` characters."""
    chunks = []
    total = 0
    i = 0
    while total < n:
        chunk = hashlib.sha256(f"{salt}:{i}".encode()).hexdigest()
        chunks.append(chunk)
        total += len(chunk)
        i += 1
    return "".join(chunks)[:n]


# --------------------------------------------------------------------------
# Fixed samples: repetition at the beginning / middle / end
# --------------------------------------------------------------------------


def test_repetition_at_beginning_is_detected():
    text = "spam" * (WINDOW // 4) + _incompressible(3 * WINDOW, "begin")
    report = scan_repetition(text)

    assert report.has_repetition
    assert report.hit_intervals[0][0] == 0


def test_repetition_in_middle_with_clean_ending_is_detected():
    """The fixed "middle repetition, normal ending" sample the old suffix-only
    check could not see."""
    head = _incompressible(3 * WINDOW, "head")
    middle = "repeat this sentence over and over. " * 600  # ~21.6k chars
    tail = _incompressible(3 * WINDOW, "tail")
    text = head + middle + tail

    # The old behaviour: only the final 10k chars were inspected, so a clean
    # ending hid the middle repetition entirely.
    assert metric_utils.window_compression_ratio(text[-WINDOW:]) <= 10.0

    report = scan_repetition(text)
    assert report.has_repetition
    start, end = report.hit_intervals[0]
    # The hit lands inside the repetitive span, not in the clean head/tail.
    assert start >= len(head) - WINDOW
    assert end <= len(head) + len(middle) + WINDOW


def test_repetition_at_end_is_detected():
    text = _incompressible(3 * WINDOW, "end") + "spam" * (WINDOW // 4)
    report = scan_repetition(text)

    assert report.has_repetition
    assert report.hit_intervals[-1][1] == len(text)


def test_clean_response_is_a_negative_control():
    text = _incompressible(5 * WINDOW, "clean")
    report = scan_repetition(text)

    assert not report.has_repetition
    assert report.hit_intervals == []
    assert report.max_compression_ratio is not None
    assert report.max_compression_ratio <= report.threshold


# --------------------------------------------------------------------------
# Boundaries: empty, short, exact window, unaligned tail
# --------------------------------------------------------------------------


def test_empty_string_is_not_repetitive():
    report = scan_repetition("")

    assert not report.has_repetition
    assert report.text_length == 0
    assert report.num_windows_scanned == 0
    assert report.max_compression_ratio is None
    assert not has_repetition("")


def test_short_response_below_one_window_is_scanned():
    report = scan_repetition("x" * (WINDOW - 1))

    assert report.has_repetition
    assert report.num_windows_scanned == 1
    assert report.hit_intervals == [(0, WINDOW - 1)]
    assert repetition_window_bounds(WINDOW - 1) == [(0, WINDOW - 1)]


def test_exactly_one_window_is_scanned_once():
    report = scan_repetition("x" * WINDOW)

    assert report.num_windows_scanned == 1
    assert report.has_repetition
    assert report.hit_intervals == [(0, WINDOW)]


def test_window_bounds_are_stride_aligned_and_overlapping():
    bounds = repetition_window_bounds(3 * WINDOW)

    assert bounds[0] == (0, WINDOW)
    assert bounds[1] == (STRIDE, STRIDE + WINDOW)
    assert bounds[-1] == (2 * WINDOW, 3 * WINDOW)
    # Overlapping: each window starts before the previous one ends.
    assert all(nxt[0] < cur[1] for cur, nxt in zip(bounds, bounds[1:]))


def test_unaligned_final_window_covers_the_exact_suffix():
    length = 2 * WINDOW + 1234  # not stride-aligned
    bounds = repetition_window_bounds(length)

    assert bounds[-1] == (length - WINDOW, length)
    assert bounds[-2][0] % STRIDE == 0


def test_unaligned_tail_repetition_is_detected():
    text = _incompressible(2 * WINDOW + 1234, "pre") + "spam" * (WINDOW // 4)
    report = scan_repetition(text)

    assert report.has_repetition
    assert report.hit_intervals[-1][1] == len(text)


def test_unaligned_repetition_can_be_diluted_by_clean_context():
    """Full coverage does not guarantee detection of a one-window repeated
    run."""
    text = _incompressible(2500, "offset-head") + "spam" * 2500 + _incompressible(17500, "offset-tail")
    report = scan_repetition(text)

    assert metric_utils.window_compression_ratio(text[2500:12500]) > report.threshold
    assert metric_utils._union_length(repetition_window_bounds(len(text))) == len(text)
    assert not report.has_repetition


def test_stride_wider_than_window_is_rejected():
    """A stride wider than the window would leave unscanned gaps, so repetition
    inside a gap would be missed with no indication."""
    with pytest.raises(ValueError, match="must not exceed window_size"):
        repetition_window_bounds(30_000, window_size=WINDOW, stride=2 * WINDOW)

    with pytest.raises(ValueError, match="must not exceed window_size"):
        scan_repetition("x" * 30_000, window_size=WINDOW, stride=2 * WINDOW)


def test_stride_equal_to_window_is_allowed_and_gap_free():
    bounds = repetition_window_bounds(3 * WINDOW, window_size=WINDOW, stride=WINDOW)

    assert metric_utils._union_length(bounds) == 3 * WINDOW


@pytest.mark.parametrize("bad", [{"window_size": 0}, {"window_size": -1}, {"stride": 0}, {"stride": -1}])
def test_non_positive_window_parameters_are_rejected(bad):
    with pytest.raises(ValueError, match="must be positive"):
        repetition_window_bounds(30_000, **bad)


def test_full_coverage_has_no_gaps():
    """No middle window is silently skipped: the union of scanned windows
    covers the whole text."""
    for length in (WINDOW, WINDOW + 1, 3 * WINDOW + 7, 137_000):
        bounds = repetition_window_bounds(length)
        covered = metric_utils._union_length(bounds)
        assert covered == length, f"{length=} covered={covered}"


# --------------------------------------------------------------------------
# Threshold semantics and Unicode offsets
# --------------------------------------------------------------------------


def test_ratio_exactly_at_threshold_is_not_a_hit(monkeypatch):
    """Strictly greater than the threshold, matching the original check."""
    monkeypatch.setattr(metric_utils, "window_compression_ratio", lambda _: 10.0)

    assert not scan_repetition("x" * WINDOW).has_repetition


def test_ratio_just_above_threshold_is_a_hit(monkeypatch):
    monkeypatch.setattr(metric_utils, "window_compression_ratio", lambda _: 10.0001)

    assert scan_repetition("x" * WINDOW).has_repetition


def test_chinese_repetition_is_detected_and_offsets_are_character_based():
    head = "".join(chr(0x4E00 + (i * 7919) % 0x4000) for i in range(2 * WINDOW))
    repeated = "这是一句会不断重复出现的中文句子。" * 800
    text = head + repeated

    report = scan_repetition(text)
    assert report.has_repetition
    # Offsets index characters, not UTF-8 bytes: slicing by them round-trips.
    start, end = report.hit_intervals[0]
    assert len(text[start:end]) == end - start
    assert end <= len(text) < len(text.encode("utf-8"))


# --------------------------------------------------------------------------
# Report structure
# --------------------------------------------------------------------------


def test_covered_chars_counts_the_union_of_overlapping_hits():
    text = "spam" * (5 * WINDOW // 4)  # many overlapping hit windows
    report = scan_repetition(text)

    assert len(report.hit_windows) > 1
    # Naive sum would double-count the overlaps; the union must not.
    naive = sum(w.length for w in report.hit_windows)
    assert report.covered_chars < naive
    assert report.covered_chars == len(text)


def test_stop_at_first_hit_short_circuits():
    text = "spam" * (5 * WINDOW // 4)

    full = scan_repetition(text)
    short = scan_repetition(text, stop_at_first_hit=True)

    assert short.has_repetition == full.has_repetition
    assert len(short.hit_windows) == 1
    assert short.num_windows_scanned < full.num_windows_scanned


@pytest.mark.parametrize("repetitive", [False, True])
def test_boolean_scan_visits_all_windows_or_stops_at_first_hit(monkeypatch, repetitive):
    text = "spam" * 250_000 if repetitive else _incompressible(1_000_000, "all-windows")
    ratio = metric_utils.window_compression_ratio
    calls = []

    def measured_ratio(window):
        calls.append(len(window))
        return ratio(window)

    monkeypatch.setattr(metric_utils, "window_compression_ratio", measured_ratio)
    result = has_repetition(text)
    assert result is repetitive
    assert len(calls) == (1 if repetitive else 199)
    assert result == scan_repetition(text).has_repetition


def test_short_clean_response_is_scanned_without_false_positive():
    report = scan_repetition("A short and ordinary answer.")
    assert not report.has_repetition
    assert report.num_windows_scanned == 1
    assert report.max_compression_ratio is not None


def test_has_repetition_preserves_boolean_interface():
    assert has_repetition("spam" * (WINDOW // 4)) is True
    assert has_repetition(_incompressible(3 * WINDOW, "bool")) is False


def test_report_to_dict_is_json_serializable():
    report = scan_repetition("spam" * (WINDOW // 2))
    payload = json.loads(json.dumps(report.to_dict()))

    assert payload["has_repetition"] is True
    assert payload["num_windows_scanned"] == report.num_windows_scanned
    assert payload["hit_windows"][0]["start"] == 0
    assert payload["covered_chars"] == report.covered_chars


# --------------------------------------------------------------------------
# Offline diagnostic over real rollout dumps
# --------------------------------------------------------------------------


def test_diagnose_dump_written_by_the_real_dump_writer(tmp_path):
    """End-to-end: write a dump with the production writer, read it with the
    offline entry."""
    torch = pytest.importorskip("torch")
    from relax.utils.repetition_diagnose import diagnose_dumps, main
    from relax.utils.training.train_dump_utils import save_debug_rollout_data
    from relax.utils.types import Sample

    clean = _incompressible(3 * WINDOW, "dump-clean")
    dirty = _incompressible(WINDOW, "dump-head") + "repeat me. " * 2_000 + _incompressible(WINDOW, "dump-tail")
    samples = [
        Sample(index=0, group_index=0, response=clean),
        Sample(index=1, group_index=0, response=dirty),
    ]

    dump_path = tmp_path / "rollout_data" / "7.pt"

    class _Args:
        save_debug_rollout_data = str(tmp_path / "rollout_data" / "{rollout_id}.pt")

    save_debug_rollout_data(_Args(), samples, rollout_id=7, evaluation=False)
    assert dump_path.exists()

    report = diagnose_dumps([dump_path])
    assert report["num_samples"] == 2
    assert report["num_repetitive_samples"] == 1
    assert report["repetition_frac"] == 0.5

    file_report = report["files"][0]
    assert file_report["rollout_id"] == 7
    hit = next(s for s in file_report["samples"] if s["has_repetition"])
    assert hit["index"] == 1
    assert hit["hit_windows"]

    # The offline verdict agrees with the shared boolean helper.
    assert [has_repetition(s.response) for s in samples] == [
        s["has_repetition"] for s in sorted(file_report["samples"], key=lambda s: s["position"])
    ]

    # The CLI writes a JSON report.
    out = tmp_path / "report.json"
    main([str(dump_path), "-o", str(out)])
    assert json.loads(out.read_text())["num_repetitive_samples"] == 1

    del torch


@pytest.fixture
def cpu_rollout_module(monkeypatch):
    """Execute the complete production module with deployment-only imports
    stubbed.

    Metrics, Sample, serialization and dump readers remain real. Stubs fail if
    invoked: the test must never accidentally exercise a deployment operation.
    The private module name and monkeypatch cleanup avoid poisoning later tests.
    """
    before_modules = set(sys.modules)
    try:
        ray = pytest.importorskip("ray", reason="the production rollout metric path needs ray")

        def deployment_only(*args, **kwargs):
            raise AssertionError("CPU metrics test invoked a deployment dependency")

        imports = {
            "sglang.srt.constants": [
                "GPU_MEMORY_TYPE_CUDA_GRAPH",
                "GPU_MEMORY_TYPE_KV_CACHE",
                "GPU_MEMORY_TYPE_WEIGHTS",
            ],
            "relax.backends.sglang.sglang_engine": ["SGLangEngine"],
            "relax.engine.rollout.base_types": ["call_rollout_fn"],
            "relax.utils.health_monitor": ["RolloutHealthMonitor"],
            "relax.utils.s3_model_loader": ["build_runai_streamer_env_for_load", "prepare_model_maybe_update_args"],
            "relax.utils.utils": ["get_ray_accelerator_kwargs"],
            "relax.distributed.ray.utils": ["NOSET_VISIBLE_DEVICES_ENV_VARS_LIST", "Lock"],
            "relax.utils.tracking_utils": ["init_tracking"],
            "transfer_queue": [],
        }
        for name, attributes in imports.items():
            module = ModuleType(name)
            for attribute in attributes:
                setattr(module, attribute, deployment_only)
            monkeypatch.setitem(sys.modules, name, module)
        monkeypatch.setattr(ray, "remote", lambda **kwargs: lambda cls: cls)
        monkeypatch.setattr(ray, "method", lambda **kwargs: lambda fn: fn)
        import relax.utils

        monkeypatch.setattr(relax.utils, "tracking_utils", sys.modules["relax.utils.tracking_utils"], raising=False)
        path = Path(__file__).resolve().parents[2] / "relax/distributed/ray/rollout.py"
        name = "relax.distributed.ray._repetition_cpu_test"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        yield module
    finally:
        monkeypatch.undo()
        # Imported helpers can retain deployment stubs after monkeypatch restores
        # sys.modules. Remove new Relax modules and their cached package attrs.
        added_modules = [name for name in sys.modules if name.startswith("relax.") and name not in before_modules]
        for name in sorted(added_modules, key=lambda name: name.count("."), reverse=True):
            imported = sys.modules.pop(name)
            parent_name, _, attribute = name.rpartition(".")
            parent = sys.modules.get(parent_name)
            if parent is not None and getattr(parent, attribute, None) is imported:
                delattr(parent, attribute)


@pytest.mark.parametrize("evaluation", [False, True], ids=["train", "eval"])
@pytest.mark.parametrize("dump_format", ["jsonl", "pt"])
def test_rollout_metric_entry_matches_real_dump_diagnosis(tmp_path, cpu_rollout_module, evaluation, dump_format):
    """Real metrics -> real train/eval dump -> fresh-process CLI, without
    GPUs."""
    from relax.utils.types import Sample

    clean = _incompressible(30_000, "entry-clean")
    repeated = "spam" * 5000
    texts = [clean, repeated + clean, clean + repeated + clean, clean + repeated, "x" * 9999, ""]
    samples = [
        Sample(index=100 + i, group_index=20 + i, response=text, response_length=len(text), reward=float(i))
        for i, text in enumerate(texts)
    ]
    snapshot = copy.deepcopy([sample.to_dict() for sample in samples])
    args = SimpleNamespace(
        log_reward_category=None,
        reward_key=None,
        log_passrate=False,
        advantage_estimator="ppo",
        save_debug_rollout_data=str(tmp_path / "{rollout_id}.pt"),
        rollout_result_dir=str(tmp_path),
    )
    online = cpu_rollout_module.compute_metrics_from_samples(args, samples, include_rloo_diagnostics=False)
    data = {"dataset-a": {"samples": samples[:3]}, "dataset-b": {"samples": samples[3:]}}
    if dump_format == "pt":
        cpu_rollout_module.save_debug_rollout_data(args, data if evaluation else samples, 7, evaluation)
        dump_path = tmp_path / ("eval_7.pt" if evaluation else "7.pt")
    elif evaluation:
        cpu_rollout_module.save_eval_summary_jsonl(args, 7, data)
        dump_path = tmp_path / "eval/7.jsonl"
    else:
        cpu_rollout_module.save_rollout_result_jsonl(args, 7, samples)
        dump_path = tmp_path / "train/7.jsonl"
    before = dump_path.read_bytes()
    output = tmp_path / "report.json"
    subprocess.run(
        [sys.executable, "-m", "relax.entrypoints.diagnose_repetition", str(dump_path), "-o", str(output)],
        check=True,
        capture_output=True,
        text=True,
    )
    offline = json.loads(output.read_text())
    assert online["repetition_frac"] == pytest.approx(4 / 6)
    assert offline["repetition_frac"] == online["repetition_frac"]
    records = offline["files"][0]["samples"]
    assert [sample["has_repetition"] for sample in records] == [False, True, True, True, True, False]
    assert [record["group_index"] for record in records] == [20 + i for i in range(6)]
    if dump_format == "pt":
        assert [record["index"] for record in records] == [100 + i for i in range(6)]
    else:
        expected_indices = [0, 1, 2, 0, 1, 2] if evaluation else list(range(6))
        assert [record["sample_index"] for record in records] == expected_indices
        if evaluation:
            assert [record["dataset"] for record in records] == ["dataset-a"] * 3 + ["dataset-b"] * 3
    assert [sample.to_dict() for sample in samples] == snapshot
    assert dump_path.read_bytes() == before


def test_diagnose_reads_the_real_jsonl_rollout_result(tmp_path):
    """End-to-end over the always-on JSONL dump, which needs no torch to read.

    ``save_rollout_result_jsonl`` writes on every rollout step whenever
    ``--rollout-result-dir`` is set, so this is the dump usually at hand.
    """
    pytest.importorskip("torch", reason="the production JSONL writer imports Sample via torch")
    from relax.utils.repetition_diagnose import diagnose_dumps
    from relax.utils.training.train_dump_utils import save_rollout_result_jsonl
    from relax.utils.types import Sample

    clean = _incompressible(30_000, "jsonl-clean")
    dirty = clean + "spam" * 5000
    samples = [
        Sample(index=i, group_index=0, response=text, response_length=len(text))
        for i, text in enumerate([clean, dirty])
    ]

    save_rollout_result_jsonl(SimpleNamespace(rollout_result_dir=str(tmp_path)), rollout_id=3, samples=samples)
    path = tmp_path / "train" / "3.jsonl"
    assert path.exists()

    report = diagnose_dumps([path])

    assert report["num_samples"] == 2
    assert report["repetition_frac"] == 0.5
    hit = next(s for s in report["files"][0]["samples"] if s["has_repetition"])
    assert hit["sample_index"] == 1
    assert hit["rollout_id"] == 3


def test_diagnose_reads_jsonl_without_importing_torch(tmp_path):
    """The JSONL path must stay usable on a box with no training
    dependencies."""
    import subprocess
    import sys

    path = tmp_path / "rollout.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"sample_index": i, "rollout_id": 5, "response": r})
            for i, r in enumerate([_incompressible(30_000, "no-torch"), "spam" * 5000])
        ),
        encoding="utf-8",
    )

    script = (
        "import sys, json;"
        "from relax.utils.repetition_diagnose import diagnose_dumps;"
        f"report = diagnose_dumps([{str(path)!r}]);"
        "print(json.dumps({'frac': report['repetition_frac'], 'torch': 'torch' in sys.modules}))"
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    result = json.loads(out.stdout)

    assert result["frac"] == 0.5
    assert result["torch"] is False


def test_diagnose_rejects_an_unsupported_dump_suffix(tmp_path):
    from relax.utils.repetition_diagnose import load_rollout_dump

    path = tmp_path / "rollout.txt"
    path.write_text("not a dump", encoding="utf-8")

    with pytest.raises(ValueError, match="expected a .jsonl or .pt rollout dump"):
        load_rollout_dump(path)


def test_diagnose_reports_malformed_jsonl_with_its_line_number(tmp_path):
    from relax.utils.repetition_diagnose import diagnose_dumps

    path = tmp_path / "bad.jsonl"
    path.write_text('{"response": "ok"}\nnot json\n', encoding="utf-8")

    with pytest.raises(ValueError, match="bad.jsonl:2: invalid JSON"):
        diagnose_dumps([path])


def test_cli_refuses_to_overwrite_an_input_dump(tmp_path, capsys):
    """The dumps are training output that cannot be re-created."""
    from relax.utils.repetition_diagnose import main

    path = tmp_path / "rollout.jsonl"
    path.write_text(json.dumps({"response": "ok"}) + "\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        main([str(path), "-o", str(path)])

    assert exc.value.code == 2
    assert "must not overwrite an input dump" in capsys.readouterr().err
    # The input is still intact.
    assert json.loads(path.read_text()) == {"response": "ok"}


def test_diagnose_rejects_a_file_that_is_not_a_rollout_dump(tmp_path):
    torch = pytest.importorskip("torch")
    from relax.utils.repetition_diagnose import load_rollout_dump

    path = tmp_path / "bad.pt"
    torch.save({"not_samples": []}, path)

    with pytest.raises(ValueError, match="not a rollout dump"):
        load_rollout_dump(path)


def test_cli_rejects_bad_window_parameters_as_a_usage_error(capsys):
    """Bad parameters must fail loudly with a non-zero exit code, not raise a
    traceback that a calling script would read as success."""
    from relax.utils.repetition_diagnose import main

    with pytest.raises(SystemExit) as exc:
        main(["unused.pt", "--stride", str(2 * WINDOW)])

    assert exc.value.code == 2
    assert "must not exceed window_size" in capsys.readouterr().err


@pytest.mark.parametrize(
    "record", [{}, {"response": None}, {"response": False}, {"response": 0}, {"response": 1}, {"response": []}]
)
def test_diagnose_rejects_missing_or_non_string_responses(record):
    with pytest.raises(ValueError, match="response"):
        diagnose_samples([{"response": "x" * WINDOW}, record])


def test_diagnose_accepts_empty_response():
    summary = diagnose_samples([{"index": 1, "response": ""}])
    assert summary["num_samples"] == 1
    assert summary["repetition_frac"] == 0.0


@pytest.mark.parametrize("threshold", [0, -1, float("nan"), float("inf"), -float("inf")])
def test_non_positive_or_non_finite_threshold_is_rejected_even_for_empty_input(threshold):
    for text in ("", "x" * WINDOW):
        with pytest.raises(ValueError, match="threshold"):
            scan_repetition(text, threshold=threshold)
    with pytest.raises(ValueError, match="threshold"):
        diagnose_samples([], threshold=threshold)


@pytest.mark.parametrize("threshold", ["0", "-1", "nan", "inf", "-inf"])
def test_cli_rejects_invalid_threshold_before_reading_dump(threshold, capsys):
    from relax.utils.repetition_diagnose import main

    with pytest.raises(SystemExit) as exc:
        main(["unused.pt", f"--threshold={threshold}"])
    assert exc.value.code == 2
    assert "threshold" in capsys.readouterr().err


@pytest.mark.parametrize("suffix", ["jsonl", "pt"])
def test_diagnose_invalid_response_identifies_source_and_sample(tmp_path, suffix):
    from relax.utils.repetition_diagnose import diagnose_dumps

    path = tmp_path / f"invalid.{suffix}"
    records = [{"response": "ok"}, {"response": None}]
    if suffix == "jsonl":
        path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    else:
        torch = pytest.importorskip("torch", reason="the .pt dump path needs torch")
        torch.save({"samples": records}, path)
    with pytest.raises(ValueError) as exc:
        diagnose_dumps([path])
    assert str(path) in str(exc.value)
    assert "response" in str(exc.value)
    assert "1" in str(exc.value) or "2" in str(exc.value)


def test_diagnose_of_no_samples_reports_zero_not_a_division_error():
    from relax.utils.repetition_diagnose import diagnose_samples

    assert diagnose_samples([])["repetition_frac"] == 0.0
