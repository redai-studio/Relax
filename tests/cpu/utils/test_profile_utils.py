# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import gzip
import hashlib
import json
import locale
import os
import platform
import shutil
import sys
from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock

import pytest

from relax.utils import profile_utils


@pytest.mark.parametrize("rank,selected", [(0, True), (48, True), (1, False)])
def test_train_profiler_honors_rank_selection(monkeypatch, rank, selected):
    args = Namespace(
        profile_ranks=[0, 48],
        use_pytorch_profiler=True,
        record_memory_history=True,
        profile_target=["train_overall", "train_actor", "train_log_probs"],
        tb_experiment_name="rank-selection",
        memory_snapshot_dir=None,
        memory_snapshot_num_steps=1,
    )
    monkeypatch.setattr(profile_utils.torch.distributed, "get_rank", lambda group: rank)
    create_torch = Mock()
    create_memory = Mock()
    monkeypatch.setattr(profile_utils, "_create_torch_profiler", create_torch)
    monkeypatch.setattr(profile_utils._BaseMemoryProfiler, "create", create_memory)
    profiler = profile_utils.TrainProfiler(args)
    profiler.on_init_end()
    profiler.step(0)
    assert list(profiler.iterate_train_actor(iter([10, 20]))) == [10, 20]
    assert list(profiler.iterate_train_log_probs(iter([30]))) == [30]
    assert create_torch.call_count == (3 if selected else 0)
    assert create_memory.call_count == int(selected)
    if selected:
        assert create_torch.return_value.start.call_count == 3
        assert create_torch.return_value.step.call_count == 4
        create_memory.return_value.start.assert_called_once()
        create_memory.return_value.stop.assert_called_once()


def test_train_profiler_without_rank_filter_preserves_legacy_behavior(monkeypatch):
    get_rank = Mock(side_effect=AssertionError("No distributed state is needed without a rank filter"))
    monkeypatch.setattr(profile_utils.torch.distributed, "get_rank", get_rank)
    assert profile_utils._should_profile_train_rank(Namespace())
    get_rank.assert_not_called()


def _python_stack_probe_leaf(tensor):
    return tensor.sin().square()


def _load_trace_events(path: Path, request: pytest.FixtureRequest, compression: str | None) -> list[dict]:
    """Keep original bytes and runtime provenance for the CI investigation."""
    evidence_root = os.environ.get("RELAX_PROFILER_EVIDENCE_DIR")
    evidence = None
    if evidence_root:
        worker = os.environ.get("PYTEST_XDIST_WORKER", "serial")
        evidence = Path(evidence_root) / f"{worker}-{os.getpid()}-gzip-{compression}"
        evidence.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, evidence / path.name)
    with gzip.open(path, "rb") as stream:
        raw = stream.read()
    metadata = {}
    if evidence is not None:
        (evidence / "trace.json").write_bytes(raw)
        torch = profile_utils.torch
        source_paths = [Path(__file__), Path(profile_utils.__file__), Path(torch.profiler.profiler.__file__)]
        metadata = {
            "nodeid": request.node.nodeid,
            "compression": compression,
            "pid": os.getpid(),
            "worker": os.environ.get("PYTEST_XDIST_WORKER", "serial"),
            "worker_count": os.environ.get("PYTEST_XDIST_WORKER_COUNT"),
            "pytest_args": list(request.config.invocation_params.args),
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "locale_encoding": locale.getpreferredencoding(False),
            "torch_version": torch.__version__,
            "torch_git_version": torch.version.git_version,
            "torch_cuda": torch.version.cuda,
            "torch_config": torch.__config__.show(),
            "torch_path": torch.__file__,
            "source_sha256": {str(source): hashlib.sha256(source.read_bytes()).hexdigest() for source in source_paths},
            "requested_checkout": os.environ.get("RELAX_CI_CHECKOUT"),
            "image": os.environ.get("RELAX_CI_IMAGE"),
            "environment": {
                key: os.environ.get(key)
                for key in (
                    "CUDA_VISIBLE_DEVICES",
                    "NVIDIA_VISIBLE_DEVICES",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "RELAX_PROFILE_GZIP_LEVEL",
                    "LANG",
                    "LC_ALL",
                )
            },
            "gzip_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "raw_size": len(raw),
        }
    try:
        # Decode the complete buffer strictly so error offsets are absolute byte offsets.
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        metadata["decode_error"] = {
            "start": exc.start,
            "end": exc.end,
            "reason": exc.reason,
            "context_start": max(0, exc.start - 96),
            "context_hex": raw[max(0, exc.start - 96) : exc.end + 96].hex(),
        }
        raise
    finally:
        if evidence is not None:
            (evidence / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return json.loads(text)["traceEvents"]


@pytest.mark.parametrize("compression", [None, "1"])
def test_train_profiler_exports_python_operator_call_stacks(monkeypatch, tmp_path, compression, request):
    """with_stack alone can omit operator Call stack fields from Chrome
    traces."""
    torch = profile_utils.torch
    if compression is None:
        monkeypatch.delenv("RELAX_PROFILE_GZIP_LEVEL", raising=False)
    else:
        monkeypatch.setenv("RELAX_PROFILE_GZIP_LEVEL", compression)
    monkeypatch.setattr(profile_utils.tempfile, "tempdir", str(tmp_path))
    original_profile = torch.profiler.profile

    def cpu_profile(**kwargs):
        return original_profile(activities=[torch.profiler.ProfilerActivity.CPU], **kwargs)

    monkeypatch.setattr(torch.profiler, "profile", cpu_profile)
    monkeypatch.setattr(profile_utils, "_get_rank_tag", lambda: "rank0_dp0_tp0_pp0")
    monkeypatch.setattr(profile_utils, "_get_train_trace_dir", lambda args: str(tmp_path))
    args = Namespace(
        profile_step_start=0,
        profile_step_end=0,
        profile_with_stack=True,
        profile_with_memory=False,
        profile_with_flops=False,
    )
    tensor = torch.randn(16)
    with profile_utils._create_torch_profiler(args, "stack-test") as profiler:
        _python_stack_probe_leaf(tensor)
        profiler.step()
    files = list(tmp_path.glob("*.pt.trace.json.gz"))
    assert len(files) == 1
    events = _load_trace_events(files[0], request, compression)
    assert any(
        event.get("cat") == "python_function" and "_python_stack_probe_leaf" in event["name"] for event in events
    )
    assert any(
        event.get("cat") == "cpu_op" and "_python_stack_probe_leaf" in event.get("args", {}).get("Call stack", "")
        for event in events
    )
    assert not list(tmp_path.glob("relax-*.json"))
    assert not list(tmp_path.glob("*.partial"))


def test_trace_diagnostics_preserve_invalid_utf8(monkeypatch, tmp_path, request):
    evidence = tmp_path / "evidence"
    monkeypatch.setenv("RELAX_PROFILER_EVIDENCE_DIR", str(evidence))
    path = tmp_path / "invalid.pt.trace.json.gz"
    raw = b'{"traceEvents": [{"name": "' + b"\x98" + b'"}]}'
    path.write_bytes(gzip.compress(raw))
    with pytest.raises(UnicodeDecodeError) as error:
        _load_trace_events(path, request, None)
    retained = next(evidence.iterdir())
    assert (retained / path.name).read_bytes() == path.read_bytes()
    assert (retained / "trace.json").read_bytes() == raw
    metadata = json.loads((retained / "metadata.json").read_text())
    assert metadata["raw_sha256"] == hashlib.sha256(raw).hexdigest()
    assert metadata["decode_error"]["start"] == error.value.start == raw.index(b"\x98")
    assert metadata["decode_error"]["context_hex"] == raw.hex()


def test_failed_trace_export_preserves_local_evidence(monkeypatch, tmp_path):
    monkeypatch.setenv("RELAX_PROFILE_GZIP_LEVEL", "1")
    monkeypatch.setattr(profile_utils.tempfile, "tempdir", str(tmp_path))
    output = tmp_path / "published"

    def interrupted_export(path):
        profile_utils.Path(path).write_text('{"traceEvents": [')
        raise RuntimeError("simulated export interruption")

    profiler = Mock()
    profiler.export_chrome_trace.side_effect = interrupted_export
    with pytest.raises(RuntimeError, match="simulated export interruption"):
        profile_utils._create_train_trace_handler(str(output), "rank0")(profiler)
    retained = list(tmp_path.glob("relax-rank0-*.json"))
    assert len(retained) == 1
    assert retained[0].read_text() == '{"traceEvents": ['
    assert not list(output.iterdir())
