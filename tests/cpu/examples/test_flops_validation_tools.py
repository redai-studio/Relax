# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Failure-sensitive checks for offline reports, using artificial test logs."""

import json

import pytest
import torch

from examples.flops_balancing.compare_step_captures import compare_captures
from examples.flops_balancing.summarize_runs import read_window, summarize
from relax.utils.training.optimizer_step_capture import OptimizerStepCapture, capture_rollout_inputs
from tests.cpu.utils.test_optimizer_step_capture import _OptimizerContract


def _capture(root):
    capture_rollout_inputs(
        {
            "tokens": [torch.tensor([1, 2])],
            "loss_masks": [torch.tensor([0, 1])],
            "total_lengths": [2],
            "response_lengths": [2],
        },
        root,
        rank=0,
        topology=dict(dp=0, tp=0, pp=0, cp=0),
        normalization="token",
        global_batch_size=1,
        device=torch.device("cpu"),
        group=None,
    )
    optimizer = _OptimizerContract()
    optimizer.parameter.grad = torch.tensor([0.5, 0.75])
    capture = OptimizerStepCapture(optimizer, root / "rank-00000", group=None, device=torch.device("cpu"))
    with capture:
        optimizer.step()
    capture.finish({"loss": 1.0}, True)


def _compare(left, right):
    return compare_captures(
        left, right, loss_atol=0, loss_rtol=0, grad_atol=0, grad_rtol=0, update_atol=0, update_rtol=0
    )


def test_complete_capture_comparison_rejects_changed_update_and_initial_state(tmp_path):
    left, right = tmp_path / "A", tmp_path / "D"
    _capture(left)
    _capture(right)
    assert _compare(left, right)["passed"]
    path = right / "rank-00000/parameter-000000.pt"
    original = torch.load(path, weights_only=True)
    for field in ("initial", "gradient", "update"):
        changed = {key: value.clone() for key, value in original.items()}
        changed[field][0] += 0.1
        torch.save(changed, path)
        assert not _compare(left, right)["passed"]


def test_capture_comparison_rejects_different_effective_inputs(tmp_path):
    left, right = tmp_path / "A", tmp_path / "D"
    _capture(left)
    _capture(right)
    path = right / "inputs-rank-00000.json"
    changed = json.loads(path.read_text())
    changed["samples"][0] = "different mask digest"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="Effective samples"):
        _compare(left, right)


def test_capture_comparison_separates_packing_metrics_from_loss(tmp_path):
    left, right = tmp_path / "A", tmp_path / "B"
    _capture(left)
    _capture(right)
    for root, count in ((left, 12.0), (right, 11.0)):
        path = root / "rank-00000/manifest.json"
        manifest = json.loads(path.read_text())
        manifest["loss"].update(num_microbatches_mean=count, pack_tokens_mean=count, pack_tokens_max=count)
        path.write_text(json.dumps(manifest))
    report = _compare(left, right)
    assert report["passed"]
    rank = report["ranks"]["rank-00000"]
    assert set(rank["loss"]) == {"loss"}
    assert rank["packing_metrics"]["num_microbatches_mean"] == {"reference": 12.0, "candidate": 11.0}
    path = right / "rank-00000/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["loss"]["loss"] = 2.0
    path.write_text(json.dumps(manifest))
    assert not _compare(left, right)["passed"]


def _log(seconds):
    return "\n".join(
        f"perf {index}: {
            {
                'perf/step_time': time,
                'perf/actor_train_tokens': tokens,
                'perf/step_resp_token_per_s': tokens / 2 / time,
            }
        }"
        for index, (time, tokens) in enumerate(seconds)
    )


@pytest.mark.parametrize("colored", [False, True])
def test_throughput_uses_total_tokens_over_total_time(colored):
    log = _log([(1, 100), (3, 600)])
    if colored:
        log = "\n".join(f"\x1b[36m(worker pid=123)\x1b[0m \x1b[1;37m{line}\x1b[0m" for line in log.splitlines())
    result = read_window(log, warmup=0, steps=2)
    assert result["input_tokens_per_s"] == 175
    assert result["supervised_tokens_per_s"] == 87.5


def test_throughput_rejects_duplicate_or_missing_steps():
    log = _log([(1, 100)])
    with pytest.raises(ValueError, match="Duplicate"):
        read_window(log + "\n" + log, 0, 1)
    with pytest.raises(ValueError, match="Missing"):
        read_window(log, 0, 2)


def _runs(tmp_path):
    directories = []
    for mode, seconds in (("A", 1.2), ("D", 1.0)):
        for repeat in range(3):
            directory = tmp_path / f"{mode}{repeat}"
            directory.mkdir()
            metadata = dict(
                mode=mode,
                capture=False,
                dry_run=False,
                invariant_command=["same args"],
                versions={},
                launch_environment={"OMP_NUM_THREADS": "1"},
                source_sha256="source",
                workload_sha256="workload",
                model_revision="model",
                data_revision="data",
                native_source_sha256="native source",
                transfer_queue_source_sha256="patched TQ source",
                devices=[{"name": "test fixture"}],
            )
            (directory / "run.json").write_text(json.dumps(metadata))
            (directory / "status.json").write_text(json.dumps({"status": "SUCCEEDED"}))
            (directory / "training.log").write_text(_log([(seconds, 120)]))
            directories.append(directory)
    return directories


def test_repeated_run_summary_checks_20_percent_boundary(tmp_path):
    report = summarize(_runs(tmp_path), warmup=0, steps=1)
    assert report["modes"]["D"]["reaches_20_percent"]


@pytest.mark.parametrize(
    "change",
    ["different_config", "different_environment", "different_tokens", "capture", "diagnostic", "failure", "same_run"],
)
def test_repeated_run_summary_rejects_unfair_or_incomplete_comparisons(tmp_path, change):
    directories = _runs(tmp_path)
    directory = directories[-1]
    path = directory / "run.json"
    metadata = json.loads(path.read_text())
    if change == "different_config":
        metadata["invariant_command"] = ["changed token budget"]
    elif change == "different_environment":
        metadata["launch_environment"]["OMP_NUM_THREADS"] = "2"
    elif change == "capture":
        metadata["capture"] = True
    elif change == "diagnostic":
        metadata["diagnostic"] = "router_trace"
    elif change == "failure":
        (directory / "status.json").write_text(json.dumps({"status": "FAILED"}))
    elif change == "same_run":
        directories[-1] = directories[-2]
    else:
        (directory / "training.log").write_text(_log([(1, 100)]))
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        summarize(directories, warmup=0, steps=1)


def test_olmoe_local_template_masks_only_answer(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    from examples.flops_balancing.download_assets import OLMOE_SFT_TEMPLATE

    backend = Tokenizer(WordLevel({"[UNK]": 0, "[EOS]": 1, "question": 2, "answer": 3}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="[EOS]", unk_token="[UNK]")
    tokenizer.chat_template = OLMOE_SFT_TEMPLATE
    tokenizer.save_pretrained(tmp_path)
    restored = AutoTokenizer.from_pretrained(tmp_path, local_files_only=True)
    rendered = restored.apply_chat_template(
        [{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}],
        tokenize=False,
    )
    assert rendered == "question\n\nanswer[EOS]"
    encoded = restored.apply_chat_template(
        [{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}],
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    assert encoded["input_ids"] == [2, 3, 1]
    assert encoded["assistant_masks"] == [0, 1, 1]
