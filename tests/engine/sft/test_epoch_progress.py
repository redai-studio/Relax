# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import ast
import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType

import pytest

from relax.engine.sft.bootstrap import resolve_sft_num_rollout


@pytest.fixture
def sized_dataset(monkeypatch):
    module = ModuleType("relax.engine.sft.dataset.streaming")

    class SizedDataset:
        def __init__(self, path: int, prefetch_max_cached: int) -> None:
            self.size = path

        def __len__(self) -> int:
            return self.size

    module.SFTStreamingDataset = SizedDataset
    monkeypatch.setitem(sys.modules, module.__name__, module)


def _reported_metrics(args: Namespace, step: int, steps_per_rollout: int = 1) -> dict:
    # Execute the actual metric-emission block without importing CUDA/Megatron.
    path = Path(__file__).resolve().parents[3] / "relax/backends/megatron/model.py"
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue
        for start, statement in enumerate(body):
            if not isinstance(statement, ast.Assign):
                continue
            if ast.unparse(statement.targets[0]) != "log_dict['train/step']":
                continue
            end = next(
                i
                for i in range(start + 1, len(body))
                if isinstance(body[i], ast.Expr)
                and isinstance(body[i].value, ast.Call)
                and ast.unparse(body[i].value.func) == "tracking_utils.log"
            )
            namespace = {
                "args": args,
                "accumulated_step_id": step,
                "num_steps_per_rollout": steps_per_rollout,
                "role_tag": "",
                "log_dict": {},
            }
            block = ast.Module(body=body[start:end], type_ignores=[])
            exec(compile(block, str(path), "exec"), namespace)
            return namespace["log_dict"]
    raise AssertionError("Training metric-emission block not found")


@pytest.mark.parametrize("step", [0, 17, 240, 241, 722])
def test_sft_epoch_progress_reports_nondivisible_dataset(sized_dataset, step):
    args = Namespace(loss_type="sft", prompt_data=30890, rollout_batch_size=128, num_epoch=3, num_rollout=1000)
    resolve_sft_num_rollout(args)

    assert args.num_rollout == 723
    assert args.num_rollout_per_epoch is None
    assert _reported_metrics(args, step)["train/cur_epoch"] == pytest.approx((step + 1) * 128 / 30890)


@pytest.mark.parametrize("eval_size,train_size,interval", [(None, 1024, 8), (128, 896, 7), (42, 982, None)])
def test_sft_epoch_progress_uses_training_split(sized_dataset, eval_size, train_size, interval):
    args = Namespace(
        loss_type="sft", prompt_data=1024, rollout_batch_size=128, num_epoch=3, num_rollout=1000, eval_size=eval_size
    )
    resolve_sft_num_rollout(args)

    assert args.num_rollout_per_epoch == interval
    assert _reported_metrics(args, 6)["train/cur_epoch"] == pytest.approx(7 * 128 / train_size)


def test_sft_epoch_progress_custom_dataset_clears_unknown_denominator():
    args = Namespace(
        loss_type="sft",
        custom_dataset_class_path="custom.Dataset",
        num_rollout=10,
        num_epoch=None,
        num_rollout_per_epoch_for_metrics=2,
    )
    resolve_sft_num_rollout(args)

    assert args.num_rollout_per_epoch_for_metrics is None
    assert "train/cur_epoch" not in _reported_metrics(args, 0)


def test_epoch_progress_preserves_rl_multiple_updates_per_rollout():
    args = Namespace(num_rollout_per_epoch=10)
    assert _reported_metrics(args, 7, steps_per_rollout=4)["train/cur_epoch"] == pytest.approx(0.2)
