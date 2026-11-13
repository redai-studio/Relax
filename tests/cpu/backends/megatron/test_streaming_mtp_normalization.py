# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise the production train loop's MTP window boundary on CPU.

The scheduler double supplies only Megatron's external scaling contract: main
and independently injected auxiliary gradients are divided by K.
"""

import ast
import sys
from argparse import Namespace
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


_ROOT = Path(__file__).resolve().parents[4]


def _load(path, names, namespace):
    nodes = [node for node in ast.parse((_ROOT / path).read_text()).body if getattr(node, "name", None) in names]
    assert {node.name for node in nodes} == set(names)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


class _Stream:
    def __init__(self, coefficients, events, window=0):
        self.loss_scale = 0.25
        self.window_quota = sum(value is not None for value in coefficients)
        self.window = window
        self.events = events
        self.offset = 0
        self.mbs = [
            (
                {
                    "pg": 999.0 if value is None else 2.0 * value,
                    "mtp": 999.0 if value is None else float(value),
                    "tokens": torch.zeros(1, 1, dtype=torch.long),
                    "full_loss_masks": torch.ones(1, 1),
                    "__is_dummy__": value is None,
                    "__loss_scale__": self.loss_scale,
                },
                object(),
            )
            for value in coefficients
        ]

    def __iter__(self):
        return self

    @property
    def mb_count(self):
        return self.offset

    def __next__(self):
        if self.offset == len(self.mbs):
            self.events.append(("drained", self.window))
            raise StopIteration
        self.events.append(("fetch", self.window, self.offset))
        batch = self.mbs[self.offset]
        self.offset += 1
        return batch

    def get_buffer(self):
        return self.mbs


def _train_loop(monkeypatch, step, *, enable_mtp=True, per_token=False):
    args = Namespace(
        enable_mtp_training=enable_mtp,
        calculate_per_token_loss=per_token,
        global_batch_size=8,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
        reset_optimizer_states=False,
        ci_test=False,
        ci_save_grad_norm=None,
        ci_load_grad_norm=None,
        mtp_num_layers=1,
        mtp_loss_scaling_factor=0.1,
    )
    logs, tracker = [], {}
    helper = SimpleNamespace(tracker=tracker, clean_loss_in_tracker=tracker.clear)
    native = ModuleType("megatron.core.transformer.multi_token_prediction")
    native.MTPLossLoggingHelper = helper
    monkeypatch.setitem(sys.modules, native.__name__, native)
    hooks = ModuleType("relax.utils.agentic_obs.train_hooks")
    hooks.observe_train_step = lambda *args, **kwargs: nullcontext()
    monkeypatch.setitem(sys.modules, hooks.__name__, hooks)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda **kwargs: 0)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)
    from relax.backends.megatron.telemetry import observe_step

    namespace = {
        "observe_step": observe_step,
        "torch": torch,
        "get_args": lambda: args,
        "get_model_config": lambda _: SimpleNamespace(),
        "DataIterator": type("UnusedDataIterator", (), {}),
        "DDP": type("UnusedDDP", (), {}),
        "StreamingTQIterator": _Stream,
        "resident_sft_param_overlap_enabled": lambda _: False,
        "has_conditional_branch_sync": lambda _: False,
        "should_disable_forward_pre_hook": lambda _: False,
        "finalize_model_grads": lambda *args: None,
        "train_one_step": step,
        "timer": lambda *args, **kwargs: nullcontext(),
        "log_step_peak_memory": lambda *args: nullcontext(),
        "logger": SimpleNamespace(info=lambda *args: None),
        "mpu": SimpleNamespace(
            get_data_parallel_rank=lambda **kwargs: 0,
            get_tensor_model_parallel_rank=lambda: 0,
            get_pipeline_model_parallel_rank=lambda: 0,
            get_pipeline_model_parallel_world_size=lambda: 1,
            get_expert_model_parallel_rank=lambda: 0,
            get_context_parallel_rank=lambda: 0,
            get_data_parallel_group=lambda **kwargs: object(),
        ),
        "tracking_utils": SimpleNamespace(
            log=lambda args, values, **kwargs: logs.append(values.copy()),
            flush_metrics=lambda *args, **kwargs: None,
        ),
    }
    _load("relax/utils/data/stream_dataloader.py", {"MicroBatchListIterator"}, namespace)
    _load(
        "relax/backends/megatron/model.py",
        {"train", "_mtp_metric_scales", "_attach_mtp_forward_kwargs", "_prepare_mtp_training_window"},
        namespace,
    )

    def run(streams, counts):
        return namespace["train"](
            0,
            [torch.nn.Identity()],
            SimpleNamespace(scale_loss=lambda value: value, param_groups=[]),
            None,
            streams,
            counts,
        )

    return run, namespace, tracker, logs


def test_streaming_mtp_windows_keep_pg_gradients_and_average_auxiliary_gradients(monkeypatch):
    events = []
    streams = [_Stream([1, None, 3], events), _Stream([2, None, 4, None, 6], events, window=1)]
    originals = [[batch.copy() for batch, _ in stream.mbs] for stream in streams]
    gradients = []

    def native_step(args, rollout_id, step_id, iterators, model, optimizer, scheduler, count, global_samples):
        stream = streams[step_id]
        assert stream.offset == 0, "Preparation must stay inside train_one_step, after optimizer offload"
        iterators, count = namespace["_prepare_mtp_training_window"](args, iterators, count, model)
        assert stream.offset == len(stream.mbs), "The current window must be collected before scheduling"
        assert all(future.offset == 0 for future in streams[step_id + 1 :]), "The next window was collected too early"
        assert count == len(stream.mbs)
        assert global_samples == stream.window_quota
        assert isinstance(iterators[0], namespace["MicroBatchListIterator"])
        events.append(("train", step_id))
        pg_weight = torch.tensor(1.0, requires_grad=True)
        mtp_weight = torch.tensor(1.0, requires_grad=True)
        auxiliary_sum = 0.0
        for index, (batch, meta) in enumerate(iterators[0]):
            assert meta is stream.mbs[index][1]
            assert batch["__loss_scale__"] == pytest.approx(stream.loss_scale * count)
            forward_kwargs = {"loss_mask": batch["full_loss_masks"]}
            namespace["_attach_mtp_forward_kwargs"](args, batch, forward_kwargs)
            pg_loss = pg_weight * batch["pg"] * batch["__loss_scale__"]
            if batch["__is_dummy__"]:
                pg_loss = 0.0 * pg_loss
            auxiliary = batch["mtp"] * forward_kwargs["loss_mask"].sum()
            auxiliary_sum += float(auxiliary)
            mtp_loss = args.mtp_loss_scaling_factor * mtp_weight * auxiliary
            # Auxiliary injection has its own backward scale, independent of PG's scalar.
            torch.autograd.backward((pg_loss, mtp_loss), (torch.tensor(1.0 / count), torch.tensor(1.0 / count)))
        gradients.append((pg_weight.grad.item(), mtp_weight.grad.item()))
        tracker["values"] = torch.tensor([args.mtp_loss_scaling_factor * auxiliary_sum])
        tracker["relax_mtp_microbatch_raw_ce"] = torch.tensor([auxiliary_sum])
        return {}, 0.0

    run, namespace, tracker, logs = _train_loop(monkeypatch, native_step)
    counts = [1, 1]
    run(streams, counts)
    # Independent oracles: PG keeps sample weights; MTP averages over all scheduled mbs.
    assert [pg for pg, _ in gradients] == pytest.approx([2.0, 6.0])
    assert [mtp for _, mtp in gradients] == pytest.approx([0.4 / 3, 1.2 / 5])
    assert [entry["train/mtp_loss"] for entry in logs] == pytest.approx([4.0 / 3, 12.0 / 5])
    assert [entry["train/mtp_objective"] for entry in logs] == pytest.approx([0.4 / 3, 1.2 / 5])
    assert events.index(("train", 0)) < events.index(("fetch", 1, 0))
    assert all(isinstance(stream, _Stream) for stream in streams)
    assert counts == [1, 1]
    for stream, snapshots in zip(streams, originals, strict=True):
        assert stream.get_buffer() is stream.mbs
        for (batch, _), snapshot in zip(stream.mbs, snapshots, strict=True):
            assert batch.keys() == snapshot.keys()
            assert all(batch[key] is value for key, value in snapshot.items())
            torch.testing.assert_close(batch["full_loss_masks"], torch.ones(1, 1))


@pytest.mark.parametrize("enable_mtp,per_token", [(False, False), (True, True)])
def test_streaming_without_microbatch_mtp_normalization_stays_lazy(monkeypatch, enable_mtp, per_token):
    stream = _Stream([1, 2, 3], [])

    def streaming_step(args, rollout_id, step_id, iterators, model, optimizer, scheduler, count, global_samples):
        iterators, count = namespace["_prepare_mtp_training_window"](args, iterators, count, model)
        assert iterators == [stream]
        assert stream.offset == 0
        assert count == 1
        assert len(list(iterators[0])) == 3
        return {}, 0.0

    run, namespace, _, _ = _train_loop(monkeypatch, streaming_step, enable_mtp=enable_mtp, per_token=per_token)
    run([stream], [1])


def test_streaming_mtp_rejects_an_empty_window_before_scheduling(monkeypatch):
    def unexpected_step(args, rollout_id, step_id, iterators, model, optimizer, scheduler, count, global_samples):
        namespace["_prepare_mtp_training_window"](args, iterators, count, model)
        pytest.fail("An empty MTP window reached the optimizer schedule")

    run, namespace, _, _ = _train_loop(monkeypatch, unexpected_step)
    with pytest.raises(RuntimeError, match="(?i)(empty|zero|microbatch)"):
        run([_Stream([], [])], [1])


def test_fixed_mtp_iterator_is_not_collected_or_scaled_twice(monkeypatch):
    _, namespace, _, _ = _train_loop(monkeypatch, None)
    namespace["get_model_config"] = lambda _: SimpleNamespace(fine_grained_activation_offloading=True)
    source = _Stream([1, None, 3], [])
    fixed = namespace["MicroBatchListIterator"](source.mbs, loss_scale=source.loss_scale)
    iterators = [fixed]
    prepared, count = namespace["_prepare_mtp_training_window"](namespace["get_args"](), iterators, 3, [object()])
    assert prepared is iterators
    assert count == 3 and fixed.offset == 0
    assert [batch["__loss_scale__"] for batch, _ in fixed] == [0.25, 0.25, 0.25]


def test_streaming_mtp_rejects_activation_offload_before_consuming_data(monkeypatch):
    def prepare_window(args, rollout_id, step_id, iterators, model, optimizer, scheduler, count, global_samples):
        namespace["_prepare_mtp_training_window"](args, iterators, count, model)
        return {}, 0.0

    run, namespace, _, _ = _train_loop(monkeypatch, prepare_window)
    # The actual model config is authoritative even when the CLI flag differs.
    namespace["get_args"]().fine_grained_activation_offloading = False
    namespace["get_model_config"] = lambda _: SimpleNamespace(fine_grained_activation_offloading=True)
    for count in (2, 4):
        stream = _Stream(range(1, count + 1), [])
        with pytest.raises(
            ValueError, match="Disable --fine-grained-activation-offloading or --use-dynamic-batch-size"
        ):
            run([stream], [1])
        assert stream.offset == 0
        assert stream.events == []


def test_streaming_mtp_collects_after_optimizer_offload_zero_grad_and_hook(monkeypatch):
    events = []
    _, namespace, _, _ = _train_loop(monkeypatch, None)
    args = namespace["get_args"]()
    args.custom_megatron_before_train_step_hook_path = "test.before_step"
    misc = ModuleType("relax.utils.misc")
    misc.load_function = lambda _: lambda *args: events.append("hook")
    monkeypatch.setitem(sys.modules, misc.__name__, misc)
    namespace.update(
        capture_hooks=SimpleNamespace(begin_step_for=lambda *args: events.append("capture")),
        collect_manual_gc=lambda _: events.append("gc"),
    )
    optimizer = SimpleNamespace(
        config=SimpleNamespace(chunked_optimizer_state_offload=True),
        optimizer_state_offload_requires_pre_forward_param_sync=lambda: False,
        offload_optimizer_state_for_forward=lambda **kwargs: events.append("offload"),
        zero_grad=lambda: events.append("optimizer_zero"),
    )
    model = SimpleNamespace(zero_grad_buffer=lambda: events.append("model_zero"))
    path = _ROOT / "relax/backends/megatron/model.py"
    function = next(
        node for node in ast.parse(path.read_text()).body if getattr(node, "name", None) == "train_one_step"
    )
    preparation_index = next(
        index
        for index, statement in enumerate(function.body)
        if isinstance(statement, ast.Assign)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == "_prepare_mtp_training_window"
    )
    # Execute the real startup prefix; the remaining body requires a CUDA model.
    function.body = function.body[: preparation_index + 1]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    for step_id in range(2):
        events.clear()
        stream = _Stream([1, None, 3], events, window=step_id)
        namespace["train_one_step"](args, 0, step_id, [stream], [model], optimizer, None, 1, 2)
        assert events[:6] == ["capture", "offload", "model_zero", "optimizer_zero", "gc", "hook"]
        assert events[6] == ("fetch", step_id, 0)
        assert stream.mb_count == 3
