# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Unit tests for ``relax/engine/rollout/base_types.py::call_rollout_fn``.

Covers the ``--rollout-sample-filter-path`` hook: filter must run on train
output, must be skipped on eval output, must be a no-op when the arg is unset,
and must also apply when the rollout fn returns the legacy (unwrapped)
``list[list[Sample]]`` form that ``call_rollout_fn`` wraps into
``RolloutFnTrainOutput``.

Run with: pytest tests/engine/rollout/test_base_types.py -v
"""

from relax.engine.rollout.base_types import (
    RolloutFnEvalOutput,
    RolloutFnTrainOutput,
    call_rollout_fn,
)
from relax.utils.types import Sample
from tests.support.args import build_relax_args


def _mark_even_index(args, groups):
    """Filter fixture: mark samples with even ``index`` as removed."""
    for group in groups:
        for sample in group:
            if sample.index % 2 == 0:
                sample.remove_sample = True


def _mark_all(args, groups):
    """Filter fixture: mark every sample as removed."""
    for group in groups:
        for sample in group:
            sample.remove_sample = True


def _raise_filter(args, groups):
    raise RuntimeError("filter invoked unexpectedly")


_FILTER_MODULE = __name__


def _build_groups(n: int) -> list[list[Sample]]:
    return [[Sample(index=i, response_length=1)] for i in range(n)]


class TestCallRolloutFnFilter:
    def test_filter_applied_on_train_output(self):
        groups = _build_groups(4)
        args = build_relax_args(rollout_sample_filter_path=f"{_FILTER_MODULE}._mark_even_index")

        def rollout_fn(a, evaluation):
            return RolloutFnTrainOutput(samples=groups)

        out = call_rollout_fn(rollout_fn, args, evaluation=False)

        assert isinstance(out, RolloutFnTrainOutput)
        assert [g[0].remove_sample for g in out.samples] == [True, False, True, False]

    def test_filter_applied_on_legacy_return(self):
        """Rollout fn returning a bare ``list[list[Sample]]`` is wrapped, then
        filtered."""
        groups = _build_groups(3)
        args = build_relax_args(rollout_sample_filter_path=f"{_FILTER_MODULE}._mark_all")

        def rollout_fn(a, evaluation):
            return groups

        out = call_rollout_fn(rollout_fn, args, evaluation=False)

        assert isinstance(out, RolloutFnTrainOutput)
        assert all(g[0].remove_sample for g in out.samples)

    def test_filter_skipped_when_path_none(self):
        groups = _build_groups(3)
        args = build_relax_args(rollout_sample_filter_path=None)

        def rollout_fn(a, evaluation):
            return RolloutFnTrainOutput(samples=groups)

        out = call_rollout_fn(rollout_fn, args, evaluation=False)

        assert not any(g[0].remove_sample for g in out.samples)

    def test_filter_skipped_when_attr_missing(self):
        """``getattr(..., None)`` must swallow the missing attribute."""
        groups = _build_groups(2)
        # A missing field is no longer a legal state: rollout_sample_filter_path keeps its declared default None.
        args = build_relax_args()

        def rollout_fn(a, evaluation):
            return RolloutFnTrainOutput(samples=groups)

        out = call_rollout_fn(rollout_fn, args, evaluation=False)

        assert not any(g[0].remove_sample for g in out.samples)

    def test_filter_skipped_on_eval(self):
        """Even if a path is set, eval must not invoke the filter."""
        args = build_relax_args(rollout_sample_filter_path=f"{_FILTER_MODULE}._raise_filter")

        def rollout_fn(a, evaluation):
            return RolloutFnEvalOutput(data={"bench": {"score": [1.0]}})

        out = call_rollout_fn(rollout_fn, args, evaluation=True)

        assert isinstance(out, RolloutFnEvalOutput)
        assert out.data == {"bench": {"score": [1.0]}}

    def test_filter_skipped_on_legacy_eval_return(self):
        """Legacy eval return (bare dict) is wrapped as EvalOutput; filter must
        not run."""
        args = build_relax_args(rollout_sample_filter_path=f"{_FILTER_MODULE}._raise_filter")

        def rollout_fn(a, evaluation):
            return {"bench": {"score": [1.0]}}

        out = call_rollout_fn(rollout_fn, args, evaluation=True)

        assert isinstance(out, RolloutFnEvalOutput)

    def test_filter_receives_args_and_samples(self):
        """Contract check: filter is called as ``filter(args, samples)`` —
        matches slime."""
        captured = {}

        def _capture(args, groups):
            captured["args"] = args
            captured["groups"] = groups

        # register on this module so load_function can find it
        globals()["_capture"] = _capture

        groups = _build_groups(1)
        args = build_relax_args(rollout_sample_filter_path=f"{_FILTER_MODULE}._capture")

        def rollout_fn(a, evaluation):
            return RolloutFnTrainOutput(samples=groups)

        call_rollout_fn(rollout_fn, args, evaluation=False)

        assert captured["args"] is args
        assert captured["groups"] is groups
