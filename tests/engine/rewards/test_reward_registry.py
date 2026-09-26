#!/usr/bin/env python3

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Pure-CPU tests for the reward registry and rm_type resolver.

Loads relax/engine/rewards/registry.py standalone (spec_from_file_location, the
same fallback pattern test_reward_worker.py uses for openr1mm.py) so the tests
run without ray / torch / math_verify. Built-in dotted-path specs are checked
for presence only and never resolved here: resolving them would import the
heavy rewards package. Run with: pytest
tests/engine/rewards/test_reward_registry.py -v
"""

import importlib.util
import logging
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) in sys.path:
    sys.path.remove(str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT))

_REGISTRY_PATH = _REPO_ROOT / "relax" / "engine" / "rewards" / "registry.py"
_spec = importlib.util.spec_from_file_location("reward_registry", _REGISTRY_PATH)
registry = importlib.util.module_from_spec(_spec)
sys.modules["reward_registry"] = registry
_spec.loader.exec_module(registry)

_LEGACY_SYNC_TYPES = [
    "deepscaler",
    "geo3k",
    "openr1mm",
    "multiple_choice",
    "dapo",
    "math",
    "mopd",
    "f1",
    "gpqa",
    "ifbench",
    "random",
]


@pytest.fixture(autouse=True)
def _preconfigure_logging():
    """Run the one-time root-logger configuration before any in-test warning.

    configure_logger() removes all root handlers on its first effective call;
    doing that during setup (instead of mid-test on the first logger.warning)
    keeps pytest's caplog capture handler intact during the test call phase.
    """
    from relax.utils.logging_utils import configure_logger

    configure_logger()
    yield


@pytest.fixture(autouse=True)
def _registry_guard():
    """Snapshot registrations and reset warn/counter state around each test."""
    saved = dict(registry._REWARDS)
    registry.reset_router_state()
    yield
    registry._REWARDS.clear()
    registry._REWARDS.update(saved)
    registry.reset_router_state()


def _resolve(metadata_rm_type=None, args_rm_type=None, label=None, **kwargs):
    return registry.resolve_rm_type(
        metadata_rm_type=metadata_rm_type,
        args_rm_type=args_rm_type,
        label=label,
        **kwargs,
    )


class TestRegistryRegistration:
    """One-line registration, duplicate rejection, and built-in presence."""

    def test_register_one_line_then_lookup(self):
        registry.register_reward("t18_dummy", lambda response, label: 0.5)
        spec = registry.get_reward_spec("t18_dummy")
        assert spec is not None
        assert spec.mode == "sync"
        assert spec.resolve()("any response", "any label") == 0.5

    def test_register_duplicate_rejected(self):
        registry.register_reward("t18_dup", lambda response, label: 0.0)
        with pytest.raises(ValueError, match="already registered"):
            registry.register_reward("t18_dup", lambda response, label: 1.0)

    def test_register_invalid_mode_rejected(self):
        with pytest.raises(ValueError, match="mode"):
            registry.register_reward("t18_bad_mode", lambda response, label: 0.0, mode="threaded")

    def test_all_legacy_sync_types_registered(self):
        for name in _LEGACY_SYNC_TYPES:
            spec = registry.get_reward_spec(name)
            assert spec is not None, f"legacy rm_type {name!r} missing from registry"
            assert spec.mode == "sync"
        assert set(_LEGACY_SYNC_TYPES) <= set(registry.list_reward_types("sync"))

    def test_guard_fixture_restores_baseline(self):
        before = set(registry.list_reward_types())
        registry.register_reward("t18_transient", lambda response, label: 0.0)
        assert "t18_transient" in registry.list_reward_types()
        registry._REWARDS.pop("t18_transient")
        assert set(registry.list_reward_types()) == before


class TestResolverPrecedence:
    """Baseline-parity resolution with all new behavior switched off."""

    def test_metadata_overrides_explicit(self):
        route = _resolve(metadata_rm_type="math", args_rm_type="multiple_choice")
        assert route.rm_type == "math"
        assert not route.boxed and not route.zero_reward

    def test_explicit_used_when_no_metadata(self):
        assert _resolve(args_rm_type="multiple_choice").rm_type == "multiple_choice"

    def test_empty_metadata_value_falls_through_to_args(self):
        assert _resolve(metadata_rm_type="", args_rm_type="math").rm_type == "math"

    def test_whitespace_stripped(self):
        assert _resolve(args_rm_type="  math  ").rm_type == "math"

    def test_missing_no_fallback_keeps_baseline_empty_route(self):
        route = _resolve()
        assert route.rm_type == ""
        assert not route.zero_reward
        assert registry.get_router_stats() == {}

    def test_unknown_no_fallback_flows_through_unchanged(self):
        route = _resolve(args_rm_type="totally_unknown_type")
        assert route.rm_type == "totally_unknown_type"
        assert not route.zero_reward
        assert registry.get_router_stats() == {}

    def test_boxed_prefix_stripped_and_flagged(self):
        route = _resolve(metadata_rm_type="boxed_openr1mm")
        assert route.rm_type == "openr1mm"
        assert route.boxed is True

    def test_label_matcher_requires_opt_in(self):
        assert _resolve(label="42").rm_type == ""
        assert _resolve(label="42", infer=True).rm_type == "math"


class TestFallback:
    """Configured fallback for unknown and missing types."""

    def test_missing_with_named_fallback(self):
        route = _resolve(fallback="math")
        assert route.rm_type == "math"
        assert registry.get_router_stats() == {("missing", ""): 1}

    def test_unknown_with_named_fallback(self):
        route = _resolve(args_rm_type="mystery", fallback="math")
        assert route.rm_type == "math"
        assert registry.get_router_stats() == {("unknown", "mystery"): 1}

    def test_unknown_with_zero_fallback(self):
        route = _resolve(args_rm_type="mystery", fallback="zero")
        assert route.zero_reward is True
        assert route.rm_type == ""
        assert registry.get_router_stats() == {("unknown", "mystery"): 1}

    def test_valid_explicit_with_flags_on_no_degradation(self):
        route = _resolve(metadata_rm_type="math", label="42", infer=True, fallback="zero")
        assert route.rm_type == "math"
        assert not route.zero_reward
        assert registry.get_router_stats() == {}


class TestConflictAndInference:
    """Label inference, conflict policy, and ambiguity handling."""

    def test_unique_inference_routes_sample(self):
        route = _resolve(label="<answer>B</answer>", infer=True)
        assert route.rm_type == "multiple_choice"
        assert registry.get_router_stats() == {}

    def test_registered_matcher_participates_in_inference(self):
        registry.register_reward(
            "t18_tagged",
            lambda response, label: 1.0,
            label_matcher=lambda label: label == "TAG",
        )
        assert _resolve(label="TAG", infer=True).rm_type == "t18_tagged"

    def test_conflict_prefers_explicit_and_warns(self, caplog):
        caplog.set_level(logging.WARNING, logger=registry.logger.name)
        route = _resolve(metadata_rm_type="multiple_choice", label="42", infer=True)
        assert route.rm_type == "multiple_choice"
        assert registry.get_router_stats() == {("conflict", ("multiple_choice", "math")): 1}
        conflict_records = [r for r in caplog.records if "conflict" in r.message]
        assert len(conflict_records) == 1
        assert "multiple_choice" in conflict_records[0].message
        assert "math" in conflict_records[0].message

    def test_inference_never_overrides_unknown_explicit(self):
        route = _resolve(args_rm_type="mystery", label="42", infer=True)
        assert route.rm_type == "mystery"
        route = _resolve(args_rm_type="mystery", label="42", infer=True, fallback="math")
        assert route.rm_type == "math"

    def test_ambiguous_inference_yields_nothing_and_warns(self):
        registry.register_reward(
            "t18_also_math",
            lambda response, label: 1.0,
            label_matcher=lambda label: isinstance(label, str) and label.strip().isdigit(),
        )
        route = _resolve(label="42", infer=True)
        assert route.rm_type == ""
        assert registry.get_router_stats() == {("ambiguous_inference", ("math", "t18_also_math")): 1}

    def test_crashing_matcher_degrades_to_no_match(self):
        def _bomb(label):
            raise RuntimeError("boom")

        registry.register_reward("t18_bomb", lambda response, label: 1.0, label_matcher=_bomb)
        assert _resolve(label="<answer>C</answer>", infer=True).rm_type == "multiple_choice"


class TestWarningDedup:
    """One log line per identity; counters stay exact."""

    def test_warning_logged_once_counter_exact(self, caplog):
        caplog.set_level(logging.WARNING, logger=registry.logger.name)
        for index in range(5):
            _resolve(args_rm_type="mystery", fallback="zero", sample_index=index)
        assert registry.get_router_stats() == {("unknown", "mystery"): 5}
        mystery_records = [r for r in caplog.records if "mystery" in r.message]
        assert len(mystery_records) == 1

    def test_distinct_identities_warn_separately(self, caplog):
        caplog.set_level(logging.WARNING, logger=registry.logger.name)
        for _ in range(3):
            _resolve(args_rm_type="alpha", fallback="zero")
        for _ in range(2):
            _resolve(args_rm_type="beta", fallback="zero")
        assert registry.get_router_stats() == {("unknown", "alpha"): 3, ("unknown", "beta"): 2}
        degradations = [r for r in caplog.records if "degradation" in r.message]
        assert len(degradations) == 2

    def test_counters_isolated_between_tests(self):
        assert registry.get_router_stats() == {}


class TestLabelMatchers:
    """Built-in matchers are conservative and reject non-string labels."""

    def test_math_matcher_accepts_strict_numerics(self):
        for label in ["42", "-3", "0.5", " 7 ", "-3/4"]:
            assert registry._looks_like_math_label(label), label

    def test_math_matcher_rejects_non_numerics(self):
        for label in ["\\boxed{42}", "x + 1", "<answer>A</answer>", "", "42 apples"]:
            assert not registry._looks_like_math_label(label), label

    def test_multiple_choice_matcher_accepts_single_letter_tag(self):
        for label in ["<answer>A</answer>", " <answer> B </answer> "]:
            assert registry._looks_like_multiple_choice_label(label), label

    def test_multiple_choice_matcher_rejects_others(self):
        for label in ["A", "<answer>AB</answer>", "<answer>a</answer>", "<answer>42</answer>", ""]:
            assert not registry._looks_like_multiple_choice_label(label), label

    def test_matchers_reject_non_str_labels(self):
        for label in [None, 42, 3.14, {"answer": "A"}, ["A"]]:
            assert not registry._looks_like_math_label(label)
            assert not registry._looks_like_multiple_choice_label(label)

    def test_matchers_are_disjoint_on_flagship_labels(self):
        assert registry._looks_like_math_label("42") and not registry._looks_like_multiple_choice_label("42")
        mc_label = "<answer>B</answer>"
        assert registry._looks_like_multiple_choice_label(mc_label) and not registry._looks_like_math_label(mc_label)
