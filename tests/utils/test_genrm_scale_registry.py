# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for the GenRM scale-operation registry (Task 4 contract core).

Covers the state machine, idempotency (replay / fingerprint conflict / keyed
NOOP verbatim replay), mutual exclusion, absolute-target validation and the
terminal-status alignment with the autoscaler's existing terminal sets. The
scenarios mirror demos/task4_genrm/contract_demo.py.

Run: python -m unittest tests.utils.test_genrm_scale_registry -v
"""

import unittest

from tests.utils._dep_stubs import install_web_framework_stubs


install_web_framework_stubs()  # relax.utils.autoscaler's package __init__ imports fastapi

from relax.utils.autoscaler.scaling_decision import (  # noqa: E402
    SCALE_IN_TERMINAL_STATUSES,
    SCALE_OUT_TERMINAL_STATUSES,
    is_scale_request_terminal,
)
from relax.utils.genrm_scale_registry import (  # noqa: E402
    GENRM_SCALE_IN_TERMINAL_STATUSES,
    GENRM_SCALE_OUT_TERMINAL_STATUSES,
    GenRMScaleRegistry,
)


def _registry(model: str = "__default__", initial: int = 1) -> GenRMScaleRegistry:
    registry = GenRMScaleRegistry()
    registry.register_initial(model, initial)
    return registry


def _submit_out(registry: GenRMScaleRegistry, target: int, **kwargs):
    kwargs.setdefault("model_name", "__default__")
    kwargs.setdefault("current", 1)
    kwargs.setdefault("ready", 1)
    return registry.submit("scale_out", target=target, **kwargs)


def _submit_in(registry: GenRMScaleRegistry, target: int, **kwargs):
    kwargs.setdefault("model_name", "__default__")
    kwargs.setdefault("current", 2)
    kwargs.setdefault("ready", 2)
    return registry.submit("scale_in", target=target, **kwargs)


class TestStateMachine(unittest.TestCase):
    def test_scale_out_happy_path_reaches_active(self):
        registry = _registry()
        decision = _submit_out(registry, 2)
        request_id = decision["request_id"]
        for status in ("CREATING", "HEALTH_CHECKING", "READY", "ACTIVE"):
            registry.advance(request_id, status)
        result = registry.get_status("scale_out", request_id)
        self.assertEqual(result["status"], "ACTIVE")

    def test_scale_out_illegal_transition_rejected(self):
        registry = _registry()
        request_id = _submit_out(registry, 2)["request_id"]
        with self.assertRaises(ValueError):
            registry.advance(request_id, "ACTIVE")  # PENDING -> ACTIVE skips the chain

    def test_scale_out_failure_reachable_from_every_live_state(self):
        for stop in ("PENDING", "CREATING", "HEALTH_CHECKING", "READY"):
            registry = _registry()
            request_id = _submit_out(registry, 2)["request_id"]
            chain = {
                "PENDING": [],
                "CREATING": ["CREATING"],
                "HEALTH_CHECKING": ["CREATING", "HEALTH_CHECKING"],
                "READY": ["CREATING", "HEALTH_CHECKING", "READY"],
            }[stop]
            for status in chain:
                registry.advance(request_id, status)
            registry.finish(request_id, status="FAILED", current=1, ready=1, failed=1)
            self.assertEqual(registry.get_status("scale_out", request_id)["status"], "FAILED")

    def test_scale_in_chain_completes(self):
        registry = _registry()
        request_id = _submit_in(registry, 1)["request_id"]
        for status in ("DRAINING", "REMOVING", "COMPLETED"):
            registry.advance(request_id, status)
        self.assertEqual(registry.get_status("scale_in", request_id)["status"], "COMPLETED")

    def test_scale_in_illegal_transition_rejected(self):
        registry = _registry()
        request_id = _submit_in(registry, 1)["request_id"]
        with self.assertRaises(ValueError):
            registry.advance(request_id, "REMOVING")  # PENDING -> REMOVING skips DRAINING

    def test_terminal_alignment_with_autoscaler_sets(self):
        """The autoscaler's existing terminal sets must keep classifying GenRM
        operation statuses correctly (success terminal = ACTIVE, no
        CANCELLED on purpose; scale-in sets are identical)."""
        self.assertEqual(GENRM_SCALE_OUT_TERMINAL_STATUSES, {"ACTIVE", "PARTIAL", "FAILED"})
        self.assertTrue(GENRM_SCALE_OUT_TERMINAL_STATUSES <= SCALE_OUT_TERMINAL_STATUSES)
        self.assertEqual(GENRM_SCALE_IN_TERMINAL_STATUSES, SCALE_IN_TERMINAL_STATUSES)
        for status in GENRM_SCALE_OUT_TERMINAL_STATUSES:
            self.assertTrue(is_scale_request_terminal("scale_out", status))
        for status in GENRM_SCALE_IN_TERMINAL_STATUSES:
            self.assertTrue(is_scale_request_terminal("scale_in", status))
        # Non-terminal GenRM states must stay non-terminal for the autoscaler.
        for status in ("PENDING", "CREATING", "HEALTH_CHECKING", "READY"):
            self.assertFalse(is_scale_request_terminal("scale_out", status))
        for status in ("PENDING", "DRAINING", "REMOVING"):
            self.assertFalse(is_scale_request_terminal("scale_in", status))

    def test_finish_carries_terminal_result_fields(self):
        """Terminal status must report
        target/current/ready/created/removed/failed/ cleanup_required (RFC API
        contract)."""
        registry = _registry()
        request_id = _submit_out(registry, 3)["request_id"]
        registry.advance(request_id, "CREATING")
        registry.advance(request_id, "HEALTH_CHECKING")
        result = registry.finish(
            request_id, status="PARTIAL", current=2, ready=2, created=1, failed=1, cleanup_required=False
        )
        self.assertEqual(result["target"], 3)
        self.assertEqual(result["current"], 2)
        self.assertEqual(result["ready"], 2)
        self.assertEqual(result["created"], 1)
        self.assertEqual(result["failed"], 1)
        self.assertFalse(result["cleanup_required"])
        self.assertEqual(registry.get_status("scale_out", request_id)["status"], "PARTIAL")

    def test_finish_rejects_non_terminal_status(self):
        registry = _registry()
        request_id = _submit_out(registry, 2)["request_id"]
        with self.assertRaises(ValueError):
            registry.finish(request_id, status="CREATING", current=1, ready=1)

    def test_unknown_request_id_is_none(self):
        self.assertIsNone(_registry().get_status("scale_out", "no-such-id"))

    def test_directions_do_not_leak_into_each_others_queries(self):
        registry = _registry()
        out_id = _submit_out(registry, 2)["request_id"]
        self.assertIsNone(registry.get_status("scale_in", out_id))


class TestIdempotency(unittest.TestCase):
    def test_same_key_same_fingerprint_returns_original_operation(self):
        registry = _registry()
        first = _submit_out(registry, 2, idempotency_key="k1", timeout_secs=60.0)
        registry.advance(first["request_id"], "CREATING")
        replay = _submit_out(registry, 2, idempotency_key="k1", timeout_secs=60.0)
        self.assertEqual(replay["http"], 200)
        self.assertEqual(replay["request_id"], first["request_id"])
        self.assertEqual(replay["status"], "CREATING")  # current status, not PENDING
        self.assertFalse(replay.get("dispatch", False))  # replay never re-dispatches

    def test_same_key_different_fingerprint_is_409(self):
        registry = _registry()
        _submit_out(registry, 2, idempotency_key="k1")
        conflict = _submit_out(registry, 3, idempotency_key="k1")
        self.assertEqual(conflict["http"], 409)
        self.assertEqual(conflict["status"], "IDEMPOTENCY_CONFLICT")
        # timeout_secs participates in the fingerprint too.
        other_timeout = _submit_out(registry, 2, idempotency_key="k1", timeout_secs=90.0)
        self.assertEqual(other_timeout["http"], 409)

    def test_keyed_noop_replayed_verbatim_after_capacity_change(self):
        """A keyed NOOP decision is recorded; after capacity changes, the same
        key and body must replay the original response (with the current
        observed at decision time) instead of executing a new operation."""
        registry = _registry()
        noop = _submit_out(registry, 2, idempotency_key="recheck", current=2, ready=2)
        self.assertEqual(noop["status"], "NOOP")
        self.assertEqual(noop["current"], 2)
        self.assertNotIn("request_id", noop)
        # Capacity dropped to 1; the retry still replays the recorded NOOP.
        replay = _submit_out(registry, 2, idempotency_key="recheck", current=1, ready=1)
        self.assertEqual(replay, noop)

    def test_keyed_noop_conflicts_on_different_fingerprint(self):
        registry = _registry()
        _submit_out(registry, 2, idempotency_key="recheck", current=2, ready=2)
        conflict = _submit_out(registry, 4, idempotency_key="recheck", current=2, ready=2)
        self.assertEqual(conflict["http"], 409)

    def test_key_is_scoped_per_direction(self):
        """The same key on scale_out and scale_in are independent records:

        after the scale-out reaches a terminal state, the same key on scale_in
        admits a *new* operation instead of replaying the scale-out one.
        """
        registry = _registry()
        out = _submit_out(registry, 2, idempotency_key="same-key")
        registry.finish(out["request_id"], status="ACTIVE", current=2, ready=2, created=1)
        inward = _submit_in(registry, 1, idempotency_key="same-key")
        self.assertEqual(inward["http"], 200)
        self.assertNotEqual(inward["request_id"], out["request_id"])

    def test_in_flight_operation_beats_idempotency_of_other_direction(self):
        """A same-key scale-in while the keyed scale-out is still in flight is
        rejected by mutual exclusion (409), never executed."""
        registry = _registry()
        _submit_out(registry, 2, idempotency_key="same-key")
        blocked = _submit_in(registry, 1, idempotency_key="same-key")
        self.assertEqual(blocked["http"], 409)
        self.assertEqual(blocked["status"], "CONFLICT")


class TestMutualExclusion(unittest.TestCase):
    def test_second_in_flight_request_is_409(self):
        registry = _registry()
        first = _submit_out(registry, 2)
        self.assertEqual(first["http"], 200)
        second = _submit_out(registry, 3)
        self.assertEqual(second["http"], 409)
        self.assertEqual(second["status"], "CONFLICT")

    def test_exclusion_released_after_terminal(self):
        registry = _registry()
        request_id = _submit_out(registry, 2)["request_id"]
        registry.finish(request_id, status="ACTIVE", current=2, ready=2, created=1)
        # Now a scale-in towards the new capacity is admissible again.
        decision = _submit_in(registry, 1, current=2, ready=2)
        self.assertEqual(decision["http"], 200)

    def test_unresolved_cleanup_blocks_new_requests(self):
        """A terminal operation whose cleanup is still required keeps blocking
        the model until reconciled (RFC: 资源持续保留是失败隔离手段)."""
        registry = _registry()
        request_id = _submit_out(registry, 2)["request_id"]
        registry.finish(request_id, status="FAILED", current=1, ready=1, failed=1, cleanup_required=True)
        blocked = _submit_out(registry, 2)
        self.assertEqual(blocked["http"], 409)
        self.assertEqual(registry.active_operation("__default__")["request_id"], request_id)

    def test_exclusion_is_per_model(self):
        registry = _registry()
        registry.register_initial("quality", 1)
        _submit_out(registry, 2, model_name="__default__")
        other = _submit_out(registry, 2, model_name="quality", current=1, ready=1)
        self.assertEqual(other["http"], 200)


class TestAbsoluteTargetValidation(unittest.TestCase):
    def test_non_integer_target_is_422(self):
        registry = _registry()
        self.assertEqual(_submit_out(registry, "2")["http"], 422)
        self.assertEqual(_submit_out(registry, 1.5)["http"], 422)
        self.assertEqual(_submit_out(registry, True)["http"], 422)  # bool is not an int target

    def test_target_below_initial_is_400(self):
        registry = _registry(initial=1)
        self.assertEqual(_submit_in(registry, 0, current=2)["http"], 400)
        self.assertEqual(_submit_in(registry, 0, current=2)["status"], "INVALID_RANGE")

    def test_scale_out_noop_when_target_satisfied(self):
        registry = _registry()
        decision = _submit_out(registry, 1, current=1, ready=1)  # target == current
        self.assertEqual(decision["status"], "NOOP")
        self.assertEqual(decision["current"], 1)
        self.assertNotIn("request_id", decision)

    def test_scale_in_noop_when_target_satisfied(self):
        registry = _registry()
        decision = _submit_in(registry, 2, current=2, ready=2)  # target == current
        self.assertEqual(decision["status"], "NOOP")

    def test_unkeyed_noop_is_not_recorded_for_replay(self):
        """Without a key, a NOOP after a capacity change re-evaluates (demo:

        unkeyed NOOP is a fresh decision each time).
        """
        registry = _registry()
        first = _submit_out(registry, 2, current=2, ready=2)
        second = _submit_out(registry, 2, current=1, ready=1)
        self.assertEqual(first["status"], "NOOP")
        self.assertEqual(second["http"], 200)
        self.assertEqual(second["status"], "PENDING")  # re-evaluated, now actionable

    def test_set_detail_round_trips(self):
        registry = _registry()
        request_id = _submit_out(registry, 2)["request_id"]
        registry.set_detail(request_id, "manager_scale_not_implemented")
        self.assertEqual(registry.get_status("scale_out", request_id)["detail"], "manager_scale_not_implemented")


class TestListOperations(unittest.TestCase):
    def test_list_filters_by_model_and_status(self):
        registry = _registry()
        registry.register_initial("quality", 1)
        first = _submit_out(registry, 2, model_name="__default__")
        second = _submit_out(registry, 2, model_name="quality")
        registry.advance(second["request_id"], "CREATING")
        self.assertEqual(len(registry.list_operations("scale_out")), 2)
        self.assertEqual(len(registry.list_operations("scale_out", model_name="quality")), 1)
        pending = registry.list_operations("scale_out", status="PENDING")
        self.assertEqual([op["request_id"] for op in pending], [first["request_id"]])


if __name__ == "__main__":
    unittest.main()
