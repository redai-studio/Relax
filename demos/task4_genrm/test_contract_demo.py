# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Contract invariants for the mock GenRM lifecycle."""

import unittest

from contract_demo import ContractDemo, scripted_scenarios


class ContractDemoTest(unittest.TestCase):
    def test_scripted_contract(self) -> None:
        checks = scripted_scenarios()["checks"]
        self.assertTrue(all(value is True for key, value in checks.items() if key != "final_capacity"))
        self.assertEqual(checks["final_capacity"], [1, 1])

    def test_health_gates_route_and_late_ack(self) -> None:
        demo = ContractDemo()
        request_id = demo.request("out", 2)["request_id"]
        self.assertEqual(demo.scale_out(request_id, health_ok=False), "FAILED")
        self.assertEqual(demo.current, 1)
        self.assertEqual(demo.ready, 1)
        self.assertNotIn("manager-pg-1", demo.pg_live)
        self.assertFalse(demo.late_health_ack("genrm-1"))

    def test_existing_replicas_do_not_make_a_failed_operation_partial(self) -> None:
        demo = ContractDemo()
        demo.scale_out(demo.request("out", 2)["request_id"])
        request_id = demo.request("out", 3)["request_id"]
        self.assertEqual(demo.scale_out(request_id, health_ok=False), "FAILED")
        self.assertEqual((demo.current, demo.ready), (2, 2))

    def test_partial_means_this_operation_added_capacity(self) -> None:
        demo = ContractDemo()
        request_id = demo.request("out", 3)["request_id"]
        self.assertEqual(demo.scale_out(request_id, health_ok=(True, False)), "PARTIAL")
        self.assertEqual((demo.current, demo.ready), (2, 2))

    def test_idempotency_key_replays_only_the_same_request(self) -> None:
        demo = ContractDemo()
        first = demo.request("out", 2, idempotency_key="client-7")
        replay = demo.request("out", 2, idempotency_key="client-7")
        conflict = demo.request("out", 3, idempotency_key="client-7")
        self.assertEqual(replay["request_id"], first["request_id"])
        self.assertEqual(replay["status"], "PENDING")
        self.assertEqual(conflict["http"], 409)
        demo.scale_out(first["request_id"])
        terminal_replay = demo.request("out", 2, idempotency_key="client-7")
        self.assertEqual(terminal_replay["request_id"], first["request_id"])
        self.assertEqual(terminal_replay["status"], "ACTIVE")

    def test_keyed_noop_replay_never_executes_after_capacity_change(self) -> None:
        """Regression: a keyed NOOP must be recorded and replayed verbatim.

        Before the fix, the NOOP path returned without writing an idempotency
        record, so a retry after capacity changed executed a new scale-out.
        """
        demo = ContractDemo()
        demo.scale_out(demo.request("out", 2)["request_id"])
        first = demo.request("out", 2, idempotency_key="retry")
        self.assertEqual(first["status"], "NOOP")
        inward = demo.request("in", 1)["request_id"]
        demo.close_admission(inward)
        demo.finish_scale_in(inward)
        self.assertEqual(demo.current, 1)
        replay = demo.request("out", 2, idempotency_key="retry")
        self.assertEqual(replay["status"], "NOOP")
        self.assertEqual(replay["current"], 2)
        self.assertNotIn("request_id", replay)
        self.assertIsNone(demo.active_request)
        # A new request without the key still executes by actual difference.
        fresh = demo.request("out", 2)
        self.assertEqual(fresh["status"], "PENDING")

    def test_idempotency_fingerprint_covers_model_and_timeout(self) -> None:
        demo = ContractDemo()
        first = demo.request("out", 2, model="judge-a", timeout_secs=60, idempotency_key="k")
        self.assertEqual(first["status"], "PENDING")
        self.assertEqual(demo.request("out", 2, model="judge-b", timeout_secs=60, idempotency_key="k")["http"], 409)
        self.assertEqual(demo.request("out", 2, model="judge-a", timeout_secs=120, idempotency_key="k")["http"], 409)
        replay = demo.request("out", 2, model="judge-a", timeout_secs=60, idempotency_key="k")
        self.assertEqual(replay["request_id"], first["request_id"])

    def test_idempotent_replay_of_failed_operation(self) -> None:
        demo = ContractDemo()
        request_id = demo.request("out", 2, idempotency_key="f")["request_id"]
        demo.scale_out(request_id, health_ok=False)
        replay = demo.request("out", 2, idempotency_key="f")
        self.assertEqual(replay["status"], "FAILED")
        self.assertEqual(replay["request_id"], request_id)

    def test_drain_waits_for_accepted_call_and_backend(self) -> None:
        demo = ContractDemo()
        demo.scale_out(demo.request("out", 2)["request_id"])
        self.assertTrue(demo.admit("genrm-1", "score"))
        request_id = demo.request("in", 1)["request_id"]
        demo.close_admission(request_id)
        self.assertEqual(demo.current, 2)
        self.assertEqual(demo.ready, 1)
        self.assertFalse(demo.admit("genrm-1", "stale"))
        self.assertEqual(demo.finish_scale_in(request_id), "DRAINING")
        demo.complete("genrm-1", "score")
        self.assertEqual(demo.finish_scale_in(request_id, backend_idle=False), "DRAINING")
        self.assertEqual(demo.finish_scale_in(request_id), "COMPLETED")
        self.assertEqual(demo.current, 1)
        self.assertIn("training-pg", demo.pg_live)

    def test_absolute_target_with_multiple_new_replicas(self) -> None:
        demo = ContractDemo()
        demo.scale_out(demo.request("out", 3)["request_id"])
        self.assertEqual((demo.current, demo.ready), (3, 3))
        request_id = demo.request("in", 1)["request_id"]
        demo.close_admission(request_id)
        self.assertEqual(demo.finish_scale_in(request_id), "PENDING")
        self.assertEqual(demo.request("out", 3)["http"], 409)
        demo.close_admission(request_id)
        self.assertEqual(demo.finish_scale_in(request_id), "COMPLETED")
        self.assertEqual((demo.current, demo.ready), (1, 1))

    def test_invalid_and_unresolved_cleanup(self) -> None:
        demo = ContractDemo()
        self.assertEqual(demo.request("out", True)["http"], 422)
        self.assertEqual(demo.request("in", 0)["http"], 400)
        self.assertEqual(demo.get_status("missing")["http"], 404)
        request_id = demo.request("out", 2)["request_id"]
        demo.scale_out(request_id, health_ok=False, cleanup_ok=False)
        self.assertIn("manager-pg-1", demo.pg_live)
        self.assertEqual(demo.request("out", 2)["http"], 409)
        self.assertEqual(demo.retry_cleanup(request_id), "RECONCILED")
        retry = demo.request("out", 2)
        self.assertEqual(demo.scale_out(retry["request_id"]), "ACTIVE")

    def test_failed_drain_requires_explicit_retry(self) -> None:
        demo = ContractDemo()
        demo.scale_out(demo.request("out", 2)["request_id"])
        request_id = demo.request("in", 1)["request_id"]
        demo.close_admission(request_id)
        demo.fail_drain(request_id)
        self.assertTrue(demo.paused)
        self.assertIn("manager-pg-1", demo.pg_live)
        self.assertEqual(demo.retry_scale_in(request_id), "COMPLETED")
        self.assertFalse(demo.paused)
        self.assertEqual((demo.current, demo.ready), (1, 1))


if __name__ == "__main__":
    unittest.main()
