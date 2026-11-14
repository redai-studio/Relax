import unittest

from relax.distributed.weight_sync.fence import WorkerFence


class FenceTests(unittest.TestCase):
    def test_old_command_never_reaches_device_operation(self):
        fence = WorkerFence()
        old = fence.begin(fence.instance, 1, "one", "root-one")
        current = fence.begin(fence.instance, 2, "two", "root-two")
        calls = []
        with self.assertRaises(ValueError):
            fence.execute(old, "load", "LOADING", lambda: calls.append("old"))
        self.assertEqual(calls, [])
        reply = fence.execute(current, "load", "LOADING", lambda: calls.append("new"))
        self.assertEqual(fence.execute(current, "load", "LOADING", lambda: calls.append("repeat")), reply)
        self.assertEqual(calls, ["new"])
        with self.assertRaises(ValueError):
            fence.execute(current, "load", "OPEN", lambda: calls.append("wrong"))

    def test_restart_requires_fresh_instance_binding(self):
        old = WorkerFence()
        old_identity = old.begin(old.instance, 4, "four", "root")
        restarted = WorkerFence()
        with self.assertRaises(ValueError):
            restarted.begin(old.instance, 5, "five", "root")
        with self.assertRaises(ValueError):
            restarted.execute(old_identity, "resume", "RESUME", lambda: self.fail("stale command executed"))

    def test_failed_operation_can_be_retried_without_claiming_success(self):
        fence = WorkerFence()
        identity = fence.begin(fence.instance, 0, "one", "root")

        def fail():
            raise OSError("injected")

        with self.assertRaises(OSError):
            fence.execute(identity, "load", "LOADING", fail)
        reply = fence.execute(identity, "load", "LOADING", lambda: "complete")
        self.assertEqual(reply["result"], "complete")
