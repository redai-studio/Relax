import json
import tempfile
import threading
import unittest
from pathlib import Path

from relax.distributed.weight_sync.sglang_loader import EmbeddedSGLangGroup
from relax.distributed.weight_sync.store import atomic_write, canonical


class PipelineRPCTests(unittest.TestCase):
    def group(self, directory):
        group = object.__new__(EmbeddedSGLangGroup)
        group.tp_size, group.pp_size = 1, 2
        group.audit_directory = Path(directory)
        group._rpc_failed = False
        group._workers = {str(rank): {"instance": str(rank)} for rank in range(2)}
        return group

    def test_stale_later_stage_prevents_sending_a_load_command(self):
        with tempfile.TemporaryDirectory() as temporary:
            group, phases = self.group(temporary), []

            def send(payload):
                request = json.loads(payload)
                phases.append(request["phase"])
                for rank in range(2):
                    atomic_write(
                        Path(request["receipts"]) / f"rank-{rank}.json",
                        canonical(
                            {
                                "rank": rank,
                                "request_id": request["request_id"],
                                "success": rank == 0,
                                "identity": {"instance": str(rank)},
                                "error": "stale later stage" if rank else None,
                            }
                        ),
                    )

            group._send_rpc = send
            with self.assertRaisesRegex(RuntimeError, "stale later stage"):
                group._rpc("load", checkpoint="candidate", version="next")
            self.assertEqual(phases, ["check"])

    def test_first_stage_reply_waits_for_the_last_stage_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            group, writers = self.group(temporary), []

            def send(payload):
                request = json.loads(payload)

                def reply(rank):
                    atomic_write(
                        Path(request["receipts"]) / f"rank-{rank}.json",
                        canonical(
                            {
                                "rank": rank,
                                "request_id": request["request_id"],
                                "success": True,
                                "identity": {"instance": str(rank)},
                                "layout": {},
                            }
                        ),
                    )

                reply(0)
                timer = threading.Timer(0.03, reply, args=(1,))
                timer.start()
                writers.append(timer)

            group._send_rpc = send
            try:
                self.assertEqual(set(group._rpc("identify")), {"0", "1"})
            finally:
                for writer in writers:
                    writer.join()
