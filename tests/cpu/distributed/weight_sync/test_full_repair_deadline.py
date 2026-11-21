import threading
import time
from concurrent.futures import ThreadPoolExecutor

from relax.distributed.weight_sync.store import VersionStore, load_json
from relax.distributed.weight_sync.transport import ObjectServer, TCPTransport


def test_full_repair_has_operation_budget_and_concurrent_requests_share_work(tmp_path):
    store = VersionStore(tmp_path / "store", disk_budget=1024**2, max_weight_bytes=1024)
    path = tmp_path / "w.bin"
    path.write_bytes(bytes(64))
    store.publish("run", 0, {"w": ("U8", [64], path)}, model_config={}, converter_revision="test")
    path.write_bytes(b"\x01" + bytes(63))
    target = store.publish("run", 1, {"w": ("U8", [64], path)}, model_config={}, converter_revision="test")
    original = store.ensure_full
    calls = []
    entered = threading.Event()

    def slow(*args, **kwargs):
        calls.append(1)
        entered.set()
        time.sleep(0.6)
        return original(*args, **kwargs)

    store.ensure_full = slow
    with ObjectServer(("127.0.0.1", 0), store, timeout=0.2) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:

            def fetch():
                return load_json(TCPTransport(server.server_address, timeout=0.2).manifest("run", 1, full=True))

            with ThreadPoolExecutor(2) as pool:
                first = pool.submit(fetch)
                assert entered.wait(2)
                second = pool.submit(fetch)
                left, right = first.result(5), second.result(5)
            assert len(calls) == 1
            assert left == right
            assert left["kind"] == "full" and left["root"] == target["root"]
            # A later repair rechecks an existing FULL artifact, rather than
            # permanently trusting an in-memory successful result.
            again = fetch()
            assert again == left and len(calls) == 2
        finally:
            server.shutdown()
            thread.join(3)
