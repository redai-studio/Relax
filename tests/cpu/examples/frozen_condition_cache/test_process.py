import json
import os
import subprocess
import sys

import pytest
from runner.common import CANCEL_REQUESTED, execute


def test_own_timeout_preserves_exit_receipt(tmp_path):
    receipt = tmp_path / "stage.json"
    with pytest.raises(subprocess.TimeoutExpired):
        execute([sys.executable, "-c", "import time; time.sleep(30)"], tmp_path, os.environ.copy(), receipt, 1)
    result = json.loads(receipt.read_text())
    assert result["status"] == "timed_out_or_interrupted"
    assert result["native_returncode"] is not None
    with pytest.raises(ProcessLookupError):
        os.kill(result["pid"], 0)


def test_owned_cancellation_is_not_global_process_kill(tmp_path):
    CANCEL_REQUESTED.set()
    try:
        with pytest.raises(KeyboardInterrupt):
            execute(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                tmp_path,
                os.environ.copy(),
                tmp_path / "cancel.json",
                30,
            )
    finally:
        CANCEL_REQUESTED.clear()
    assert json.loads((tmp_path / "cancel.json").read_text())["native_returncode"] is not None
