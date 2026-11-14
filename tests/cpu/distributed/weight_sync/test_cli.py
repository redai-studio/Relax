import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class CLITests(unittest.TestCase):
    def test_publish_and_offline_replay_in_separate_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "original"
            original.mkdir()
            (original / "config.json").write_text("{}")
            header = json.dumps({"w": {"dtype": "BF16", "shape": [4], "data_offsets": [0, 8]}}).encode()
            header += b" " * (-len(header) % 8)
            data = bytes.fromhex("00000080017fc17f")
            (original / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + data)
            common = [
                sys.executable,
                "-m",
                "relax.distributed.weight_sync",
                "--disk-budget",
                "1048576",
                "--max-weight-bytes",
                "1024",
            ]
            published = subprocess.run(
                [
                    *common,
                    "--store",
                    str(root / "published"),
                    "publish",
                    "--checkpoint",
                    str(original),
                    "--stream",
                    "run",
                    "--version",
                    "0",
                ],
                check=True,
                capture_output=True,
            )
            fetched = subprocess.run(
                [
                    *common,
                    "--store",
                    str(root / "receiver"),
                    "fetch",
                    "--source-store",
                    str(root / "published"),
                    "--stream",
                    "run",
                    "--version",
                    "0",
                    "--checkpoint-output",
                    str(root / "rebuilt"),
                    "--config",
                    str(original / "config.json"),
                ],
                check=True,
                capture_output=True,
            )
            self.assertEqual(json.loads(published.stdout)["root"], json.loads(fetched.stdout)["root"])
            self.assertFalse(json.loads(fetched.stdout)["installed"])
            self.assertEqual((root / "rebuilt/model.safetensors").read_bytes()[-8:], data)


if __name__ == "__main__":
    unittest.main()
