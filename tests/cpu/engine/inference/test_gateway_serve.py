# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""ASGI tests cannot validate Serve's serialized constructor and app binding.

One isolated local Ray runtime deploys all three CPU ingresses. The inference
engine, tokenizer and rollout workload are external-boundary stubs; real Serve
routing, constructors and task-Manager admission execute unchanged.
"""

import os
import subprocess
import sys
from pathlib import Path


def test_three_role_gateways_construct_and_serve_real_http():
    root = Path(__file__).resolve().parents[4]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(root))
    result = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("gateway_serve_smoke.py"))],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "REAL_SERVE_THREE_ROLES_OK" in result.stdout
