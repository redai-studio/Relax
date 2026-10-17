# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import os
import sys
from types import ModuleType


def test_http_utils_run_router_drops_inherited_proxies(monkeypatch):
    from relax.utils import http_utils

    seen = {}
    launch = ModuleType("sglang_router.launch_router")
    launch.launch_router = lambda args: (
        seen.update({k: os.environ.get(k) for k in http_utils._PROXY_ENV_VARS}) or object()
    )
    monkeypatch.setitem(sys.modules, "sglang_router.launch_router", launch)
    for name in http_utils._PROXY_ENV_VARS:
        monkeypatch.setenv(name, "http://127.0.0.1:1080")
    monkeypatch.setenv("NO_PROXY", "*")

    assert http_utils.run_router(object()) == 0

    # The Router's client ignores a wildcard NO_PROXY, so the proxies themselves must go.
    assert seen == dict.fromkeys(http_utils._PROXY_ENV_VARS)
