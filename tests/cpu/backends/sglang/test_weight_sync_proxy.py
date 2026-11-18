# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from collections import deque
from time import monotonic
from types import SimpleNamespace

import pytest

from relax.backends.sglang.weight_sync_proxy import WeightSyncProxy


def test_proxy_consumes_entire_failed_reply_batch_before_reporting_error():
    class Channel:
        def __init__(self):
            self.replies = deque([dict(success=False, error="pending IPC"), dict(success=True)])

        def poll(self, timeout):
            return bool(self.replies)

        def recv(self):
            return self.replies.popleft()

    proxy = WeightSyncProxy(SimpleNamespace())
    proxy.children = [(None, Channel()), (None, Channel())]
    with pytest.raises(RuntimeError, match="batch failed"):
        proxy._receive_replies(deadline=monotonic() + 1)
    assert proxy._receive_replies(deadline=monotonic() + 1) == [dict(success=True), dict(success=True)]
