# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""ART lookahead: start the next generation before decoding this response.

The backend's open_response callback must return after generation and body
encoding complete. Only one unread response is retained; request/policy
identity must match exactly.
"""

import copy
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable


class HeaderLookahead:
    def __init__(self, open_response: Callable[[dict], Any]) -> None:
        self.open_response = open_response
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="art-http-lookahead")
        self.pending = None
        self.closed = False

    @staticmethod
    def key(body: dict, policy: Any) -> str:
        return hashlib.sha256(
            json.dumps([body, policy], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()

    def _open(self, body: dict) -> Any:
        response = self.open_response(body)
        try:
            response.raise_for_status()
        except BaseException:
            response.close()
            raise
        return response

    def consume(self, body: dict, next_body: dict | None, policy: Any, decode: Callable[[Any], Any]) -> Any:
        if self.closed:
            raise RuntimeError("Lookahead is closed")
        response = None
        try:
            key = self.key(body, policy)
            if self.pending is None:
                response = self._open(body)
            else:
                expected, future = self.pending
                if expected != key:
                    raise RuntimeError("ART lookahead request or policy mismatch; refusing resampling")
                self.pending = None
                response = future.result()
            # The previous GPU generation is now complete. Schedule before any body access.
            if next_body is not None:
                frozen = copy.deepcopy(next_body)
                self.pending = self.key(frozen, policy), self.pool.submit(self._open, frozen)
            return decode(response)
        except BaseException:
            # The failing current response must be released before waiting on the next.
            if response is not None:
                response.close()
                response = None
            self.close()
            raise
        finally:
            if response is not None:
                response.close()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            if self.pending is not None:
                _, future = self.pending
                self.pending = None
                # Cannot cancel running HTTP: wait for server completion, then close.
                response = future.result()
                response.close()
        finally:
            self.pool.shutdown(wait=True, cancel_futures=True)
