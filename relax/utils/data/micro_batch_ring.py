# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU-only prefetch helpers for the SFT prepack pipeline.

Enabled by ``--sft-async-prepack``. The prefetch worker fetches the whole
rollout partition from TransferQueue, runs the same seqlen-balanced K-way
partition dev uses, CPU-packs each micro-batch, pins CPU memory, and enqueues
the first H2D copy — all off the training thread. The training thread only pays
for one H2D wait per micro-batch.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Any


def _raise_pin_memory_failure(context: str, exc: BaseException) -> None:
    """Fail fast on pin_memory errors.

    Falling back to a pageable CPU tensor would push the H2D cost onto the
    training thread (``.to(..., non_blocking=True)`` implicitly stages via a
    pinned buffer or blocks) and silently erase the whole point of the prepack
    pipeline. Surface the failure instead so the user can tune pinned memory
    limits or drop --sft-async-prepack.
    """
    raise RuntimeError(
        f"pin_memory failed in {context} ({exc}); the SFT prepack pipeline requires pinned CPU "
        "tensors so the training thread never issues pageable H2D copies. Increase the pinned "
        "memory limit (e.g. ulimit -l) or disable --sft-async-prepack."
    ) from exc


def is_sft_async_prepack_enabled(args: Any) -> bool:
    """The SFT prepack pipeline is opt-in via --sft-async-prepack."""
    return bool(getattr(args, "sft_async_prepack", False))


@dataclass(frozen=True)
class PrefetchedSFTWindow:
    rollout_id: int
    rollout_data: dict[str, Any]
    packed_micro_batches: list[tuple[dict[str, Any], Any]]
    first_device_micro_batch: tuple[dict[str, Any], Any]
    first_ready_event: Any


class SFTWindowPrefetcher:
    """Keep at most one future SFT window in a daemon worker.

    Re-invoking ``prefetch`` with the same ``rollout_id`` is idempotent: the
    already-running worker's result will be returned by the next ``get``. Pass
    an ``identity`` key so a lookahead ``prefetch`` (typically triggered after
    a successful ``get``) cannot silently reuse a stale closure.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._rollout_id: int | None = None
        self._identity: Hashable | None = None
        self._result: PrefetchedSFTWindow | None = None
        self._error: BaseException | None = None
        self._thread: threading.Thread | None = None

    def prefetch(
        self,
        rollout_id: int,
        fetch_fn: Callable[[], PrefetchedSFTWindow],
        *,
        identity: Hashable | None = None,
    ) -> None:
        with self._condition:
            if self._rollout_id is not None:
                if self._rollout_id != rollout_id:
                    raise RuntimeError(
                        f"SFT prefetcher already owns rollout {self._rollout_id}; cannot start rollout {rollout_id}"
                    )
                if identity is not None and self._identity is not None and identity != self._identity:
                    raise RuntimeError(
                        f"SFT prefetcher rollout {rollout_id} re-invoked with different identity "
                        f"{identity!r} (previous={self._identity!r}); the second fetch_fn would "
                        "silently be ignored."
                    )
                return
            self._rollout_id = rollout_id
            self._identity = identity
            self._result = None
            self._error = None

        def _run() -> None:
            try:
                result = fetch_fn()
                if result.rollout_id != rollout_id:
                    raise RuntimeError(
                        f"SFT prefetch worker returned rollout {result.rollout_id}; expected {rollout_id}"
                    )
                with self._condition:
                    self._result = result
                    self._condition.notify_all()
            except BaseException as exc:  # noqa: BLE001
                with self._condition:
                    self._error = exc
                    self._condition.notify_all()

        self._thread = threading.Thread(target=_run, name=f"sft-window-{rollout_id}", daemon=True)
        self._thread.start()

    def get(self, rollout_id: int) -> PrefetchedSFTWindow:
        with self._condition:
            if self._rollout_id != rollout_id:
                raise RuntimeError(f"SFT prefetcher has rollout {self._rollout_id}, requested {rollout_id}")
            while self._result is None and self._error is None:
                self._condition.wait()
            if self._error is not None:
                error = self._error
                self._reset_locked()
                raise error
            assert self._result is not None
            result = self._result
            self._reset_locked()
            return result

    def _reset_locked(self) -> None:
        self._rollout_id = None
        self._identity = None
        self._result = None
        self._error = None
        # The daemon thread has already completed by the time get() returns;
        # dropping the reference is safe.
        self._thread = None


__all__ = [
    "PrefetchedSFTWindow",
    "SFTWindowPrefetcher",
    "is_sft_async_prepack_enabled",
]
