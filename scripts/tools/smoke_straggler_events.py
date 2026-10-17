# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""One-GPU check that stage events read back without synchronizing inside
start/stop."""

import torch

from relax.utils.straggler.timer_shim import NonBlockingTimers, reset_timers_for_tests


def main() -> None:
    reset_timers_for_tests()
    timers = NonBlockingTimers()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise SystemExit("cuda is required")
    x = torch.randn(1024, 1024, device=device)
    timers("forward-compute").start()
    y = x @ x
    timers("forward-compute").stop()
    # The shim itself must not sync. This explicit sync is only for the smoke check.
    torch.cuda.synchronize()
    totals = timers.drain()
    print(f"fwd_ms={totals['fwd']:.3f} out={tuple(y.shape)}")


if __name__ == "__main__":
    main()
