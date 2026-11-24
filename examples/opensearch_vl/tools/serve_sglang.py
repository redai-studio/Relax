# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Launch SGLang with rendezvous ports outside the kernel ephemeral range."""

import random
import runpy
import socket
from pathlib import Path


def get_control_port() -> int:
    ephemeral_start = int(Path("/proc/sys/net/ipv4/ip_local_port_range").read_text().split()[0])
    for port in random.sample(range(1024, ephemeral_start), 64):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            try:
                listener.bind(("0.0.0.0", port))
            except OSError:
                continue
            return port
    raise RuntimeError("No free SGLang control port below the kernel ephemeral range")


if __name__ in {"__main__", "__mp_main__"}:
    import sglang.srt.server_args as server_args

    # DP workers bind their rendezvous sockets later. bind(0) ports can be
    # claimed by outgoing connections during that gap. Patch spawned workers too.
    server_args.get_free_port = get_control_port
    if __name__ == "__main__":
        runpy.run_module("sglang.launch_server", run_name="__main__")
