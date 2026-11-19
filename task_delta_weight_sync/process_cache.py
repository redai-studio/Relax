# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Keep concurrent benchmark compilers in separate shared-filesystem
directories."""

import os
import socket

from triton.runtime.cache import FileCacheManager


class ProcessCacheManager(FileCacheManager):
    def __init__(self, key: str, override: bool = False, dump: bool = False) -> None:
        # Every spawned scheduler resolves its own PID here, even when it
        # inherits the parent engine's environment. Preserve Triton's normal
        # atomic writes, group metadata, and caller-selected cache roots.
        key = f"{socket.gethostname()}-{os.getpid()}/{key}"
        super().__init__(key, override=override, dump=dump)
