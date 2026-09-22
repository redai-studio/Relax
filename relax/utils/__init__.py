# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib
import logging
import sys
from typing import Any

from relax.utils.env import Envs


def report_train_start(config: Any) -> None:
    pass


def try_import_telemetry_hook(logger: Any = None) -> None:
    """Import the optional telemetry hook without affecting training."""
    hook = Envs.RELAX_TELEMETRY_HOOK
    if not hook:
        return
    # Skip if the hook (or its parent package) has already been imported by an
    # earlier channel — e.g. a site-packages `.pth` autoload that fires at
    # Python site init, before this entrypoint. Re-importing here would install
    # the same monkey-patches twice and can corrupt Ray's captured actor-method
    # signatures.
    root = hook.split(".", 1)[0]
    if hook in sys.modules or root in sys.modules:
        return
    try:
        importlib.import_module(hook)
    except Exception as e:
        logger = logger or logging.getLogger(__name__)
        logger.warning(f"Telemetry hook {hook!r} failed to import: {type(e).__name__}: {e}")
