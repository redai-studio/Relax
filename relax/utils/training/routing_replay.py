# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Compatibility entry for existing external model patches."""

import sys

from relax.distributed.replay_manager.backend import megatron as moe


sys.modules[__name__] = moe
