# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Import alias for Megatron images built before the replay manager migration.

Retire this path once those images and their patches are out of support.
"""

import sys

from relax.distributed.replay_manager.backend import megatron


sys.modules[__name__] = megatron
