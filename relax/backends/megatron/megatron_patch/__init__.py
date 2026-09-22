# Runtime Megatron monkey-patches applied for their import side effects. Mirrors
# the slime ``megatron_utils/megatron_patch`` package: importing this subpackage
# swaps patched implementations into ``megatron.core`` before any training step.
from . import (
    chunked_grad_coalesce_patch,  # noqa: F401
    chunked_mtp_loss_patch,  # noqa: F401
)
