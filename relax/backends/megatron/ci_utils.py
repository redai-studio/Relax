"""CI utilities for Megatron backend testing."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from megatron.core.distributed import DistributedDataParallel as DDP

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


@dataclass
class CriticValueHeadUpdateSnapshot:
    params: list[tuple[str, torch.nn.Parameter, torch.Tensor]]
    has_nonzero_grad: bool


def capture_critic_value_head_update(model: Sequence[DDP]) -> CriticValueHeadUpdateSnapshot:
    snapshots = []
    has_nonzero_grad = False
    for model_chunk in model:
        for name, param in model_chunk.named_parameters():
            if not name.endswith(("output_layer.weight", "output_layer.bias")):
                continue
            if param.ndim == 0 or param.shape[0] != 1:
                continue

            grad = getattr(param, "main_grad", None)
            if grad is None:
                grad = param.grad
            assert grad is not None, f"critic value head parameter {name} has no gradient"
            assert bool(torch.isfinite(grad).all()), f"critic value head parameter {name} has non-finite gradient"
            has_nonzero_grad = has_nonzero_grad or bool(torch.count_nonzero(grad))
            snapshots.append((name, param, param.detach().clone()))
    return CriticValueHeadUpdateSnapshot(params=snapshots, has_nonzero_grad=has_nonzero_grad)


def assert_critic_value_head_updated(
    snapshot: CriticValueHeadUpdateSnapshot,
    *,
    update_successful: bool,
    learning_rates: Sequence[float],
) -> None:
    if not snapshot.params or not update_successful or not snapshot.has_nonzero_grad:
        return
    if not any(learning_rate > 0 for learning_rate in learning_rates):
        return
    assert any(not torch.equal(param.detach(), before) for _name, param, before in snapshot.params), (
        "critic value head did not change after a successful optimizer step with non-zero gradient"
    )


def check_mtp_only_grad(model: Sequence[DDP], step_id: int, require_non_mtp_zero: bool = True) -> None:
    """Check MTP gradients and optionally require only MTP parameters to have
    gradients.

    This is used for CI testing to verify that when all outputs are truncated,
    only the MTP layers receive gradients (since only mtp_loss contributes). For
    normal SFT batches, non-MTP parameters are expected to receive main-loss gradients.

    Args:
        model: Sequence of DDP-wrapped model chunks.
        step_id: Current step index for logging.
        require_non_mtp_zero: Whether non-MTP gradients should be rejected.

    Raises:
        AssertionError: If any non-MTP parameter has a non-zero gradient.
    """
    non_mtp_nonzero_grads = []
    mtp_nonzero_grads = []
    mtp_param_count = 0

    for model_chunk in model:
        for name, param in model_chunk.named_parameters():
            # Get the main_grad from the distributed optimizer if available
            grad = getattr(param, "main_grad", None)
            if grad is None:
                grad = param.grad
            if grad is None:
                continue

            grad_norm = grad.abs().max().item()
            is_mtp = ".mtp." in name

            if is_mtp:
                mtp_param_count += 1
                if grad_norm > 0:
                    mtp_nonzero_grads.append((name, grad_norm))
            else:
                if grad_norm > 0:
                    non_mtp_nonzero_grads.append((name, grad_norm))

    # Log the results
    logger.info(
        f"[CI MTP Grad Check] Step {step_id}: "
        f"MTP params on this rank: {mtp_param_count}, "
        f"MTP params with non-zero grad: {len(mtp_nonzero_grads)}, "
        f"non-MTP params with non-zero grad: {len(non_mtp_nonzero_grads)}, "
        f"require_non_mtp_zero={require_non_mtp_zero}"
    )

    if require_non_mtp_zero and non_mtp_nonzero_grads:
        # Log the first few non-MTP params with non-zero gradients for debugging
        for name, grad_norm in non_mtp_nonzero_grads[:5]:
            logger.error(f"[CI MTP Grad Check] Non-MTP param with non-zero grad: {name}, max_grad={grad_norm}")

        assert len(non_mtp_nonzero_grads) == 0, (
            f"Expected all non-MTP parameters to have zero gradients, "
            f"but found {len(non_mtp_nonzero_grads)} with non-zero gradients. "
            f"First few: {non_mtp_nonzero_grads[:5]}"
        )

    if mtp_param_count == 0:
        logger.info(f"[CI MTP Grad Check] Step {step_id}: no local MTP parameters; skipping MTP grad assertion")
        return

    # Also verify that MTP params do have gradients (otherwise the test is not valid)
    assert len(mtp_nonzero_grads) > 0, (
        "Expected MTP parameters to have non-zero gradients, but all were zero. "
        "This may indicate the MTP loss is not being computed."
    )


def check_mtp_loss(mtp_loss: float, max_mtp_loss: float = 1.0) -> None:
    """Check that MTP loss is within expected bounds.

    Args:
        mtp_loss: The computed MTP loss value.
        max_mtp_loss: Maximum allowed MTP loss (default: 1.0).

    Raises:
        AssertionError: If MTP loss exceeds the maximum allowed value.
    """
    assert mtp_loss < max_mtp_loss, (
        f"MTP loss {mtp_loss} exceeds maximum allowed value {max_mtp_loss}. "
        "This may indicate an issue with MTP training."
    )
