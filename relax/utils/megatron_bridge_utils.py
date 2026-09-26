import contextvars
import dataclasses
from contextlib import contextmanager

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


try:
    from megatron.core.utils import unwrap_model
except ImportError:
    unwrap_model = None


def _patch_progress_tracking_show_elapsed():
    """Replace ``MegatronModelBridge._with_progress_tracking`` to show elapsed
    time instead of remaining time.

    Upstream uses ``TimeRemainingColumn`` which counts down to 00:00 and erases
    itself the moment the bar finishes, so the final wall-clock cost of the
    conversion is invisible. We swap in ``TimeElapsedColumn`` so the rendered
    duration stays on screen after completion.
    """
    try:
        import torch
        from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
        from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
    except ImportError:
        return

    def _with_progress_tracking(self, tasks, description: str, show_progress: bool = True):
        is_main_rank = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        if not show_progress:
            yield from tasks
            return

        bridge_name = self.__class__.__name__
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TextColumn("({task.completed}/{task.total})"),
            TextColumn("{task.fields[bridge]}"),
            disable=not is_main_rank,
        ) as progress:
            task_id = progress.add_task(description, total=len(tasks), bridge=bridge_name)
            for task in tasks:
                yield task
                progress.update(task_id, advance=1)

    MegatronModelBridge._with_progress_tracking = _with_progress_tracking


_patch_progress_tracking_show_elapsed()


@contextmanager
def patch_megatron_model(model):
    unwrapped_model = unwrap_model(model)[0]
    model_config = unwrapped_model.config
    attribute_was_added = False
    if not hasattr(model_config, "share_embeddings_and_output_weights"):
        model_config.share_embeddings_and_output_weights = unwrapped_model.share_embeddings_and_output_weights
        attribute_was_added = True

    try:
        yield
    finally:
        if attribute_was_added:
            delattr(model_config, "share_embeddings_and_output_weights")


_adapter_splice_weights: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "relax_adapter_splice_weights", default=None
)
_adapter_splice_patched = False


def _splice_task_weight(task, backup):
    """Return ``task`` with ``param_weight`` swapped for a fresh backup copy.

    Left unchanged when there is no local weight (non-owning rank) or no backup
    entry (the live, paused weight stays in place).
    """
    if task is None or task.param_weight is None:
        return task

    key = f"vp_stages.{task.vp_stage}.{task.param_name}"
    fresh = backup.get(key)
    if fresh is None:
        logger.warning("[adapter-splice] no backup entry for %s; leaving live (paused) weight in place", key)
        return task

    from relax.utils.device import make_current_torch_device

    return dataclasses.replace(task, param_weight=fresh.to(make_current_torch_device()))


def _make_adapter_splice_wrapper(orig_fn):
    def wrapped(self, megatron_model):
        tasks_by_base = orig_fn(self, megatron_model)

        backup = _adapter_splice_weights.get()
        if backup is None:
            # Not inside an adapter-splice context (e.g. standard export); leave
            # the bridge's behavior untouched.
            return tasks_by_base

        for adapter_tasks in tasks_by_base.values():
            for i, adapter_task in enumerate(adapter_tasks):
                in_task = _splice_task_weight(adapter_task.linear_in_task, backup)
                out_task = _splice_task_weight(adapter_task.linear_out_task, backup)
                adapter_tasks[i] = dataclasses.replace(
                    adapter_task,
                    linear_in_task=in_task,
                    linear_out_task=out_task,
                )

        return tasks_by_base

    return wrapped


def _ensure_adapter_splice_patched():
    global _adapter_splice_patched
    if _adapter_splice_patched:
        return
    from megatron.bridge.models.conversion.peft_bridge import MegatronPeftBridge

    MegatronPeftBridge.build_adapter_conversion_tasks = _make_adapter_splice_wrapper(
        MegatronPeftBridge.build_adapter_conversion_tasks
    )
    _adapter_splice_patched = True


@contextmanager
def splice_adapter_weights(backup_weights):
    """Make ``build_adapter_conversion_tasks`` splice fresh backup copies into
    the adapter tasks' ``param_weight`` for the duration of the block.

    Args:
        backup_weights: dict keyed by ``vp_stages.{vp}.{megatron_param_name}``
            holding fresh (non-paused) weight tensors — the same dict the base
            conversion-task splice uses.
    """
    _ensure_adapter_splice_patched()
    token = _adapter_splice_weights.set(backup_weights)
    try:
        yield
    finally:
        _adapter_splice_weights.reset(token)
