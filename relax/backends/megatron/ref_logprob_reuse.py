# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace


class WeightSnapshotVersions:
    """Track known snapshot provenance without comparing or hashing model
    tensors.

    An independent checkpoint load always gets a new identity, even when its
    path matches another load. Only backup/restore/copy operations establish
    equality. Unknown snapshots never compare equal, including two missing
    tags.
    """

    def __init__(self) -> None:
        self._next_version = 0
        self._current: int | None = None
        self._tags: dict[str, int] = {}

    def mark_modified(self) -> None:
        """Invalidate the live model identity after training or checkpoint
        loading."""
        self._next_version += 1
        self._current = self._next_version

    def backup(self, tag: str) -> None:
        if self._current is None:
            self._tags.pop(tag, None)
        else:
            self._tags[tag] = self._current

    def restore(self, tag: str) -> None:
        self._current = self._tags.get(tag)

    def copy(self, *, src_tag: str, dst_tag: str) -> None:
        version = self._tags.get(src_tag)
        if version is None:
            self._tags.pop(dst_tag, None)
        else:
            self._tags[dst_tag] = version

    def same_snapshot(self, first_tag: str, second_tag: str) -> bool:
        version = self._tags.get(first_tag)
        return version is not None and version == self._tags.get(second_tag)

    def current_matches(self, tag: str) -> bool:
        return self._current is not None and self._current == self._tags.get(tag)


def can_reuse_ref_log_probs(args: Namespace, versions: WeightSnapshotVersions) -> bool:
    """Reuse only an equivalent training-engine forward, never rollout
    probabilities."""
    actor_tag = "old_actor" if getattr(args, "keep_old_actor", False) else "actor"
    if not versions.same_snapshot("ref", actor_tag):
        return False
    if getattr(args, "use_rollout_logprobs", False) and not getattr(args, "get_mismatch_metrics", False):
        return False
    # These paths can change forward semantics or require their own forward side effects.
    if any(
        getattr(args, name, False)
        for name in (
            "use_routing_replay",
            "use_rollout_routing_replay",
            "custom_megatron_before_log_prob_hook_path",
            "custom_megatron_before_train_step_hook_path",
            "use_opd",
        )
    ):
        return False
    # Keep the same microbatch shapes as the reference forward, including numerical
    # behavior of multimodal packing and attention kernels.
    if getattr(args, "use_dynamic_batch_size", False):
        if args.log_probs_max_tokens_per_gpu != args.max_tokens_per_gpu:
            return False
    return True
