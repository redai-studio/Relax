# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Order complete SFT microbatches using CPU supervision counts within equal
padded shapes."""

from functools import wraps

import torch
import torch.distributed as dist


# Separate the CPU schedule from other P2P traffic on the shared process group.
_SCHEDULE_SIZE_TAG = 0x5150
_SCHEDULE_DATA_TAG = 0x5151


def permutation(batches, lengths, counts, dummy, pad, chunks):
    if sum(chunks) != len(batches) or len(lengths) != len(counts):
        raise ValueError("SFT counts and microbatch schedule disagree")
    order = []
    offset = 0
    for k in chunks:

        def key(i):
            rows = sum(lengths[j] for j in batches[i])
            supervised = sum(counts[j] for j in batches[i]) if i not in dummy else 0
            return (-((rows + pad - 1) // pad), -supervised, i)

        order.extend(sorted(range(offset, offset + k), key=key))
        offset += k
    return order


def apply_order(iterators, order):
    for iterator in iterators:
        original = iterator.micro_batch_indices
        dummy = iterator.dummy_micro_batch_offsets
        if iterator.offset != 0 or sorted(order) != list(range(len(original))):
            raise ValueError("Invalid initial microbatch order")
        iterator.micro_batch_indices = [original[i] for i in order]
        iterator.dummy_micro_batch_offsets = {j for j, i in enumerate(order) if i in dummy}


def install(args):
    from megatron.core import parallel_state as ps

    from relax.backends.megatron import actor, data
    from relax.utils.distributed_utils import get_gloo_group
    from relax.utils.logging_utils import get_logger

    if (
        args.loss_type != "sft"
        or args.context_parallel_size != 1
        or getattr(args, "virtual_pipeline_model_parallel_size", None) not in (None, 1)
    ):
        raise RuntimeError("Supervision wave ordering requires SFT, CP1 and native pipeline stages")
    native = data.get_data_iterator
    if actor.get_data_iterator is not native:
        raise RuntimeError("Unexpected SFT iterator owner")
    # Use the train actors' existing Gloo connections for the small CPU plan.
    # Different DP replicas may have different plans and call this at different times.
    group = ps.get_model_parallel_group()
    members = dist.get_process_group_ranks(group)
    cpu_group = get_gloo_group()

    @wraps(native)
    def build(*args_, **kwargs):
        iterators, chunks = native(*args_, **kwargs)
        iterator = iterators[0]
        batches = iterator.micro_batch_indices
        if batches is None:
            return iterators, chunks
        size = torch.tensor([len(batches) + 1], dtype=torch.int64, device="cpu")
        if group.rank() == 0:
            lengths = iterator.rollout_data["total_lengths"] if batches else []
            masks = iterator.rollout_data.get("selective_loss_masks_cpu") if batches else None
            active = (
                masks is not None
                and len(masks) == len(lengths)
                and all(isinstance(m, torch.Tensor) and m.device.type == "cpu" for m in masks)
            )
            if active:
                counts = [int(torch.count_nonzero(m)) for m in masks]
                order = permutation(
                    batches,
                    lengths,
                    counts,
                    iterator.dummy_micro_batch_offsets,
                    args.tensor_model_parallel_size * args.data_pad_size_multiplier,
                    chunks,
                )
            else:
                order = list(range(len(batches)))
            plan = torch.tensor([int(active), *order], dtype=torch.int64, device="cpu")
            for peer in members[1:]:
                dist.send(size, dst=peer, group=cpu_group, tag=_SCHEDULE_SIZE_TAG)
                dist.send(plan, dst=peer, group=cpu_group, tag=_SCHEDULE_DATA_TAG)
        else:
            dist.recv(size, src=members[0], group=cpu_group, tag=_SCHEDULE_SIZE_TAG)
            plan = torch.empty(int(size[0]), device="cpu", dtype=torch.int64)
            dist.recv(plan, src=members[0], group=cpu_group, tag=_SCHEDULE_DATA_TAG)
            if plan.numel() != len(batches) + 1:
                raise ValueError("Model-parallel ranks have inconsistent microbatch counts")
        active, *order = plan.tolist()
        if active:
            apply_order(iterators, order)
            if not build.reported:
                get_logger(__name__).info(
                    "Ordered %d native SFT microbatches by physical size and supervised tokens on the model-parallel group",
                    len(order),
                )
                build.reported = True
        return iterators, chunks

    build.reported = False
    actor.get_data_iterator = data.get_data_iterator = build
