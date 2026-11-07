# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Pinned legacy serializer used to replay the actual checkpoint patch in
behavioral regressions."""

from __future__ import annotations

from typing import Optional

import torch
from megatron.core.dist_checkpointing.mapping import (
    LocalNonpersistentObject,
    ShardedObject,
    ShardedStateDict,
    ShardedTensor,
)


# fmt: off
def sharded_param_state_dp_reshardable(
        self,
        model_sharded_state_dict: ShardedStateDict,
        is_loading: bool = False,
        metadata: Optional[dict] = None,
    ):
        """Sharded state dict where each noncontiguous buffer is a separate
        ShardedTensor.

        Results in fully parallel save and load without any inter-process
        communication or intermediate buffers/copies.

        Stores optimizer state in the format that corresponds to the internal Distributed
        Optimizer format, i.e. in buckets. Each buckets consists of state parameters and
        potentially some padding:
        - intra-param padding
        - param 1
        - intra-param padding
        - param 2
        - intra-param padding
        - param ...
        - intra-param padding
        - param N
        - intra-param padding
        - bucket padding to some DP multiple

        Different buckets are assigned a different ShardedTensor key. Within each bucket,
        each param and each padding above is represented with a different ShardedTensor object
        sharing the same key (so, corresponding to the same tensor in the checkpoint).

        For checkpointing, we include the intra-param padding for correctness
        but we must discard the last padding to DP multiple, because that might
        change during DP resharding - we want the checkpoint tensor to always have size
        `gbuf_world_numel_unpadded` which means everything except for the last padding above.
        """
        data_parallel_rank = self.data_parallel_group.rank()
        data_parallel_world_size = self.data_parallel_group.size()

        state = self.get_parameter_state_dp_reshardable()
        # per_bucket_numel metadata is saved separately for each TPxPP domain.
        for per_bucket_key in ('per_bucket_numel', 'per_bucket_numel_unpadded'):
            key = (
                f'optimizer.distributed.dp_group_idx_{self.data_parallel_group_idx}'
                f'.{per_bucket_key}'
            )
            state[per_bucket_key] = ShardedObject(
                key,
                state[per_bucket_key],
                (1,),
                (0,),
                replica_id=(self.distributed_optimizer_instance_id, 0, data_parallel_rank),
            )

        for gbuf_idx, gbuf_range_maps in enumerate(self.gbuf_ranges):
            for dtype, gbuf_range_map_for_all_buckets in state[gbuf_idx].items():
                for bucket_idx, bucket_state in enumerate(gbuf_range_map_for_all_buckets):
                    # Compute local DP contiguous shard's size.
                    gbuf_world_numel_unpadded = (
                        self.buffers[gbuf_idx].buckets[bucket_idx].numel_unpadded
                    )
                    gbuf_world_numel = self.buffers[gbuf_idx].buckets[bucket_idx].grad_data.numel()
                    assert gbuf_world_numel_unpadded <= gbuf_world_numel
                    assert gbuf_world_numel % data_parallel_world_size == 0
                    gbuf_local_numel = gbuf_world_numel // data_parallel_world_size

                    sharded_bucket_key = (
                        f'optimizer.distributed.dp_group_idx_{self.data_parallel_group_idx}'
                        f'.gbuf_idx_{gbuf_idx}.dtype_{dtype}.bucket_idx_{bucket_idx}'
                    )

                    # The global ckpt tensors must be fully covered.
                    # We add extra empty padding if necessary
                    assert bucket_state, 'empty bucket encountered'

                    # Insert padding between parameter tensors to ensure full coverage as needed.
                    all_pad_tensors = {}
                    for i in range(-1, len(bucket_state)):
                        if i == len(bucket_state) - 1:
                            # Potential padding at the end
                            next_param_start = gbuf_local_numel
                        else:
                            next_param_start = bucket_state[i + 1]['gbuf_local_start']
                        if i == -1:
                            # Potential padding at the front
                            cur_param_end = 0
                        else:
                            cur_param_end = bucket_state[i]['gbuf_local_end']
                        world_param_end = data_parallel_rank * gbuf_local_numel + cur_param_end
                        # Insert padding if there is a gap between next param,
                        # but not exceeding unpadded gbuf size
                        if (
                            next_param_start != cur_param_end
                            and world_param_end < gbuf_world_numel_unpadded
                        ):
                            pad_tensors = {
                                k: torch.empty(
                                    next_param_start - cur_param_end, dtype=v.dtype, device=v.device
                                )
                                for k, v in bucket_state[i].items()
                                if isinstance(v, torch.Tensor)
                            }
                            all_pad_tensors[i + 1] = {
                                **pad_tensors,
                                'gbuf_local_start': cur_param_end,
                                'gbuf_local_end': next_param_start,
                                'padding': True,
                            }

                    # Insert from end so that insertion positions are still correct.
                    indices_to_insert = sorted(list(all_pad_tensors.keys()))  # noqa: C414 - preserve the pinned source for patch replay
                    for index_to_insert in reversed(indices_to_insert):
                        bucket_state.insert(index_to_insert, all_pad_tensors[index_to_insert])

                    # Each tensor is mapped to a slice
                    # of a DP-local shard of size `gbuf_local_numel`.
                    for bucket_params_idx in range(len(bucket_state)):
                        tensors = bucket_state[bucket_params_idx]
                        gbuf_local_start = tensors.pop('gbuf_local_start')
                        gbuf_local_end = tensors.pop('gbuf_local_end')
                        if 'padding' not in tensors:
                            tensors['padding'] = False

                        for key in tensors:
                            if key == 'padding':
                                tensors[key] = LocalNonpersistentObject(tensors[key])
                                continue
                            if key == 'step':
                                # The optimizer state of STEP is a 0-dim tensor and is handled
                                # separately via param_groups, not as part of the gradient buffer.
                                tensors[key] = LocalNonpersistentObject(tensors[key])
                                continue
                            if key == 'step':
                                continue
                            assert tensors[key].shape == (gbuf_local_end - gbuf_local_start,), (
                                tensors[key].shape,
                                gbuf_local_start,
                                gbuf_local_end,
                            )

                            tensors[key] = ShardedTensor(
                                f'{sharded_bucket_key}.{key}',
                                tensors[key],
                                tensors[key].dtype,
                                tensors[key].shape,
                                (gbuf_world_numel_unpadded,),
                                (data_parallel_rank * gbuf_local_numel + gbuf_local_start,),
                                axis_fragmentations=None,
                                flattened_range=None,
                                allow_shape_mismatch=False,
                                replica_id=(self.distributed_optimizer_instance_id, 0, 0),
                            )
        return state
