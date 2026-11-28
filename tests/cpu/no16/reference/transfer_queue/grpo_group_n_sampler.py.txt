# Copyright 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2025 The TransferQueue Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

from transfer_queue.sampler import BaseSampler


class GRPOGroupNSampler(BaseSampler):
    """Group-based sampler for reinforcement learning and multi-sample generation workflows.

    This sampler implements grouped sampling without replacement, specifically designed
    for scenarios where multiple samples need to be generated from the same input prompt
    or where grouped sampling is required. It ensures that all samples belonging to the
    same prompt are either selected together or not at all, maintaining the integrity
    of prompt groups throughout the training process.

    The sampler is commonly used in GRPO (Group Relative Policy Optimization)
    training scenarios where you need to generate multiple responses from the same
    prompt and train the policy on all of them together.

    The sampler is configured through TransferQueueController and receives parameters
    via the sampling_config in get_meta calls:

    ```python
    # Initialize controller with GRPO sampler
    from transfer_queue import TransferQueueController, GRPOGroupNSampler, AsyncTransferQueueClient

    controller = TransferQueueController.remote(sampler=GRPOGroupNSampler(n_samples_per_prompt=4))
    controller_info = process_zmq_server_info(controller)

    client = AsyncTransferQueueClient(
        client_id="rl_client",
        controller_info=controller_info,
    )

    # Get metadata with grouped sampling configuration
    meta = await client.async_get_meta(
        data_fields=["input_ids", "attention_mask", "generated_text", "reward"],
        batch_size=16,  # Total samples requested
        partition_id="train_0",
        task_name="rl_training",
    )
    # This will return 16 samples organized as 4 groups of 4 samples each
    ```

    Data Organization:
    This sampler assumes the user puts the prompts in consecutive orders, such as
    [prompt1_sample1, prompt1_sample2, prompt2_sample1, prompt2_sample2, ...]
    belong to the same prompt group:
    ```
    ready_indexes = [prompt1_sample1, prompt1_sample2, prompt1_sample3, prompt1_sample4,
                    prompt2_sample1, prompt2_sample2, prompt2_sample3, prompt2_sample4, ...]
    ```
    """

    def __init__(
        self,
        n_samples_per_prompt: int = 1,
    ):
        """Initialize the GRPOGroupNSampler.

        The sampler maintains minimal internal state and relies on runtime
        configuration through the sampling_config parameter.
        Args:
            n_samples_per_prompt: Number of samples per prompt group. Must be > 0.

        """
        super().__init__()

        # Basic validation
        if n_samples_per_prompt <= 0:
            raise ValueError(f"n_samples_per_prompt must be positive, got {n_samples_per_prompt}")
        self.n_samples_per_prompt = n_samples_per_prompt

    def sample(
        self,
        ready_indexes: list[int],
        batch_size: int,
        task_name: str = "",
        partition_id: str = "",
        *args: Any,
        **kwargs: Any,
    ) -> tuple[list[int], list[int]]:
        """Sample groups of indices from the ready indices.

        This method implements group completeness validation and ensures that only complete
        groups are sampled. It returns empty lists if insufficient complete groups are available.

        Args:
            ready_indexes: List of global indices for which all required fields have been
                produced and samples are not labeled as consumed. These should be organized
                such that consecutive indices belong to the same prompt group.
            batch_size: Total number of samples to select. Must be divisible by n_samples_per_prompt.
            task_name: Unique identifier for the training task. Used for state caching and
                tracking consumed samples.
            partition_id: Partition ID for data versioning. Used for state organization.
            *args: Additional positional arguments (ignored in current implementation).
            **kwargs: Additional keyword arguments, key ones are:
                - dp_rank: Data parallel rank for multi-GPU training. Used for state cache organization.
                - batch_index: Current batch index for tracking consumption progress.

        Returns:
            Tuple of (sampled_indexes, consumed_indexes):
            - sampled_indexes: List of selected global indices, length = batch_size or empty if
              insufficient complete groups are available.
            - consumed_indexes: List of indices to mark as consumed, identical to sampled_indexes
              (without replacement semantics).

        Raises:
            ValueError: batch_size is not divisible by n_samples_per_prompt.

        Examples:
            >>> sampler = GRPOGroupNSampler(n_samples_per_prompt=3)
            >>> ready_indexes = [0, 1, 3, 4, 6, 7]  # No complete groups after sorting
            >>> sampled, consumed = sampler.sample(ready_indexes, 6)
            >>> sampled
            []
            >>> consumed
            []

            >>> ready_indexes = [0, 1, 3, 4, 5, 6, 7, 9, 10, 11]  # Has complete groups after sorting
            >>> sampled, consumed = sampler.sample(ready_indexes, 6)
            >>> sampled
            [3, 4, 5, 9, 10, 11]
            >>> consumed
            [3, 4, 5, 9, 10, 11]
        """
        states = self._states.get(partition_id, {}).get(task_name, {})
        dp_rank = kwargs.get("dp_rank", None)
        batch_index = kwargs.get("batch_index", None)

        # Return cached result if available
        if dp_rank in states.keys() and batch_index in states[dp_rank].keys():
            return states[dp_rank][batch_index]

        if batch_size % self.n_samples_per_prompt != 0:
            raise ValueError(
                f"batch_size ({batch_size}) must be a multiple of n_samples_per_prompt ({self.n_samples_per_prompt})"
            )

        required_groups = batch_size // self.n_samples_per_prompt
        sorted_ready_indexes = sorted(ready_indexes)

        complete_group_indices = []
        found_groups = 0

        # Scan for consecutive groups
        i = 0
        while i <= len(sorted_ready_indexes) - self.n_samples_per_prompt and found_groups < required_groups:
            potential_group = sorted_ready_indexes[i : i + self.n_samples_per_prompt]
            # Check if this forms a complete group (consecutive indices)
            is_consecutive = all(
                potential_group[j + 1] - potential_group[j] == 1 for j in range(len(potential_group) - 1)
            )
            if is_consecutive:
                complete_group_indices.extend(potential_group)
                found_groups += 1
                i += self.n_samples_per_prompt
            else:
                i += 1

        # Return empty if insufficient complete groups
        if found_groups < required_groups:
            return [], []

        sampled_indexes = complete_group_indices
        consumed_indexes = sampled_indexes.copy()

        # Cache the sampling result for deterministic future calls
        if dp_rank is not None:
            if dp_rank not in states:
                states[dp_rank] = {}
                states[dp_rank][batch_index] = (sampled_indexes, consumed_indexes)
            elif batch_index not in states[dp_rank]:
                states[dp_rank][batch_index] = (sampled_indexes, consumed_indexes)
            if partition_id not in self._states:
                self._states[partition_id] = {}
            self._states[partition_id][task_name] = states
        return sampled_indexes, consumed_indexes
