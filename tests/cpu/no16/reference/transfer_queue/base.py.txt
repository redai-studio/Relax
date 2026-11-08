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

from abc import ABC, abstractmethod
from typing import Any


class BaseSampler(ABC):
    """Base class for samplers that control how data is consumed from TransferQueue.

    A sampler defines the logic for selecting which samples to retrieve from the
    available samples, and which should be labeled as consumed (will never be retrieved in the future).
    Based on this abstraction, users can implement various data consumption strategies
    for different training scenarios, such as sequential sampling, grouped sampling for
    reinforcement learning, or custom sampling patterns.

    The sampler interface provides a clean separation between data production status
    (handled by TransferQueueController) and data consumption strategy (implemented by samplers).
    This allows users to customize data consumption behavior without modifying the TransferQueue codes.

    Available Samplers:
    - **SequentialSampler**: Default sampler, selects samples sequentially without replacement
    - **GRPOGroupNSampler**: A sampler that performs sampling on continuous N samples only when all of them are ready.
                            It assumes the N samples associated with the same prompt are stored contiguously
    - **RankAwareSampler**: Rank-aware sampling for distributed training where each rank retrieves data independently.
                            This sampler will guarantee ranks of the same DP group consume identical samples.

    NOTE: Always return both sampled and consumed indexes (may be identical).
    """

    def __init__(self):
        self._states: dict[Any, Any] = {}

    @abstractmethod
    def sample(
        self,
        ready_indexes: list[int],
        batch_size: int,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[list[int], list[int]]:
        """Sample a batch of indices from the ready indices.

        Args:
            ready_indexes: List of global indices for which all required fields of the
            corresponding samples have been produced, and the samples are not labeled as
            consumed in the corresponding task.
            batch_size: Number of samples to select
            *args: Additional positional arguments for specific sampler implementations
            **kwargs: Additional keyword arguments for specific sampler implementations

        Returns:
            List of sampled global indices of length batch_size

            List of global indices of length batch_size that should be labeled as consumed
            (will never be retrieved in the future)

        Raises:
            ValueError: If batch_size is invalid or ready_indexes is insufficient
        """
        raise NotImplementedError("Subclasses must implement sample")

    def __call__(self, *args: Any, **kwargs: Any) -> tuple[list[int], list[int]]:
        return self.sample(*args, **kwargs)

    def has_cached_result(
        self,
        partition_id: str,
        task_name: str,
        sampling_config: dict[str, Any] | None = None,
    ) -> bool:
        """Check whether the sampler has a cached sampling result for the given context.

        This is used by the controller in polling mode to determine if a previously
        computed sampling result is already available, so that it can skip waiting
        for more data and directly proceed to return the cached result.

        The check is based on the ``_states`` cache structure:
        ``_states[partition_id][task_name][dp_rank][batch_index]``.

        Args:
            partition_id: The partition identifier.
            task_name: The consumer task name.
            sampling_config: Optional sampling configuration dict that may contain
                ``dp_rank`` and ``batch_index`` keys used to locate the cached result.

        Returns:
            True if a cached result exists for the specified
            ``(partition_id, task_name, dp_rank, batch_index)`` combination,
            False otherwise. Also returns False if ``dp_rank`` is not provided
            in ``sampling_config``.
        """
        sampling_config = sampling_config or {}
        dp_rank = sampling_config.get("dp_rank", None)
        batch_index = sampling_config.get("batch_index", None)

        if dp_rank is None:
            return False

        states = self._states.get(partition_id, {}).get(task_name, {})
        return dp_rank in states and batch_index in states[dp_rank]

    def clear_cache(self, partition_id: str):
        """Clear cached states.

        This method removes any cached sampling results that include the specified
        global indexes, ensuring that future sampling operations do not reference
        stale data.

        Args:
            partition_id: The partition ID associated with the task.
        """
        if partition_id in self._states.keys():
            self._states.pop(partition_id)

    def save_checkpoint(self) -> dict:
        """Return the sampler's serializable state for checkpointing.

        By default this returns ``_states``, the shared cache used by all
        samplers. Subclasses that maintain additional runtime state (e.g.
        a separate balanced cache) should override this method and include
        their extra fields.

        Returns:
            A dict that can be passed to ``restore_state`` to recreate the
            current sampling state.
        """
        return {"_states": self._states}

    def load_checkpoint(self, state: dict) -> None:
        """Restore the sampler's state from a checkpoint.

        By default this restores ``_states``. Subclasses that override
        ``save_checkpoint`` must also override this method to restore their extra
        fields.

        Args:
            state: A dict previously returned by ``save_checkpoint``.
        """
        self._states = state["_states"]
