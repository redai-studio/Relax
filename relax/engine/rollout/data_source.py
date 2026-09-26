# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import abc
import os
from pathlib import Path

import torch

from relax.utils.data.data import Dataset
from relax.utils.data.processing_utils import load_processor, load_tokenizer
from relax.utils.logging_utils import get_logger
from relax.utils.misc import load_function
from relax.utils.multimodal.config import MultimodalConfig
from relax.utils.s3_model_loader import prepare_model_maybe_update_args
from relax.utils.types import Sample


logger = get_logger(__name__)


def _shallow_copy_sample(src: Sample) -> Sample:
    """Create a lightweight copy of a Sample that *shares* heavy read-only
    payloads (``multimodal_inputs``) with the source."""
    new = Sample.__new__(Sample)
    new.__dict__.update(src.__dict__)
    # Shallow-copy mutable containers that downstream code mutates in-place.
    new.tokens = list(src.tokens)
    new.rollout_tokens = list(src.rollout_tokens)
    new.weight_versions = list(src.weight_versions)
    new.metadata = dict(src.metadata)
    # Per-sample accumulators — create fresh instances.
    new.spec_info = Sample.SpecInfo()
    new.prefix_cache_info = Sample.PrefixCacheInfo()
    # ``multimodal_inputs`` is read-only downstream — share the reference.
    # ``multimodal_train_inputs`` is *set* (not mutated) per-sample by the
    # processor, so sharing the initial ``None`` is fine.
    return new


def _create_dataset(args, tokenizer, processor, multimodal_config=None):
    """Factory function to create dataset based on configuration.

    If args.use_streaming_dataset is True, uses the memory-efficient StreamingDataset.
    Otherwise, uses the traditional Dataset that loads all data into memory.

    Args:
        args: Arguments containing dataset configuration
        tokenizer: Tokenizer for processing text
        processor: Processor for multimodal inputs
        multimodal_config: Config for processing multimodal data

    Returns:
        Dataset or StreamingDataset instance
    """
    custom_prompt_path = getattr(args, "custom_prompt_path", None)
    custom_prompt_func = load_function(custom_prompt_path) if custom_prompt_path else None

    use_streaming = getattr(args, "use_streaming_dataset", False)

    # === OPSD: resolve teacher-side multimodal_keys ===
    # --opd-teacher-image-key / -video-key / -audio-key are convenience flags
    # that get assembled into the teacher_multimodal_keys dict consumed by
    # build_opd_teacher_sample_fields (which populates sample.teacher_multimodal_inputs).
    teacher_mm_keys = None
    _t_img = getattr(args, "opd_teacher_image_key", None)
    if _t_img is not None:
        teacher_mm_keys = {"image": _t_img}
        logger.info(f"OPSD: teacher_multimodal_keys = {teacher_mm_keys}")

    if use_streaming:
        from relax.utils.data.streaming_dataset import StreamingDataset

        buffer_size = getattr(args, "streaming_buffer_size", 10000)
        prefetch_chunk_size = getattr(args, "prefetch_chunk_size", 32)
        prefetch_max_cached = getattr(args, "prefetch_max_cached", 256)
        prefetch_num_workers = getattr(args, "prefetch_num_workers", 1)

        logger.info(
            f"Using StreamingDataset with buffer_size={buffer_size}, "
            f"prefetch_chunk_size={prefetch_chunk_size}, prefetch_max_cached={prefetch_max_cached}, "
            f"prefetch_num_workers={prefetch_num_workers}"
        )
        return StreamingDataset(
            path=args.prompt_data,
            tokenizer=tokenizer,
            processor=processor,
            max_length=args.rollout_max_prompt_len,
            prompt_key=args.input_key,
            multimodal_keys=args.multimodal_keys,
            label_key=args.label_key,
            tool_key=args.tool_key,
            metadata_key=args.metadata_key,
            system_prompt=args.system_prompt,
            apply_chat_template=args.apply_chat_template,
            apply_chat_template_kwargs=args.apply_chat_template_kwargs,
            use_audio_in_video=args.use_audio_in_video,
            seed=args.rollout_seed,
            buffer_size=buffer_size,
            prefetch_chunk_size=prefetch_chunk_size,
            prefetch_max_cached=prefetch_max_cached,
            prefetch_num_workers=prefetch_num_workers,
            multimodal_config=multimodal_config,
            custom_prompt_func=custom_prompt_func,
            teacher_prompt_key=getattr(args, "opd_teacher_prompt_key", None),
            teacher_multimodal_keys=teacher_mm_keys,
        )
    else:
        logger.info("Using traditional Dataset (eager loading)")
        return Dataset(
            args.prompt_data,
            tokenizer=tokenizer,
            processor=processor,
            max_length=args.rollout_max_prompt_len,
            prompt_key=args.input_key,
            multimodal_keys=args.multimodal_keys,
            label_key=args.label_key,
            metadata_key=args.metadata_key,
            system_prompt=args.system_prompt,
            tool_key=args.tool_key,
            apply_chat_template=args.apply_chat_template,
            apply_chat_template_kwargs=args.apply_chat_template_kwargs,
            use_audio_in_video=args.use_audio_in_video,
            seed=args.rollout_seed,
            multimodal_config=multimodal_config,
            custom_prompt_func=custom_prompt_func,
            teacher_prompt_key=getattr(args, "opd_teacher_prompt_key", None),
            teacher_multimodal_keys=teacher_mm_keys,
        )


class DataSource(abc.ABC):
    @abc.abstractmethod
    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        """Return num_samples samples."""

    @abc.abstractmethod
    def add_samples(self, samples: list[list[Sample]]):
        """Add samples to the data source."""

    @abc.abstractmethod
    def save(self, rollout_id):
        """Save the state of the data source."""

    @abc.abstractmethod
    def load(self, rollout_id=None):
        """Load the state of the data source."""

    @abc.abstractmethod
    def __len__(self) -> int:
        """Length of the data source.

        May change when samples are added/fetched.
        """


# TODO may further refactor data-loading part later
class RolloutDataSource(DataSource):
    def __init__(self, args):
        self.args = args

        self.epoch_id = 0
        self.sample_group_index = 0
        self.sample_index = 0
        self.sample_offset = 0
        # TODO remove this
        self.metadata = {}

        # Check if using streaming dataset
        self._use_streaming = getattr(args, "use_streaming_dataset", False)
        self.dataset = None

        if args.rollout_global_dataset:
            if getattr(args, "hf_checkpoint", None):
                prepare_model_maybe_update_args(args, completeness="metadata")
                tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
                processor = load_processor(args.hf_checkpoint, trust_remote_code=True)
            else:
                # Native generation (diffusion/flow): prompts are strings consumed
                # by the rollout engine; no LLM tokenizer/processor is needed at the
                # data source. Condition media is loaded via multimodal_keys.
                tokenizer = None
                processor = None

            # TODO move (during the refactor)
            if (d := args.dump_details) is not None and tokenizer is not None:
                tokenizer.save_pretrained(Path(d) / "tokenizer")
                if processor:
                    processor.save_pretrained(Path(d) / "processor")

            # Initialize multimodal config from args
            multimodal_config = MultimodalConfig.from_args(args)

            # Use factory function to create dataset
            self.dataset = _create_dataset(args, tokenizer, processor, multimodal_config)

            if self.args.rollout_shuffle:
                self.dataset.shuffle(self.epoch_id)

    def lengths(self):
        return len(self.dataset)

    def get_samples(self, num_samples):
        # TODO further improve code
        if self.dataset is not None:
            if self._use_streaming:
                # Use streaming dataset's get_batch method
                prompt_samples, crossed_epoch = self.dataset.get_batch(num_samples)
                if crossed_epoch:
                    self.epoch_id += 1
                    logger.info(f"Epoch boundary crossed, now at epoch {self.epoch_id}")
            else:
                # Original logic for traditional Dataset
                if self.sample_offset + num_samples <= len(self.dataset):
                    prompt_samples = self.dataset.samples[self.sample_offset : self.sample_offset + num_samples]
                    self.sample_offset += num_samples
                else:
                    prompt_samples = self.dataset.samples[self.sample_offset :]
                    num_samples -= len(prompt_samples)
                    self.epoch_id += 1
                    if self.args.rollout_shuffle:
                        self.dataset.shuffle(self.epoch_id)
                    prompt_samples += self.dataset.samples[:num_samples]
                    self.sample_offset = num_samples
        else:
            prompt_samples = [Sample() for _ in range(num_samples)]

        samples = []
        for prompt_sample in prompt_samples:
            group = []
            for _ in range(self.args.n_samples_per_prompt):
                sample = _shallow_copy_sample(prompt_sample)
                sample.group_index = self.sample_group_index
                sample.index = self.sample_index
                self.sample_index += 1
                group.append(sample)
            self.sample_group_index += 1
            samples.append(group)
        return samples

    def add_samples(self, samples: list[list[Sample]]):
        raise RuntimeError(f"Cannot add samples to {self.__class__.__name__}. This is a read-only data source.")

    def save(self, rollout_id):
        if not self.args.rollout_global_dataset:
            return

        state_dict = {
            "sample_offset": self.sample_offset,
            "epoch_id": self.epoch_id,
            "sample_group_index": self.sample_group_index,
            "sample_index": self.sample_index,
            "metadata": self.metadata,
            "use_streaming": self._use_streaming,
        }

        # Add streaming dataset state if applicable
        if self._use_streaming and self.dataset is not None:
            state_dict["streaming_state"] = self.dataset.get_state()

        path = os.path.join(self.args.save, f"dataset/global_dataset_state_dict_{rollout_id}.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state_dict, path)

    def load(self, rollout_id=None):
        if not self.args.rollout_global_dataset:
            return

        if self.args.load is None:
            return

        if rollout_id < 0:
            return

        path = os.path.join(self.args.load, f"dataset/global_dataset_state_dict_{rollout_id}.pt")
        if not os.path.exists(path):
            # Backwards compat: older checkpoints wrote to rollout/ instead of dataset/.
            legacy_path = os.path.join(self.args.load, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
            if os.path.exists(legacy_path):
                logger.warning(f"Loading dataset state from legacy path {legacy_path} (new path: dataset/)")
                path = legacy_path
            else:
                logger.error(f"Checkpoint {path} does not exist.")
                return

        logger.info(f"load metadata from {path}")
        state_dict = torch.load(path)
        self.sample_offset = state_dict.get("sample_offset", 0)
        self.epoch_id = state_dict.get("epoch_id", 0)
        self.sample_group_index = state_dict.get("sample_group_index", 0)
        self.sample_index = state_dict.get("sample_index", 0)
        self.metadata = state_dict.get("metadata", {})
        logger.info(f"load metadata: {self.metadata}")

        # Load streaming dataset state if applicable
        if self._use_streaming and self.dataset is not None:
            streaming_state = state_dict.get("streaming_state")
            if streaming_state:
                self.dataset.load_state(streaming_state)
                logger.info(
                    f"Loaded streaming dataset state: epoch={streaming_state.get('epoch_id')}, position={streaming_state.get('position')}"
                )
        elif self.args.rollout_global_dataset and self.args.rollout_shuffle:
            self.dataset.shuffle(self.epoch_id)

    def __len__(self) -> int:
        return 0 if self.dataset is None else len(self.dataset)


class RolloutDataSourceWithBuffer(RolloutDataSource):
    def __init__(self, args):
        super().__init__(args)
        self.buffer = []
        if self.args.buffer_filter_path is None:
            self.buffer_filter = pop_first
        else:
            self.buffer_filter = load_function(self.args.buffer_filter_path)

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        """Return exactly num_samples groups, buffer-first then dataset top-
        up."""
        samples = self._get_samples_from_buffer(num_samples)
        num_samples -= len(samples)

        if num_samples == 0:
            return samples

        samples += super().get_samples(num_samples=num_samples)
        return samples

    def _get_samples_from_buffer(self, num_samples: int) -> list[list[Sample]]:
        if len(self.buffer) == 0 or num_samples == 0:
            return []

        samples = self.buffer_filter(self.args, None, self.buffer, num_samples)
        return samples

    def add_samples(self, samples: list[list[Sample]]):
        """Add a sample group to buffer."""
        if not samples:
            return
        assert isinstance(samples, list), f"samples must be a list, got {type(samples)}"
        assert isinstance(samples[0], list), f"the elements of samples must be list, got {type(samples[0])}"
        for i in range(0, len(samples)):
            assert len(samples[i]) == self.args.n_samples_per_prompt, (
                f"the length of the elements of samples must be equal to n_samples_per_prompt, got {len(samples[i])} != {self.args.n_samples_per_prompt}"
            )
            group = samples[i]  # type: ignore
            self.buffer.append(group)

    # TODO remove
    def update_metadata(self, metadata: dict):
        self.metadata.update(metadata)

    # TODO remove
    def get_metadata(self):
        return self.metadata

    def get_buffer_length(self):
        return len(self.buffer)


def pop_first(args, rollout_id, buffer: list[list[Sample]], num_samples: int) -> list[list[Sample]]:
    num_to_pop = min(len(buffer), num_samples)
    samples = buffer[:num_to_pop]
    del buffer[:num_to_pop]
    return samples
