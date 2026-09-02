# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""SFT data producer component.

Pulls samples from `SFTStreamingDataset`, renders with chat template (lazily),
and pushes a per-sample batch (RolloutBatch shape) to TransferQueue at
`partition_id=f"sft_{step}"`. Sequence packing for the trainer happens online
inside `backends/megatron/data.py:get_batch` — the producer only filters
oversized samples and emits per-sample lists.

When ``--eval-interval`` is set, the producer also pushes an eval batch on
every step where ``(step + 1) % eval_interval == 0``. The eval set is split
into ``ceil(n_eval / global_batch_size)`` chunks and pushed serially under
``partition_id=f"sft_eval_{step}_n{N}_{i}"`` (with backpressure between
chunks) so each partition fits within TQ's per-step storage cap. The
Megatron actor parses ``N`` from the partition name to know how many chunks
to consume. The eval source is one of:

- ``--eval-prompt-data NAME PATH`` — load a separate prompt-data dataset.
- ``--eval-size N`` — deterministically shuffle row IDs once and carve out an
  eval subset (``N<1`` is a fraction, ``N>=1`` an absolute count); the held-out
  rows are excluded from every shuffled training epoch.

Mirrors `relax/components/advantages.py` in shape: no FastAPI ingress, plain
`@serve.deployment` + async `run()` loop.
"""

import asyncio
import random
from typing import Any

import torch.nn.functional as F
import transfer_queue as tq
from ray import serve
from transformers import AutoConfig, AutoTokenizer

from relax.components.base import Base
from relax.engine.sft.dataset.streaming import ProcessedSample, SFTStreamingDataset, pack_samples_for_tq
from relax.engine.sft.debug_print import print_first_sample
from relax.engine.sft.runtime import resolve_sft_split_indices
from relax.utils.data.processor_pool import ProcessorPool
from relax.utils.misc import load_function
from relax.utils.s3_model_loader import prepare_model_maybe_update_args
from relax.utils.training.eval_config import build_named_prompt_data_configs
from relax.utils.utils import dict_to_tensordict


_PAD_TOKEN_ID_KEYS = ("image_token_id", "video_token_id", "audio_token_id")


def _load_custom_dataset_class(path: str | None) -> type | None:
    if path is None:
        return None
    cls = load_function(path)
    if not hasattr(cls, "from_args"):
        raise TypeError(f"--custom-dataset-class {path!r} must point to a class with from_args(...).")
    return cls


def _resolve_pad_token_ids_from_config(model_path: str) -> frozenset[int]:
    """Pull the model's multimodal pad-token ids from its ``config.json`` —
    these are the tokens the HF processor expands into per-image / per-video /
    per-audio runs (and that the model itself uses for ``image_mask =
    (input_ids == self.image_token_id)``).

    Returns an empty set for text-only models or when the keys are absent.
    """
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    ids: list[int] = []
    for key in _PAD_TOKEN_ID_KEYS:
        v = getattr(cfg, key, None)
        if isinstance(v, int) and v >= 0:
            ids.append(v)
    return frozenset(ids)


def _resolve_classification_sentinel_token_id(tokenizer) -> int:
    for attr in ("eos_token_id", "pad_token_id"):
        token_id = getattr(tokenizer, attr, None)
        if isinstance(token_id, int) and token_id >= 0:
            return token_id
    raise ValueError("--task-type seq_cls requires a tokenizer with a valid EOS or PAD token id.")


@serve.deployment
class SFT(Base):
    def __init__(self, healthy, pgs, num_gpus, config, role, runtime_env=None):  # noqa: ARG002
        super().__init__()
        self.config = config
        self.role = role
        self.healthy = healthy
        self.step = getattr(config, "start_rollout_id", 0)

        tq.init(self.config.tq_config)
        self.data_system_client = tq.get_client()

        self._dataset: Any | None = None
        self._eval_dataset: Any | None = None
        self._eval_indices: tuple[int, ...] | None = None
        self._train_size: int = 0
        self._tokenizer = None
        self._processor_pool: ProcessorPool | None = None
        self._stop_event = asyncio.Event()
        self._run_task: asyncio.Task | None = None

    def _init_data_pipeline(self) -> None:
        if self._dataset is not None:
            return
        prepare_model_maybe_update_args(self.config, completeness="metadata")
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.hf_checkpoint, trust_remote_code=True)
        try:
            self._processor_pool = ProcessorPool(self.config.hf_checkpoint, pool_size=None, trust_remote_code=True)
        except Exception as exc:
            self._logger.warning(f"Could not init ProcessorPool ({exc}); multimodal samples will fail at push.")
            self._processor_pool = None
        pad_token_ids = _resolve_pad_token_ids_from_config(self.config.hf_checkpoint)
        self._logger.info(f"Resolved multimodal pad token ids from model config: {sorted(pad_token_ids)}")

        cp_size = max(1, getattr(self.config, "context_parallel_size", 1) or 1)
        capacity = self.config.max_tokens_per_gpu * cp_size
        prefetch_buffer_size = getattr(self.config, "sft_prefetch_buffer_size", 256)
        prefetch_chunk_size = getattr(self.config, "sft_prefetch_chunk_size", 32)
        prefetch_num_workers = getattr(self.config, "sft_prefetch_num_workers", 4)
        seed = getattr(self.config, "seed", 42)
        task_type = getattr(self.config, "task_type", "causal_lm")
        classification_sentinel_token_id = (
            _resolve_classification_sentinel_token_id(self._tokenizer) if task_type == "seq_cls" else None
        )

        oversize_strategy = getattr(self.config, "sft_oversize_strategy", "keep")
        invalid_multimodal_strategy = getattr(self.config, "sft_invalid_multimodal_strategy", "error")
        oversize_custom_path = getattr(self.config, "sft_oversize_custom_function_path", None)
        oversize_custom_fn = None
        if oversize_strategy == "custom":
            if not oversize_custom_path:
                raise ValueError("--sft-oversize-strategy custom requires --sft-oversize-custom-function-path.")
            oversize_custom_fn = load_function(oversize_custom_path)
            self._logger.info(f"SFT oversize strategy: custom (loaded {oversize_custom_path})")
        else:
            self._logger.info(f"SFT oversize strategy: {oversize_strategy}")
        self._logger.info(f"SFT invalid multimodal strategy: {invalid_multimodal_strategy}")

        dataset_cls = _load_custom_dataset_class(getattr(self.config, "custom_dataset_class_path", None))
        if dataset_cls is None:
            self._dataset = SFTStreamingDataset(
                path=self.config.prompt_data,
                tokenizer=self._tokenizer,
                processor_pool=self._processor_pool,
                capacity=capacity,
                prompt_key=self.config.input_key,
                label_key=self.config.label_key,
                multimodal_keys=self.config.multimodal_keys,
                conversation_key_map=getattr(self.config, "conversation_key_map", None),
                metadata_key=self.config.metadata_key,
                tool_key=self.config.tool_key,
                system_prompt=self.config.system_prompt,
                seed=seed,
                prefetch_max_cached=prefetch_buffer_size,
                prefetch_chunk_size=prefetch_chunk_size,
                prefetch_num_workers=prefetch_num_workers,
                pad_token_ids=pad_token_ids,
                oversize_strategy=oversize_strategy,
                oversize_custom_fn=oversize_custom_fn,
                invalid_multimodal_strategy=invalid_multimodal_strategy,
                apply_chat_template_kwargs=getattr(self.config, "apply_chat_template_kwargs", None),
                require_response=task_type != "seq_cls",
                task_type=task_type,
                num_labels=getattr(self.config, "num_labels", None),
                problem_type=getattr(self.config, "problem_type", "single_label_classification"),
                classification_sentinel_token_id=classification_sentinel_token_id,
            )
        else:
            self._dataset = dataset_cls.from_args(
                self.config,
                tokenizer=self._tokenizer,
                processor_pool=self._processor_pool,
                pad_token_ids=pad_token_ids,
            )
        n_avail = len(self._dataset)
        self._train_size = n_avail

        eval_prompt_data = build_named_prompt_data_configs(getattr(self.config, "eval_prompt_data", None))
        eval_size_arg = getattr(self.config, "eval_size", None)
        if eval_size_arg is not None:
            # Randomize the split once with a fixed seed. Each epoch reshuffles
            # only the resulting train row IDs; eval membership stays fixed.
            train_indices, eval_indices = resolve_sft_split_indices(n_avail, eval_size_arg, seed)
            self._train_size = len(train_indices)
            n_eval = len(eval_indices)
            if n_eval == 0:
                self._logger.warning(
                    f"--eval-size {eval_size_arg} resolves to 0 samples on a dataset of size {n_avail}; "
                    "eval will be skipped."
                )
            else:
                self._eval_indices = eval_indices
                restrict_training_indices = getattr(self._dataset, "restrict_training_indices", None)
                get_batch_by_indices = getattr(self._dataset, "get_batch_by_indices", None)
                if not callable(restrict_training_indices) or not callable(get_batch_by_indices):
                    raise TypeError(
                        "--eval-size requires the SFT dataset to implement restrict_training_indices(indices) "
                        "and get_batch_by_indices(indices) for a deterministic random split."
                    )
                restrict_training_indices(train_indices)
                self._logger.info(
                    f"--eval-size randomly held out {n_eval} samples with seed={seed}; "
                    f"train pool size now {self._train_size}."
                )
        elif eval_prompt_data:
            eval_input_key = getattr(self.config, "eval_input_key", None) or self.config.input_key
            eval_label_key = getattr(self.config, "eval_label_key", None) or self.config.label_key
            eval_tool_key = getattr(self.config, "eval_tool_key", None) or self.config.tool_key
            # Eval is small + runs every `eval_interval`; disable prefetch so
            # we don't consume worker threads idly between eval rounds.
            self._eval_dataset = SFTStreamingDataset(
                path=[d.path for d in eval_prompt_data],
                tokenizer=self._tokenizer,
                processor_pool=self._processor_pool,
                capacity=capacity,
                prompt_key=eval_input_key,
                label_key=eval_label_key,
                multimodal_keys=self.config.multimodal_keys,
                conversation_key_map=getattr(self.config, "conversation_key_map", None),
                metadata_key=self.config.metadata_key,
                tool_key=eval_tool_key,
                system_prompt=self.config.system_prompt,
                source_name="+".join(d.name for d in eval_prompt_data),
                seed=seed,
                prefetch_max_cached=0,
                pad_token_ids=pad_token_ids,
                oversize_strategy=oversize_strategy,
                oversize_custom_fn=oversize_custom_fn,
                invalid_multimodal_strategy=invalid_multimodal_strategy,
                apply_chat_template_kwargs=getattr(self.config, "apply_chat_template_kwargs", None),
                require_response=task_type != "seq_cls",
                task_type=task_type,
                num_labels=getattr(self.config, "num_labels", None),
                problem_type=getattr(self.config, "problem_type", "single_label_classification"),
                classification_sentinel_token_id=classification_sentinel_token_id,
            )

        # Resume: align IndexManager with `start_rollout_id` so a restart sees
        # the same shuffled order it would on a fresh run.
        if self._train_size > 0:
            consumed = self.step * self.config.global_batch_size
            start_epoch = consumed // self._train_size
            position = consumed % self._train_size
        else:
            start_epoch, position = 0, 0
        self._dataset.shuffle(start_epoch, position=position)

    async def _wait_for_partition_drained(self, partition_id: str, timeout_sec: float | None = None) -> bool:
        """Backpressure: hold off until the consumer has cleared ``partition_id``
        from TQ. Used both for train-step gating and for serial eval-chunk push.

        TQ storage is sized for one step (max_staleness=0); pushing a new
        partition before the previous one drains overflows the buffer.

        Returns ``True`` if the partition drained, ``False`` if ``timeout_sec``
        elapsed first (only meaningful when a timeout is provided).
        """
        deadline = None if timeout_sec is None else asyncio.get_event_loop().time() + timeout_sec
        while not self._stop_event.is_set():
            partitions = await self.data_system_client.async_get_partition_list()
            if partitions is None or partition_id not in partitions:
                return True
            if deadline is not None and asyncio.get_event_loop().time() >= deadline:
                return False
            await asyncio.sleep(1)
        return False

    async def _wait_for_buffer_capacity(self) -> None:
        """Backpressure for the next train PUT.

        Holds off until the number of in-flight train partitions (``sft_<N>``)
        is strictly less than ``max_staleness + 1``, i.e. there is room for
        one more without overflowing the TQ buffer.  TQ total storage is
        sized as ``rollout_batch_size * (max_staleness + 1) * n_samples_per_prompt``
        in ``controller._initialize_data_system``, so this ceiling and the
        actual storage capacity stay in lockstep.

        With ``max_staleness=0`` the ceiling is 1, which reproduces the
        original "wait for previous partition to drain" behavior.  With
        ``max_staleness>0`` the producer can run up to ``max_staleness``
        steps ahead of the consumer, hiding consumer compute behind the
        producer's audio I/O pipeline.

        Eval partitions (``sft_eval_*``) are excluded from the in-flight
        count: they are pushed and drained synchronously by
        ``_maybe_produce_eval`` and do not consume the train backpressure
        budget.

        Bounded by ``--sft-tq-timeout-minutes`` (falls back to
        ``--distributed-timeout-minutes``): if the consumer dies, the
        producer raises ``TimeoutError`` instead of spinning forever.
        """
        if self.step == 0:
            return
        max_in_flight = self.config.max_staleness + 1
        timeout_sec = float(getattr(self.config, "sft_tq_timeout_minutes", None) or 30) * 60.0
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout_sec
        wait_count = 0
        while not self._stop_event.is_set():
            partitions = await self.data_system_client.async_get_partition_list()
            if partitions is None:
                return
            in_flight = sum(1 for p in partitions if p.startswith("sft_") and not p.startswith("sft_eval_"))
            if in_flight < max_in_flight:
                if wait_count > 0:
                    self._logger.info(
                        f"SFT producer step {self.step}: TQ buffer freed after {wait_count}s "
                        f"(in_flight={in_flight}/{max_in_flight})"
                    )
                return
            if loop.time() >= deadline:
                raise TimeoutError(
                    f"SFT producer step {self.step}: TQ buffer stuck for "
                    f">{timeout_sec:.0f}s (in_flight={in_flight}/{max_in_flight}, "
                    f"partitions={partitions}); consumer likely dead. Raise "
                    f"--sft-tq-timeout-minutes if this is a slow consumer."
                )
            if wait_count % 60 == 0:
                self._logger.info(
                    f"SFT producer step {self.step}: TQ buffer full "
                    f"(in_flight={in_flight}/{max_in_flight}, partitions={partitions}); waited {wait_count}s"
                )
            wait_count += 1
            await asyncio.sleep(1)

    def _maybe_print_first_sample(self, samples: list[ProcessedSample]) -> None:
        if self.step != 0 or not samples:
            return
        s = samples[0]
        try:
            loss_mask = s.loss_mask
            if s.classification_label is not None:
                loss_mask = F.pad(loss_mask, (s.total_length - 2, 1), value=0)
                self._logger.info(f"First classification sample label: {s.classification_label.tolist()}")
            print_first_sample(
                step=self.step,
                sample_idx=s.source_idx,
                input_ids=s.tokens,
                loss_mask=loss_mask,
                multimodal_train_inputs=s.multimodal_train_inputs,
                tokenizer=self._tokenizer,
            )
        except Exception as exc:
            self._logger.warning(f"print_first_sample failed: {exc}")

    async def _produce_one_step(self) -> None:
        assert self._dataset is not None and self._tokenizer is not None
        await self._wait_for_buffer_capacity()
        if self._train_size == 0:
            raise RuntimeError("SFT train pool is empty (check --eval-size relative to dataset size).")

        # When prefetch is on, get_batch_async delegates to the sync prefetch
        # path (already parallel via background threads). When prefetch is off,
        # it parallelises multimodal preprocess via asyncio.gather over the pool.
        samples, crossed_epoch = await self._dataset.get_batch_async(self.config.global_batch_size)
        if len(samples) != self.config.global_batch_size:
            raise RuntimeError(
                f"SFT step {self.step}: dataset returned {len(samples)}/{self.config.global_batch_size} samples "
                "after bounded refill attempts. Refusing to push a partial TQ partition because the Megatron "
                "consumer requires a full global batch. Check invalid-multimodal and oversize skip warnings."
            )
        self._maybe_print_first_sample(samples)
        backend_batch = pack_samples_for_tq(samples, force_multimodal_field=self.config.multimodal_keys is not None)
        assert backend_batch is not None
        await self.data_system_client.async_put(
            data=dict_to_tensordict(backend_batch, batch_size=len(backend_batch["tokens"])),
            partition_id=f"sft_{self.step}",
            custom_meta=[{"total_lengths": int(length)} for length in backend_batch["total_lengths"]],
        )
        if crossed_epoch:
            self._logger.info(
                f"SFT step {self.step}: epoch boundary crossed (epoch={self._dataset.index_manager.current_epoch})"
            )
        await self._maybe_produce_eval()
        self.step += 1

    def _build_eval_batches(self) -> list[ProcessedSample] | None:
        """Render the entire eval set in deterministic index order.

        Returns None when eval is not configured.
        """
        if self._eval_indices is not None:
            assert self._dataset is not None
            return self._dataset.get_batch_by_indices(self._eval_indices)
        if self._eval_dataset is not None:
            return self._eval_dataset.get_batch_in_order(0, len(self._eval_dataset))
        return None

    async def _maybe_produce_eval(self) -> None:
        """Push the eval set under partitions ``sft_eval_<step>_n<N>_<i>`` when
        due, chunked into ``global_batch_size`` pieces and serially drained.

        TQ per-partition storage is sized for ``global_batch_size`` (one train
        step). Pushing the entire eval set at once overflows the buffer when
        the eval pool is larger than one batch. Instead we slice the rendered
        batch into N chunks, embed N in each partition name so the consumer can
        discover it, and push-then-wait-for-drain serially. Eval blocks the
        producer here, but only on eval steps.
        """
        eval_interval = getattr(self.config, "eval_interval", None)
        if not eval_interval or eval_interval <= 0:
            return
        if (self.step + 1) % eval_interval != 0:
            return
        samples = self._build_eval_batches()
        if samples is None:
            return
        if not samples:
            raise RuntimeError(
                f"Eval @ step {self.step}: source produced 0 valid samples. Refusing to skip the eval push because "
                "the Megatron consumer is waiting for an eval partition. Check invalid-multimodal and oversize "
                "skip warnings."
            )

        # Pad sub-gbs eval pools with random resamples so the eval set always
        # forms at least one full ``global_batch_size`` chunk. Without this the
        # chunking loop below would skip eval entirely (n_chunks==0), and the
        # consumer — which enters ``run_sft_eval`` purely on interval — would
        # block forever waiting for partitions that never come. Seeded by step
        # so the padding is reproducible across restarts.
        gbs = self.config.global_batch_size
        n_original = len(samples)
        is_classification = getattr(self.config, "task_type", "causal_lm") == "seq_cls"
        sample_weights = None
        if is_classification:
            n_chunks = (n_original + gbs - 1) // gbs
            pad_count = n_chunks * gbs - n_original
            if pad_count:
                samples = list(samples) + [samples[i % n_original] for i in range(pad_count)]
            sample_weights = [1.0] * n_original + [0.0] * pad_count
            self._logger.info(
                f"Classification eval @ step {self.step}: {n_original} real samples, "
                f"{pad_count} zero-weight padding samples, {n_chunks} chunk(s)."
            )
        elif n_original < gbs:
            rng = random.Random(self.step)
            pad_count = gbs - n_original
            samples = list(samples) + rng.choices(samples, k=pad_count)
            self._logger.warning(
                f"Eval @ step {self.step}: eval pool of {n_original} samples is smaller than "
                f"global_batch_size ({gbs}); random-padded with {pad_count} resampled (with "
                f"replacement) samples to fill one batch. PPL counts duplicated samples — "
                f"interpret with caution."
            )

        backend_batch = pack_samples_for_tq(
            samples,
            force_multimodal_field=self.config.multimodal_keys is not None,
            sample_weights=sample_weights,
        )
        assert backend_batch is not None
        n_samples = len(backend_batch["tokens"])

        # Drain the current train partition so the eval chunks have the full
        # TQ capacity to themselves.
        await self._wait_for_partition_drained(f"sft_{self.step}")

        chunk_size = self.config.global_batch_size
        # Causal-LM eval drops trailing samples that don't fill a full chunk;
        # classification eval was padded above and therefore has no trailing
        # real samples. The consumer's
        # `_get_data_from_transfer_queue` calls `tq.get_meta(batch_size=...)`
        # which returns size=0 when the partition has fewer than batch_size
        # samples, so a partial last chunk would never be marked consumed and
        # the actor's `while not all_consumed` loop would spin forever (it
        # already burned a full eval round in the wild — see the
        # `[get_data_profile] samples=0` log spam).
        n_chunks = n_samples // chunk_size
        n_dropped = n_samples - n_chunks * chunk_size
        if n_chunks == 0:
            raise RuntimeError(
                f"Eval @ step {self.step}: eval pool of {n_samples} samples is smaller than "
                f"global_batch_size ({chunk_size}); cannot push the full partition expected by the consumer."
            )
        if n_dropped > 0:
            self._logger.warning(
                f"Eval @ step {self.step}: dropping {n_dropped} trailing sample(s) so eval "
                f"chunks align to global_batch_size ({chunk_size}); raise eval pool size or "
                f"reduce global_batch_size if this matters."
            )
        # Per-chunk drain timeout. If consumers crash mid-eval (the actor's
        # try/except swallows the failure), the producer would otherwise spin
        # on _wait_for_partition_drained forever and starve the next train
        # step. On timeout we clear our own pending chunk and bail.
        chunk_drain_timeout = float(getattr(self.config, "sft_eval_chunk_drain_timeout_sec", 600.0))
        self._logger.info(
            f"Eval @ step {self.step}: pushing {n_chunks * chunk_size} samples in {n_chunks} chunk(s) of {chunk_size}."
        )
        for chunk_idx in range(n_chunks):
            s = chunk_idx * chunk_size
            e = s + chunk_size
            chunk = {k: v[s:e] for k, v in backend_batch.items()}
            partition_id = f"sft_eval_{self.step}_n{n_chunks}_{chunk_idx}"
            await self.data_system_client.async_put(
                data=dict_to_tensordict(chunk, batch_size=len(chunk["tokens"])),
                partition_id=partition_id,
                custom_meta=[{"total_lengths": int(length)} for length in chunk["total_lengths"]],
            )
            drained = await self._wait_for_partition_drained(partition_id, timeout_sec=chunk_drain_timeout)
            if not drained:
                self._logger.warning(
                    f"Eval @ step {self.step}: chunk {chunk_idx}/{n_chunks} ({partition_id}) did not drain "
                    f"within {chunk_drain_timeout}s; aborting eval push and clearing TQ."
                )
                await self.data_system_client.async_clear_partition(partition_id=partition_id)
                return

    async def run(self) -> None:
        if self._run_task is not None:
            return
        self._init_data_pipeline()
        self._run_task = asyncio.create_task(self._async_run())

    async def _async_run(self) -> None:
        try:
            while self.step < self.config.num_rollout and not self._stop_event.is_set():
                await self._produce_one_step()
        except Exception as exc:
            error_msg = f"SFT producer crashed at step {self.step}: {type(exc).__name__}: {str(exc)}"
            self._logger.exception(error_msg)
            # SFT producer failures are deterministic by nature — data schema
            # mismatches, malformed rows, alignment errors. Restarting the
            # replica reads the same data and crashes again. Mark fatal so
            # the controller exits immediately instead of grinding through
            # ~12 service restarts before _global_restart hits its limit.
            self.healthy.report_error.remote("sft", error_msg, fatal=True)
            raise

    async def stop(self) -> None:
        self._stop_event.set()
        if self._dataset is not None:
            self._dataset.stop()
        if self._eval_dataset is not None:
            self._eval_dataset.stop()
        if self._run_task:
            await self._run_task
