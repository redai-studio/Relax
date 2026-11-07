# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Strict per-batch source quotas with independent, resumable shuffled
epochs."""

import copy
import hashlib
import json
import os
from bisect import bisect_right
from collections import Counter
from dataclasses import asdict
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any

import torch

from relax.engine.rollout.data_source import DataSource, RolloutDataSource, _shallow_copy_sample
from relax.utils.data.data_utils import parse_generalized_path
from relax.utils.data.streaming_dataset import CompositeStreamingReader, IndexManager, PrefetchBuffer
from relax.utils.logging_utils import get_logger
from relax.utils.types import Sample


logger = get_logger(__name__)
SOURCE_MIXTURE_PATH = "relax.engine.rollout.source_mixture.SourceMixtureDataSource"
BATCH_ID_KEY = "_source_mixture_batch_id"
_DEFAULT_SOURCE = "relax.engine.rollout.data_source.RolloutDataSourceWithBuffer"
_DEFAULT_ROLLOUT = "relax.engine.rollout.sglang_rollout.generate_rollout"


def validate_source_mixture_args(args: Any) -> None:
    """Normalize the final (including custom YAML overrides) mixture
    configuration."""
    config_path = getattr(args, "rollout_source_mixture", None)
    if not config_path:
        return

    required = {
        "colocate": True,
        "use_opd": True,
        "opd_type": "sglang",
        "use_streaming_dataset": True,
        "rollout_global_dataset": True,
        "n_samples_per_prompt": 1,
        "opd_teacher_key": "data_source",
        "rollout_function_path": _DEFAULT_ROLLOUT,
    }
    for name, expected in required.items():
        if getattr(args, name, None) != expected:
            raise ValueError(f"--rollout-source-mixture requires {name}={expected!r}")
    for name in (
        "fully_async",
        "hybrid",
        "partial_rollout",
        "use_dynamic_global_batch_size",
        "use_agentic_rollout",
        "debug_train_only",
    ):
        if getattr(args, name, False):
            raise ValueError(f"--rollout-source-mixture does not support {name}")
    for name in (
        "dynamic_sampling_filter_path",
        "buffer_filter_path",
        "num_epoch",
        "prompt_data",
        "custom_convert_samples_to_train_data_path",
        "custom_reward_post_process_path",
        "rollout_sample_filter_path",
        "load_debug_rollout_data",
    ):
        if getattr(args, name, None) is not None:
            raise ValueError(f"--rollout-source-mixture is incompatible with {name}")
    if getattr(args, "data_source_path", _DEFAULT_SOURCE) not in (_DEFAULT_SOURCE, SOURCE_MIXTURE_PATH):
        raise ValueError("--rollout-source-mixture does not support a custom data_source_path")
    if getattr(args, "num_steps_per_rollout", None) not in (None, 1):
        raise ValueError("--rollout-source-mixture requires one training batch per rollout")
    if getattr(args, "num_iters_per_train_update", 1) != 1:
        raise ValueError("--rollout-source-mixture requires num_iters_per_train_update=1")
    for name in ("num_rollout", "global_batch_size"):
        value = getattr(args, name, None)
        if type(value) is not int or value <= 0:
            raise ValueError(f"--rollout-source-mixture requires an explicit positive {name}")
    batch_size = args.global_batch_size
    if args.rollout_batch_size != batch_size or args.over_sampling_batch_size != batch_size:
        raise ValueError(
            "--rollout-source-mixture requires rollout_batch_size = over_sampling_batch_size = global_batch_size"
        )

    import yaml

    config_path = Path(config_path).resolve()
    with config_path.open(encoding="utf-8") as stream:
        # This schema needs scalar text; preserve decimal weights without float rounding.
        config = yaml.load(stream, Loader=yaml.BaseLoader)
    sources = config.get("sources") if isinstance(config, dict) else None
    if not isinstance(sources, dict) or not sources:
        raise ValueError("Source mixture configuration requires a non-empty 'sources' mapping")
    try:
        routes = json.loads(args.opd_teacher_routes)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("--rollout-source-mixture requires JSON opd_teacher_routes") from exc
    if not isinstance(routes, dict) or set(routes) != set(sources):
        raise ValueError("Source mixture names must exactly match opd_teacher_routes keys")

    spec = []
    weights = {}
    for name, source in sources.items():
        if not isinstance(name, str) or not name or not isinstance(source, dict):
            raise ValueError("Each source must have a non-empty string name and a mapping with path/weight")
        path = source.get("path")
        if not isinstance(path, str) or not path:
            raise ValueError(f"Source {name!r} requires a dataset path")
        real_path, row_slice = parse_generalized_path(os.path.expandvars(path))
        resolved = Path(real_path)
        if not resolved.is_absolute():
            resolved = config_path.parent / resolved
        resolved = resolved.resolve()
        if not resolved.is_file() or resolved.suffix not in (".jsonl", ".parquet"):
            raise ValueError(f"Source {name!r} must reference an existing JSONL or Parquet file: {resolved}")
        path = str(resolved)
        if row_slice is not None:
            start = "" if row_slice.start is None else str(row_slice.start)
            stop = "" if row_slice.stop is None else str(row_slice.stop)
            path += f"@[{start}:{stop}]"
        try:
            weight = Decimal(str(source.get("weight")))
        except InvalidOperation as exc:
            raise ValueError(f"Source {name!r} weight must be finite and positive") from exc
        if not weight.is_finite() or weight <= 0:
            raise ValueError(f"Source {name!r} weight must be finite and positive")
        weights[name] = Fraction(weight)
        spec.append({"name": name, "path": path, "weight": str(weights[name])})

    total_weight = sum(weights.values())
    exact_quotas = {name: batch_size * weight / total_weight for name, weight in weights.items()}
    quotas = {name: int(quota) for name, quota in exact_quotas.items()}
    remaining = batch_size - sum(quotas.values())
    # Largest remainder preserves the batch size. Stable sorting resolves ties
    # in configuration order; Fraction keeps decimal weights exact throughout.
    by_remainder = sorted(quotas, key=lambda name: exact_quotas[name] - quotas[name], reverse=True)
    for name in by_remainder[:remaining]:
        quotas[name] += 1
    logger.info("Source mixture largest-remainder quotas for global_batch_size=%s: %s", batch_size, quotas)
    zero_sources = [name for name, quota in quotas.items() if quota == 0]
    if zero_sources:
        logger.warning(
            "Source mixture sources %s have zero quota and will not be sampled in any batch; "
            "increase global_batch_size or their weights to include them.",
            zero_sources,
        )
    args.source_mixture_spec = spec
    args.source_mixture_quotas = quotas
    args.data_source_path = SOURCE_MIXTURE_PATH


def validate_source_mixture_batch(
    args: Any, groups: list[list[Sample]], rollout_id: int, *, completed: bool = False
) -> dict[str, int]:
    """Fail before training enqueue if a strict mixture batch lost its
    contract."""
    if not getattr(args, "rollout_source_mixture", None):
        return {}
    counts = Counter({name: 0 for name in args.source_mixture_quotas})
    indices = set()
    group_indices = set()
    for group in groups:
        if len(group) != 1:
            raise ValueError("Source mixture requires exactly one sample per prompt group")
        sample = group[0]
        if sample.metadata.get(BATCH_ID_KEY) != rollout_id:
            raise ValueError(f"Source mixture batch membership mismatch for rollout {rollout_id}")
        if (
            sample.index is None
            or sample.index in indices
            or sample.group_index is None
            or sample.group_index in group_indices
        ):
            raise ValueError("Source mixture requires unique sample and group indices")
        indices.add(sample.index)
        group_indices.add(sample.group_index)
        counts[sample.metadata.get("data_source")] += 1
        if completed and (
            sample.status not in (Sample.Status.COMPLETED, Sample.Status.TRUNCATED) or sample.remove_sample
        ):
            raise ValueError(f"Source mixture rollout {rollout_id} contains an incomplete or removed sample")
    if len(groups) != args.global_batch_size or dict(counts) != args.source_mixture_quotas:
        raise ValueError(
            f"Source mixture quota mismatch at rollout {rollout_id}: {dict(counts)} != {args.source_mixture_quotas}"
        )
    return dict(counts)


class SourceMixtureDataSource(DataSource):
    """One shared streaming processor/cache, one independent cursor per
    source."""

    args: Any
    spec: list[dict[str, str]]
    quotas: dict[str, int]
    dataset: Any
    _source_ends: list[int]
    index_managers: dict[str, IndexManager]
    fingerprint: str
    _next_rollout_id: int
    _snapshots: dict[int, dict[str, Any]]

    def __init__(self, args: Any):
        from xxhash import xxh3_128  # Provided by the datasets dependency; no extra package required.

        self.args: Any = args
        if not getattr(args, "source_mixture_spec", None):
            validate_source_mixture_args(args)
        self.spec: list[dict[str, str]] = args.source_mixture_spec
        self.quotas: dict[str, int] = args.source_mixture_quotas

        dataset_args = copy.copy(args)
        dataset_args.prompt_data = [parse_generalized_path(source["path"])[0] for source in self.spec]
        # The per-source cursors below own the order; do not start a global shuffle/prefetch.
        dataset_args.rollout_shuffle = False
        self.dataset = RolloutDataSource(dataset_args).dataset
        if isinstance(self.dataset.reader, CompositeStreamingReader):
            readers = self.dataset.reader.readers
        else:
            readers = [self.dataset.reader]

        self._source_ends: list[int] = []
        self.index_managers: dict[str, IndexManager] = {}
        manifest = []
        offset = 0
        for source, reader in zip(self.spec, readers, strict=True):
            name = source["name"]
            path, row_slice = parse_generalized_path(source["path"])
            size = len(reader)
            positions = range(size) if row_slice is None else range(size)[row_slice]
            if not positions:
                raise ValueError(f"Source {name!r} is empty")
            seed = int.from_bytes(hashlib.sha256(f"{args.rollout_seed}:{name}".encode()).digest()[:4], "big")
            manager = IndexManager(len(positions), seed=seed, index_pool=(offset + i for i in positions))
            manager.shuffle(0)
            self.index_managers[name] = manager
            offset += size
            self._source_ends.append(offset)
            # Hash all bytes once at initialization; file location and timestamps
            # do not identify the data. Keep memory bounded for large datasets.
            digest = xxh3_128()
            logger.info("Computing source mixture content fingerprint for %s", name)
            with open(path, "rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            manifest.append(
                {
                    "name": name,
                    "xxh3_128": digest.hexdigest(),
                    "rows": size,
                    "slice": [positions.start, positions.stop, positions.step],
                    "seed": seed,
                }
            )
        fingerprint = {
            "sources": manifest,
            "quotas": self.quotas,
            "routes": json.loads(args.opd_teacher_routes),
            "multimodal_config": asdict(self.dataset.multimodal_config),
            "processing": {
                key: getattr(args, key, None)
                for key in (
                    "hf_checkpoint",
                    "input_key",
                    "label_key",
                    "metadata_key",
                    "tool_key",
                    "multimodal_keys",
                    "system_prompt",
                    "apply_chat_template",
                    "apply_chat_template_kwargs",
                    "custom_prompt_path",
                    "rollout_max_prompt_len",
                    "use_audio_in_video",
                    "opd_teacher_prompt_key",
                    "opd_teacher_image_key",
                )
            },
        }
        self.fingerprint: str = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()

        self._next_rollout_id: int = 0
        self._snapshots: dict[int, dict[str, Any]] = {}
        if self.dataset._prefetch_buffer is not None:
            self.dataset._prefetch_buffer._process_fn = self._read_sample

    def _read_sample(self, index: int) -> Sample | None:
        """Read and validate a row using the shared streaming processor.

        Processing failures and configured length filtering return None.
        """
        name = self.spec[bisect_right(self._source_ends, index)]["name"]  # Binary Search
        raw = self.dataset.reader[index]
        metadata = raw.get(self.dataset.metadata_key) or {}
        for value in (raw.get("data_source"), metadata.get("data_source")):
            if value is not None and value != name:
                raise ValueError(f"Source {name!r} conflicts with record data_source={value!r} at index {index}")
        raw = dict(raw)
        raw[self.dataset.metadata_key] = dict(metadata)
        return self.dataset._process_raw_data(raw)

    def _get_sample(self, index: int) -> Sample | None:
        prefetch = self.dataset._prefetch_buffer
        if prefetch is not None:
            sample = prefetch.get(index)
        else:
            sample = self.dataset.buffer.get(index)
            if sample is None:
                sample = self._read_sample(index)
                if sample is not None:
                    self.dataset.buffer.put(index, sample)
        return sample

    def get_samples(self, num_samples: int, rollout_id: int | None = None) -> list[list[Sample]]:
        if num_samples != self.args.global_batch_size:
            raise ValueError("Source mixture must be fetched as one complete global batch")
        rollout_id = self._next_rollout_id if rollout_id is None else rollout_id
        if rollout_id != self._next_rollout_id:
            raise ValueError(f"Expected source mixture rollout {self._next_rollout_id}, got {rollout_id}")
        if self.dataset._prefetch_buffer is not None:
            upcoming = []
            for name, manager in self.index_managers.items():
                # A shallow cursor copy shares immutable permutations until it crosses an epoch.
                upcoming.extend(copy.copy(manager).get_next_indices(self.quotas[name])[0])
            self.dataset._prefetch_buffer.set_index_order(list(dict.fromkeys(upcoming)))

        groups = []
        for name, manager in self.index_managers.items():
            remaining = self.quotas[name]
            invalid = 0
            while remaining:
                index = manager.get_next_indices(1)[0][0]
                prompt_sample = self._get_sample(index)
                if prompt_sample is None:
                    invalid += 1
                    # From an arbitrary cursor, two source traversals contain a full epoch.
                    if invalid >= 2 * manager.total_size:
                        raise ValueError(f"Source {name!r} has no usable records; cannot fill its quota")
                    continue
                invalid = 0
                sample = _shallow_copy_sample(prompt_sample)
                if sample.metadata.get("data_source", name) != name:
                    raise ValueError(f"Source {name!r} conflicts with processed sample data_source")
                sample.metadata["data_source"] = name
                sample.metadata[BATCH_ID_KEY] = rollout_id
                sample.index = sample.group_index = rollout_id * num_samples + len(groups)
                groups.append([sample])
                remaining -= 1
        self._next_rollout_id += 1
        self._snapshots[rollout_id] = self._get_state()
        while len(self._snapshots) > 2:
            del self._snapshots[next(iter(self._snapshots))]
        return groups

    def _get_state(self) -> dict:
        return {
            "fingerprint": self.fingerprint,
            "sources": {name: manager.get_state() for name, manager in self.index_managers.items()},
            "next_rollout_id": self._next_rollout_id,
        }

    def save(self, rollout_id: int) -> None:

        if rollout_id not in self._snapshots:
            raise ValueError(f"No retained source mixture snapshot for rollout {rollout_id}")
        path = Path(self.args.save) / "dataset" / f"global_dataset_state_dict_{rollout_id}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._snapshots[rollout_id], path)

    def load(self, rollout_id: int | None = None) -> None:

        if self.args.load is None or rollout_id is None or rollout_id < 0:
            return
        path = Path(self.args.load) / "dataset" / f"global_dataset_state_dict_{rollout_id}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing source mixture checkpoint: {path}")
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("fingerprint") != self.fingerprint:
            raise ValueError("Source mixture checkpoint does not match the configuration or dataset fingerprint")
        if state["next_rollout_id"] != rollout_id + 1 or set(state["sources"]) != set(self.index_managers):
            raise ValueError("Source mixture checkpoint has inconsistent rollout/source state")
        for name, manager in self.index_managers.items():
            cursor = state["sources"][name]
            if not 0 <= cursor["position"] <= manager.total_size or cursor["epoch_id"] < 0:
                raise ValueError(f"Invalid source mixture cursor for {name!r}")
        # Stop and replace the prefetch object, so a draining old worker cannot populate the restored cache.
        self.close()
        self.dataset.buffer.clear()
        if self.dataset._prefetch_buffer is not None:
            self.dataset._prefetch_buffer = PrefetchBuffer(
                self._read_sample,
                chunk_size=getattr(self.args, "prefetch_chunk_size", 32),
                max_cached=getattr(self.args, "prefetch_max_cached", 256),
                num_workers=getattr(self.args, "prefetch_num_workers", 1),
            )
        for name, manager in self.index_managers.items():
            manager.load_state(state["sources"][name])
        self._next_rollout_id = state["next_rollout_id"]
        self._snapshots.clear()
        self._snapshots[rollout_id] = self._get_state()

    def close(self) -> None:
        if self.dataset._prefetch_buffer is not None:
            self.dataset._prefetch_buffer.stop()

    def add_samples(self, samples: list[list[Sample]]) -> None:
        if samples:
            raise ValueError("Source mixture does not support returning samples to a buffer")

    def __len__(self) -> int:
        return sum(manager.total_size for manager in self.index_managers.values())

    def lengths(self) -> int:
        return self.__len__()
