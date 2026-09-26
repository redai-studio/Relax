# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Data Processing Utilities.

This module contains shared utility functions and base classes for data processing,
used by both Dataset and StreamingDataset classes.

Classes:
- BaseDataset: Abstract base class for dataset implementations

Functions:
- read_file: Read data from JSONL or Parquet files
- parse_generalized_path: Parse path with optional slice notation
- build_messages: Build message format from raw data
- collect_message_multimodal_data: Collect inline media in message order
- process_raw_sample: Process raw data dict into a Sample
- check_sample_length: Check if sample exceeds max_length
- filter_long_prompts: Filter samples by length (batch version)
"""

import abc
import ast
import itertools
import json
import logging
import multiprocessing
import os
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Optional

import numpy as np
import tqdm


try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

from relax.utils.multimodal.config import MultimodalConfig
from relax.utils.opd.opd_utils import build_opd_teacher_sample_fields
from relax.utils.types import MultimodalTypes, Sample


logger = logging.getLogger(__name__)

__all__ = [
    "BaseDataset",
    "read_file",
    "parse_generalized_path",
    "resolve_path_plan",
    "build_messages",
    "collect_message_multimodal_data",
    "count_message_multimodal_items",
    "process_raw_sample",
    "check_sample_length",
    "filter_long_prompts",
]

_MULTIMODAL_CONTENT_TYPE_ALIASES = {
    "image": "image",
    "image_url": "image",
    "video": "video",
    "audio": "audio",
}


def _extract_inline_multimodal_value(item: dict[str, Any]) -> tuple[str | None, Any | None]:
    item_type = item.get("type")
    media_name = _MULTIMODAL_CONTENT_TYPE_ALIASES.get(item_type) if isinstance(item_type, str) else None
    if media_name is None:
        return None, None

    if item_type == "image_url":
        value = item.get("image_url")
        if isinstance(value, dict):
            value = value.get("url")
        if not isinstance(value, str):
            return media_name, None
    elif media_name == "image":
        value = item.get("image")
    else:
        value = item.get(media_name)
    if value is None or (isinstance(value, str) and not value.strip()):
        return media_name, None
    return media_name, value


def _normalize_inline_multimodal_items(prompt: Any) -> Any:
    if not isinstance(prompt, list):
        return prompt

    normalized_messages = []
    for message in prompt:
        if not isinstance(message, dict) or not isinstance(message.get("content"), list):
            normalized_messages.append(message)
            continue
        normalized_content = []
        for item in message["content"]:
            if not isinstance(item, dict):
                normalized_content.append(item)
                continue
            media_name, value = _extract_inline_multimodal_value(item)
            if media_name is None or value is None:
                normalized_content.append(item)
                continue
            normalized_item = dict(item)
            normalized_item["type"] = media_name
            if media_name == "image":
                normalized_item.pop("image_url", None)
            normalized_item[media_name] = value
            normalized_content.append(normalized_item)
        normalized_message = dict(message)
        normalized_message["content"] = normalized_content
        normalized_messages.append(normalized_message)
    return normalized_messages


def count_message_multimodal_items(prompt: str | list[Any]) -> dict[str, int]:
    """Count literal placeholders and structured media items in messages."""
    counts = {media.name: 0 for media in MultimodalTypes.all()}
    if isinstance(prompt, str):
        contents: list[Any] = [prompt]
    else:
        contents = [message.get("content") for message in prompt if isinstance(message, dict)]

    for content in contents:
        if isinstance(content, str):
            for media in MultimodalTypes.all():
                counts[media.name] += content.count(media.placeholder)
        elif isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                media_name, _ = _extract_inline_multimodal_value(item)
                if media_name is not None:
                    counts[media_name] += 1
    return counts


def collect_message_multimodal_data(prompt: str | list[Any]) -> dict[str, list[Any]]:
    """Collect valid inline image/video/audio payloads from messages in
    encounter order."""
    collected = {media.name: [] for media in MultimodalTypes.all()}
    if not isinstance(prompt, list):
        return collected

    for message in prompt:
        if not isinstance(message, dict) or not isinstance(message.get("content"), list):
            continue
        for item in message["content"]:
            if not isinstance(item, dict):
                continue
            media_name, value = _extract_inline_multimodal_value(item)
            if media_name is not None and value is not None:
                collected[media_name].append(value)
    return collected


def parse_generalized_path(s: str) -> tuple[str, Optional[slice]]:
    """Parse path with optional slice notation.

    Supports format like 'file.jsonl@[0:100]' to specify a subset of rows.

    Args:
        s: Path string, optionally with slice notation

    Returns:
        Tuple of (real_path, slice_or_none)

    Examples:
        >>> parse_generalized_path("data.jsonl")
        ("data.jsonl", None)
        >>> parse_generalized_path("data.jsonl@[0:100]")
        ("data.jsonl", slice(0, 100))
    """
    if (m := re.match(r"^(?P<real_path>.*)@\[(?P<start>-?\d*):(?P<end>-?\d*)\]$", s)) is not None:
        path = m.group("real_path")
        start = int(x) if (x := m.group("start")) != "" else None
        end = int(x) if (x := m.group("end")) != "" else None
        return path, slice(start, end)
    return s, None


def build_messages(
    data: dict,
    prompt_key: str,
    system_prompt: Optional[str],
    as_conversation: bool,
    multimodal_keys: Optional[dict] = None,
    custom_prompt_func: Optional[Callable[[Any, dict], Any]] = None,
) -> Any:
    """Build message format from raw data.

    Handles conversion of string prompts to conversation format,
    multimodal content insertion, and system prompt addition.

    Args:
        data: Raw data dictionary
        prompt_key: Key to extract prompt from data
        system_prompt: System prompt key or content
        as_conversation: Whether to convert to conversation format
        multimodal_keys: Mapping of multimodal types to data keys
        custom_prompt_func: Optional callable ``(prompt, data) -> prompt`` applied
            immediately after extracting the prompt from *data*, before any
            conversation / multimodal processing.

    Returns:
        Processed prompt (string or list of message dicts)
    """
    prompt = data.get(prompt_key)

    if custom_prompt_func is not None:
        prompt = custom_prompt_func(prompt, data)

    if isinstance(prompt, str):
        # If prompt is a string and we don't apply chat template, return as is
        if not as_conversation:
            return prompt
        else:
            prompt = [{"role": "user", "content": prompt}]

    prompt = _normalize_inline_multimodal_items(prompt)

    # TODO(xide): audio in video special case
    if multimodal_keys:
        # Build mapping: placeholder -> (MultimodalType, content_list)
        multimodals = {}
        multimodals_by_name = {}
        remain_data = defaultdict(int)
        for type_name, data_key in multimodal_keys.items():
            mt = MultimodalTypes.get(type_name)
            if mt is None:
                raise ValueError(f"Unsupported multimodal type: {type_name}")

            placeholder = mt.placeholder
            multimodal_data = list(data.get(data_key) or [])
            multimodals[placeholder] = (mt, multimodal_data)
            multimodals_by_name[mt.name] = (mt, multimodal_data)
            remain_data[mt.name] += len(multimodal_data)

        if multimodals:
            pattern = "(" + "|".join(re.escape(p) for p in multimodals.keys()) + ")"
            built_prompt = []
            for message in prompt:
                if isinstance(message["content"], str):
                    content_list = []
                    for segment in re.split(pattern, message["content"]):
                        if not segment:
                            continue
                        if segment in multimodals:
                            mt, content = multimodals[segment]
                            remain_data[mt.name] -= 1
                            if remain_data[mt.name] < 0:
                                logger.warning(
                                    f"The number of placeholder {segment} in prompt is more than data number."
                                )
                            if content:
                                content_list.append({"type": mt.name, mt.name: content.pop(0)})
                        else:
                            content_list.append({"type": "text", "text": segment})
                    built_message = dict(message)
                    built_message["content"] = content_list
                    built_prompt.append(built_message)
                elif isinstance(message["content"], list):
                    content_list = []
                    for item in message["content"]:
                        if not isinstance(item, dict):
                            content_list.append(item)
                            continue
                        media_name, inline_value = _extract_inline_multimodal_value(item)
                        if media_name is None or inline_value is not None or media_name not in multimodals_by_name:
                            content_list.append(item)
                            continue
                        _, content = multimodals_by_name[media_name]
                        remain_data[media_name] -= 1
                        if remain_data[media_name] < 0:
                            logger.warning(
                                f"The number of structured {media_name} placeholders in prompt "
                                "is more than data number."
                            )
                        if content:
                            built_item = dict(item)
                            built_item["type"] = media_name
                            if media_name == "image":
                                built_item.pop("image_url", None)
                            built_item[media_name] = content.pop(0)
                            content_list.append(built_item)
                        else:
                            content_list.append(item)
                    built_message = dict(message)
                    built_message["content"] = content_list
                    built_prompt.append(built_message)
                else:
                    raise ValueError(
                        f"Unsupported content type: {type(message['content'])}, expected str or list of dicts"
                    )

            prompt = built_prompt

            if any(v > 0 for v in remain_data.values()):
                raise RuntimeError(
                    f"placeholder lost! The number of remain mutimodal data is {remain_data}. Please check your dataset prompt."
                )

    if system_prompt is not None:
        final_message = [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}]
        final_message.extend(prompt)
        return final_message

    return prompt


def process_raw_sample(
    data: dict,
    tokenizer: Any,
    processor: Any,
    *,
    prompt_key: str = "text",
    multimodal_keys: Optional[dict] = None,
    label_key: Optional[str] = None,
    tool_key: Optional[str] = None,
    metadata_key: str = "metadata",
    system_prompt: Optional[str] = None,
    apply_chat_template: bool = False,
    apply_chat_template_kwargs: Optional[dict] = None,
    use_audio_in_video: Optional[bool] = False,
    multimodal_config: MultimodalConfig = None,
    custom_prompt_func: Optional[Callable[[Any, dict], Any]] = None,
    teacher_prompt_key: Optional[str] = None,
    teacher_multimodal_keys: Optional[dict] = None,
) -> Sample:
    """Process a raw data dictionary into a Sample object.

    Args:
        data: Raw data dictionary from file
        tokenizer: Tokenizer for chat template application
        processor: Processor for multimodal inputs
        prompt_key: Key for prompt in data
        multimodal_keys: Mapping of multimodal types to data keys
        label_key: Key for labels in data
        tool_key: Key for tools in data
        metadata_key: Key for metadata in data
        system_prompt: System prompt key or content
        apply_chat_template: Whether to apply chat template
        apply_chat_template_kwargs: Additional kwargs for chat template
        use_audio_in_video: Whether to extract audio from video files for multimodal processing

    Returns:
        Processed Sample object
    """
    # Both chat templates and multimodal inputs require conversation format
    as_conversation = apply_chat_template or (multimodal_keys is not None)
    prompt = build_messages(data, prompt_key, system_prompt, as_conversation, multimodal_keys, custom_prompt_func)

    metadata = data.get(metadata_key) or {}

    # MOPD: surface top-level ``data_source`` column into metadata so the
    # per-sample teacher router (``_pick_teacher_url``) can look it up via
    # ``sample.metadata["data_source"]``.  Only injected when the column
    # exists and the metadata dict does not already carry it.
    if "data_source" not in metadata and "data_source" in data:
        if isinstance(metadata, dict):
            metadata["data_source"] = data["data_source"]
        else:
            metadata = {"data_source": data["data_source"]}

    tools = None

    if tool_key is not None and tool_key in data:
        tools = data[tool_key]
        if isinstance(tools, str):
            tools = json.loads(tools)
        elif isinstance(tools, np.ndarray):
            tools = tools.tolist()
        assert isinstance(tools, list), f"tools must be a list, got {type(tools)} instead"
        metadata["tools"] = tools

    # Apply chat template if needed.
    # Per-sample override: a sample may carry its own ``apply_chat_template_kwargs``
    # in ``metadata`` (e.g. per-sample ``enable_thinking``). It is merged on top of
    # the global kwargs so the global value is the default and the per-sample value
    # wins. Samples without this metadata key keep the global behavior unchanged.
    per_sample_kwargs = metadata.get("apply_chat_template_kwargs") if isinstance(metadata, dict) else None
    merged_chat_template_kwargs = {**(apply_chat_template_kwargs or {}), **(per_sample_kwargs or {})}
    if apply_chat_template:
        output_prompt = tokenizer.apply_chat_template(
            prompt,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            **merged_chat_template_kwargs,
        )
    else:
        output_prompt = prompt

    # Process multimodal inputs
    if processor:
        from relax.utils.data.processing_utils import process_vision_info

        assert isinstance(prompt, list), (
            f"prompt must be a list when processor is not None, got {type(prompt)} instead"
        )
        multimodal_inputs = process_vision_info(
            prompt, processor, use_audio_in_video=use_audio_in_video, config=multimodal_config
        )
    else:
        multimodal_inputs = None

    teacher_prompt_str, teacher_multimodal_inputs = build_opd_teacher_sample_fields(
        data,
        tokenizer,
        processor,
        prompt_key=prompt_key,
        system_prompt=system_prompt,
        as_conversation=as_conversation,
        multimodal_keys=multimodal_keys,
        teacher_prompt_key=teacher_prompt_key,
        teacher_multimodal_keys=teacher_multimodal_keys,
        custom_prompt_func=custom_prompt_func,
        apply_chat_template=apply_chat_template,
        apply_chat_template_kwargs=apply_chat_template_kwargs,
        tools=tools,
        use_audio_in_video=use_audio_in_video,
        multimodal_config=multimodal_config,
        build_messages_fn=build_messages,
    )

    return Sample(
        prompt=output_prompt,
        label=data[label_key] if label_key is not None else None,
        metadata=metadata,
        multimodal_inputs=multimodal_inputs,
        teacher_prompt=teacher_prompt_str,
        teacher_multimodal_inputs=teacher_multimodal_inputs,
    )


def check_sample_length(
    sample: Sample,
    tokenizer: Any,
    processor: Any,
    max_length: int,
    *,
    apply_chat_template_kwargs: Optional[dict] = None,
) -> bool:
    """Check if a sample's prompt length is within the allowed limit.

    Args:
        sample: Sample to check
        tokenizer: Tokenizer for encoding
        processor: Processor for multimodal inputs
        max_length: Maximum allowed length

    Returns:
        True if sample is valid (not too long), False otherwise
    """
    try:
        tools = sample.metadata.get("tools") if isinstance(sample.metadata, dict) else None
        # Honor per-sample apply_chat_template_kwargs (same merge as process_raw_sample)
        # so length filtering renders the prompt exactly as the rollout will.
        per_sample_kwargs = (
            sample.metadata.get("apply_chat_template_kwargs") if isinstance(sample.metadata, dict) else None
        )
        merged_chat_template_kwargs = {**(apply_chat_template_kwargs or {}), **(per_sample_kwargs or {})}
        if isinstance(sample.prompt, str):
            prompt_text = sample.prompt
            input_ids = None
        elif isinstance(sample.prompt, list):
            if not hasattr(tokenizer, "apply_chat_template"):
                logger.warning("Skipping max_length check for list prompt because tokenizer has no chat template.")
                return True
            prompt_text = tokenizer.apply_chat_template(
                sample.prompt,
                tools=tools,
                tokenize=False,
                add_generation_prompt=True,
                **merged_chat_template_kwargs,
            )
            input_ids = None
        else:
            return True

        if processor and sample.multimodal_inputs:
            from relax.utils.data.processing_utils import adapt_processor_kwargs

            adapted = adapt_processor_kwargs(processor, sample.multimodal_inputs)
            processor_output = processor(text=prompt_text, **adapted)
            input_ids = processor_output["input_ids"][0]
        elif input_ids is None:
            if isinstance(sample.prompt, list):
                input_ids = tokenizer.apply_chat_template(
                    sample.prompt,
                    tools=tools,
                    tokenize=True,
                    add_generation_prompt=True,
                    **merged_chat_template_kwargs,
                )
            else:
                input_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        return len(input_ids) <= max_length

    except Exception as e:
        logger.warning(f"Error checking length: {e}")
        return True


# Global variables for multiprocessing workers (required for ProcessPoolExecutor initializer)
_worker_tokenizer = None
_worker_processor = None
_worker_apply_chat_template_kwargs = None


def _init_check_length_worker(
    tokenizer: Any,
    processor: Any,
    apply_chat_template_kwargs: Optional[dict],
) -> None:
    """Initialize worker process with tokenizer and processor."""
    global _worker_tokenizer, _worker_processor
    global _worker_apply_chat_template_kwargs
    _worker_tokenizer = tokenizer
    _worker_processor = processor
    _worker_apply_chat_template_kwargs = apply_chat_template_kwargs


def _check_sample_length_worker(sample: Sample, max_length: int) -> tuple[Sample, bool]:
    """Worker function for parallel sample length checking."""
    global _worker_tokenizer, _worker_processor
    global _worker_apply_chat_template_kwargs
    is_valid = check_sample_length(
        sample,
        _worker_tokenizer,
        _worker_processor,
        max_length,
        apply_chat_template_kwargs=_worker_apply_chat_template_kwargs,
    )
    return sample, is_valid


def filter_long_prompts(
    samples: list[Sample],
    tokenizer: Any,
    processor: Any,
    max_length: int,
    num_workers: int = 0,
    *,
    apply_chat_template_kwargs: Optional[dict] = None,
) -> list[Sample]:
    """Filter out samples that exceed the maximum length.

    This is the batch version for eager-loading datasets.

    Args:
        samples: List of samples to filter
        tokenizer: Tokenizer for encoding
        processor: Processor for multimodal inputs
        max_length: Maximum allowed length
        num_workers: Number of CPU workers for parallel processing. Default 0 (auto-detect).

    Returns:
        Filtered list of samples
    """
    if not samples:
        return samples

    if not isinstance(samples[0].prompt, (str, list)):
        logger.warning("Skipping max_length check for unsupported prompt type.")
        return samples

    if processor:
        actual_workers = min(multiprocessing.cpu_count(), len(samples), 8) if num_workers <= 0 else num_workers

        if actual_workers > 1 and len(samples) > 1:
            filtered_samples = []
            with ProcessPoolExecutor(
                max_workers=actual_workers,
                initializer=_init_check_length_worker,
                initargs=(tokenizer, processor, apply_chat_template_kwargs),
            ) as executor:
                futures = {
                    executor.submit(_check_sample_length_worker, sample, max_length): sample for sample in samples
                }
                for future in tqdm.tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"check sample length...(num_proc={actual_workers})",
                ):
                    try:
                        sample, is_valid = future.result()
                        if is_valid:
                            filtered_samples.append(sample)
                        else:
                            logger.debug(f"Filtered sample exceeding max_length={max_length}")
                    except Exception as e:
                        logger.warning(f"Error checking sample length: {e}")
                        filtered_samples.append(futures[future])
        else:
            filtered_samples = []
            for sample in tqdm.tqdm(samples, desc="check sample length..."):
                if check_sample_length(
                    sample,
                    tokenizer,
                    processor,
                    max_length,
                    apply_chat_template_kwargs=apply_chat_template_kwargs,
                ):
                    filtered_samples.append(sample)
                else:
                    logger.debug(f"Filtered sample exceeding max_length={max_length}")
    else:
        if isinstance(samples[0].prompt, list):
            filtered_samples = []
            for sample in samples:
                if check_sample_length(
                    sample,
                    tokenizer,
                    processor,
                    max_length,
                    apply_chat_template_kwargs=apply_chat_template_kwargs,
                ):
                    filtered_samples.append(sample)
        else:
            prompts = [sample.prompt for sample in samples]
            input_ids_list = tokenizer(prompts, add_special_tokens=False)["input_ids"]
            filtered_samples = [
                sample
                for sample, input_ids in zip(samples, input_ids_list, strict=True)
                if len(input_ids) <= max_length
            ]

    filtered_count = len(samples) - len(filtered_samples)
    if filtered_count > 0:
        logger.info(f"Filtered {filtered_count} samples longer than max_length={max_length}.")

    return filtered_samples


def _expand_directory(dir_path: str) -> list[str]:
    """Recursively find all supported data files (.jsonl, .parquet) under a
    directory.

    Files are sorted by full path to ensure deterministic ordering across runs.
    """
    supported_extensions = (".jsonl", ".parquet")
    files = []
    for root, _, filenames in os.walk(dir_path):
        for fname in filenames:
            if fname.endswith(supported_extensions):
                files.append(os.path.join(root, fname))
    files.sort()
    if not files:
        logger.warning(f"No supported data files (.jsonl, .parquet) found in directory: {dir_path}")
    else:
        logger.info(f"Found {len(files)} data files in directory: {dir_path}")
    return files


def _normalize_paths(path: Any) -> list[str]:
    def _as_list(value: Any) -> list[str]:
        if isinstance(value, (list, tuple)):
            items = [str(p) for p in value]
        else:
            items = [str(value)]
        # Expand directories into individual file paths
        expanded = []
        for p in items:
            if os.path.isdir(p):
                expanded.extend(_expand_directory(p))
            else:
                expanded.append(p)
        return expanded

    if not isinstance(path, str):
        return _as_list(path)

    s = path.strip()
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            parsed = ast.literal_eval(s)
            return _as_list(parsed)
        except (ValueError, SyntaxError):
            body = s[1:-1].strip()
            if body:
                return [p.strip().strip("\"'") for p in body.split(",") if p.strip()]
    return _as_list(path)


def resolve_path_plan(path: Any) -> tuple[list[str], Optional[slice]]:
    """Resolve a dataset path specification into physical file paths and an
    optional outer slice over the concatenated sample stream.

    Examples:
        "a.parquet@[0:4]" -> (["a.parquet"], slice(0, 4))
        "[a.parquet,b.parquet]" -> (["a.parquet", "b.parquet"], None)
        "[a.parquet,b.parquet]@[0:4]" -> (["a.parquet", "b.parquet"], slice(0, 4))
        "/data/dir" -> (["/data/dir/a.parquet", ...], None)
        "[/data/dir, b.parquet]@[0:100]" -> (["/data/dir/a.parquet", ... , "b.parquet"], slice(0, 100))
    """
    if isinstance(path, str):
        path_spec, row_slice = parse_generalized_path(path)
        return _normalize_paths(path_spec), row_slice
    return _normalize_paths(path), None


def _build_reader_for_path(path: str):
    path, row_slice = parse_generalized_path(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt dataset path '{path}' does not exist.")

    if path.endswith(".jsonl"):

        def jsonl_reader(p):
            with open(p, encoding="utf-8") as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON decode error at line {line_num}: {e}")
                        continue

        reader = jsonl_reader(path)
        if row_slice is not None:
            reader = itertools.islice(reader, row_slice.start, row_slice.stop, row_slice.step)
        return reader

    if path.endswith(".parquet"):
        if pq is None:
            raise ImportError("pyarrow is required for parquet support")

        def parquet_reader(p):
            pf = pq.ParquetFile(p)

            # Read row groups individually instead of using iter_batches().
            # iter_batches() creates chunked arrays for multi-row-group files,
            # which fails with ArrowNotImplementedError on nested types
            # (e.g. list<struct<...>>, struct<...>).
            for i in range(pf.metadata.num_row_groups):
                yield from pf.read_row_group(i).to_pylist()

        reader = parquet_reader(path)
        if row_slice is not None:
            reader = itertools.islice(reader, row_slice.start, row_slice.stop, row_slice.step)
        return reader

    raise ValueError(f"Unsupported file format: {path}. Supported formats are .jsonl and .parquet.")


def read_file(path: str):
    """Read data from JSONL or Parquet files.

    # Supports path with slice notation like 'file.jsonl@[0:100]'.
    # Also supports a list of files, e.g. '[file1.jsonl, file2.jsonl]' or '(file1.jsonl, file2.jsonl)'.
    # Also supports a directory, e.g. '/data/dir' '/data/dir@[0:100]'.
    # Also supports directories and files hybrid, e.g. '[/data/dir, a.parquet]@[0:100]'.

    Args:
        path: Path to data file (JSONL or Parquet)

    Yields:
        Parsed data dictionaries
    """
    paths, row_slice = resolve_path_plan(path)
    reader = itertools.chain.from_iterable(_build_reader_for_path(item) for item in paths)

    if row_slice is not None:
        logger.info("read_file paths=%s applying slice row_slice=%s", paths, row_slice)
        reader = itertools.islice(reader, row_slice.start, row_slice.stop, row_slice.step)

    yield from reader


class BaseDataset(abc.ABC):
    """Abstract base class for dataset implementations.

    Provides common interface for both eager-loading Dataset and lazy-loading
    StreamingDataset.
    """

    def __init__(
        self,
        tokenizer: Any,
        processor: Any,
        max_length: Optional[int],
        *,
        prompt_key: str = "text",
        multimodal_keys: Optional[dict] = None,
        label_key: Optional[str] = None,
        tool_key: Optional[str] = None,
        metadata_key: str = "metadata",
        system_prompt: Optional[str] = None,
        seed: int = 42,
        apply_chat_template: bool = False,
        apply_chat_template_kwargs: Optional[dict] = None,
        use_audio_in_video: bool = False,
        multimodal_config: MultimodalConfig = None,
        custom_prompt_func: Optional[Callable[[Any, dict], Any]] = None,
        teacher_prompt_key: Optional[str] = None,
        teacher_multimodal_keys: Optional[dict] = None,
    ):
        """Initialize base dataset configuration.

        Args:
            tokenizer: Tokenizer for processing text
            processor: Processor for multimodal inputs
            max_length: Maximum prompt length for filtering
            prompt_key: Key for prompt in data
            multimodal_keys: Mapping of multimodal types to data keys
            label_key: Key for labels in data
            tool_key: Key for tools in data
            metadata_key: Key for metadata in data
            system_prompt: System prompt key or content
            seed: Random seed for shuffling
            apply_chat_template: Whether to apply chat template
            apply_chat_template_kwargs: Additional kwargs for chat template
            use_audio_in_video: Whether to extract audio from video files for multimodal processing
        """
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length
        self.prompt_key = prompt_key
        self.multimodal_keys = multimodal_keys
        self.label_key = label_key
        self.tool_key = tool_key
        self.metadata_key = metadata_key
        self.system_prompt = system_prompt
        self.seed = seed
        self.apply_chat_template = apply_chat_template
        self.apply_chat_template_kwargs = apply_chat_template_kwargs or {}
        self.use_audio_in_video = use_audio_in_video
        self.multimodal_config = multimodal_config
        self.custom_prompt_func = custom_prompt_func
        self.teacher_prompt_key = teacher_prompt_key
        self.teacher_multimodal_keys = teacher_multimodal_keys

        self.epoch_id = -1

    @abc.abstractmethod
    def __len__(self) -> int:
        """Return total number of samples."""
        pass

    @abc.abstractmethod
    def __getitem__(self, idx: int) -> Sample:
        """Get a sample by index."""
        pass

    @abc.abstractmethod
    def shuffle(self, epoch_id: int) -> None:
        """Shuffle the dataset for a new epoch."""
        pass

    def _process_data(self, data: dict) -> Sample:
        """Process a raw data dictionary into a Sample.

        Uses shared utility function for consistency.
        """
        return process_raw_sample(
            data,
            self.tokenizer,
            self.processor,
            prompt_key=self.prompt_key,
            multimodal_keys=self.multimodal_keys,
            label_key=self.label_key,
            tool_key=self.tool_key,
            metadata_key=self.metadata_key,
            system_prompt=self.system_prompt,
            apply_chat_template=self.apply_chat_template,
            apply_chat_template_kwargs=self.apply_chat_template_kwargs,
            use_audio_in_video=self.use_audio_in_video,
            multimodal_config=self.multimodal_config,
            custom_prompt_func=self.custom_prompt_func,
            teacher_prompt_key=self.teacher_prompt_key,
            teacher_multimodal_keys=self.teacher_multimodal_keys,
        )
