# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import os
import socket
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import ray
import torch
from tensordict import TensorDict

from relax.utils.device import get_ray_accelerator_name
from relax.utils.env import Envs, validate_env
from relax.utils.logging_utils import get_logger
from relax.utils.misc import load_function
from relax.utils.training.ppo_utils import compute_rloo_leave_one_out_rewards
from relax.utils.types import Sample


logger = get_logger(__name__)
CURRENT_ROLLOUT_BATCH = []


def _extract_images_seqlens(multimodal_train_inputs) -> list[int]:
    """Extract per-image ViT token counts from multimodal_train_inputs.

    Accepts either:
      - ``list[dict | None]``: per-sample dicts (pre-batch format)
      - ``dict``: concatenated tensors (post-``prepare_batch`` format)

    For each image, the ViT input sequence length = H * W (repeated T times
    along the temporal axis).
    """
    if isinstance(multimodal_train_inputs, dict):
        grid_thw = multimodal_train_inputs.get("image_grid_thw")
        if grid_thw is None:
            return []
        if isinstance(grid_thw, torch.Tensor):
            seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
            return seqlens.tolist()
        return [int(h * w) for t, h, w in grid_thw for _ in range(int(t))]

    images_seqlens: list[int] = []
    for mm_input in multimodal_train_inputs:
        if mm_input is None:
            continue
        grid_thw = mm_input.get("image_grid_thw")
        if grid_thw is None:
            continue
        if isinstance(grid_thw, torch.Tensor):
            seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
            images_seqlens.extend(seqlens.tolist())
        elif isinstance(grid_thw, (list, np.ndarray)):
            for t, h, w in grid_thw:
                images_seqlens.extend([int(h * w)] * int(t))
    return images_seqlens


def _extract_audio_seqlens(multimodal_train_inputs) -> list[int]:
    """Extract per-audio raw mel feature lengths from multimodal_train_inputs.

    Accepts either:
      - ``list[dict | None]``: per-sample dicts (pre-batch format)
      - ``dict``: concatenated tensors (post-``prepare_batch`` format)

    Returns the effective mel frame count for each audio clip, derived from
    ``feature_attention_mask.sum(-1)``.
    """
    if isinstance(multimodal_train_inputs, dict):
        feat_mask = multimodal_train_inputs.get("feature_attention_mask")
        if feat_mask is None:
            return []
        if isinstance(feat_mask, torch.Tensor):
            return feat_mask.sum(-1).tolist()
        return torch.tensor(feat_mask).sum(-1).tolist()

    audio_seqlens: list[int] = []
    for mm_input in multimodal_train_inputs:
        if mm_input is None:
            continue
        feat_mask = mm_input.get("feature_attention_mask")
        if feat_mask is None:
            continue
        if isinstance(feat_mask, torch.Tensor):
            lengths = feat_mask.sum(-1)
            audio_seqlens.extend(lengths.tolist())
        elif isinstance(feat_mask, (list, np.ndarray)):
            feat_mask_t = torch.tensor(feat_mask)
            lengths = feat_mask_t.sum(-1)
            audio_seqlens.extend(lengths.tolist())
    return audio_seqlens


def convert_samples_to_train_data(args: Any, samples: list[Sample] | list[list[Sample]]):
    """Convert inference generated samples to training data."""
    raw_rewards, rewards = post_process_rewards(args, samples)

    assert len(raw_rewards) == len(samples)
    assert len(rewards) == len(samples)

    sample_indices = [sample.index for sample in samples]
    train_data = {
        "tokens": [sample.tokens for sample in samples],
        "response_lengths": [sample.response_length for sample in samples],
        # some reward model, e.g. remote rm, may return multiple rewards,
        # we could use key to select the reward.
        "rewards": rewards,
        "raw_reward": raw_rewards,
        # Semantic group index (prompt group for GRPO reward normalization); needed
        # by the per-rollout replay capture to recompute reward.post_process.
        "group_index": [sample.group_index if sample.group_index is not None else 0 for sample in samples],
        "truncated": [1 if sample.status == Sample.Status.TRUNCATED else 0 for sample in samples],
        "sample_indices": sample_indices,
    }

    # loss mask
    # TODO: compress the loss mask
    loss_masks = []
    for sample in samples:
        # always instantiate loss_mask if not provided
        if sample.loss_mask is None:
            sample.loss_mask = [1] * sample.response_length
        else:
            # NOTE(jiajia): loss_mask is not None only if args.mask_offpolicy_in_partial_rollout is True, so we need to pad it to response_length.
            sample.loss_mask += [1] * (sample.response_length - len(sample.loss_mask))

        assert len(sample.loss_mask) == sample.response_length, (
            f"loss mask length {len(sample.loss_mask)} != response length {sample.response_length}"
        )
        if sample.remove_sample:
            sample.loss_mask = [0] * sample.response_length
        loss_masks.append(sample.loss_mask)
    train_data["loss_masks"] = loss_masks
    mask_sums_by_sample_index: dict[Any, int] = {}
    for sample_index, loss_mask in zip(sample_indices, loss_masks, strict=True):
        mask_sums_by_sample_index[sample_index] = mask_sums_by_sample_index.get(sample_index, 0) + int(sum(loss_mask))
    train_data["sample_index_mask_sums"] = [mask_sums_by_sample_index[sample_index] for sample_index in sample_indices]

    # For rollout buffer
    if samples[0].metadata and "round_number" in samples[0].metadata:
        train_data["round_number"] = [sample.metadata["round_number"] for sample in samples]

    # Add rollout log probabilities for off-policy correction
    if samples[0].rollout_log_probs is not None:
        train_data["rollout_log_probs"] = [sample.rollout_log_probs for sample in samples]

    if samples[0].rollout_routed_experts is not None:
        train_data["rollout_routed_experts"] = [sample.rollout_routed_experts for sample in samples]

    if samples[0].train_metadata is not None:
        train_data["metadata"] = [sample.train_metadata for sample in samples]

    if args.multimodal_keys is not None:
        train_data["multimodal_train_inputs"] = [sample.multimodal_train_inputs for sample in samples]

    if args.use_opd:
        from relax.engine.rollout.on_policy_distillation import produce_opd_transfer_data

        produce_opd_transfer_data(args, samples, train_data)

    total_lengths = [len(t) for t in train_data["tokens"]]
    train_data["total_lengths"] = total_lengths
    if args.debug_train_only:
        return train_data
    rollout_batch = dict_to_tensordict(train_data, len(total_lengths))
    return rollout_batch


def build_rollout_custom_meta(rollout_batch: Any) -> list[dict[str, int]]:
    return [
        {"total_lengths": int(total_length), "sample_index": int(sample_index)}
        for total_length, sample_index in zip(
            rollout_batch["total_lengths"], rollout_batch["sample_indices"], strict=True
        )
    ]


def post_process_rewards(args: Any, samples: list[Sample] | list[list[Sample]]):
    """Return raw rewards and post-processed rewards consumed by training.

    Returns:
        Tuple[List[float], List[float]]
    """
    if args.custom_reward_post_process_path is not None:
        custom_reward_post_process_func = load_function(args.custom_reward_post_process_path)
        processed_rewards = custom_reward_post_process_func(args, samples)
        if isinstance(processed_rewards, tuple) and len(processed_rewards) == 2:
            return processed_rewards
        raw_rewards = [sample.get_reward_value(args) for sample in samples]
        return raw_rewards, processed_rewards

    raw_rewards = [sample.get_reward_value(args) for sample in samples]

    if getattr(args, "agentic_custom_advantage_path", None) is not None:
        return raw_rewards, [sample.custom_advantage for sample in samples]

    if (
        args.advantage_estimator in ["grpo", "gspo", "sapo", "cispo", "reinforce_plus_plus_baseline", "rloo"]
        and args.rewards_normalization
    ):
        # group norm
        rewards = torch.tensor(raw_rewards, dtype=torch.float)
        positions_by_group: dict[int, list[int]] = {}
        for position, sample in enumerate(samples):
            if sample.group_index is None:
                raise ValueError("Sample.group_index is required for group reward normalization.")
            if sample.group_index not in positions_by_group:
                positions_by_group[sample.group_index] = []
            positions_by_group[sample.group_index].append(position)

        normalized_rewards = torch.empty_like(rewards)
        for group_index, positions in positions_by_group.items():
            if len(positions) != args.n_samples_per_prompt:
                raise ValueError(
                    f"Reward group {group_index} has {len(positions)} samples, expected {args.n_samples_per_prompt}."
                )
            group_rewards = rewards[positions]
            if args.advantage_estimator == "rloo":
                finite_mask = torch.isfinite(group_rewards)
                if not finite_mask.all():
                    invalid_group_positions = (~finite_mask).nonzero(as_tuple=False).flatten().tolist()
                    invalid_sample_positions = [positions[position] for position in invalid_group_positions]
                    invalid_values = group_rewards[~finite_mask].tolist()
                    raise ValueError(
                        f"RLOO group_index={group_index} contains non-finite reward(s) at "
                        f"group position(s) {invalid_group_positions}, sample position(s) "
                        f"{invalid_sample_positions}: {invalid_values}."
                    )
                group_rewards = compute_rloo_leave_one_out_rewards(group_rewards)
            else:
                group_rewards = group_rewards - group_rewards.mean()
                if args.advantage_estimator in ["grpo", "gspo", "sapo", "cispo"] and args.grpo_std_normalization:
                    group_rewards = group_rewards / (group_rewards.std() + 1e-6)
            normalized_rewards[positions] = group_rewards

        return raw_rewards, normalized_rewards.tolist()

    return raw_rewards, raw_rewards


def dict_to_tensordict(
    data: Dict[str, List],
    batch_size: Union[int, torch.Size, None] = None,
    device: Optional[torch.device] = None,
) -> TensorDict:
    """Convert a nested-list / tensor-list dictionary to a TensorDict.

    Args:
        data: Mapping of keys to nested lists (supports depth 1 or 2) or
            lists of per-sample tensors.
        batch_size: Optional batch size. If None, caller may set an appropriate
            batch size (TensorDict accepts None or an int/torch.Size).
        device: Optional target torch.device for created tensors.

    Returns:
        A TensorDict built from the input nested lists.
    """
    if not data:
        return TensorDict({}, batch_size=0 if batch_size is None else batch_size, device=device)

    def _nesting_depth(x):
        if isinstance(x, list) and x:
            return 1 + _nesting_depth(x[0])
        return 0

    def _scalar_dtype(sample) -> Optional[torch.dtype]:
        """Return an explicit dtype only for bool/float; None lets torch.tensor
        infer."""
        if isinstance(sample, bool):
            return torch.bool
        if isinstance(sample, float):
            return torch.float32
        # int or mixed int/float: let torch.tensor auto-promote (C++ level, zero overhead)
        return None

    def _to_tensor_1d(lst):
        dtype = _scalar_dtype(lst[0])
        return torch.tensor(lst, dtype=dtype, device=device)

    def _to_tensor_2d(lst):
        dtype = _scalar_dtype(lst[0][0])
        tensors = [torch.tensor(seq, dtype=dtype, device=device) for seq in lst]
        return torch.nested.as_nested_tensor(tensors, layout=torch.jagged)

    def _to_nested_tensor_from_tensor_list(lst):
        if not all(isinstance(item, torch.Tensor) for item in lst):
            raise TypeError("Mixed tensor and non-tensor values are not supported")
        tensors = [tensor.to(device=device) for tensor in lst] if device is not None else lst
        return torch.nested.as_nested_tensor(tensors, layout=torch.jagged)

    result = {}

    for key, value in data.items():
        if not isinstance(value, list):
            raise TypeError(f"Value for key '{key}' must be a list, got {type(value)}")
        if key == "classification_labels":
            depth = _nesting_depth(value)
            if depth == 1:
                result[key] = torch.tensor(value, dtype=torch.long, device=device)
            elif depth == 2:
                result[key] = torch.tensor(value, dtype=torch.float32, device=device)
            else:
                raise ValueError(
                    "classification_labels must be scalar class ids or fixed-width multi-hot vectors; "
                    f"got nesting depth {depth}"
                )
            continue
        if key == "rollout_routed_experts":
            # Flatten 3D numpy (seq_i, num_layers, topk) -> 2D tensor (seq_i, num_layers*topk)
            # so NestedTensor jagged layout can handle variable seq_len efficiently.
            # This avoids NonTensorStack wrapping which forces slow pickle serialization
            # during dist.broadcast_object_list (~377 MB pickle -> ~14s overhead).
            tensors = [
                torch.from_numpy(np.ascontiguousarray(arr.reshape(arr.shape[0], -1))).to(torch.int32) for arr in value
            ]
            result[key] = torch.nested.as_nested_tensor(tensors, layout=torch.jagged)
            continue
        if value and isinstance(value[0], torch.Tensor):
            result[key] = _to_nested_tensor_from_tensor_list(value)
            continue
        depth = _nesting_depth(value)
        if depth == 0:  # empty list []
            tensor = torch.empty(0)
        elif depth == 1:
            if key == "multimodal_train_inputs":
                tensor = value
            else:
                tensor = _to_tensor_1d(
                    value,
                )
        elif depth == 2:
            tensor = _to_tensor_2d(
                value,
            )
        else:
            raise ValueError(f"Unsupported nesting depth {depth} for key '{key}'. Max supported: 2.")

        result[key] = tensor

    return TensorDict(result, batch_size=batch_size, device=device)


def _resolve_to_ip(addr: str) -> str:
    """Resolve *addr* to an IPv4/IPv6 address string.

    If *addr* is already a valid IP literal, return it unchanged. Otherwise,
    treat it as a hostname and resolve it via DNS. Falls back to
    ``"127.0.0.1"`` if resolution fails.
    """
    import ipaddress as _ipaddress

    # Fast path: addr is already an IP literal
    try:
        _ipaddress.ip_address(addr)
        return addr
    except ValueError:
        pass

    # addr is a hostname — resolve it
    try:
        return socket.gethostbyname(addr)
    except socket.gaierror:
        logger.warning("Failed to resolve hostname %r to IP; falling back to 127.0.0.1", addr)
        return "127.0.0.1"


def post_process_env(args, env):
    """Set and return environment variables required for rollout workers.

    Populates common env keys used by the rollout processes.
    """
    cur_dir = Path(__file__).resolve().parent
    repo_dir = cur_dir.parent.parent

    if "env_vars" not in env or not isinstance(env["env_vars"], dict):
        env["env_vars"] = {}

    # Dynamic-batch streaming ends via the producer's is_last signal, not a
    # pre-allocated partition, so pre-allocate the minimum (1) and let it grow.
    # The non-dynamic path still pre-allocates the exact count for its .all() check.
    if getattr(args, "fully_async", False) and getattr(args, "use_dynamic_batch_size", False):
        env["env_vars"]["TQ_PRE_ALLOC_SAMPLE_NUM"] = str(
            args.rollout_batch_size * args.n_samples_per_prompt
        )  ## * args.max_num_agents
    else:
        batch_size_for_capacity = (
            args.over_sampling_batch_size
            if args.partial_rollout and args.use_dynamic_global_batch_size
            else args.rollout_batch_size
        )
        env["env_vars"]["TQ_PRE_ALLOC_SAMPLE_NUM"] = str(batch_size_for_capacity * args.n_samples_per_prompt)
    env["env_vars"]["TQ_ZERO_COPY_SERIALIZATION"] = "true"
    env["env_vars"]["SLIME_HOST_IP"] = _resolve_to_ip(Envs.MASTER_ADDR)

    if Envs.RAY_DEBUG == "1":
        env["env_vars"]["RAY_DEBUG_POST_MORTEM"] = "1"
        env["env_vars"]["RAY_DEBUG"] = "1"

    # Propagate PYTHONPATH to Ray workers so external packages (e.g. Megatron-LM)
    # are available in Serve replicas and remote actors.
    python_paths = [str(repo_dir)]
    if pp := os.environ.get("PYTHONPATH"):
        python_paths += pp.split(":")
    if pp := env["env_vars"].get("PYTHONPATH"):
        python_paths += pp.split(":")

    # deduplicate with order
    python_paths = list(dict.fromkeys(python_paths))

    env["env_vars"]["PYTHONPATH"] = ":".join(python_paths)

    # Propagate the extension-module hook so every Ray actor that loads
    # ``relax.backends.megatron`` re-runs the imports listed here (analogue
    # of ``--custom-generate-function-path``). Downstream packages register
    # Megatron-Bridge converters / family-token tables this way.
    extra_modules = Envs.RELAX_EXTRA_MODULES
    if extra_modules and "RELAX_EXTRA_MODULES" not in env["env_vars"]:
        env["env_vars"]["RELAX_EXTRA_MODULES"] = extra_modules

    # Producer and consumer derive the same TransferQueue partition names from
    # this value, so it must be identical in every Serve and Megatron actor.
    if "RELAX_SFT_TQ_SHARDS" not in env["env_vars"]:
        env["env_vars"]["RELAX_SFT_TQ_SHARDS"] = str(Envs.RELAX_SFT_TQ_SHARDS)

    # Generic env-var passthrough for overlay packages. Comma-separated list
    # of env-var names the driver wants forwarded to every Ray actor. Each
    # name is copied from the driver's os.environ; missing names are
    # silently skipped.
    propagate_list = Envs.RELAX_PROPAGATE_ENV_VARS
    for var in propagate_list.split(","):
        var = var.strip()
        if not var or var in env["env_vars"]:
            continue
        val = os.environ.get(var)
        if val is not None:
            env["env_vars"][var] = val

    logger.info(f"Ray runtime env: {env['env_vars']}")

    # Last point where both env sources are merged and still on the driver:
    # reject unparseable declared variables here, where a single traceback is
    # readable, rather than letting every worker fail independently later. Also
    # warns about unregistered RELAX_*/SLIME_* names, which have no effect.
    validate_env(env["env_vars"])

    return env


def merge_dict_list(dict_list):
    """Merge a list of (dict, something) pairs into a single dict of lists.

    Each input item is expected to be a (dict, <unused>) tuple. For each key,
    values that are list/tuple are extended, otherwise appended.

    Args:
        dict_list: Iterable of (dict, any) pairs.

    Returns:
        A dict mapping keys to lists of aggregated values.
    """
    merged: Dict[str, List[Any]] = {}
    for d, _ in dict_list:
        for key, value in d.items():
            # ensure target key maps to a list
            if key not in merged:
                merged[key] = []
            # extend if iterable (list/tuple) and not a string/bytes, else append
            if isinstance(value, (list, tuple)) and not isinstance(value, (str, bytes)):
                merged[key].extend(value)
            else:
                merged[key].append(value)
    return merged


def get_debug_data(args, rollout_id: int, batch_size, dp_rank: int) -> Dict[str, Any]:
    """Fetch debug data for a given rollout_id from the data system.

    Parameters:
        rollout_id: The rollout ID for which to fetch debug data.
    Returns:
        A dictionary containing the debug data for the specified rollout ID.
    """

    data = torch.load(
        open(args.load_debug_rollout_data.format(rollout_id=rollout_id), "rb"),
        weights_only=False,
    )["samples"]
    data = [Sample.from_dict(sample) for sample in data]
    if (ratio := args.load_debug_rollout_data_subsample) is not None:
        original_num_rows = len(data)
        if (
            args.custom_reward_post_process_path is None
            and args.advantage_estimator in ["grpo", "gspo", "sapo", "cispo", "reinforce_plus_plus_baseline", "rloo"]
            and args.rewards_normalization
        ):
            group_ids = list(dict.fromkeys(sample.group_index for sample in data))
            if None in group_ids:
                raise ValueError("Sample.group_index is required for group-aware debug subsampling.")
            subsample_group_count = min(max(int(len(group_ids) * ratio), 1), len(group_ids))
            front_group_count = subsample_group_count // 2
            selected_group_ids = set(group_ids[:front_group_count])
            selected_group_ids.update(group_ids[-(subsample_group_count - front_group_count) :])
            data = [sample for sample in data if sample.group_index in selected_group_ids]
        else:
            rough_subsample_num_rows = int(original_num_rows * ratio)
            data = data[: rough_subsample_num_rows // 2] + data[-rough_subsample_num_rows // 2 :]
        logger.info(
            f"Subsample loaded debug rollout data using {ratio=} and change num rows {original_num_rows} -> {len(data)}"
        )
    rows_by_sample_index: dict[int, list[Sample]] = {}
    for sample in data:
        rows_by_sample_index.setdefault(sample.index, []).append(sample)
    sample_indices = list(rows_by_sample_index)
    selected_sample_indices = sample_indices[dp_rank * batch_size : (dp_rank + 1) * batch_size]
    selected_rows = [
        sample for sample_index in selected_sample_indices for sample in rows_by_sample_index[sample_index]
    ]
    return convert_samples_to_train_data(args, selected_rows)


async def transfer_batch_to_data_system(
    args: Namespace,
    batch_samples: List,
    batch_count: int,
    rollout_id: int,
    data_system_client: Any,
    is_last: bool = False,
) -> None:
    """Helper function to transfer a batch of samples to the data system
    client.

    Args:
        batch_samples: List of sample groups
        batch_count: Batch sequence number
        rollout_id: Rollout identifier
        data_system_client: Client for async data transfer
        is_last: Mark this as the final batch of the partition train_{rollout_id}
            so the data system can detect streaming end-of-stream without a preset
            global batch size. See the is_last bookkeeping in generate_rollout.
    """
    try:
        # Guard against empty batch_samples
        if not batch_samples:
            logger.warning(
                f"transfer_batch_to_data_system called with empty batch_samples for rollout_id={rollout_id}, batch_count={batch_count}"
            )
            return
        batch_samples = sorted(
            batch_samples, key=lambda group: group[0][0].index if isinstance(group[0], list) else group[0].index
        )
        # Flatten nested groups of samples into a single list
        while isinstance(batch_samples[0], list):
            batch_samples = sum(batch_samples, [])
        global CURRENT_ROLLOUT_BATCH
        CURRENT_ROLLOUT_BATCH.extend(batch_samples)
        rollout_batch = convert_samples_to_train_data(args, batch_samples)
        logger.info(f"Prepared rollout batch {batch_count} with {rollout_batch.numel()} samples for transfer")
        logger.info(f"Transferring batch rollout_batch: {rollout_batch}")

        # Pass sampler metadata inline so it lands atomically with ready samples.
        custom_meta = build_rollout_custom_meta(rollout_batch)
        await data_system_client.async_put(
            data=rollout_batch, partition_id=f"train_{rollout_id}", custom_meta=custom_meta, is_last=is_last
        )

        logger.info(f"Batch {batch_count} transferred successfully for rollout_id: {rollout_id}")
    except Exception as e:
        logger.error(f"Error transferring batch {batch_count}: {e}")
        raise


def process_args(args: Namespace, role: str) -> None:
    """Process args for reference actor and actor fwd."""
    # Adjust max tokens per GPU for reference actor and actor fwd
    if args.ref_actor_config is not None:
        for key in args.ref_actor_config:
            setattr(args, key, args.ref_actor_config[key])
    args.max_tokens_per_gpu = args.log_probs_max_tokens_per_gpu
    args.only_load_weight = True
    if role == "reference":
        args.load = args.ref_load
    if role == "actor_fwd":
        args.load = args.ref_load


def get_serve_url(route_prefix: str = "") -> str:
    """Return an accessible HTTP URL for the current Ray Serve deployment.

    Notes:
        - Call after `serve.run()` from a client that can reach the Ray
          cluster (typically the head node).
        - Returns a URL like: http://<head-node-ip>:<http-port><route_prefix>

    Args:
        route_prefix: Optional route prefix to append to the base URL.
    """
    # 1. Determine head node IP. Prefer Ray cluster state; fall back to
    #    local hostname resolution for client-on-head scenarios.
    try:
        # ray.nodes() returns info for all nodes. Ray 2.x auto-registers
        # "node:__internal_head__" on the head node; some legacy setups also
        # mark it with a custom "head" resource. Accept either.
        for node in ray.nodes():
            if not node["Alive"]:
                continue
            resources = node.get("Resources", {})
            if "node:__internal_head__" in resources or resources.get("head"):
                head_ip = node["NodeManagerAddress"]
                break
        else:
            # If no head marker, fall back to the first alive node
            head_ip = ray.nodes()[0]["NodeManagerAddress"]
    except Exception:
        # Fallback: resolve local hostname (works when client runs on head)
        head_ip = socket.gethostbyname(socket.gethostname())

    # 2. 格式化 route_prefix
    if route_prefix and not route_prefix.startswith("/"):
        route_prefix = "/" + route_prefix

    serve_url = f"http://{head_ip}:{8000}{route_prefix}"
    logger.info("Serve URL: %s", serve_url)
    return serve_url


def recovery_load_path(args: Namespace) -> Optional[str]:
    """Determine the checkpoint path to load for recovery, if applicable."""
    if args.save is not None and os.path.exists(os.path.join(args.save, "latest_checkpointed_iteration.txt")):
        args.no_load_optim = args.no_save_optim
        args.no_load_rng = args.no_save_rng
        args.finetune = False
        args.start_rollout_id = None
        args.load = args.save


def compute_dp_size(config) -> int:
    """Compute data-parallel size from config for the actor role.

    For Megatron backend: dp_size = total_actor_gpus / (tp * pp * cp)
    """
    _, actor_total_gpus = config.resource.get("actor", (1, 1))
    tp = getattr(config, "tensor_model_parallel_size", 1)
    pp = getattr(config, "pipeline_model_parallel_size", 1)
    cp = getattr(config, "context_parallel_size", 1)
    dp_size = actor_total_gpus // (tp * pp * cp)
    if dp_size <= 0:
        raise ValueError(
            f"Computed dp_size={dp_size} is invalid. actor_total_gpus={actor_total_gpus}, tp={tp}, pp={pp}, cp={cp}"
        )
    return dp_size


def get_ray_accelerator_kwargs(num_accelerator: int | float) -> Dict:
    accelerator_kwargs = {}
    accelerator_name = get_ray_accelerator_name()
    if accelerator_name == "GPU":
        accelerator_kwargs["num_gpus"] = num_accelerator
    else:
        accelerator_kwargs["resources"] = {"NPU": num_accelerator}
    return accelerator_kwargs
