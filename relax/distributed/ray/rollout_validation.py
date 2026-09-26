def validate_server_group_gpu_indices(
    *,
    worker_type: str,
    gpu_offset: int,
    num_gpus_per_engine: int,
    num_gpu_per_engine: int,
    num_engines: int,
    num_available_gpus: int,
    rollout_num_gpus: int,
    rollout_num_gpus_per_engine: int,
) -> None:
    """Fail fast when the rollout engine layout would index past the placement
    group's GPU list.

    Without this, a misaligned config surfaces later as an opaque
    ``IndexError`` at ``reordered_gpu_ids[gpu_index]``; here it raises a
    ``ValueError`` carrying the relevant rollout config instead.
    """
    if num_engines == 0:
        return

    required_gpu_slots = gpu_offset + num_engines * num_gpu_per_engine
    if gpu_offset >= 0 and num_gpu_per_engine > 0 and required_gpu_slots <= num_available_gpus:
        return

    raise ValueError(
        "Invalid rollout server group GPU placement: "
        f"worker_type={worker_type}, "
        f"gpu_offset={gpu_offset}, "
        f"num_gpus_per_engine={num_gpus_per_engine}, "
        f"num_gpu_per_engine_on_node={num_gpu_per_engine}, "
        f"num_engines={num_engines}, "
        f"required_gpu_slots={required_gpu_slots}, "
        f"len(reordered_gpu_ids)={num_available_gpus}, "
        f"rollout_num_gpus={rollout_num_gpus}, "
        f"rollout_num_gpus_per_engine={rollout_num_gpus_per_engine}. "
        "Align --rollout-num-gpus, --rollout-num-gpus-per-engine, and --sglang-config server_groups."
    )
