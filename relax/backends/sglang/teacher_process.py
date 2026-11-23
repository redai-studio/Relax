# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import os


def launch_bound_teacher(server_args: dict, gpu_ids: list[int], parent_pid: int, reference_publication: bool) -> None:
    # Importing SGLang can probe CUDA. Bind this spawned process first.
    devices = [str(gpu) for gpu in gpu_ids]
    inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
    if inherited is not None and not set(devices).issubset(device.strip() for device in inherited.split(",")):
        raise ValueError(f"Teacher GPUs {devices} are outside CUDA_VISIBLE_DEVICES={inherited!r}")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices)

    from relax.backends.sglang.sglang_engine import _launch_server_with_patches

    _launch_server_with_patches({**server_args, "base_gpu_id": 0, "gpu_id_step": 1}, parent_pid, reference_publication)
