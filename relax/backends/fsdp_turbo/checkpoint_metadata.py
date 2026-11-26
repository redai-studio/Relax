# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Checkpoint directory selection and recovery load policy without training
imports."""

import os


_LATEST = "latest_checkpointed_iteration.txt"
_DCP_METADATA = ".metadata"


def find_latest_complete_checkpoint(save_root: str) -> str | None:
    """Find a published step even if the root recovery pointer is stale."""
    if not os.path.isdir(save_root):
        return None
    with os.scandir(save_root) as entries:
        candidates = sorted(
            (
                (int(entry.name), entry.path)
                for entry in entries
                if entry.name.isdecimal() and entry.is_dir(follow_symlinks=False)
            ),
            reverse=True,
        )
    for step, path in candidates:
        marker = os.path.join(path, _LATEST)
        if not os.path.isfile(marker):
            continue
        with open(marker, encoding="utf-8") as file:
            if file.read().strip() != str(step):
                raise ValueError(f"Checkpoint completion marker does not match its step: {path}")
        has_model = any(os.path.isfile(os.path.join(path, name)) for name in ("model.safetensors", "model.bin"))
        has_state = os.path.isfile(os.path.join(path, _DCP_METADATA)) or has_model
        if not has_state or not os.path.isfile(os.path.join(path, "config.json")):
            raise ValueError(f"Published checkpoint is incomplete: {path}")
        return path
    return None


def _resolve_checkpoint_directory(path: str) -> str:
    if not any(os.path.isfile(os.path.join(path, name)) for name in (_DCP_METADATA, "model.safetensors", "model.bin")):
        latest = find_latest_complete_checkpoint(path)
        if latest is not None:
            path = latest
        else:
            marker = os.path.join(path, _LATEST)
            if os.path.isfile(marker):
                with open(marker, encoding="utf-8") as file:
                    path = os.path.join(path, file.read().strip())
    return path


def remember_initial_load_policy(args) -> None:
    if hasattr(args, "_turbo_initial_load_policy"):
        return
    args._turbo_initial_load_policy = {}
    if args.load is not None:
        selected = _resolve_checkpoint_directory(args.load)
        args._turbo_initial_load_policy[os.path.realpath(selected)] = args.no_load_optim


def checkpoint_recovery_load_policy(args, path: str) -> tuple[str, bool]:
    # Match aliases in the driver; only the logical path travels to other nodes.
    selected = _resolve_checkpoint_directory(path)
    return selected, args._turbo_initial_load_policy.get(os.path.realpath(selected), args.no_save_optim)
