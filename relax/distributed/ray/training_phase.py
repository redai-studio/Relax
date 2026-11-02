# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Collective receipts for training's use of shared inference bundles."""

from functools import wraps
from typing import Any, Callable

import ray
import torch.distributed as dist

from relax.engine.inference.phase_client import _CURRENT, PhaseSession
from relax.utils.distributed_utils import get_gloo_group


def begin_training_phase(actor: Any) -> PhaseSession | None:
    authority = getattr(actor.args, "_inference_authority", None)
    if authority is None:
        return None
    group = get_gloo_group()
    rank = dist.get_rank(group)
    receipt = [None, None]
    session = getattr(actor, "_training_session", None)
    if rank == 0:
        try:
            if session is None or session.closed:
                claims = ray.get(authority.claims_for.remote(actor.role))
                if claims:
                    session = PhaseSession(authority, actor.role, f"train:{actor.role}")
                    session.lease = ray.get(
                        authority.acquire.remote(session.owner, session.operation_id, claims, role=actor.role)
                    )
                    session.switch("train")
            receipt[0] = session.lease if session is not None and not session.closed else None
        except Exception as error:
            receipt[1] = f"{type(error).__name__}: {error}"
    # All ranks observe both the grant and any failed handoff before wake_up.
    dist.broadcast_object_list(receipt, src=0, group=group)
    if receipt[1]:
        raise RuntimeError(f"Training ownership handoff failed: {receipt[1]}")
    if receipt[0] is None:
        return None
    if rank != 0:
        session = PhaseSession(authority, actor.role, f"train:{actor.role}", lease=receipt[0])
    actor._training_session = session
    return session


def release_training_phase(actor: Any) -> None:
    """Every rank must acknowledge its local GPU release before handoff."""
    session = getattr(actor, "_training_session", None)
    if session is None or session.closed:
        return
    group = get_gloo_group()
    resident = [None] * dist.get_world_size(group)
    dist.all_gather_object(resident, bool(getattr(actor, "_inference_train_resident", True)), group=group)
    if any(resident):
        # Resident SFT keeps the same lease across steps until predict/offload.
        return
    receipt = [None]
    if dist.get_rank(group) == 0:
        try:
            session.close()
        except Exception as error:
            receipt[0] = f"{type(error).__name__}: {error}"
    dist.broadcast_object_list(receipt, src=0, group=group)
    if receipt[0]:
        raise RuntimeError(f"Training release was not confirmed: {receipt[0]}")
    session.closed = True


def resume_training_phase(actor: Any) -> None:
    """Replace the context after an SFT predict cycle returned its old lease.

    The enclosing ``leased_training`` scope restores the original context on
    exit; subsequent operations in this call must use the newly granted lease.
    """
    _CURRENT.set(begin_training_phase(actor))


def leased_training(function: Callable) -> Callable:
    @wraps(function)
    def run(actor: Any, *args: Any, **kwargs: Any) -> Any:
        parent = _CURRENT.get()
        if parent is not None and not parent.closed:
            return function(actor, *args, **kwargs)
        session = begin_training_phase(actor)
        token = _CURRENT.set(session)
        try:
            if (
                function.__name__ == "train"
                and session is not None
                and not getattr(actor, "_per_step_rollout", True)
                and not getattr(actor, "_inference_train_resident", True)
            ):
                actor.wake_up()
            result = function(actor, *args, **kwargs)
            if (
                function.__name__ == "train"
                and args
                and args[0] + 1 == getattr(actor.args, "num_rollout", None)
                and getattr(actor.args, "offload_train", False)
                and getattr(actor, "_inference_train_resident", False)
            ):
                actor.sleep()
            release_training_phase(actor)
            return result
        except BaseException as error:
            # No collective here: a peer may still be inside NCCL. Retain the
            # lease and quarantine it; a timeout is not a release receipt.
            active = getattr(actor, "_training_session", session)
            if active is not None and not active.closed:
                active.suspect(f"Training failed without all-rank release: {error}")
            raise
        finally:
            _CURRENT.reset(token)

    return run
