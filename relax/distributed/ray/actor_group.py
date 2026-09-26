# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import os
from typing import Any

import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from relax.core.node_group_affinity import with_control_plane_affinity
from relax.distributed.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST, Lock
from relax.utils.env import Envs
from relax.utils.utils import get_ray_accelerator_kwargs


def _resolve_train_actor_class(args):
    """Return the train actor class for the configured ``--train-backend``.

    Defaults to Megatron so existing token-RL runs are unaffected; ``fsdp``
    selects the FSDP2 actor used by native generative RL (design doc 8.3).
    """
    backend = getattr(args, "train_backend", "megatron")
    if backend == "fsdp":
        from relax.backends.fsdp.actor import FSDPTrainRayActor

        return FSDPTrainRayActor
    from relax.backends.megatron.actor import MegatronTrainRayActor

    return MegatronTrainRayActor


class RayTrainGroup:
    """A group of ray actors Functions start with 'async' should return list of
    object refs.

    Args:
        args (Namespace): Arguments for the actor group.
        num_nodes (int): Number of nodes for this actor group.
        num_gpus_per_node (int): Number of gpus for this actor group.
        pg (PlacementGroup, optional): Placement group to schedule actor on.
            If none, create new placement group automatically. Defaults to None.
        num_gpus_per_actor (float, optional): Number of gpus allocated for each actor.
            If < 1.0, multiple models can share same gpu. Defaults to 1.
        resources (Dict[str, float], optional): Custom resources to allocate for each actor.
            See https://docs.ray.io/en/latest/ray-core/scheduling/resources.html
        num_resources_per_node (int, optional): Number of custom resources to allocate for each node.
            See https://docs.ray.io/en/latest/ray-core/scheduling/resources.html
    """

    def __init__(
        self,
        args,
        num_gpus,
        pg: tuple[PlacementGroup, list[int], list[int]],
        num_gpus_per_actor: float = 1,
        role: str = "actor",
        runtime_env: dict = None,
    ) -> None:
        self.args = args
        self._num_gpus = num_gpus
        self.role = role
        self.runtime_env = runtime_env
        # Allocate the GPUs for actors w/o instantiating them
        self._allocate_gpus_for_actor(pg, num_gpus_per_actor)

    def _allocate_gpus_for_actor(self, pg, num_gpus_per_actor):
        world_size = self._num_gpus

        # Use placement group to lock resources for models of same type
        assert pg is not None
        pg, reordered_bundle_indices, _reordered_gpu_ids = pg

        env_vars = {
            # because sglang will always set NCCL_CUMEM_ENABLE to 0
            # we need also set it to 0 to prevent nccl error.
            "NCCL_CUMEM_ENABLE": Envs.NCCL_CUMEM_ENABLE,
            "NVTE_FP8_BLOCK_SCALING_FP32_SCALES": Envs.NVTE_FP8_BLOCK_SCALING_FP32_SCALES,
            **{name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST},
            **self.runtime_env.get("env_vars", {}),
            **self.args.train_env_vars,
        }
        # Compiler cache paths are train-actor-only.  Putting them in the Job
        # runtime env would make Serve/TQ/SGLang processes pollute the cache.
        kernel_cache_local_dir = env_vars.get("RELAX_KERNEL_CACHE_LOCAL_DIR")
        if kernel_cache_local_dir:
            env_vars["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(kernel_cache_local_dir, "inductor")
            env_vars["TRITON_CACHE_DIR"] = os.path.join(kernel_cache_local_dir, "triton")

        # Only preload the torch_memory_saver hook when it is actually the offload
        # mechanism. --selective-offload uses application-level selective CPU offload
        # and must NOT LD_PRELOAD the TMS hook (its global allocator hook + whole-pool
        # CPU backup would defeat the purpose and confuse torch_memory_saver_preloaded()).
        if (
            self.args.offload_train
            and self.args.train_backend == "megatron"
            and not getattr(self.args, "selective_offload", False)
        ):
            import torch_memory_saver

            dynlib_path = os.path.join(
                os.path.dirname(os.path.dirname(torch_memory_saver.__file__)),
                "torch_memory_saver_hook_mode_preload.abi3.so",
            )
            assert os.path.exists(dynlib_path), f"LD_PRELOAD so file {dynlib_path} does not exist."

            env_vars["LD_PRELOAD"] = dynlib_path
            env_vars["TMS_INIT_ENABLE"] = "1"
            env_vars["TMS_INIT_ENABLE_CPU_BACKUP"] = "1"

        # We cannot do routing replay for critic.
        if self.args.use_routing_replay and self.role == "actor":
            env_vars["ENABLE_ROUTING_REPLAY"] = "1"

        actor_impl = _resolve_train_actor_class(self.args)

        TrainRayActor = ray.remote(runtime_env={"env_vars": env_vars})(actor_impl)
        lock = Lock.options(**with_control_plane_affinity(self.args, {"num_cpus": 1, "num_gpus": 0})).remote()

        # Create worker actors
        self._actor_handlers = []
        master_addr, master_port = None, None
        accelerator_kwargs = get_ray_accelerator_kwargs(num_gpus_per_actor)
        for rank in range(world_size):
            actor = TrainRayActor.options(
                num_cpus=num_gpus_per_actor,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=reordered_bundle_indices[rank],
                ),
                **accelerator_kwargs,
            ).remote(world_size, rank, master_addr, master_port, lock)
            if rank == 0:
                master_addr, master_port = ray.get(actor.get_master_addr_and_port.remote())
            self._actor_handlers.append(actor)

    def async_init(self, args, role, with_ref=False, with_opd_teacher=False):
        """Allocate GPU resourced and initialize model, optimzier, local ckpt,
        etc."""
        self.args = args
        return [
            actor.init.remote(args, role, with_ref=with_ref, with_opd_teacher=with_opd_teacher)
            for actor in self._actor_handlers
        ]

    def async_train(self, rollout_id):
        """Do one rollout training."""
        return [actor.train.remote(rollout_id) for actor in self._actor_handlers]

    def async_compute_ref_log_prob(self, rollout_id):
        """Compute reference log prob for routing replay."""
        return [actor.compute_ref_log_prob.remote(rollout_id) for actor in self._actor_handlers]

    def async_compute_actor_log_prob(self, rollout_id):
        """Compute actor log prob for routing replay."""
        return [actor.compute_actor_log_prob.remote(rollout_id) for actor in self._actor_handlers]

    def train_fully_async(self, rollout_id):
        """Do one rollout training without ref log prob computation."""
        return [actor.train_async.remote(rollout_id) for actor in self._actor_handlers]

    def train_hybrid(self, rollout_id):
        """Hybrid mode: actor handles ref/actor_fwd/adv internally."""
        return [actor.train_hybrid.remote(rollout_id) for actor in self._actor_handlers]

    def save_model(self, rollout_id, force_sync=False):
        """Save actor model."""
        ray.get([actor.save_model.remote(rollout_id, force_sync=force_sync) for actor in self._actor_handlers])

    def update_weights(self):
        """Broadcast weights from rank 0 to all other ranks."""
        ray.get([actor.update_weights.remote() for actor in self._actor_handlers])

    def update_weights_fully_async(self, rollout_id, rollout_only=False, actor_fwd_only=False) -> None:
        """Update weights in fully async mode (sends to rollout and
        actor_fwd)."""
        ray.get(
            [
                actor.update_weights_fully_async.remote(
                    rollout_id, rollout_only=rollout_only, actor_fwd_only=actor_fwd_only
                )
                for actor in self._actor_handlers
            ]
        )

    def recv_weight_fully_async(self, rollout_id) -> None:
        ray.get([actor.recv_weight_fully_async.remote(rollout_id) for actor in self._actor_handlers])

    def onload(self):
        ray.get([actor.wake_up.remote() for actor in self._actor_handlers])

    def offload(self):
        ray.get([actor.sleep.remote() for actor in self._actor_handlers])

    def clear_memory(self):
        ray.get([actor.clear_memory.remote() for actor in self._actor_handlers])

    def set_rollout_manager(self, rollout_manager: Any):
        ray.get([actor.set_rollout_manager.remote(rollout_manager) for actor in self._actor_handlers])

    def set_genrm_manager(self, genrm_manager: Any):
        """Set the genRM manager for coordinated offload/onload.

        In colocated mode, the genRM manager is used to offload genRM engines
        before training and onload them before rollout, since they share GPU
        resources.
        """
        ray.get([actor.set_genrm_manager.remote(genrm_manager) for actor in self._actor_handlers])
