# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import abc
import os
import random
from datetime import timedelta

import ray
import torch
import torch.distributed as dist

import relax.utils.training.eval_config
from relax.distributed.ray.ray_actor import RayActor
from relax.utils import device as device_utils
from relax.utils.device import device_module, ray_get_device_ids
from relax.utils.distributed_utils import init_gloo_group
from relax.utils.env import Envs
from relax.utils.logging_utils import get_logger
from relax.utils.memory_utils import clear_memory, print_memory
from relax.utils.memory_utils import set_role as set_memory_role


logger = get_logger(__name__)


def get_local_gpu_id():
    cvd = os.environ.get(device_utils.get_visible_devices_env_var(), None)
    device_ids = ray_get_device_ids()
    if cvd is None:
        return device_ids[0]
    else:
        return cvd.split(",").index(str(device_ids[0]))


class TrainRayActor(RayActor):
    def __init__(self, world_size, rank, master_addr, master_port, lock):
        self._world_size = world_size
        self._rank = rank
        self.lock = lock
        if master_addr:
            self.master_addr, self.master_port = master_addr, master_port
        else:
            self.master_addr, self.master_port = self._get_current_node_ip_and_free_port(
                start_port=random.randint(20000, 21000)
            )

        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = str(self.master_port)
        os.environ["WORLD_SIZE"] = str(self._world_size)
        os.environ["RANK"] = str(self._rank)
        # TODO: currently this doesn't work as ray has already set torch.cuda.device_count().
        # os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        # os.environ["LOCAL_RANK"] = str(ray.get_gpu_ids()[0])
        os.environ["LOCAL_RANK"] = str(get_local_gpu_id())

    def init(self, args, role, with_ref=False, with_opd_teacher=False):
        self.args = args
        self.role = role
        set_memory_role(role)
        self.with_ref = with_ref
        self.with_opd_teacher = with_opd_teacher

        torch.serialization.add_safe_globals([relax.utils.training.eval_config.EvalDatasetConfig])

        local_rank = Envs.LOCAL_RANK
        device_module.set_device(f"{device_utils.get_device_name()}:{local_rank}")

        backend = args.distributed_backend

        dist.init_process_group(
            backend=backend,
            timeout=timedelta(minutes=args.distributed_timeout_minutes),
        )
        init_gloo_group(distributed_timeout_minutes=args.distributed_timeout_minutes)

        args.rank = dist.get_rank()
        args.world_size = dist.get_world_size()

        numa_local_rank = Envs.RANK % args.num_gpus_per_node
        device_utils.set_numa_affinity(numa_local_rank)

    def clear_memory(self):
        print_memory("before TrainRayActor.clear_memory")
        clear_memory()
        print_memory("after TrainRayActor.clear_memory")

    @abc.abstractmethod
    def sleep(self, tags):
        raise NotImplementedError

    @abc.abstractmethod
    def wake_up(self, tags):
        raise NotImplementedError

    @abc.abstractmethod
    def train(self, rollout_id, rollout_data_ref):
        raise NotImplementedError

    @abc.abstractmethod
    def save_model(self, rollout_id, force_sync=False):
        raise NotImplementedError

    @abc.abstractmethod
    def update_weights(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _get_parallel_config(self):
        raise NotImplementedError

    def set_rollout_manager(self, rollout_manager):
        self.rollout_manager = rollout_manager
        if not self.args.debug_rollout_only and self.args.rank == 0:
            ray.get(self.rollout_manager.set_train_parallel_config.remote(self.train_parallel_config))
        # Retrieve the distributed lock that serialises DCS weight sync with
        # P2P direct sync (_sync_weights_from_seed_engine on RolloutManager).
        self._weight_sync_lock = ray.get(self.rollout_manager.get_weight_sync_lock.remote())

    def set_genrm_manager(self, genrm_manager):
        """Set the genRM manager for coordinated offload/onload.

        In colocated mode, the genRM manager is used to offload genRM engines
        before training and onload them before rollout, since they share GPU
        resources.
        """
        self.genrm_manager = genrm_manager

    def set_teacher_manager(self, teacher_manager):
        """Set the managed OPD teacher manager for coordinated
        offload/onload."""
        self.teacher_manager = teacher_manager
