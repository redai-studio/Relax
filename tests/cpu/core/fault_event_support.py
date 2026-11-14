# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU seams for full production orchestration and real local Ray/Serve.

Model construction, datasets, placement and external managers are substituted.
The Controller/Service bodies, health actor/checker, training scheduler, task
cancellation, component completion methods and RayTrainGroup wait methods run.
"""

import ast
import asyncio
import concurrent.futures
import copy
import os
import threading
import traceback
from argparse import Namespace
from enum import Enum, IntEnum
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import ray
from ray import serve

from relax.utils.async_utils import run as training_run
from relax.utils.async_utils import shutdown_async_loop
from relax.utils.fault_events import recovery
from relax.utils.fault_events.models import RecoveryEvidence
from relax.utils.health_system import HealthChecker, HealthManager
from relax.utils.logging_utils import get_logger


ROOT = Path(__file__).resolve().parents[3]
ROLES = SimpleNamespace(
    actor="actor", actor_fwd="actor_fwd", reference="reference", rollout="rollout", critic="critic"
)


def load_definitions(path, namespace, selected=None):
    tree = ast.parse((ROOT / path).read_text())
    nodes = [
        n
        for n in tree.body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and (selected is None or n.name in selected)
    ]
    module = ast.Module(body=ast.parse("from __future__ import annotations").body + nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(ROOT / path), "exec"), namespace)
    return namespace


def production_method(path, cls, method):
    tree = ast.parse((ROOT / path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    function = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == method)
    module = ast.Module(body=ast.parse("from __future__ import annotations").body + [function], type_ignores=[])
    ns = {"ray": ray}
    exec(compile(ast.fix_missing_locations(module), str(ROOT / path), "exec"), ns)
    return ns[method]


SEND = production_method("relax/components/actor.py", "Actor", "update_weights_fully_async")
RECV = production_method("relax/components/actor_fwd.py", "ActorFwd", "recv_weight_fully_async")
GROUP_SEND = production_method("relax/distributed/ray/actor_group.py", "RayTrainGroup", "update_weights_fully_async")
GROUP_RECV = production_method("relax/distributed/ray/actor_group.py", "RayTrainGroup", "recv_weight_fully_async")


@ray.remote(num_cpus=0)
class WeightGate:
    def __init__(self):
        self._release = asyncio.Event()
        self._entered = set()

    async def wait(self, role):
        self._entered.add(role)
        await self._release.wait()

    def entered(self):
        return sorted(self._entered)

    def release(self):
        self._release.set()


@ray.remote(num_cpus=0.05)
class CPUWeightWorker:
    def __init__(self, role, fail, gate):
        self.role, self.fail, self.gate = role, fail, gate

    def weight(self):
        if self.gate is not None:
            ray.get(self.gate.wait.remote(self.role))
        if self.fail:
            raise RuntimeError("injected CPU weight operation failure")

    def update_weights_fully_async(self, *args, **kwargs):
        self.weight()

    def recv_weight_fully_async(self, *args):
        self.weight()


class CPUTrainGroup:
    def __init__(self, role, fail, gate):
        self._actor_handlers = [CPUWeightWorker.remote(role, fail, gate)]

    def update_weights_fully_async(self, *args, **kwargs):
        return GROUP_SEND(self, *args, **kwargs)

    def recv_weight_fully_async(self, *args):
        return GROUP_RECV(self, *args)


@serve.deployment(
    ray_actor_options={"num_cpus": 0.1}, graceful_shutdown_timeout_s=0.1, graceful_shutdown_wait_loop_s=0.1
)
class CPUReplica:
    def __init__(self, healthy, role, fail_weights=False, gate=None):
        self.healthy, self.role = healthy, role
        self.step = 0
        self.done = asyncio.Event()
        self.actor_model = CPUTrainGroup(role, fail_weights, gate)

    def update_weights_fully_async(self, **kwargs):
        return SEND(self, **kwargs)

    def recv_weight_fully_async(self, **kwargs):
        return RECV(self, **kwargs)

    async def run(self):
        await self.done.wait()

    async def finish(self):
        self.done.set()

    def report_fault(self):
        return ray.get(self.healthy.report_error.remote(self.role, "CPU component fault"))

    def get_step(self):
        return self.step

    def set_step(self, step):
        self.step = step


class ControlledHealthManager(HealthManager):
    def start(self, on_unhealthy, on_fatal=None):
        self._checker = HealthChecker(
            self.status,
            on_unhealthy,
            on_fatal=on_fatal,
            check_interval=0.01,
            fault_events=self._fault_events,
        )


def cpu_controller_class(ray_adapter, serve_adapter, *, native=False):
    async def noop(*args, **kwargs):
        return None

    namespace = dict(globals())
    namespace.update(
        GlobalStartup=recovery.GlobalStartup,
        RecoveryEvidence=RecoveryEvidence,
        concurrent=concurrent,
        copy=copy,
        os=os,
        threading=threading,
        traceback=traceback,
        Enum=Enum,
        IntEnum=IntEnum,
        partial=partial,
        ray=ray_adapter,
        serve=serve_adapter,
        HealthManager=HealthManager if native else ControlledHealthManager,
        logger=get_logger("fault-cpu-controller"),
        GENRM_ROLE="genrm",
        ACTOR_ROLLOUT_PG_ROLES=[],
        _S3_MODEL_CLEANUP_SAFE_LOAD_FORMATS={"dummy"},
        require_control_plane_resource=lambda _: None,
        with_control_plane_affinity=lambda *args: {},
        resolve_sft_num_rollout=lambda _: None,
        create_dcs_deployment=lambda **kwargs: (None, "unused"),
        shutdown_async_loop=shutdown_async_loop if native else lambda: None,
        tq=SimpleNamespace(close=lambda: None),
        close_tq_owner=lambda _owner: None,  # Data-system initialization is also replaced below.
        recovery_load_path=lambda _: None,
        time=SimpleNamespace(sleep=lambda _: None),
        resolve_sft_algo_key=lambda _: "cpu",
        ALGOS={"cpu": {role: CPUReplica for role in ("actor", "actor_fwd", "reference", "advantages")}},
        register_extra_roles=lambda *args: None,
        run=training_run if native else asyncio.run,
        set_managed_opd_teacher_on_actor_service=noop,
    )
    cls = load_definitions("relax/core/controller.py", namespace)["Controller"]
    cls._initialize_data_system = lambda self: None
    cls._shutdown_agentic_rollout_services = lambda self: None
    cls._cleanup_s3_model_weights_after_init = lambda *args, **kwargs: None
    cls._report_error_to_metrics_service = lambda *args, **kwargs: None
    service_namespace = dict(globals())
    service_namespace.update(
        RecoveryEvidence=RecoveryEvidence,
        observe_recovery=recovery.observe_recovery,
        logger=get_logger("fault-cpu-service"),
        recovery_load_path=lambda _: None,
        time=SimpleNamespace(sleep=lambda _: None),
    )
    Service = load_definitions("relax/core/service.py", service_namespace)["Service"]

    def register(self):
        self._cpu_gate = WeightGate.remote() if getattr(self.config, "hold_weights", False) else None
        for role in self.config.cpu_roles:
            self._health_manager.mark_healthy(role)
            svc = object.__new__(Service)
            svc.role, svc.healthy, svc.config = role, self._health_manager.status, self.config
            svc._start_heartbeat = lambda: None
            svc._stop_heartbeat_thread = lambda: None
            svc._ensure_placement_group = lambda: None
            svc._task_ref = None

            def deploy(pgs=None, svc=svc):
                svc.handle = serve.run(
                    CPUReplica.bind(
                        svc.healthy, svc.role, getattr(self.config, "fail_weights", False), self._cpu_gate
                    ),
                    name=svc.role,
                    route_prefix=None,
                )

            def http(role, path, svc=svc, **kwargs):
                if path == "/get_step":
                    return {"step": 0}
                if path == "/set_step" and getattr(svc.config, "fail_step", False):
                    raise RuntimeError("injected step restore failure")
                return {"status": "ok"}

            svc._deploy, svc._http_call = deploy, http
            svc._deploy()
            self.serve_dict[role] = svc

    cls.register_all_serve = register
    return cls


def config(**kwargs):
    values = dict(
        use_health_check=True,
        max_global_restart=3,
        use_metrics_service=False,
        use_agentic_rollout=False,
        debug_train_only=False,
        debug_rollout_only=False,
        fully_async=True,
        hybrid=False,
        loss_type=None,
    )
    return Namespace(**{**values, **kwargs})
