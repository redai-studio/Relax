# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import threading
import time
from argparse import Namespace
from typing import Any, Dict, Optional

import ray
import transfer_queue as tq
from fastapi import FastAPI, HTTPException
from ray import serve

from relax.components.base import Base
from relax.distributed.coordination import PeerStepBarrier, RolloutOffloadBarrier
from relax.distributed.ray.placement_group import allocate_train_group
from relax.engine.sft.runtime import is_sft_mode, sft_partition_ids, sft_task_name
from relax.utils.async_utils import run
from relax.utils.opd.opd_utils import set_managed_opd_teacher_on_train_group


app = FastAPI()


def _resolve_start_rollout_id(
    configured_step: Optional[int],
    backend_step: int,
    *,
    explicitly_set: bool,
) -> int:
    """Resolve the service step after the backend has loaded its checkpoint."""
    if configured_step is None:
        return backend_step
    if not explicitly_set and configured_step == 0 and backend_step > 1:
        return backend_step
    return configured_step


@serve.deployment(max_ongoing_requests=10, max_queued_requests=20)
@serve.ingress(app)
class Actor(Base):
    """Actor service for training the policy model.

    Supports two execution modes:
    - fully_async: Asynchronous training without waiting for rollout data
    - sync: Waits for rollout data before each training step
    """

    def __init__(
        self,
        healthy: Any,
        pgs: Any,
        num_gpus: int,
        config: Namespace,
        role: str,
        runtime_env: dict = None,
    ) -> None:
        super().__init__()

        self.config = config
        self._lock = threading.RLock()
        self.healthy = healthy
        self.role = role
        self.genrm_manager = None  # Set later via set_genrm_manager if genRM is enabled

        # Threading primitives so the training loop doesn't block the Serve/FastAPI thread
        self._stop_event = threading.Event()
        self._run_thread = None
        self._done_event: Optional[asyncio.Event] = None
        self._thread_error: Optional[Exception] = None

        self.actor_model = allocate_train_group(args=config, num_gpus=num_gpus, pg=pgs, runtime_env=runtime_env)

        tq.init(self.config.tq_config)
        self.data_system_client = tq.get_client()

        self._lora_profile = None
        self._lora_exports: dict[str, dict] = {}
        self._lora_export_pending = None
        self._lora_base_digest = None
        if getattr(config, "lora_publication_config", None):
            from uuid import uuid4

            from relax.engine.lora.publication import PublicationConfig

            self._lora_profile = PublicationConfig.read(config.lora_publication_config)
            self._lora_profile.validate_args(config)
            self._lora_export_run = uuid4().hex

        self.step = config.start_rollout_id or 0
        self._lora_bootstrap_export = None
        initial_export = None
        if self._lora_profile is not None and self._lora_profile.bootstrap_version_id is None:
            intent = "bootstrap-" + self._lora_export_run
            self._accept_lora_export({"request_id": intent, "version_id": intent, "publish": True})
            initial_export = self._prepare_lora_export(self.step)

        self.steps = ray.get(
            self.actor_model.async_init(
                config,
                role=self.role,
                with_ref=config.kl_coef != 0 or config.use_kl_loss,
                with_opd_teacher=self.config.opd_teacher_load,
                **({"lora_export": initial_export} if initial_export is not None else {}),
            )
        )

        assert len(set(self.steps)) == 1
        configured_step = self.config.start_rollout_id
        self.config.start_rollout_id = _resolve_start_rollout_id(
            configured_step,
            self.steps[0],
            explicitly_set=getattr(self.config, "_start_rollout_id_explicit", configured_step is not None),
        )
        if configured_step is not None and configured_step != self.config.start_rollout_id:
            self._logger.warning(
                "Checkpoint backend restored step %s; overriding auto-derived start_rollout_id=%s",
                self.config.start_rollout_id,
                configured_step,
            )
        self.step = self.config.start_rollout_id
        self._logger.info(f"Actor initialized with starting step {self.step}")
        if initial_export is not None:
            self._lora_bootstrap_export = self._finish_lora_export(submit_publication=False)

        # Wired by controller in colocate mode. In fully_async / hybrid both
        # stay ``None`` and every barrier-guarded site short-circuits.
        self._rollout_barrier: Optional[RolloutOffloadBarrier] = None
        self._peer_barrier: Optional[PeerStepBarrier] = None

    def set_rollout_manager(self, rollout_manager: Any) -> None:
        """Set the rollout manager and initialize weights."""
        self.rollout_manager = rollout_manager
        self.actor_model.set_rollout_manager(self.rollout_manager)
        self._bootstrap_lora_publication()

        # Call update_weights when weight_updater exists (sync colocate or hybrid mode).
        # In pure fully_async mode weight_updater is not created and weights are synced via DCS.
        # SFT: skip the init-time weight sync. SFT only sync weights to SGLang
        # right before periodic predict (gated in `train_actor`); between
        # predicts SGLang stays fully offloaded. Sync-at-init would leave
        # SGLang with `weights` resumed but no follow-up offload, causing
        # the first predict-step `onload_weights` to crash on a non-idempotent
        # `set.remove`. NCCL group setup is lazy — `connect_rollout_engines`
        # fires on the first real `update_weights` instead.
        if (
            not self._lora_profile
            and (not self.config.fully_async or self.config.hybrid)
            and not is_sft_mode(self.config)
        ):
            self.actor_model.update_weights()

    def set_barriers(
        self,
        *,
        rollout: Optional[RolloutOffloadBarrier] = None,
        peers: Optional[PeerStepBarrier] = None,
    ) -> None:
        self._rollout_barrier = rollout
        self._peer_barrier = peers

    def set_genrm_manager(self, genrm_manager: Any) -> None:
        """Set the genRM manager(s) for coordinated offload/onload.

        ``genrm_manager`` is a list of manager handles -- one per genRM
        instance (a single-instance config still passes a one-element list). In
        colocated mode, they are used to offload genRM engines before training
        and onload them before rollout, since they share GPU resources.
        """
        self.genrm_manager = genrm_manager
        self.actor_model.set_genrm_manager(self.genrm_manager)
        self._logger.info("GenRM manager(s) set on Actor for coordinated offload/onload")

    def set_teacher_manager(self, teacher_manager: Any) -> None:
        """Set the managed OPD teacher manager for coordinated
        offload/onload."""
        set_managed_opd_teacher_on_train_group(self.actor_model, teacher_manager)
        self._logger.info("Teacher manager set on Actor for coordinated offload/onload")

    def update_weights_fully_async(self, rollout_only: bool = False, actor_fwd_only: bool = False) -> None:
        self.actor_model.update_weights_fully_async(0, rollout_only=rollout_only, actor_fwd_only=actor_fwd_only)

    async def run(self) -> None:
        """Start the training loop in a background thread and async-wait until
        it completes.

        Uses an asyncio.Event so that this coroutine yields control back to the
        Ray Serve event loop while waiting. This keeps the Serve replica
        responsive to concurrent HTTP requests while the long-running training
        loop executes in a background thread.
        """
        if self._run_thread is not None and self._run_thread.is_alive():
            if self._done_event is not None:
                await self._done_event.wait()
            return
        for partition_id in sft_partition_ids(self.config, self.step):
            self.data_system_client.reset_consumption(
                partition_id=partition_id,
                task_name=sft_task_name(self.config, component="actor"),
            )
        # Create an asyncio.Event bound to the current event loop so the
        # background thread can signal completion without blocking the loop.
        loop = asyncio.get_running_loop()
        self._done_event = asyncio.Event()

        def _thread_target():
            try:
                self._background_run()
            except Exception as exc:
                self._thread_error = exc
            finally:
                # Thread-safe way to set the asyncio event from a non-async context
                loop.call_soon_threadsafe(self._done_event.set)

        self._thread_error = None
        self._run_thread = threading.Thread(target=_thread_target, daemon=True)
        self._run_thread.start()
        # Async-wait: yields control so other requests can be served
        await self._done_event.wait()
        if self._thread_error is not None:
            raise self._thread_error

    def _background_run(self) -> None:
        """The actual training loop running in a background thread.

        This is a near-direct translation of the original run() logic but uses
        thread-safe access to `self.step` and respects a stop event.
        """

        try:
            while True:
                if self._stop_event.is_set():
                    self._logger.info("Actor background loop stopping by request")
                    break

                with self._lock:
                    local_step = self.step

                if local_step >= self.config.num_rollout:
                    self._logger.info("All training steps finished")
                    break

                if not self.config.fully_async and self.config.colocate and not self.config.debug_train_only:
                    if not self._wait_for_rollout_data():
                        continue

                self._logger.info(f"Actor training step {local_step}/{self.config.num_rollout}")
                did_train = self._execute_training()

                self._logger.info(f"Actor training completed step {local_step}/{self.config.num_rollout}")

                if did_train:
                    for partition_id in sft_partition_ids(self.config, local_step):
                        run(self.data_system_client.async_clear_partition(partition_id=partition_id))
                    self._logger.info(f"Actor cleared data for step {local_step}/{self.config.num_rollout}")

                try:
                    self.healthy.update_heartbeat.remote("actor", local_step + 1)
                except Exception:
                    pass

                # increment step with lock
                with self._lock:
                    self.step += 1
                    if self.step >= self.config.num_rollout and self._lora_export_pending is not None:
                        if self._lora_export_pending["state"] == "WAITING_BOUNDARY":
                            self._lora_export_pending["state"] = "TRAINING_FINISHED"
                            self._lora_export_pending = None

        except Exception as e:
            error_msg = f"Actor training failed at step {self.step}: {type(e).__name__}: {str(e)}"
            self._logger.exception(error_msg)
            self.healthy.report_error.remote("actor", error_msg)
            if not getattr(self.config, "use_health_check", False):
                raise

    def _wait_for_rollout_data(self) -> bool:
        """Wait for rollout data to be ready in async colocate mode.

        Returns:
            True if data is ready and training can proceed,
            False if should continue waiting (caller should skip this iteration)
        """
        partition_ids = sft_partition_ids(self.config, self.step)
        partition_list = run(self.data_system_client.async_get_partition_list())
        if partition_list is None or any(partition_id not in partition_list for partition_id in partition_ids):
            time.sleep(1)
            return False

        # Colocate: block until rollout(SGLang) has offloaded so wake_up/onload
        # of actor weights doesn't collide with SGLang's static KV pool.
        # SFT skips the barrier — its rollout is a passive HTTP server driven
        # by /predict, no async GPU contention.
        if is_sft_mode(self.config):
            return True
        if self.config.offload_rollout and self._rollout_barrier is not None:
            self._rollout_barrier.wait_offloaded_sync()

        # PPO colocate: also block until critic has finished this round
        # (sleep + step increment) so critic's resident weights are off the
        # shared card before actor wakes up. Skip during critic-only warmup —
        # ``_execute_training`` handles that case explicitly.
        if (
            self._peer_barrier is not None
            and not self._peer_barrier.is_empty()
            and self.step >= getattr(self.config, "num_critic_only_steps", 0)
        ):
            self._peer_barrier.wait_completed_round_sync(self.step)

        return True

    def _execute_training(self) -> bool:
        """Execute training for the current step.

        Handles critic-only phase, training method selection (sync vs async),
        and model saving based on configuration.

        Returns:
            True when actor training ran and this service owns partition cleanup.
        """
        # Skip training during critic-only phase.
        # But we still must trigger actor.update_weights: it is the only path
        # that calls rollout_manager.onload_weights / onload_kv on SGLang.
        # Without it, SGLang's resume_memory_occupation only remaps the weight
        # region — the contents are uninitialized, so the next rollout's
        # generate produces garbage tokens. The NCCL broadcast inside
        # update_weights is what actually populates SGLang's weight memory
        # (push is idempotent since actor didn't train, but not optional).
        if self.step < self.config.num_critic_only_steps:
            if (
                getattr(self.config, "use_critic", False)
                and getattr(self.config, "offload_rollout", False)
                and (not self.config.fully_async or self.config.hybrid)
            ):
                # Wait for critic to finish this round before triggering
                # update_weights, otherwise SGLang onload collides with
                # critic still training on the same GPUs.
                if self._peer_barrier is not None:
                    self._peer_barrier.wait_completed_round_sync(self.step)
                self.actor_model.update_weights()
            return False

        # Capture the intent before dispatch. Requests arriving during this
        # RPC remain queued for the next consistent training boundary.
        descriptor = self._prepare_lora_export(self.step + 1)
        options = {"lora_export": descriptor} if descriptor is not None else {}
        try:
            if self.config.hybrid:
                ray.get(self.actor_model.train_hybrid(self.step, **options))
            elif self.config.fully_async:
                ray.get(self.actor_model.train_fully_async(self.step, **options))
                if self._lora_profile is None:
                    self._maybe_save_model()
            else:
                ray.get(self.actor_model.async_train(self.step, **options))
        except Exception as error:
            if descriptor is not None:
                self._finish_lora_export(error=error)
            raise
        if descriptor is not None:
            self._finish_lora_export()
        return True

    def _accept_lora_export(self, payload: dict) -> dict:
        from pathlib import Path

        from relax.engine.lora.snapshot import AdapterSnapshot

        if self._lora_profile is None:
            raise HTTPException(409, "PUBLICATION_NOT_CONFIGURED")
        request_id, version_id = payload["request_id"], payload["version_id"]
        AdapterSnapshot(version_id, "0" * 64, "0" * 64, Path("."))
        if not isinstance(request_id, str) or not request_id or len(request_id) > 128:
            raise HTTPException(400, "INVALID_REQUEST_ID")
        publish = payload.get("publish", False)
        if type(publish) is not bool:
            raise HTTPException(400, "INVALID_PUBLISH_FLAG")
        with self._lock:
            previous = self._lora_exports.get(request_id)
            if previous is not None:
                if (previous["version_id"], previous["publish"]) != (version_id, publish):
                    raise HTTPException(409, "REQUEST_ID_CONFLICT")
                return dict(previous)
            if self._lora_export_pending is not None:
                raise HTTPException(409, "EXPORT_BUSY")
            if self.step >= self.config.num_rollout or self._stop_event.is_set():
                raise HTTPException(409, "TRAINING_FINISHED")
            if len(self._lora_exports) >= self._lora_profile.max_lifecycle_records:
                raise HTTPException(507, "LIFECYCLE_CAPACITY_EXCEEDED")
            record = {
                "request_id": request_id,
                "version_id": version_id,
                "publish": publish,
                "requested_step": self.step,
                "exported_step": None,
                "state": "WAITING_BOUNDARY",
            }
            self._lora_exports[request_id] = record
            self._lora_export_pending = record
            return dict(record)

    @app.post("/lora/exports")
    async def submit_lora_export(self, payload: dict) -> dict:
        # No collective from the Serve thread: training consumes this intent.
        return self._accept_lora_export(payload)

    @app.get("/lora/exports/{request_id}")
    async def lora_export_status(self, request_id: str) -> dict:
        with self._lock:
            if request_id not in self._lora_exports:
                raise HTTPException(404, "EXPORT_NOT_FOUND")
            return dict(self._lora_exports[request_id])

    def _prepare_lora_export(self, completed_step: int) -> dict | None:
        import tempfile
        from pathlib import Path

        from relax.engine.lora.snapshot import fingerprint_model

        profile = self._lora_profile
        if profile is None:
            return None
        with self._lock:
            record = self._lora_export_pending
            if record is None:
                every = profile.export_every_n_steps
                if every is None or completed_step % every:
                    return None
                intent = f"{self._lora_export_run}-{completed_step}"
                self._accept_lora_export({"request_id": intent, "version_id": intent, "publish": profile.auto_publish})
                record = self._lora_export_pending
            if record["state"] != "WAITING_BOUNDARY":
                raise RuntimeError("previous export has not settled")
            record.update(state="EXPORTING", started=time.monotonic())
        try:
            store = Path(profile.artifact_store).resolve()
            staging = store / ".staging"
            staging.mkdir(parents=True, exist_ok=True)
            output = tempfile.mkdtemp(prefix="train-export-", dir=staging)
            record["staging_path"] = output
            if self._lora_base_digest is None:
                self._lora_base_digest = fingerprint_model(self.config.hf_checkpoint)
            return dict(
                output_dir=output,
                version_id=record["version_id"],
                store_dir=str(store),
                base_model_digest=self._lora_base_digest,
                artifact_max_bytes=profile.artifact_max_bytes,
            )
        except Exception as error:
            self._finish_lora_export(error=error)
            raise

    def _publish_lora_export(self, record: dict) -> None:
        descriptor = record["snapshot"]
        result = ray.get(
            self.rollout_manager.lora_control.remote(
                "publish",
                {
                    "version_id": descriptor["version_id"],
                    "digest": descriptor["digest"],
                    "request_id": record["request_id"],
                },
            )
        )
        with self._lock:
            record["publication"] = result
            self._lora_exports[record["request_id"]]["publication"] = result

    def _finish_lora_export(self, *, error: Exception | None = None, submit_publication: bool = True) -> dict:
        import shutil

        record = self._lora_export_pending
        try:
            if error is not None:
                raise error
            descriptor = self.actor_model.lora_export_result()
            if descriptor["version_id"] != record["version_id"]:
                raise ValueError("export result belongs to a different intent")
            with self._lock:
                record.update(state="SEALED", snapshot=descriptor, exported_step=descriptor["source_train_step"])
            try:
                shutil.rmtree(record["staging_path"])
            except OSError as cleanup_error:
                record["staging_cleanup_error"] = str(cleanup_error)
            if record["publish"] and submit_publication:
                self._publish_lora_export(record)
        except Exception as export_error:
            with self._lock:
                if record["state"] != "SEALED":
                    record["state"] = "EXPORT_UNKNOWN"
                record["error"] = f"{type(export_error).__name__}: {export_error}"
        finally:
            with self._lock:
                record["export_seconds"] = time.monotonic() - record.pop("started")
                if record["state"] != "EXPORT_UNKNOWN":
                    self._lora_export_pending = None
        if record["state"] == "EXPORT_UNKNOWN":
            raise RuntimeError(f"adapter export did not reach an all-rank result: {record}")
        return dict(record)

    def _bootstrap_lora_publication(self) -> None:
        if self._lora_profile is None:
            return
        state = ray.get(self.rollout_manager.lora_control.remote("status"))
        if state.get("default") is not None:
            return
        exported = self._lora_bootstrap_export
        if exported is None or exported["state"] != "SEALED":
            raise RuntimeError("initial adapter was not exported before releasing training GPUs")
        self._publish_lora_export(exported)
        result = exported.get("publication", {})
        if exported["state"] != "SEALED" or "operation_id" not in result:
            raise RuntimeError(f"initial adapter export/publication failed: {exported}")
        deadline = time.monotonic() + self._lora_profile.prepare_timeout_seconds + 5
        while result["state"] == "PREPARING" and time.monotonic() < deadline:
            time.sleep(0.1)
            result = ray.get(
                self.rollout_manager.lora_control.remote("status", {"operation_id": result["operation_id"]})
            )
        if result["state"] != "PUBLISHED":
            raise RuntimeError(f"initial adapter was not published: {result}")

    def _maybe_save_model(self) -> None:
        """Save model checkpoint if save interval is reached."""
        if self.config.save is None or self.config.save_interval is None:
            return

        is_save_step = (self.step + 1) % self.config.save_interval == 0
        is_final_step = (self.step + 1) == self.config.num_rollout

        if self.config.rotate_ckpt or is_save_step or is_final_step:
            self.actor_model.save_model(self.step, force_sync=is_final_step)

    def train(self, step: int, clear_data: bool = True) -> Dict[str, Any]:
        """Execute a single training step (for external control).

        This method is called by ServiceController/RLSP for fine-grained control.
        It supports interactive debugging by allowing step-by-step execution.

        Args:
            step: The training step number to execute
            clear_data: Whether to clear data partition after training (default: True)

        Returns:
            Dict containing training metrics and status
        """
        import time

        self._logger.info(f"Actor.train called with step={step}, clear_data={clear_data}")
        self.step = step

        start_time = time.time()
        metrics = {}

        try:
            # Check if rollout data is available for this step
            partition_ids = sft_partition_ids(self.config, step)
            partition_list = run(self.data_system_client.async_get_partition_list())

            if partition_list is not None and all(partition_id in partition_list for partition_id in partition_ids):
                self._logger.info(f"Data available for step {step}, executing training")

                # Execute training
                did_train = self._execute_training()

                # Only clear partition data if clear_data is True
                if clear_data and did_train:
                    for partition_id in partition_ids:
                        run(self.data_system_client.async_clear_partition(partition_id=partition_id))
                    self._logger.info(f"Cleared data partitions: {partition_ids}")
                elif clear_data:
                    self._logger.info(f"Skipped clearing partitions after skipped actor step: {partition_ids}")
                else:
                    self._logger.info(f"Keeping data partitions (clear_data=False): {partition_ids}")

                metrics["data_consumed"] = True
                metrics["elapsed_time"] = time.time() - start_time
                metrics["success"] = True
            else:
                self._logger.warning(f"No data available for step {step}, skipping training")
                metrics["data_consumed"] = False
                metrics["success"] = True
                metrics["message"] = f"No data in partitions {partition_ids}"

        except Exception as e:
            self._logger.error(f"Training failed at step {step}: {e}")
            metrics["success"] = False
            metrics["error"] = str(e)

        return metrics

    async def stop(self) -> None:
        """Signal the background training loop to stop and wait for the thread
        to join.

        This is optional but useful for graceful shutdown in tests or service
        stops.
        """
        self._stop_event.set()
        if self._run_thread is not None:
            self._run_thread.join(timeout=5)

    # --- HTTP endpoints for restart / recovery (bypass Ray Serve handle) ---

    @app.get("/get_step")
    def http_get_step(self) -> dict:
        return {"step": self.get_step()}

    @app.post("/set_step")
    def http_set_step(self, step: int) -> dict:
        self.set_step(step)
        return {"status": "ok"}

    @app.post("/stop_service")
    async def http_stop(self) -> dict:
        await self.stop()
        return {"status": "ok"}
