# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Ray owner for publication I/O, Session references, and reconciliation."""

import asyncio
import zlib
from typing import Any

import httpx
import ray

from relax.backends.sglang.lora_client import SGLangLoRAClient

from .artifact import read_adapter_artifact
from .identity import VersionIdentity
from .manager import PublicationError, SessionKey, VersionState
from .publisher import LoRAPublisher


@ray.remote(max_concurrency=128, max_restarts=0)
class LoRAVersionRuntime:
    """One owner per fixed engine pool; metadata transitions never await I/O.

    Files passed to publish must be visible on every engine and remain alive
    through reclamation. This actor deliberately accepts file artifacts only.
    """

    def __init__(self) -> None:
        self._publisher: LoRAPublisher | None = None
        self._urls: dict[str, str] = {}
        self._http: list[httpx.AsyncClient] = []
        self._shards: dict[tuple[str, str], Any] = {}
        self._bindings: dict[SessionKey, VersionIdentity] = {}
        self._maintenance: asyncio.Task | None = None
        self._operations: set[asyncio.Task] = set()
        self._closing = False
        self._errors: dict[str, str] = {}
        self._replacing: set[str] = set()
        self._replacement_clients: dict[tuple[str, str], SGLangLoRAClient] = {}
        self._replacement_tasks: dict[str, asyncio.Task] = {}
        self._replacement_urls: dict[str, str] = {}
        self._replacement_instances: dict[str, str] = {}
        self._close_task: asyncio.Task | None = None

    async def start(
        self,
        engine_urls: dict[str, str],
        probe_input_ids: list[int],
        max_versions: int = 2,
        reconcile_interval_s: float = 1.0,
    ) -> None:
        if self._publisher is not None or self._http or self._closing:
            raise RuntimeError("LoRA runtime has already started or closed")
        if not engine_urls or reconcile_interval_s <= 0:
            raise ValueError("engines and a positive reconciliation interval are required")
        self._urls = {name: url.rstrip("/") for name, url in engine_urls.items()}
        if len(set(self._urls.values())) != len(self._urls):
            raise ValueError("logical engines must have distinct endpoints")
        try:
            clients = {}
            for name, url in self._urls.items():
                http = httpx.AsyncClient(base_url=url, timeout=30.0)
                self._http.append(http)
                clients[name] = SGLangLoRAClient(http)
            self._publisher = await LoRAPublisher.connect(clients, probe_input_ids, max_versions)
            instances = [self._publisher.manager.engine_instance(name) for name in self._urls]
            if len(set(instances)) != len(instances):
                raise ValueError("logical engines must have distinct process instances")
            if self._closing:
                raise RuntimeError("LoRA runtime closed during startup")
        except BaseException:
            await asyncio.gather(*(http.aclose() for http in self._http))
            self._http.clear()
            self._publisher = None
            raise
        self._maintenance = asyncio.create_task(self._maintain(reconcile_interval_s))

    def _ready(self) -> LoRAPublisher:
        if self._publisher is None or self._closing:
            raise RuntimeError("LoRA runtime is not accepting work")
        return self._publisher

    def _track_operation(self, task: asyncio.Task) -> None:
        self._operations.add(task)

        def completed(done: asyncio.Task) -> None:
            self._operations.discard(done)
            if not done.cancelled():
                error = done.exception()
                if error is not None:
                    self._errors[done.get_name()] = str(error)
                else:
                    self._errors.pop(done.get_name(), None)

        task.add_done_callback(completed)

    async def publish(self, artifact_path: str) -> Any:
        publisher = self._ready()

        async def publish_artifact() -> Any:
            artifact = await asyncio.to_thread(read_adapter_artifact, artifact_path)
            self._ready()
            return await publisher.publish(artifact)

        task = asyncio.create_task(publish_artifact(), name="operation:publish")
        self._track_operation(task)
        # A cancelled RPC must not orphan a still-running publication.
        return await asyncio.shield(task)

    async def publish_for_training(self, artifact_path: str) -> dict[str, Any]:
        """Return a serializable verdict, including nonfatal capacity
        pressure."""
        try:
            publication = await self.publish(artifact_path)
        except PublicationError as error:
            default = self._publisher.manager.default_version if self._publisher else None
            return {
                "status": (
                    "capacity_skipped" if error.code == "CAPACITY_EXHAUSTED" and default is not None else "failed"
                ),
                "default_version": default,
                "error": str(error),
            }
        except Exception as error:
            return {"status": "failed", "error": f"{type(error).__name__}: {error}"}
        default = self._publisher.manager.default_version
        if publication.state is not VersionState.PUBLISHED or default != publication.identity.version_id:
            return {
                "status": "unconfirmed",
                "default_version": default,
                "error": f"publication is {publication.state.value}, not the committed default",
            }
        return {
            "status": "published",
            "version_id": publication.identity.version_id,
            "default_version": default,
        }

    async def register_shard(self, shard_id: str, instance_id: str, handle: Any) -> None:
        self._ready().manager.register_shard(shard_id, instance_id)
        self._shards[(shard_id, instance_id)] = handle

    async def bind(self, key: SessionKey) -> VersionIdentity:
        identity = self._ready().manager.bind(key)
        self._bindings[key] = identity
        return identity

    async def route(self, key: SessionKey, identity: VersionIdentity) -> str:
        publisher = self._ready()
        if self._bindings.get(key) != identity:
            raise PublicationError("SESSION_CLOSED", "routing requires the Session's live binding")
        engines = tuple(engine for engine in publisher.manager.serving_engines if engine not in self._replacing)
        if not engines:
            raise PublicationError("LORA_VERSION_UNAVAILABLE", "no engine has all live versions ready")
        # Stable Session affinity while the ready set is unchanged.
        engine = engines[zlib.crc32(f"{key.shard_id}/{key.instance_id}/{key.sequence}".encode()) % len(engines)]
        return self._urls[engine]

    async def replace_engine(
        self, engine_id: str, old_instance: str, new_url: str, *, old_process_exited: bool = False
    ) -> None:
        """Restore an explicitly replaced process in the same logical pool.

        The caller must have confirmed process exit; a timeout is not proof.
        Keep this engine out of routing until every live version is restored.
        Retrying after a lost ACK reuses the same client and endpoint.
        """
        publisher = self._ready()
        if not old_process_exited:
            raise ValueError("replacement requires confirmation that the old process exited")
        if engine_id not in self._urls:
            raise ValueError("replacement cannot add a new logical engine")
        previous = self._replacement_tasks.get(engine_id)
        if previous is not None and not previous.done():
            raise RuntimeError("an engine replacement is already in progress")
        new_url = new_url.rstrip("/")
        if any(
            other != engine_id and endpoint == new_url
            for other, endpoint in (*self._urls.items(), *self._replacement_urls.items())
        ):
            raise ValueError("replacement endpoint belongs to another logical engine")
        current = publisher.manager.engine_instance(engine_id)
        if current != old_instance:
            retry_client = self._replacement_clients.get((engine_id, new_url))
            if retry_client is None or retry_client is not publisher.engines[engine_id]:
                raise PublicationError("ENGINE_INSTANCE_MISMATCH", "stale replacement cannot quarantine an engine")
            if engine_id not in self._replacing:
                return  # Successful replacement whose RPC reply was lost.
        self._replacement_urls[engine_id] = new_url
        self._replacing.add(engine_id)

        async def replace() -> None:
            client_key = (engine_id, new_url)
            client = self._replacement_clients.get(client_key)
            if client is None:
                http = httpx.AsyncClient(base_url=new_url, timeout=30.0)
                self._http.append(http)
                client = SGLangLoRAClient(http)
                self._replacement_clients[client_key] = client
            inventory = await client.inventory()
            if inventory["instance_id"] == old_instance:
                raise ValueError("replacement endpoint still serves the old process")
            new_instance = inventory["instance_id"]
            if any(
                other != engine_id and publisher.manager.engine_instance(other) == new_instance for other in self._urls
            ) or any(
                other != engine_id and instance == new_instance
                for other, instance in self._replacement_instances.items()
            ):
                raise ValueError("replacement process belongs to another logical engine")
            self._replacement_instances[engine_id] = new_instance
            await publisher.replace_engine(engine_id, old_instance, new_instance, client=client)
            self._urls[engine_id] = new_url
            self._replacing.discard(engine_id)
            self._replacement_urls.pop(engine_id, None)
            self._replacement_instances.pop(engine_id, None)
            self._errors.pop(f"replacement:{engine_id}", None)

        task = asyncio.create_task(replace(), name=f"replacement:{engine_id}")
        self._replacement_tasks[engine_id] = task
        self._track_operation(task)
        await asyncio.shield(task)

    async def release(self, key: SessionKey) -> bool:
        if self._publisher is None:
            raise RuntimeError("LoRA runtime has not started")
        released = self._publisher.manager.release(key)
        self._bindings.pop(key, None)
        return released

    async def advance_watermark(self, shard_id: str, instance_id: str, sequence: int) -> bool:
        return self._publisher.manager.advance_watermark(shard_id, instance_id, sequence)

    async def close_shard(self, shard_id: str, instance_id: str) -> bool:
        released = self._publisher.manager.close_shard(shard_id, instance_id)
        self._bindings = {
            key: value
            for key, value in self._bindings.items()
            if (key.shard_id, key.instance_id) != (shard_id, instance_id)
        }
        self._shards.pop((shard_id, instance_id), None)
        return released

    async def _check_shard(self, identity: tuple[str, str], handle: Any) -> None:
        try:
            await asyncio.wait_for(handle.lora_heartbeat.remote(), timeout=5.0)
            self._errors.pop(f"shard:{identity}", None)
        except ray.exceptions.ActorDiedError:
            # Timeouts and transient actor unavailability are not death proof.
            await self.close_shard(*identity)
            self._errors.pop(f"shard:{identity}", None)
        except Exception as error:
            self._errors[f"shard:{identity}"] = str(error)

    async def _maintain(self, interval: float) -> None:
        while True:
            try:
                errors = await self._publisher.reconcile()
                self._errors = {
                    key: value
                    for key, value in self._errors.items()
                    if key.startswith(("shard:", "replacement:", "operation:"))
                }
                self._errors.update({key: str(error) for key, error in errors.items()})
                await asyncio.gather(*(self._check_shard(key, handle) for key, handle in tuple(self._shards.items())))
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self._errors["maintenance"] = str(error)
            await asyncio.sleep(interval)

    async def status(self) -> dict[str, Any]:
        publisher = self._publisher
        return {
            "closing": self._closing,
            "default_version": publisher.manager.default_version if publisher else None,
            "occupied_versions": publisher.manager.occupied_versions if publisher else 0,
            "serving_engines": (
                tuple(engine for engine in publisher.manager.serving_engines if engine not in self._replacing)
                if publisher
                else ()
            ),
            "bound_sessions": len(self._bindings),
            "errors": dict(self._errors),
        }

    async def close(self) -> None:
        """Join Session cleanup before clients close or the owner kills
        engines.

        Failure retains clients and ownership so the caller can retry close.
        Release RPCs remain available while this method awaits shard shutdown.
        """
        if self._close_task is None or (self._close_task.done() and self._close_task.exception() is not None):
            self._close_task = asyncio.create_task(self._close_once())
        await asyncio.shield(self._close_task)

    async def _close_once(self) -> None:
        self._closing = True
        for identity, handle in tuple(self._shards.items()):
            try:
                await handle.shutdown.remote()
            except ray.exceptions.ActorDiedError:
                pass
            await self.close_shard(*identity)
        if self._operations:
            await asyncio.gather(*tuple(self._operations), return_exceptions=True)
        if self._maintenance is not None:
            self._maintenance.cancel()
            await asyncio.gather(self._maintenance, return_exceptions=True)
            self._maintenance = None
        await asyncio.gather(*(http.aclose() for http in self._http))
        self._http.clear()
