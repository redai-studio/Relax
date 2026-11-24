# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Immutable LoRA control client for shared files or same-host shared
memory."""

from typing import Any

import httpx

from relax.engine.lora_versions.artifact import AdapterArtifact
from relax.engine.lora_versions.identity import AdapterSnapshot


class SGLangLoRAClient:
    def __init__(self, http: httpx.AsyncClient) -> None:
        """The caller owns an HTTP client with the engine base_url and timeout.

        Keep this object alive until reconciliation confirms fenced absence. A
        timed-out load may still be reading shared memory on the engine.
        """
        self.http = http
        self._payloads: dict[tuple[str, str, int], tuple[dict, str]] = {}

    async def inventory(self) -> dict[str, Any]:
        response = await self.http.get("/relax/lora/inventory")
        response.raise_for_status()
        return response.json()

    async def load(
        self, snapshot: AdapterSnapshot | AdapterArtifact, instance_id: str, attempt_id: int
    ) -> dict[str, Any]:
        inventory = await self.inventory()
        if inventory["instance_id"] != instance_id:
            raise ValueError("ENGINE_INSTANCE_MISMATCH")
        if isinstance(snapshot, AdapterArtifact):
            source = {"relax_artifact_path": str(snapshot.path), "serialized_named_tensors": []}
        else:
            from sglang.srt.utils import MultiprocessingSerializer
            from torch.multiprocessing import get_sharing_strategy, set_sharing_strategy

            key = (instance_id, snapshot.identity.lora_name, attempt_id)
            if key not in self._payloads:
                tensors = snapshot.copy_tensors()
                previous = get_sharing_strategy()
                try:
                    set_sharing_strategy("file_system")
                    serialized = MultiprocessingSerializer.serialize(tensors, output_str=True)
                finally:
                    set_sharing_strategy(previous)
                self._payloads[key] = tensors, serialized
            source = {"serialized_named_tensors": [self._payloads[key][1]] * inventory["tp_size"]}
        response = await self.http.post(
            "/relax/lora/load",
            json={
                "lora_name": snapshot.identity.lora_name,
                "config_dict": snapshot.config,
                **source,
                "pinned": True,
                "relax_instance_id": instance_id,
                "relax_attempt_id": attempt_id,
                "relax_digest": snapshot.identity.digest,
                "relax_manifest": snapshot.manifest_json,
            },
        )
        response.raise_for_status()
        return response.json()

    async def probe(self, lora_name: str, input_ids: list[int]) -> None:
        response = await self.http.post(
            "/generate",
            json={
                "input_ids": input_ids,
                "lora_path": lora_name,
                "sampling_params": {"temperature": 0, "max_new_tokens": 1, "ignore_eos": True},
            },
        )
        response.raise_for_status()
        result = response.json()
        if result.get("meta_info", {}).get("completion_tokens") != 1:
            raise ValueError("LoRA generation probe did not produce one token")
        reason = result.get("meta_info", {}).get("finish_reason") or {}
        if reason.get("type") == "abort":
            raise ValueError("LoRA generation probe aborted")

    async def control(
        self, action: str, instance_id: str, lora_name: str, attempt_id: int, lora_id: str | None = None
    ) -> dict[str, Any]:
        response = await self.http.post(
            "/relax/lora/control",
            json={
                "action": action,
                "instance_id": instance_id,
                "lora_name": lora_name,
                "attempt_id": attempt_id,
                "lora_id": lora_id,
            },
        )
        response.raise_for_status()
        receipt = response.json()
        if receipt["instance_id"] != instance_id or receipt["attempt_id"] != attempt_id:
            raise ValueError("control receipt identity mismatch")
        if receipt["lora_id"] is None and receipt["attempt_fenced"]:
            self._payloads.pop((instance_id, lora_name, attempt_id), None)
        return receipt

    def forget_replaced_instance(self, instance_id: str) -> None:
        """Only after the host confirms this process has exited."""
        for key in tuple(self._payloads):
            if key[0] == instance_id:
                del self._payloads[key]
