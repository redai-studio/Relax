# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""In-process manager transport exercising the real admission gate."""

from relax.engine.inference.admission import AdmissionGate
from relax.engine.inference.client import InferenceRequestClient
from relax.engine.inference.discovery import RoleSnapshot


class AdmissionManager:
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.gate = AdmissionGate()
        self.acquired = []
        self.released = []

    def fetch(self):
        snapshot = self.snapshot() if callable(self.snapshot) else self.snapshot
        return snapshot if isinstance(snapshot, RoleSnapshot) else RoleSnapshot.from_dict(snapshot)

    def acquire_inference(self, request_id, **selection):
        permit = self.gate.acquire(self.fetch, request_id, **selection)
        self.acquired.append(permit)
        return permit

    def release_inference(self, permit):
        self.released.append(permit)
        self.gate.release(permit)

    def confirm_inference_dispatch(self, permit):
        self.gate.confirm_dispatch(permit)

    def client(self, role="rollout", **extra):
        async def resources():
            snapshot = self.fetch()
            return {"snapshot": snapshot.to_dict(), "managers": {m.name: self for m in snapshot.models}, **extra}

        return InferenceRequestClient(role, resources)
