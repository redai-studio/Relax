# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Fixed fault identities owned by the existing HealthStatus actor."""

from dataclasses import asdict, replace

from .context import (
    MAX_TRACE_CONTEXTS,
    FaultTraceContext,
    capture_trace_context,
    parse_trace_context,
    parse_trace_contexts,
)
from .models import ROLES, FaultContext, new_id, service_role, valid_uint


class FaultSource:
    def __init__(self, generation: int | None = None) -> None:
        if generation is not None and not valid_uint(generation):
            raise ValueError("INVALID_GENERATION: generation must be an unsigned 64-bit int")
        self.generation = generation
        self.faults: dict[str, FaultContext] = {}
        self._trace_contexts: dict[str, dict[str, FaultTraceContext]] = {}
        self.observation_error = False

    def error(self, role: str, reason: str, fatal: bool = False) -> None:
        """Keep D and the first reason; allocate U only on a nonfatal-to-fatal
        edge."""
        if self.generation is None:
            return
        event_id = None
        try:
            role = service_role(role)
            if type(reason) is not str or reason not in ("reported_error", "heartbeat_timeout", "fatal_error"):
                raise ValueError("INVALID_REASON: reason must be a fault_detected reason code")
            if type(fatal) is not bool or (reason == "fatal_error" and not fatal):
                raise ValueError("INVALID_FATAL: fatal must be a bool and true for fatal_error")
            old = self.faults.get(role)
            if old is None:
                fault = FaultContext(
                    role, new_id(), self.generation, new_id(), "fatal_error" if fatal else reason, fatal
                )
                event_id = fault.event_id
            else:
                FaultContext.validate(old)
                fault = replace(old, fatal=True, fatal_event_id=new_id()) if fatal and not old.fatal else old
                if fatal and not old.fatal:
                    event_id = fault.fatal_event_id
            fault.validate()
            self.faults[role] = fault
        except Exception:
            self.observation_error = True
            return
        if event_id is not None:
            # Context capture is independent of the already-valid local fault.
            # An absent or broken optional OTel API cannot reject that fault.
            try:
                if old is None:
                    self._trace_contexts.pop(role, None)
                context = capture_trace_context()
                if type(context) is FaultTraceContext:
                    sidecars = self._trace_contexts.setdefault(role, {})
                    if len(sidecars) < 2:
                        sidecars[event_id] = context
            except Exception:
                pass

    def clear(self, role: str) -> None:
        """End this observation period; Runtime owns any pending
        publication."""
        if self.generation is None:
            return
        try:
            role = service_role(role)
            self.faults.pop(role, None)
        except Exception:
            self.observation_error = True
            return
        try:
            self._trace_contexts.pop(role, None)
        except Exception:
            pass

    def get(self, role: str) -> dict | None:
        if self.generation is None:
            return None
        try:
            fault = self.faults.get(service_role(role))
            if fault is not None:
                FaultContext.validate(fault)
                return asdict(fault)
        except Exception:
            self.observation_error = True
        return None

    def trace_contexts(self, role: str) -> dict[str, dict[str, object]]:
        """Return validated D/U context copies, never SDK objects."""
        try:
            role = service_role(role)
            fault = self.faults.get(role)
            contexts = self._trace_contexts.get(role, {})
            if type(fault) is not FaultContext or type(contexts) is not dict or len(contexts) > 2:
                return {}
            result = {}
            for event_id in (fault.event_id, fault.fatal_event_id):
                context = contexts.get(event_id)
                if type(context) is FaultTraceContext:
                    validated = parse_trace_context(context.to_payload())
                    if validated is not None:
                        result[event_id] = validated.to_payload()
            return result
        except Exception:
            return {}

    def snapshot(self, roles: list[str]) -> dict:
        """Return only current unhealthy contexts, preserving the business
        list."""
        faults = []
        seen = set()
        if type(roles) is not list or len(roles) > len(ROLES):
            self.observation_error = True
        if type(roles) is list and self.generation is not None:
            for role in roles[: len(ROLES)]:
                try:
                    role = service_role(role)
                    if role in seen:
                        raise ValueError("DUPLICATE_ROLE")
                    seen.add(role)
                    fault = self.get(role)
                    if fault is None:
                        raise ValueError("UNOBSERVED_FAULT")
                    faults.append(fault)
                except Exception:
                    self.observation_error = True
        result = {
            "roles": roles,
            "source_generation": self.generation,
            "faults": faults,
            "observation_error": self.observation_error or self.generation is None,
        }
        try:
            contexts = {
                event_id: payload
                for fault in faults
                for event_id, payload in self.trace_contexts(fault["role"]).items()
            }
            if contexts and len(contexts) <= MAX_TRACE_CONTEXTS:
                result["trace_contexts"] = contexts
        except Exception:
            pass
        return result


def _validated_trace_contexts(payload: object) -> dict:
    """Discard invalid optional metadata without changing local health."""
    try:
        return {key: value.to_payload() for key, value in parse_trace_contexts(payload).items()}
    except Exception:
        return {}


def fault_details(fault: object, source: FaultSource | None, role: str, observation_error: bool) -> dict:
    """Assemble one observed fault and its optional D/U trace sidecars."""
    # A source replaced by None after a failed observation stays unavailable.
    unavailable = observation_error or source is None or source.observation_error
    if fault is not None and type(fault) is not dict:
        fault, unavailable = None, True
    result = {
        "fault": fault,
        "observation_error": unavailable,
    }
    if fault is not None:
        try:
            contexts = _validated_trace_contexts(source.trace_contexts(role))
            ids = (fault.get("event_id"), fault.get("fatal_event_id"))
            contexts = {key: value for key, value in contexts.items() if key in ids}
            if contexts:
                result["trace_contexts"] = contexts
        except Exception:
            pass
    return result


def snapshot_details(observed: object, generation: int | None, observation_error: bool) -> dict:
    """Normalize observation metadata before it joins authoritative health
    roles."""
    if type(observed) is not dict or type(observed.get("faults")) is not list or len(observed["faults"]) > 9:
        observed = {"faults": [], "observation_error": True}
    result = {
        "source_generation": generation,
        "faults": observed["faults"],
        "observation_error": observation_error or observed.get("observation_error") is not False,
    }
    contexts = _validated_trace_contexts(observed.get("trace_contexts"))
    if contexts:
        result["trace_contexts"] = contexts
    return result
