# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Immutable receiver membership, independent of model and parallel layout."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class SyncPlan:
    groups: tuple[tuple[str, tuple[str, ...]], ...]
    tensor_schema: str
    source_ranks: tuple[int, ...]
    receiver_capabilities: tuple[tuple[str, str], ...] = ()

    @classmethod
    def build(
        cls,
        groups: Mapping[str, Iterable[str]],
        *,
        tensor_schema: str,
        source_ranks: Iterable[int],
        receiver_capabilities: Mapping[str, str] | None = None,
    ) -> "SyncPlan":
        if not groups:
            raise ValueError("A synchronization plan requires receiver groups")
        if len(tensor_schema) != 64 or any(value not in "0123456789abcdef" for value in tensor_schema):
            raise ValueError("A synchronization plan requires its exported tensor schema")
        ranks = tuple(source_ranks)
        if not ranks or any(type(rank) is not int or rank < 0 for rank in ranks) or len(set(ranks)) != len(ranks):
            raise ValueError("A synchronization plan requires distinct source ranks")
        owned: set[str] = set()
        normalized = []
        for name, values in groups.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Receiver group identities must be nonempty strings")
            members = tuple(values)
            if not members or any(not isinstance(value, str) or not value for value in members):
                raise ValueError("Receiver groups require nonempty worker identities")
            if len(set(members)) != len(members) or owned.intersection(members):
                raise ValueError("Each receiver worker must belong to exactly one group")
            owned.update(members)
            normalized.append((name, members))
        capabilities = receiver_capabilities or {}
        if receiver_capabilities is not None and (
            set(capabilities) != set(groups)
            or any(
                not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
                for value in capabilities.values()
            )
        ):
            raise ValueError("A synchronization plan requires one verified capability per receiver group")
        return cls(tuple(sorted(normalized)), tensor_schema, ranks, tuple(sorted(capabilities.items())))

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            (self.groups, self.tensor_schema, self.source_ranks, self.receiver_capabilities), separators=(",", ":")
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    @property
    def members(self) -> set[str]:
        return {member for _, members in self.groups for member in members}

    def group_members(self, identity: str) -> tuple[str, ...]:
        for name, members in self.groups:
            if name == identity:
                return members
        raise ValueError(f"Unknown receiver group: {identity}")
