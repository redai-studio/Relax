# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Model-source descriptors and optional provider registration."""

import os
import re
from argparse import Namespace
from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSource:
    """Serializable description of where a model comes from."""

    uri: str
    endpoint: str | None = None
    credential_mode: str = "default"
    addressing_style: str = "auto"
    provider_name: str = "cli"

    def __post_init__(self) -> None:
        if not self.uri:
            raise ValueError("ModelSource.uri must be non-empty")
        if self.credential_mode not in {"default", "placeholder"}:
            raise ValueError(f"Unsupported model credential mode: {self.credential_mode!r}")
        if self.addressing_style not in {"auto", "path"}:
            raise ValueError(f"Unsupported model addressing style: {self.addressing_style!r}")


@dataclass(frozen=True)
class LocalModel:
    """Node-local view of a model source."""

    source: ModelSource
    path: str
    completeness: str


@dataclass(frozen=True)
class SGLangLoadPlan:
    """Resolved model path and load format for one SGLang engine group."""

    model_path: str
    load_format: str
    source: ModelSource


ModelSourceProvider = Callable[[Sequence[str]], ModelSource | None]

_PROVIDERS: dict[str, ModelSourceProvider] = {}
_FROZEN = False
_MODEL_URI_PREFIX = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")


def is_model_uri(path: str) -> bool:
    """Return whether a model path has a hierarchical URI-shaped prefix."""
    return _MODEL_URI_PREFIX.match(path) is not None


def normalize_model_path(path: str) -> str:
    """Normalize local path spelling without rewriting model URIs."""
    return path if is_model_uri(path) else os.path.normpath(path)


def model_source_path_aliases(args: Namespace, source_uri: str) -> set[str]:
    """Return non-empty paths that identified a model before localization."""
    original_hf_checkpoint = getattr(args, "_model_source_original_hf_checkpoint", None)
    return {normalize_model_path(path) for path in (source_uri, original_hf_checkpoint) if path}


def is_model_source_alias(args: Namespace, path: str | None) -> bool:
    """Check whether a path identified the model before localization."""
    source = getattr(args, "model_source", None)
    return bool(
        path and source is not None and normalize_model_path(path) in model_source_path_aliases(args, source.uri)
    )


def register_model_source_provider(name: str, provider: ModelSourceProvider) -> None:
    """Register a provider before the first model-source resolution."""
    global _FROZEN

    if _FROZEN:
        raise RuntimeError(f"Model source provider registry is frozen; cannot register {name!r}")
    if not name:
        raise ValueError("Model source provider name must be non-empty")
    existing = _PROVIDERS.get(name)
    if existing is provider:
        return
    if existing is not None:
        raise RuntimeError(f"Model source provider {name!r} is already registered")
    _PROVIDERS[name] = provider


def resolve_model_source(argv: Sequence[str]) -> ModelSource | None:
    """Resolve exactly one optional provider result and freeze registration."""
    global _FROZEN

    _FROZEN = True
    resolved: list[tuple[str, ModelSource]] = []
    for name in sorted(_PROVIDERS):
        try:
            source = _PROVIDERS[name](tuple(argv))
            if source is not None and not isinstance(source, ModelSource):
                raise TypeError(f"expected ModelSource or None, got {type(source).__name__}")
        except Exception as exc:
            raise RuntimeError(f"Model source provider {name!r} failed: {exc}") from exc
        if source is not None:
            resolved.append((name, source))
    if len(resolved) > 1:
        names = ", ".join(name for name, _ in resolved)
        raise RuntimeError(f"Multiple model source providers matched: {names}")
    if not resolved:
        return None
    name, source = resolved[0]
    if source.provider_name != name:
        source = ModelSource(
            uri=source.uri,
            endpoint=source.endpoint,
            credential_mode=source.credential_mode,
            addressing_style=source.addressing_style,
            provider_name=name,
        )
    return source
