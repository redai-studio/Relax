#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Offline OpenAPI JSON generator for Relax Ray Serve services.

This script **dynamically imports** the FastAPI app objects defined in
``relax/components/`` and calls ``app.openapi()`` to produce the specification.
Heavy runtime dependencies (Ray, PyTorch, etc.) are shimmed so that only
FastAPI, Pydantic, and the standard library are required.

Usage::

    python scripts/tools/generate_openapi.py

The generated files are consumed by the VitePress documentation site to
render interactive Swagger UI for each service's HTTP API.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent
DOCS_PUBLIC = ROOT / "docs" / "public" / "openapi"

# Ensure the project root is on sys.path so ``relax.*`` can be imported.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ═══════════════════════════════════════════════════════════════════════════
# Step 1: Shim heavy dependencies before importing any relax.impl module
# ═══════════════════════════════════════════════════════════════════════════


def _make_module(name: str, **attrs) -> types.ModuleType:
    """Create a lightweight stand-in module."""
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    return mod


def _mock_deployment(*args, **kwargs):
    """Stand-in for ``ray.serve.deployment``.

    Handles both ``@serve.deployment`` (no parens) and
    ``@serve.deployment(key=value)`` (with parens).
    """
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]  # bare @serve.deployment
    return lambda cls: cls  # @serve.deployment(...)


def _mock_ingress(_app):
    """Stand-in for ``ray.serve.ingress`` — identity decorator."""
    return lambda cls: cls


# -- ray & ray.serve -------------------------------------------------------
_ray = _make_module("ray")
_ray.get = MagicMock()
_ray.remote = lambda *a, **kw: (lambda fn: fn) if not a else a[0]

_ray_serve = _make_module("ray.serve")
_ray_serve.deployment = _mock_deployment
_ray_serve.ingress = _mock_ingress

_ray_serve_schema = _make_module("ray.serve.schema")
_ray_serve_schema.LoggingConfig = MagicMock()

sys.modules.update(
    {
        "ray": _ray,
        "ray.serve": _ray_serve,
        "ray.serve.schema": _ray_serve_schema,
    }
)

# -- relax heavy sub-packages (shimmed to avoid importing torch, etc.) ------
for _name in [
    "relax.distributed.ray",
    "relax.distributed.ray.placement_group",
    "relax.engine.sft.runtime",
    "relax.utils.async_utils",
    "relax.utils.data.processing_utils",
    "relax.utils.opd.opd_utils",
]:
    sys.modules[_name] = MagicMock()

# -- transfer_queue ---------------------------------------------------------
for _name in ["transfer_queue", "transfer_queue.client"]:
    sys.modules[_name] = MagicMock()

# -- httpx (used by genrm.py) ----------------------------------------------
sys.modules["httpx"] = MagicMock()


# ═══════════════════════════════════════════════════════════════════════════
# Step 2: Monkey-patch FastAPI.add_api_route to strip ``self`` parameter
#
# Ray Serve's @serve.ingress handles ``self`` binding at runtime. Since we
# shimmed it away, the raw class methods still carry ``self`` in their
# signature. We intercept route registration and remove ``self`` so that
# the OpenAPI spec is clean.
# ═══════════════════════════════════════════════════════════════════════════

from fastapi import FastAPI  # noqa: E402 – must come after shim injection
from fastapi.routing import APIRouter  # noqa: E402


_original_add_api_route = APIRouter.add_api_route


def _strip_self(endpoint):
    """Return a new endpoint function with ``self`` removed from signature."""
    sig = inspect.signature(endpoint)
    params = list(sig.parameters.values())

    if not params or params[0].name != "self":
        return endpoint  # nothing to strip

    new_sig = sig.replace(parameters=params[1:])

    if inspect.iscoroutinefunction(endpoint):

        @functools.wraps(endpoint)
        async def wrapper(*args, **kw):  # noqa: ARG001
            return {}
    else:

        @functools.wraps(endpoint)
        def wrapper(*args, **kw):  # noqa: ARG001
            return {}

    wrapper.__signature__ = new_sig
    return wrapper


def _patched_add_api_route(self_router, path, endpoint, **kwargs):
    """Wrapper that strips the leading ``self`` parameter from class
    methods."""
    return _original_add_api_route(self_router, path, _strip_self(endpoint), **kwargs)


APIRouter.add_api_route = _patched_add_api_route


# ═══════════════════════════════════════════════════════════════════════════
# Step 3: Import each module, grab its ``app``, and emit openapi.json
# ═══════════════════════════════════════════════════════════════════════════

# (module_path, output_filename, title, description)
SERVICES: list[tuple[str, str, str, str]] = [
    (
        "relax.components.actor",
        "actor",
        "Actor Service",
        (
            "Actor service for training the policy model.\n\n"
            "Supports two execution modes:\n"
            "- **fully_async**: Asynchronous training without waiting for rollout data\n"
            "- **sync**: Waits for rollout data before each training step\n\n"
            "These HTTP endpoints are used for lifecycle management, restart, "
            "and recovery.  They bypass the Ray Serve handle to avoid deadlocks."
        ),
    ),
    (
        "relax.components.rollout",
        "rollout",
        "Rollout Service",
        (
            "Rollout service for data generation via SGLang engines.\n\n"
            "Generates training samples by running the policy model against prompts, "
            "computing rewards, and publishing data to the TransferQueue for the Actor "
            "to consume.\n\n"
            "These HTTP endpoints are used for lifecycle management, evaluation "
            "triggering, and async weight-update coordination."
        ),
    ),
    (
        "relax.components.genrm",
        "genrm",
        "GenRM Service",
        (
            "Generative Reward Model (GenRM) service for LLM-based response evaluation.\n\n"
            "Uses SGLang engines to perform preference evaluation by comparing model "
            "responses against ground truth or quality standards.  The service accepts "
            "OpenAI-style chat messages and returns raw model judgements.\n\n"
            "GenRM is a passive HTTP service — unlike Actor or Rollout it does not run "
            "a background loop.  It only responds to incoming ``/generate`` requests."
        ),
    ),
    (
        "relax.components.actor_fwd",
        "actor_fwd",
        "ActorFwd Service",
        (
            "ActorFwd service for computing actor/reference log-probabilities.\n\n"
            "This service runs a forward-only copy of the policy model to compute "
            "log-probabilities for rollout data.  In fully-async mode it also receives "
            "weight updates from the Actor service.\n\n"
            "These HTTP endpoints are used for lifecycle management and weight "
            "synchronisation."
        ),
    ),
]


def main() -> None:
    DOCS_PUBLIC.mkdir(parents=True, exist_ok=True)

    for module_path, name, title, description in SERVICES:
        # Import the module – this triggers class definition & route registration
        mod = importlib.import_module(module_path)
        app: FastAPI = getattr(mod, "app")

        # Inject metadata (the source code typically uses bare FastAPI())
        app.title = title
        app.description = description
        app.version = "1.0.0"

        # Reset cached schema so our title/description changes take effect
        app.openapi_schema = None

        spec = app.openapi()
        out = DOCS_PUBLIC / f"{name}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(spec, f, indent=2, ensure_ascii=False)
        print(f"  ✓ {out.relative_to(ROOT)}")

    print(f"\nGenerated {len(SERVICES)} OpenAPI specs in {DOCS_PUBLIC.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
