# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Immutable artifacts and clients for the existing Actor/Rollout services."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

from relax.engine.lora.artifact import PROVENANCE, ModelContract, canonical, provenance, read_object
from relax.engine.lora.snapshot import AdapterSnapshot, fingerprint_model, snapshot_adapter


def seal_export(
    export: Path, model: Path, store: Path, version: str, *, producer: str, export_step: int | None
) -> AdapterSnapshot:
    """Producer-owned synchronous handoff; source may be cleaned after
    return."""
    config = read_object(export / "adapter_config.json")
    contract = ModelContract(
        fingerprint_model(model),
        read_object(model / "config.json"),
        config["r"],
        tuple(config["target_modules"]),
    )
    manifest = provenance(contract, export, producer=producer, export_step=export_step)
    (export / PROVENANCE).write_bytes(canonical(manifest))
    return snapshot_adapter(
        export, store, version_id=version, base_model_digest=contract.base_digest, max_bytes=8 * 1024**3
    )


def service_url(url: str, role: str) -> str:
    base = url.rstrip("/")
    return base if base.endswith("/" + role) else base + "/" + role


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    seal = commands.add_parser("seal")
    seal.add_argument("--source", type=Path, required=True)
    seal.add_argument("--model", type=Path, required=True)
    seal.add_argument("--store", type=Path, required=True)
    seal.add_argument("--version-id", required=True)
    seal.add_argument("--source-step", type=int, default=None)
    export = commands.add_parser("export")
    export.add_argument("--actor-url", required=True)
    export.add_argument("--version-id", required=True)
    export.add_argument("--request-id")
    export.add_argument("--publish", action="store_true")
    for name in ("publish", "status", "cancel", "collect"):
        command = commands.add_parser(name)
        command.add_argument("--rollout-url", required=True)
        if name == "publish":
            command.add_argument("--version-id", required=True)
            command.add_argument("--request-id", required=True)
            command.add_argument("--retry-of")
            command.add_argument("--expected-default-epoch", type=int)
        if name in {"status", "cancel"}:
            command.add_argument("--operation-id", required=name == "cancel")
    for command in (export, commands.choices["publish"]):
        command.add_argument("--wait-seconds", type=float, default=0)
    args = parser.parse_args()
    if args.command == "export" and args.request_id is None:
        args.request_id = uuid4().hex
    try:
        if args.command == "seal":
            snapshot = seal_export(
                args.source,
                args.model,
                args.store,
                args.version_id,
                producer="relax.completed-peft-export",
                export_step=args.source_step,
            )
            result = {
                "state": "SEALED",
                "version_id": snapshot.version_id,
                "digest": snapshot.digest,
                "path": str(snapshot.path),
            }
        else:
            with httpx.Client(timeout=15, trust_env=False) as client:
                result = run_command(client, args)
        sys.stdout.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        if result.get("state") in {"PREPARING", "WAITING_BOUNDARY", "EXPORTING", "RETIRING"}:
            raise SystemExit(3)
        if result.get("state") in {"ABORTED", "EXPORT_UNKNOWN"}:
            raise SystemExit(2)
    except (httpx.HTTPError, ValueError, OSError) as error:
        sys.stdout.write(
            json.dumps({"error": str(error), "state": "UNKNOWN", "request_id": getattr(args, "request_id", None)})
            + "\n"
        )
        raise SystemExit(2) from error


def run_command(client: httpx.Client, args: Any) -> dict:
    name = args.command
    if name == "export":
        base = service_url(args.actor_url, "actor")
        request_id = args.request_id or uuid4().hex
        response = client.post(
            base + "/lora/exports",
            json={
                "request_id": request_id,
                "version_id": args.version_id,
                "publish": args.publish,
            },
        )
        poll = base + "/lora/exports/" + request_id
    else:
        base = service_url(args.rollout_url, "rollout")
        if name == "publish":
            response = client.post(
                base + "/lora/publications",
                json={
                    "request_id": args.request_id,
                    "version_id": args.version_id,
                    "retry_of": args.retry_of,
                    "expected_default_epoch": args.expected_default_epoch,
                },
            )
        elif name == "status":
            suffix = "/lora/publications/" + args.operation_id if args.operation_id else "/lora/versions"
            response = client.get(base + suffix)
        elif name == "cancel":
            response = client.post(base + "/lora/publications/" + args.operation_id + "/cancel")
        else:
            response = client.post(base + "/lora/collect")
        poll = None
    response.raise_for_status()
    result = response.json()
    if name == "publish":
        poll = base + "/lora/publications/" + result["operation_id"]
    deadline = time.monotonic() + getattr(args, "wait_seconds", 0)
    while poll and result.get("state") in {"PREPARING", "WAITING_BOUNDARY", "EXPORTING", "RETIRING"}:
        if time.monotonic() >= deadline:
            break
        time.sleep(min(0.2, max(0, deadline - time.monotonic())))
        response = client.get(poll)
        response.raise_for_status()
        result = response.json()
    return result


if __name__ == "__main__":
    main()
