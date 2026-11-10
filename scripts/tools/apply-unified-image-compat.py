# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Install checked source deltas on an idle node; runtime paths come from the
caller."""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def package_root(name: str) -> Path:
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None:
        raise ValueError(f"Required installed package not found: {name}")
    return Path(spec.origin).parent


def replace(path: Path, content: bytes) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
    try:
        shutil.copymode(path, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def backup_path(path: Path, original: bytes) -> Path:
    backup = path.with_name(path.name + ".pre-unified-image-compat")
    if backup.exists() and backup.read_bytes() != original:
        backup = backup.with_name(backup.name + "." + digest(original))
    if backup.exists() and backup.read_bytes() != original:
        raise ValueError(f"Existing backup differs; no files installed: {backup}")
    return backup


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--megatron-root", type=Path, required=True)
    parser.add_argument("--sglang-root", type=Path, help="Python package directory containing srt/")
    parser.add_argument("--transferqueue-root", type=Path, help="Python package directory containing sampler/")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    roots = {
        "megatron": args.megatron_root,
        "sglang": args.sglang_root or package_root("sglang"),
        "transferqueue": args.transferqueue_root or package_root("transfer_queue"),
    }
    catalog = Path(__file__).resolve().parents[2] / "docker/patch/image-compat/qs-20261021.json"
    records = json.loads(catalog.read_text())["files"]
    original_manifest = args.manifest.read_bytes()
    manifest = json.loads(original_manifest)
    changes = {}
    with tempfile.TemporaryDirectory() as directory:
        for index, record in enumerate(records):
            path = roots[record["component"]] / record["path"]
            original = path.read_bytes()
            sha = digest(original)
            patches = {record["before"]: record["patch"]}
            patches.update({upgrade["before"]: upgrade["patch"] for upgrade in record.get("upgrades", [])})
            accepted = {*patches, record["after"]}
            if sha not in accepted:
                raise ValueError(f"Unknown source version; no files installed: {path}")
            if record["component"] == "megatron":
                recorded = manifest["megatron_sources"].get(record["path"])
                if recorded is not None and recorded not in accepted:
                    raise ValueError(f"Manifest differs from validated source: {path}")
                manifest["megatron_sources"][record["path"]] = record["after"]
            if sha == record["after"]:
                continue
            staged = Path(directory) / str(index)
            staged.write_bytes(original)
            subprocess.run(
                ["patch", "--batch", "--forward", "--fuzz=0", str(staged)],
                input=patches[sha],
                text=True,
                capture_output=True,
                check=True,
                timeout=30,
            )
            content = staged.read_bytes()
            if digest(content) != record["after"]:
                raise ValueError(f"Patch output hash mismatch: {path}")
            compile(content, str(path), "exec")
            changes[path] = (original, content)
    if json.loads(original_manifest) != manifest:
        changes[args.manifest] = (original_manifest, (json.dumps(manifest, indent=2) + "\n").encode())
    if args.check_only:
        print(f"Validated {len(records)} source files; {len(changes)} file updates required")
        return
    # Validate and create all backups before replacing any installed file.
    backups = {path: backup_path(path, original) for path, (original, _) in changes.items()}
    for path in changes:
        backup = backups[path]
        if not backup.exists():
            shutil.copy2(path, backup)
    replaced = []
    try:
        for path, (_, content) in changes.items():
            replace(path, content)
            replaced.append(path)
    except BaseException as error:
        failed = []
        for path in reversed(replaced):
            try:
                replace(path, changes[path][0])
            except OSError as rollback_error:
                failed.append(f"{path}: {rollback_error}")
        if failed:
            raise RuntimeError(f"Installation failed; restore these files from backups: {failed}") from error
        raise
    print(f"Validated {len(records)} source files; installed {len(changes)} updates")


if __name__ == "__main__":
    main()
