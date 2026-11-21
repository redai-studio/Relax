# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Cache the shared Agent venv on every live Ray node before job submission."""

import argparse
import hashlib
import os
import sys
from pathlib import Path


def prepare_archive(source: Path, archive: Path) -> None:
    """Create one reusable archive to avoid remote per-file filesystem
    reads."""
    import fcntl
    import subprocess
    import tempfile

    archive.parent.mkdir(parents=True, exist_ok=True)
    with archive.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if archive.is_file():
            return
        with tempfile.NamedTemporaryFile(dir=archive.parent, suffix=".tar") as temporary:
            subprocess.run(["tar", "-C", str(source), "-cf", temporary.name, "."], check=True)
            # Keep the temporary filename for the context manager's cleanup.
            os.link(temporary.name, archive)


def cache_environment(archive: str, destination: str) -> str:
    """Publish a complete local copy under a lock; never expose a partial
    copy."""
    import fcntl
    import shutil
    import subprocess
    import tempfile

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (target / ".relax-local-ready").is_file():
            if target.exists():
                raise RuntimeError(f"Refusing to overwrite incomplete cache: {target}")
            staging = Path(tempfile.mkdtemp(prefix=f"{target.name}-", dir=target.parent))
            try:
                # This archive is produced by prepare_archive from our trusted venv.
                subprocess.run(["tar", "-xf", archive, "-C", str(staging)], check=True)
                # Relocating a venv is valid here only if its Python uses the new prefix.
                subprocess.run(
                    [
                        str(staging / "bin/python"),
                        "-B",
                        "-c",
                        "import pathlib, sys; assert pathlib.Path(sys.prefix) == pathlib.Path(sys.argv[1])",
                        str(staging),
                    ],
                    check=True,
                    timeout=60,
                )
                (staging / ".relax-local-ready").touch()
                staging.rename(target)
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
        python = target / "bin/python"
        if not os.access(python, os.X_OK):
            raise RuntimeError(f"Cached Agent Python is not executable: {python}")
    return str(python)


def main() -> None:
    import fcntl

    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    from relax.utils.logging_utils import get_logger

    logger = get_logger(__name__)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    with (args.source.parent / ".osworld-agent-venv.lock").open("a") as source_lock:
        fcntl.flock(source_lock, fcntl.LOCK_SH)
        source = args.source.resolve(strict=True)
        marker = source / ".relax-agent-ready"
        # Touch the shared completion marker after intentional dependency updates.
        identity = f"{source}:{marker.stat().st_mtime_ns}:".encode() + (source / "pyvenv.cfg").read_bytes()
        version = hashlib.sha256(identity).hexdigest()[:20]
        destination = f"/tmp/relax-osworld-agent-envs/{version}"
        archive = args.source.parent / ".osworld-agent-env-archives" / f"{version}.tar"
        logger.info("Preparing shared Agent environment archive: %s", archive)
        prepare_archive(source, archive)
        ray.init(address=os.environ.get("RAY_ADDRESS", "auto"), log_to_driver=False)
        try:
            nodes = [node for node in ray.nodes() if node["Alive"]]
            if not nodes:
                raise RuntimeError("No live Ray nodes available for Agent environment staging")
            stage = ray.remote(num_cpus=0)(cache_environment)
            pending = {}
            for node in nodes:
                ref = stage.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(node["NodeID"], soft=False)
                ).remote(str(archive), destination)
                pending[ref] = node["NodeManagerAddress"]
            while pending:
                ready, _ = ray.wait(list(pending), timeout=30)
                if not ready:
                    logger.info("Copying Agent environment on: %s", ", ".join(pending.values()))
                for ref in ready:
                    python = ray.get(ref)
                    logger.info("Agent environment ready on %s: %s", pending.pop(ref), python)
            # stdout is consumed by the launcher's command substitution.
            sys.stdout.write(f"{destination}/bin/python\n")
        finally:
            ray.shutdown()


if __name__ == "__main__":
    main()
