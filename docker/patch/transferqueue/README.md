# TransferQueue patches

The source is pinned to `9784ad0d8a5f0db93ff8326379ab2a6d06e640a0`.
Its archive SHA256 is
`5da896714784e4f3c03f54ef6454cc9893f6e143d447cf520e0c3bafd17aab31`.

The existing `9784ad0-assignment-transform.patch` now contains all fixes for
this version, applied once with zero fuzz:

- Preserve the existing sequence-length sampler assignment transform.
- Reuse one logical insert ID across client socket retries, returning the original
  indices after a lost reply. Reusing an ID with different arguments is rejected.
- Preserve completed tasks' drain status after partition cleanup, including across
  checkpoints. Unknown/future partitions and unfinished tasks stay incomplete.
- Return empty metadata immediately for empty token-budget polling requests, so
  the Controller request thread can acknowledge production notifications.
  Non-polling callers retain their bounded wait.

These incorporate the former source-branch
`9784ad0-insert-idempotency-drain-terminal.patch` and
`9784ad0-controller-notify-fairness.patch` into the existing version patch.

## Long-running controllers

There is no lifetime insert or partition-creation quota. Cleared insert requests
retain only their IDs with `None` tombstones; their metadata and signatures are
released. A late retry remains rejected even after a same-name partition is
recreated. Completed-task sets are retained per cleared partition and reset on
recreation. Both histories are checkpointed, and older checkpoints remain readable.

History is owned by the Controller and released when it exits. It grows with the
number of distinct logical inserts and cleared partitions; it is not silently
evicted because that would let an arbitrarily late retry allocate again. CPU
validation measured approximately 98 MiB per million retired 32-character IDs and
282 MiB per million retired partitions with one completed task (CPython 3.12;
allocator/RSS and active metadata can add overhead).

Consumers must back off between empty polls. The patch preserves the production
notification ACK contract and does not retry an ambiguous data write.

## Reproducible wheel

Both CUDA Docker recipes build an isolated wheel from the same `TRAIN_IMAGE`
and mount that wheel into the final stage; they do not patch installed packages.
To build without installing:

```bash
TQ_BUILD_PYTHON=/path/to/build-venv/bin/python \
  bash docker/scripts/build_transfer_queue.sh /tmp/tq-wheel
```

The environment needs `build`, `setuptools` and `wheel`. The script verifies
the archive, patch and baseline source hashes, applies the single patch, and builds
`0.1.10.dev0+relax.9784ad0.lifecycle.ack`. It verifies the wheel RECORD and
packaged source hashes, includes `_relax_build.json`, and emits a build receipt.
Optional second and third arguments provide the patch and a downloaded archive.

## Install for a source checkout

Use the patched wheel for source installations too. Installing the upstream
`9784ad0` commit alone omits the fixes above, even though its version satisfies
Relax's minimum-version check.

Run these commands from the Relax repository root with the intended training
environment active (`python` must be Python 3.10 or newer). The build requires
`bash`, `curl`, `patch`, Python's `venv` module and network access. Build tooling
is installed into a separate environment. The first training-environment install
resolves TransferQueue's runtime dependencies, retaining already satisfied versions;
the second replaces only TransferQueue with the wheel just built:

```bash
tq_build_env="$(mktemp -d)"
tq_wheel_dir="$(mktemp -d)"
python -m venv "$tq_build_env"
"$tq_build_env/bin/python" -m pip install build setuptools wheel
TQ_BUILD_PYTHON="$tq_build_env/bin/python" \
  bash docker/scripts/build_transfer_queue.sh "$tq_wheel_dir"
python -m pip install "$tq_wheel_dir"/*.whl
python -m pip install --force-reinstall --no-deps "$tq_wheel_dir"/*.whl
```

If the training environment already contains all required TransferQueue runtime
dependencies, the final `--no-deps` command alone is sufficient (as in the CUDA
Docker recipes). Otherwise, keep the dependency-resolving install first.
`--force-reinstall --no-deps` replaces an existing wheel even when its version
string is unchanged, without reinstalling its dependencies. Keep the wheel and
its `.build.json` receipt together; use a fresh output directory for each build.

To verify the installed artifact, run the following with the same training
Python, from the repository root, retaining `tq_wheel_dir` from the build above.
The build script has already checked the source, patch and wheel. This checks
that the installed manifest and patched source files match that build's receipt;
`pip show transferqueue` alone cannot establish that the patches are installed.

```bash
python - "$tq_wheel_dir" <<'PY'
import hashlib
import json
import sys
from importlib.metadata import distribution
from pathlib import Path

receipt_path, = Path(sys.argv[1]).glob("*.build.json")
receipt = json.loads(receipt_path.read_text())
dist = distribution("transferqueue")
package = Path(dist.locate_file("transfer_queue"))
build = json.loads((package / "_relax_build.json").read_text())
assert build == receipt["build"]
assert dist.version == build["version"]
for name, expected in build["source_files_sha256"].items():
    assert hashlib.sha256((package / name).read_bytes()).hexdigest() == expected, name
print(f"Verified TransferQueue {dist.version} at {package}")
PY
```
