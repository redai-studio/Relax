#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Refresh source/Python layers in the pinned CUDA 12 training image.
# Usage: bash bootstrap-cu129.sh /persistent/runtime [--stage-only]
set -euo pipefail
BENCH_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
STAGING_DIR="$(realpath -m "${1:?Provide a dedicated runtime directory}")"
mkdir -p "$STAGING_DIR"
BRIDGE_REV=6862ee170a2569607e8ebc5dde91f119779093ca
MCORE_REV=4a4de865795f123eda422ebf0a4c13d2418cf3cb
TQ_REV=9784ad0d8a5f0db93ff8326379ab2a6d06e640a0
NVRX_REV=6c5f2a13c7688d92a7ac7ee6e464721eb8b7345d
if [[ ! -f "$STAGING_DIR/source-sha256.txt" ]]; then
    python3 - "$STAGING_DIR" "$BRIDGE_REV" "$MCORE_REV" <<'PYSOURCE'
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

root, bridge, mcore = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
for name, repository, revision in (
    ("bridge", "NVIDIA-NeMo/Megatron-Bridge", bridge),
    ("mcore", "NVIDIA/Megatron-LM", mcore),
):
    archive = root / f"{name}-{revision}.tar.gz"
    urllib.request.urlretrieve(f"https://codeload.github.com/{repository}/tar.gz/{revision}", archive)
    directory = root / name
    directory.mkdir(exist_ok=False)
    with tarfile.open(archive) as source:
        source.extractall(directory, filter="data")
    code = next(directory.iterdir()) / ("src/megatron" if name == "bridge" else "megatron")
    shutil.copytree(code, root / "merged/megatron", dirs_exist_ok=True)
PYSOURCE
    for patch_file in \
        latest/megatron.patch \
        megatron/sync-save-bounded-staging.patch \
        megatron/20261016-qwen38-models.patch \
        megatron/20261020-qwen35-optimized.patch \
        megatron/20261020-model-operators.patch \
        megatron/mtp-cp-normalization.patch; do
        patch --batch --forward --fuzz=0 -d "$STAGING_DIR/merged" -p1 < "$BENCH_ROOT/docker/patch/$patch_file"
    done
    python3 -m compileall -q "$STAGING_DIR/merged/megatron"
    (
        cd "$STAGING_DIR/merged"
        find megatron -type f -name '*.py' -print0 | sort -z | xargs -0 sha256sum
    ) > "$STAGING_DIR/source-sha256.txt"
fi
(
    cd "$STAGING_DIR/merged"
    sha256sum --quiet -c "$STAGING_DIR/source-sha256.txt"
)
if [[ "${2:-}" == --stage-only ]]; then
    exit 0
fi
python3 - "$STAGING_DIR" <<'PYABI'
import importlib.metadata as metadata
import json
import sys
from pathlib import Path
import torch

assert torch.version.cuda == "12.9", f"Expected the pinned CUDA 12.9 image, got {torch.version.cuda}"
root = Path(sys.argv[1])
packages = {
    distribution.metadata["Name"]: distribution.version
    for distribution in metadata.distributions()
    if distribution.metadata["Name"].lower().replace("_", "-").startswith(
        ("torch", "triton", "nvidia-", "transformer-engine", "flash-attn", "apex", "causal-conv1d", "sgl-kernel")
    ) and distribution.metadata["Name"].lower().replace("_", "-") != "nvidia-resiliency-ext"
}
baseline = root / "native-abi.json"
if baseline.exists():
    assert json.loads(baseline.read_text()) == packages, "Native package versions changed"
else:
    baseline.write_text(json.dumps(packages, indent=2, sort_keys=True) + "\n")
(root / "native-constraints.txt").write_text("".join(f"{name}=={version}\n" for name, version in sorted(packages.items())))
PYABI
python3 -m pip install -c "$STAGING_DIR/native-constraints.txt" \
    'ray[default,serve]==2.56.1' 'uvicorn>=0.36.0' \
    'opentelemetry-sdk>=1.45.0,<1.46' 'opentelemetry-exporter-otlp>=1.45.0,<1.46' \
    'transformers==5.12.1' 'huggingface_hub==1.7.2' 'numpy<2' 'pyarrow==14.0.2'
python3 -m pip install --force-reinstall --no-deps \
    "transferqueue @ git+https://github.com/redai-studio/TransferQueue.git@$TQ_REV"
STRAGGLER_DET_SKIP_CUPTI_EXT_BUILD=1 python3 -m pip install --no-deps \
    "nvidia-resiliency-ext @ git+https://github.com/NVIDIA/nvidia-resiliency-ext.git@$NVRX_REV"
python3 -m pip install --force-reinstall --no-deps 'fla-core==0.5.2' 'flash-linear-attention==0.5.2'
TQ_SITE="$(python3 -c 'import importlib.util; from pathlib import Path; print(Path(importlib.util.find_spec("transfer_queue").origin).parent.parent)')"
patch --batch --forward --fuzz=0 -d "$TQ_SITE" -p1 < "$BENCH_ROOT/docker/patch/transferqueue/9784ad0-assignment-transform.patch"
FLA_SITE="$(python3 -c 'import importlib.util; from pathlib import Path; print(Path(importlib.util.find_spec("fla").origin).parent.parent)')"
for patch_file in \
    0.5.2-kda-backward-precision.patch \
    0.5.2-qwen38-output-config.patch \
    0.5.2-conv-static-entries.patch; do
    patch --batch --forward --fuzz=0 -d "$FLA_SITE" -p1 < "$BENCH_ROOT/docker/patch/fla/$patch_file"
done
python3 - "$STAGING_DIR/native-abi.json" <<'PYVERIFY'
import importlib.metadata as metadata
import json
import sys
from pathlib import Path

for name, version in json.loads(Path(sys.argv[1]).read_text()).items():
    assert metadata.version(name) == version, f"Native ABI package changed: {name}"
PYVERIFY
python3 -m pip freeze > "$STAGING_DIR/pip-freeze.txt"
printf 'MEGATRON=%s/merged\n' "$STAGING_DIR"
