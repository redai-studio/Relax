#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Pull once: apptainer pull sandboxfusion.sif \
#   docker://vemlp-cn-beijing.cr.volces.com/preset-images/code-sandbox:server-20250609
# Run on the rollout host: SANDBOX_IMAGE=/path/to/sandboxfusion.sif bash <this-script>
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${SANDBOX_IMAGE:?Set the pinned SandboxFusion server-20250609 SIF path}"
if [[ $(id -u) != 0 ]]; then
    echo "SandboxFusion lite requires a root driver for namespaces and delegated cgroups" >&2
    exit 1
fi
# Keep the image's runtime, but expose no host home, working directory, GPU,
# repository, model or data. Only the trusted bootstrap and cgroup controls bind.
exec apptainer exec --cleanenv --containall --no-home \
    --no-mount hostfs,bind-paths,cwd --scratch /run --pwd /root/sandbox \
    --bind /sys/fs/cgroup:/sys/fs/cgroup:rw \
    --bind "${SCRIPT_DIR}/../sandbox_server.py:/opt/retool_server.py:ro" \
    --bind "${SCRIPT_DIR}/../sandbox_python.py:/opt/retool_python.py:ro" \
    --env PYTHONPATH=/root/sandbox --env OMP_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 \
    "${SANDBOX_IMAGE}" bash -c \
    'set -e; mount --make-private /; exec python /opt/retool_server.py "$@"' sandbox \
    --host "${SANDBOX_HOST:-127.0.0.1}" --port "${SANDBOX_PORT:-8080}"
