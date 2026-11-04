# CUDA device regression — 2026-11-04

RTX 3090 (GPU1), Torch 2.9.1+cu128, torchvision 0.24.1+cu128. All 14 checks passed in 3.62 seconds; peak CUDA allocated/reserved memory was 11,264 / 2,097,152 bytes. This supplements the CPU tests with position-tensor placement and real condition-cache device, dtype, copy ownership, capacity and close checks; it is not another full-model SFT measurement.

`result.json` records the result and tested source hashes. `manifest.json` records original and public hashes; only the container working-directory string was redacted. The gzip files preserve the exact original test script and stdout bytes. The isolated container exited successfully and was removed.

From the repository root, with CUDA Torch and torchvision installed:

```bash
gzip -dc examples/frozen_condition_cache/results/cuda-20261104/3090_smoke.py.gz > /tmp/relax17-3090-smoke.py
CUDA_VISIBLE_DEVICES=1 python /tmp/relax17-3090-smoke.py --source-root "$PWD"
```

The script imports the two repository modules directly and does not load model weights. Check their recorded SHA-256 values when comparing this result with later source revisions.
