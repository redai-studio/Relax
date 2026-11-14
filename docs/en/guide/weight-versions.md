# Independent weight publication and consumption

After an optimizer update, the training side exports logical weights as a FULL version or a lossless delta. An independent consumer reconstructs a standard safetensors checkpoint and installs it through the existing SGLang Loader.

## Configuration

Use `--weight-version-config` for the publisher and `--weight-consumer-config` for managed consumers. Omitting both options keeps the existing synchronization path.

For SharedFS, set `transport.type=sharedfs`, `transport.store`, and `transport.disk_budget`. The publisher and consumer use a shared repository path and configure their own temporary directories and storage budgets.

## Publication and installation

The source exports logical tensors through the existing full converter, then selects COPY, SPARSE, or RAW per block. Submission returns a publication handle; background persistence and source snapshot confirmation complete before the weight version is returned.

The consumer downloads and reconstructs its checkpoint before pausing generation. Installation loads and verifies Dense or MoE parameters using each worker's actual TP/PP/EP layout. After loading, cache refresh, and a generation probe, it resumes requests and persists the ACTIVE receipt and consumer cursor.

A sequential subscription retains unconsumed history and resumes from its durable cursor. Retained canonical weights can supply a same-target FULL package for cold starts and installation recovery. Capacity and installation errors report the target version, current stage, and retry action.

## Verification

```bash
python -m pytest -q tests/cpu/distributed/weight_sync
python -m pytest -q tests/cpu/backends/sglang/test_weight_version_verification.py
```

The [FULL checkpoint guide](./weight-checkpoints.md) provides GPU entrypoints for training export and an independent Loader.

## TCP transport

Set `transport.type=tcp`, `host`, and `port` in the consumer configuration. SharedFS and TCP share version identities, encoded objects, and installation behavior.

```bash
python -m relax.distributed.weight_sync \
  --store /path/to/published --disk-budget 34359738368 \
  --max-weight-bytes 8589934592 \
  serve --bind 127.0.0.1 --port 9010
```

TCP transfers objects in chunks and reuses verified content after reconnection. Consumers can request a same-target FULL package from the publication owner. Transport `timeout` covers connections and frames, `control_timeout` covers control operations, and `full_timeout` covers FULL repair; corresponding server options are available. The send ledger records actual delivery and retries.

For cross-host deployments, bind to an address reachable by consumers and provide access control or an authenticated tunnel at the deployment layer.
