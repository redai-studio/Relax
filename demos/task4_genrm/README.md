# Task 4 contract demo

Run from this directory:

```bash
python -m unittest -v test_contract_demo.py
python contract_demo.py
python render_demo.py results/contract-demo.json --output results/contract-demo.html
```

Open `results/contract-demo.html` in a browser (GitHub's `blob` view shows the source; download the file or use a local checkout to run it). The page replays four deterministic scenarios: normal 1→2→1, health-check failure, backend still busy during drain, and cleanup failure followed by explicit reconciliation. Scrub the timeline to inspect route membership, admitted requests, PG ownership and the recovery boundary. The JSON is the exact event trace used by the page.

This exercises the proposed request and lifecycle contract, including unknown-request `404`, absolute-target validation, idempotency-key replay (a keyed `NOOP` is recorded and replayed verbatim after capacity changes, never executing a new operation), `409` on key/body mismatch or unresolved cleanup, and explicit retry/reconcile paths. Engines, workers, health checks, admission, backend idleness and PGs are in-memory fakes; the `workers` field is a contract signal, not a real worker process. The demo has no Ray, SGLang, GPU, Autoscaler or real scoring. It does not establish that Task 3 supplies the required non-cancelling drain or full-worker cleanup, or that text training continues during scaling. Those remain integration and acceptance work for [RFC #351](https://github.com/redai-studio/Relax/issues/351).

## GPU E2E drivers

- `e2e_genrm_scale.py` — manual `1→2→1` under continuous scoring: probes physical GPUs for the scale-out PG, records per-phase scores, per-engine served counts, GPU snapshots and machine verdicts.
- `e2e_autoscaler_load.py` — full autoscaler cycle under a `LOW→HIGH→STEADY→LOW'` load curve with per-service GenRM thresholds; samples capacity, decisions and history at ~1 Hz.

Both drivers need a single node with free GPUs and the SGLang runtime; they deploy only Ray Serve apps they own and are expected to clean up in an outer `finally`. Machine-verdict summaries of the recorded runs live under `results/`; see [`EVIDENCE.md`](EVIDENCE.md) for the run-to-commit mapping, known limits and hash pins. Full per-request logs and the failed intermediate runs are on the `evidence/task4-genrm` branch.
