# Task 4 — GenRM elastic scaling demos & evidence

Contract demo, GPU E2E drivers and machine-verdict evidence for the GenRM
elastic-scaling work (PR #370 / RFC #351). Chinese mirror: [`README_zh.md`](README_zh.md).

## Contract demo (CPU, no GPU/Ray)

Run from this directory:

```bash
python -m unittest -v test_contract_demo.py
python contract_demo.py
python render_demo.py results/contract-demo.json --output results/contract-demo.html
```

Open `results/contract-demo.html` in a browser (GitHub's `blob` view shows the source; download the file or use a local checkout to run it). The page replays four deterministic scenarios: normal 1→2→1, health-check failure, backend still busy during drain, and cleanup failure followed by explicit reconciliation. Scrub the timeline to inspect route membership, admitted requests, PG ownership and the recovery boundary. The JSON is the exact event trace used by the page.

This exercises the request and lifecycle contract, including unknown-request `404`, absolute-target validation, idempotency-key replay (a keyed `NOOP` is recorded and replayed verbatim after capacity changes, never executing a new operation), `409` on key/body mismatch or unresolved cleanup, and explicit retry/reconcile paths. Engines, workers, health checks, admission, backend idleness and PGs are in-memory fakes; the `workers` field is a contract signal, not a real worker process. The demo has no Ray, SGLang, GPU, Autoscaler or real scoring. It does not establish that Task 3 supplies the required non-cancelling drain or full-worker cleanup, or that text training continues during scaling. Those remain integration and acceptance work for [RFC #351](https://github.com/redai-studio/Relax/issues/351).

## GPU E2E drivers

All drivers need a single node with free GPUs and the SGLang runtime; they deploy only Ray Serve apps they own and clean up in an outer `finally`.

- `e2e_genrm_scale.py` — manual `1→2→1` under continuous scoring: probes physical GPUs for the scale-out PG, records per-phase scores, per-engine served counts, GPU snapshots and machine verdicts.
- `e2e_autoscaler_load.py` — full autoscaler cycle under a `LOW→HIGH→STEADY→LOW'` load curve with per-service GenRM thresholds; samples capacity, decisions and history at ~1 Hz.
- `e2e_autoscaler_preregistered.py` — preregistered acceptance round: frozen thresholds and sub-assertions (A1–A8, B1–B3), a with-traffic Round A and a true-idle Round B, plus gated TUI screenshots.
- `e2e_failure_injection.py` — drain/abort/kill scenarios: deadline-abort with a fail-closed parked drain fence, SIGKILL of an in-flight victim with zero-loss retry onto the initial engine.
- `e2e_reward_consistency.py` — production dapo-genrm judge protocol over a fixed input set, attributed per engine: greedy verdicts must agree exactly; official-sampling stability is reported alongside.
- `e2e_sampling_divergence.py` — adversarial consistency: the initial engine first serves N stochastic requests alone, the elastic engine then boots with the same server seed, and the fixed probe set must still produce identical per-engine verdict sets (divergent RNG histories).
- `e2e_train_continuity.py` — sidecar monitor for the training recipe: drives `scale_out`/`scale_in` against a live DAPO+GenRM run and asserts training progress in every observable window (before / between / after), no >120 s stall, 8/8 rollouts and final capacity back to the initial engine.

Training recipes that pair with the monitor live in `scripts/training/genrm/`.

## Evidence

Machine-verdict summaries of the recorded runs live under `results/` (final verdicts and the frozen preregistration documents only); see [`EVIDENCE.md`](EVIDENCE.md) for the run-to-commit mapping, SHA256 pins, reproduction commands and known limits. Full per-request logs, event timelines, charts and failed/superseded intermediate rounds are preserved on the `evidence/task4-genrm` branch (immutable commit links from `EVIDENCE.md`).
