# Task 4 GenRM elastic scaling — evidence manifest

Raw evidence (full run directories including per-request load logs and the two
failed intermediate autoscaler runs) is preserved on the
[`evidence/task4-genrm`](https://github.com/shanyulu/Relax/tree/evidence/task4-genrm/demos/task4_genrm/results)
branch. This directory carries the machine-verdict summaries of the two passing
runs so the acceptance claims are verifiable without large artifacts; every
file is hash-pinned below. Engine `host` fields are normalized to `node-0`
and local filesystem paths to `<…>` placeholders (single-node run; engine
identity is the port) — the raw data on the evidence branch keeps the original
values.

**Pinning policy**: each row binds a conclusion to the commit that produced
its evidence (linked in the row). The production code (`relax/`) has not
changed between those commits and the PR head — the post-squash lineage
consists of driver, evidence and documentation commits only, verifiable via
`git diff --stat <linked-commit>..HEAD -- relax/` being empty on the archive
branch. The training-continuity row is additionally re-run from the final PR
head (`train_continuity_20260925`, final-head verdict below).

## Passing runs

| Run                                                          | Code commit                                                                                | Verdict                      | Key checks                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------ | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Manual `1→2→1` under continuous scoring                      | [a2ca6cb](https://github.com/shanyulu/Relax/tree/a2ca6cb5b80fd20b372aa6e982805e345ce95dcd) | `E2E_PASS` (`verdicts.json`) | scale-out `CREATING→HEALTH_CHECKING→ACTIVE` in ~45 s on a probed free PG/GPU; scale-in `DRAINING→COMPLETED`; initial engine survived, removed engine was exactly the elastic one; elastic engine served 511 reqs; three-phase greedy short-generation prefixes identical; 4,163 load requests, 0 failures; Ray free GPUs `3→2→3`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| Autoscaler full cycle `LOW→HIGH→STEADY→LOW'`                 | [105c69b](https://github.com/shanyulu/Relax/tree/105c69b59ae3896cfca7dd0e01bf5b606f00c0a7) | `E2E_PASS` (`verdicts.json`) | auto scale-out decided inside HIGH (~16 s after onset); auto scale-in decided t≈132.5 s / completed t≈137.5 s, both inside STEADY — after scale-out the load was diluted across two engines (avg token usage ~1.8 % \< 5 % threshold), so this was a with-traffic scale-in, not an idle one; elastic engine served 516 reqs; 3,202 requests, 0 failures; final capacity == initial                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| Reward consistency, real dapo-genrm protocol                 | this branch's head (`reward_consistency_20260925`)                                         | `PASS` (`verdicts.json`)     | 50 fixed inputs (25 dataset positives + 25 corrupted negatives), exact production prompt/ICE/parser, thinking disabled, 800 engine-attributed replies across both engines: greedy verdicts identical across engines for every input; independent per-engine parse gate; zero truncated replies; exact elastic removal on scale-in. Reported (not gated): official sampling (temperature 0.1, per-engine seeds `args.seed + rank`) flips verdicts on 2/50 inputs (4 %) — a property of the deployed sampling config, reproducible across two runs (same two cases; the discovery run is on the evidence branch); judge correctness vs ground truth 95 % / 93 % (initial / elastic)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| Failure injection (drain / abort / kill)                     | [12f783a](https://github.com/shanyulu/Relax/tree/12f783a13a13646af9e7711c0e0b21a45d91682b) | `E2E_PASS` (`verdicts.json`) | every scenario starts from an observed in-flight victim (`inflight>=1` asserted, `ignore_eos` keeps decode alive); S1 empty drain completes instantly; S2 deadline-abort fails the registry `FAILED + cleanup_required` and holds the mutex, reconcile is accepted only after **607.3 s** parked drain (fail-closed by design); S3 SIGKILL of the victim retries every in-flight request onto the initial engine with zero losses; cleanup green, GPUs returned (pins `f1accf18…`/`450af55…`)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| Training continuity through scaling windows (①, real recipe) | this PR head (`train_continuity_20260925`)                                                 | `E2E_PASS` (`verdicts.json`) | same real DAPO+GenRM recipe with actor TP1×DP1 leaving one GPU for the elastic engine; a sidecar monitor drives `scale_out`→ACTIVE (55 s) and `scale_in`→COMPLETED (1 s drain) against the live service and asserts from two independent sources (job-log step timestamps + `/genrm/engines` counters): train events kept landing inside both scaling windows with no stall >120 s, the elastic engine really scored rewards (`served=1` observed on the elastic replica), training ran to completion afterwards (8/8 rollouts), final capacity back to 1 with only the initial engine alive. Zero errors. **Final-head verdict** (`train_continuity_20260925_final_head`, re-run from the PR head with the index-deduplicated rollout assertion): `rollout_count: 8`, set-equality `all_rollouts_landed`, all twelve assertions green — the earlier run's `rollout_count: 16` line-double-count is superseded. Runs 1–5 are recorded driver-iteration evidence: run 1 disk-full checkpoint write, run 2 actor DP1 OOM (fixed by halving `--max-tokens-per-gpu`, engine GPU placement verified correct), runs 3–5 monitor-tail defects (unwrapped final poll, tuple JSON keys, empty fallback list) — the training itself succeeded from run 3 onward; pins below. |
| Preregistered autoscaler r3 (v2 metric, frozen thresholds)   | this PR head (`autoscaler_prereg_v2_20260925_r3`)                                          | `E2E_PASS` (`verdicts.json`) | identical frozen configuration to r2; the only change is the preregistered v2 acceptance metric (count placement groups in a non-terminal state instead of the raw table length Ray 2.58 grows with `REMOVED` tombstones). **14/14 frozen sub-assertions pass**, including A7/B3 resources (non-terminal PG `1/1`, free GPUs `3.0/3.0`, memory within tolerance), the true-idle Round B scale-in on all three conditions with zero running requests, and the gated TUI double-screenshot; 3,214 load requests, 0 failures; cleanup green, GPUs returned. r2's recorded FAIL stands unchanged as the metric-defect evidence.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| Minimal training smoke (B2, real recipe)                     | c78e613 (`b2_train_smoke_20260925`)                                                        | `PASS` (`verdicts.json`)     | native 4×4090 recipe via `ray-job.sh` + training venv runtime-env injection (zero-GPU probe first: megatron/TE/FA2/FA3/apex import on a real Ray worker); dapo-genrm protocol, step 1 trained with full metric set, checkpoints iters 0+1 saved; **GenRM judge call really happened** (`judge_response` landed on disk for a parseable answer, the other 7 samples legitimately short-circuit `answer_missing`); weight sync `update_weights_from_distributed` 200 OK ×9; zero errors, graceful shutdown, GPUs back to 4/4 free. Run 1 (512-token budget) truncated every response inside `<think>` and never reached the judge — kept as infra-only evidence; the 2048-token budget in c78e613 is what exercises the judge path. The 0.6B judge's noisy `<think>`-preamble verdicts are model capability, not pipeline defects.                                                                                                                                                                                                                                                                                                                                                                                                                                   |

Re-render the charts from the committed summaries:

```bash
python results/autoscaler_run_20260924_v3/plot_timeline.py
python results/train_continuity_20260925/plot_continuity.py
```

## Known limits of this evidence

- Reward consistency was established with the 0.6B judge, thinking disabled,
  greedy decoding, on 50 fixed inputs: greedy verdicts are identical across
  engines, while official sampling (temperature 0.1, per-engine seeds
  `args.seed + rank`) flips 2/50 inputs — characterized as a property of the
  deployed sampling config, reported but not gated. Judge correctness vs
  ground truth is 95 % / 93 % (initial / elastic): model capability at this
  scale, not a pipeline property.
- Autoscaler thresholds are tuned for this 0.6B / 4×4090 / short-reply
  workload; they are frozen per preregistration, not claimed as a universal
  policy. Per-service cooldown overrides are not yet part of
  `ServiceScalingPolicy` (global cooldowns via `PATCH /config`).
- The continuity verdict's `rollout_count: 16` is a log-line double count:
  the job driver logs each rollout index twice (batch start and result), so
  the `>= 8` assertion passed on a doubled numerator. Deduplicated by index
  the committed `train_events.json` contains exactly `{0..7}` — 8/8, the
  conclusion stands; the driver now counts unique indices and asserts set
  equality. Disclosed rather than re-run: the corrected reading is
  machine-verifiable from the committed evidence.
- Failure-injection S2 intentionally waits for the configured 600 s drain
  fence after its five-second operation deadline. Its terminal operation
  status remains `FAILED`; `cleanup_required=false` after reconcile is the
  success criterion for physical completion, not a relabeling of that result.
- Preregistered autoscaler r2 passed A1–A6, A8 and B1–B2 but **failed** A7
  and B3: after each scale-in, Ray free GPUs and physical-memory samples were
  back at baseline while `len(ray.util.placement_group_table())` was 2 versus
  the pre-run baseline of 1. This is a resource-accounting blocker, not a
  threshold-tuning failure; the frozen policy was not changed and the run is
  not claimed as an autoscaler pass.
- Post-run classification settles that blocker without touching product
  code: a removed-and-confirmed `REMOVED` placement group still remains in
  `ray.util.placement_group_table()` (Ray 2.58 keeps tombstone entries;
  reproduced CPU-only), so the count grows by one per completed scale-in and
  can never return to the pre-run baseline. Since scale-in `COMPLETED`
  already requires the manager to poll Ray until the elastic PG is `REMOVED`
  before reporting physical completion (scale-in lifecycle in
  `relax/distributed/ray/genrm.py`), and free GPUs and memory did return to
  baseline within the run, the two failing verdicts are an acceptance-metric
  defect (counting `REMOVED` tombstones), not a placement-group lifecycle
  leak. r3 re-ran the identical frozen configuration under the corrected
  non-terminal-PG metric and passed 14/14; r2's FAIL record stands unchanged.
- Single-Gateway adapter accounting; direct-client / cross-gateway drain is
  not claimed.
- Single-node, single-GPU elastic replicas only; multi-node TP/PP is
  rejected at elastic-op admission.
- Idempotency records are retained for the component process lifetime — no
  TTL/eviction window is implemented in `genrm_scale_registry.py` (the RFC
  wording has been aligned accordingly). Long-running training with many
  distinct idempotency keys grows the registry in memory.
- The GenRM scale drain-fence timeout is read via
  `getattr(args, "genrm_scale_drain_timeout_s", 600.0)` and is **not yet
  declared as a CLI argument** in `relax/utils/arguments.py` (Ask-First
  area); from the training entrypoint it stays at the 600 s default and can
  only be overridden by drivers that construct args directly (the E2E
  drivers do).
- Autoscaler cooldowns are global per deployment (`PATCH /config`), not
  per-service fields in `ServiceScalingPolicy`: thresholds, queue, TTFT and
  variance are per-service, cooldown is not.

## Failed intermediate runs (evidence branch only)

| Run                                  | Result                                   | Lesson                                                                                                                                                                                                                                                          |
| ------------------------------------ | ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `autoscaler_run_20260924` (v1)       | scale-out OK; scale-in never triggered   | cooldown / condition-window tuning; per-request data on the evidence branch                                                                                                                                                                                     |
| `autoscaler_run_20260924_v2` (v2)    | scale-out OK; scale-in never triggered   | default `throughput_variance_threshold=0.1` never treated bursty short-request throughput as stable (measured variance 0.77 under low load); addressed with a per-service threshold of `1.0`                                                                    |
| `failure_injection_20260925_v1`–`v3` | invalid test setup, not product verdicts | v1 had scenario sequencing flaws; v2 could not identify the container-side victim PID; v3 let long requests terminate early at EOS, so it never exercised the deadline-abort path. v4 adds `ignore_eos` and verifies victim `inflight>=1` before each scenario. |
| `autoscaler_prereg_20260925_r1`      | invalid driver launch                    | `ray.get()` cannot consume Ray Serve's `DeploymentResponse`; no preregistered assertion was reached. Fixed by awaiting `.result()` without changing the frozen policy.                                                                                          |
| `autoscaler_prereg_20260925_r2`      | `FAIL` (A7, B3 only)                     | Both automatic cycles and all semantic assertions passed, but the PG-count return check failed after each scale-in. The exact failing verdict and normalized event timeline are committed; full request logs/screenshots remain evidence-branch artifacts.      |

## SHA256 (first 16 hex)

| File                                                                   | sha256             |
| ---------------------------------------------------------------------- | ------------------ |
| `e2e_run_20260924/verdicts.json`                                       | `c87ebec5df05ee7d` |
| `e2e_run_20260924/events.json`                                         | `25731e768c5e8d16` |
| `e2e_run_20260924/load_requests.json` (evidence branch only)           | `bd9ac8f5e3613a53` |
| `autoscaler_run_20260924_v3/verdicts.json`                             | `1303b57a9bef0cd1` |
| `autoscaler_run_20260924_v3/events.json`                               | `d66662debda40511` |
| `autoscaler_run_20260924_v3/timeline.json`                             | `81d97d9c992336ac` |
| `autoscaler_run_20260924_v3/scale_history.json`                        | `592a5f31860ceddc` |
| `autoscaler_run_20260924_v3/autoscaler.yaml`                           | `8ec6698a20f03295` |
| `autoscaler_run_20260924_v3/load_requests.json` (evidence branch only) | `00d2bc62f10cc0f9` |
| `reward_consistency_20260925/verdicts.json`                            | `88120c9a415082bb` |
| `reward_consistency_20260925/events.json`                              | `14f454d4835d45de` |
| `reward_consistency_20260925/replies.json` (evidence branch only)      | `2d6e7ba1cd4c0d67` |
| `failure_injection_20260925_v4/verdicts.json`                          | `f1accf18a833fc8a` |
| `failure_injection_20260925_v4/events.json`                            | `c84034453ffef9df` |
| `autoscaler_prereg_20260925_r2/verdicts.json`                          | `03f92aa80b6b0f31` |
| `autoscaler_prereg_20260925_r2/events.json`                            | `9d25426ddbe58ac8` |
| `autoscaler_prereg_20260925_r2/scale_history.json`                     | `5a0d1e39ab5222e2` |
| `autoscaler_prereg_20260925_r2/scale_history_round_b.json`             | `6559b23237f5e96f` |
| `b2_train_smoke_20260925/verdicts.json`                                | `8d5b2944510ccf3b` |
| `b2_train_smoke_20260925/rollout_result_run2.jsonl`                    | `1d0949b60c51a049` |
| `train_continuity_20260925/verdicts.json`                              | `3f8e9d6da66d667b` |
| `train_continuity_20260925/events.json`                                | `8bae33698da830f7` |
| `train_continuity_20260925/train_events.json`                          | `12344b997b3c731e` |
| `autoscaler_prereg_v2_20260925_r3/verdicts.json`                       | `41cb8b4b5b730a98` |
| `autoscaler_prereg_v2_20260925_r3/events.json`                         | `7b91c9a1698cc3d9` |
| `autoscaler_prereg_v2_20260925_r3/scale_history.json`                  | `4d4eda5dbd513853` |
| `autoscaler_prereg_v2_20260925_r3/scale_history_round_b.json`          | `b06867137b3f345d` |

| `train_continuity_20260925_final_head/verdicts.json`                   | `28ce453c9bbdf96e` |
| `train_continuity_20260925_final_head/events.json`                     | `70385b7d3751f2a0` |
| `train_continuity_20260925_final_head/train_events.json`               | `98e0b6d0929fe64b` |
