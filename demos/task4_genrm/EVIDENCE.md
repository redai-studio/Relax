# Task 4 GenRM elastic scaling — evidence manifest

Raw evidence (full run directories: per-request load logs, event timelines,
scale histories, charts, TUI screenshots, failed and superseded intermediate
rounds) is preserved on the
[`evidence/task4-genrm` branch at commit `c56c821`](https://github.com/shanyulu/Relax/tree/c56c8214c9c7590774e7a91cd15cbbcaee2a4f49/demos/task4_genrm/results)
(immutable link; every path below resolves there). The PR itself carries only
the final machine-verdict summaries and the frozen preregistration documents,
so the acceptance claims stay verifiable without large artifacts; every
committed file is hash-pinned below. In the committed summaries, engine
`host` fields are normalized to `node-0` and local filesystem paths to
`<…>` placeholders (single-node run; engine identity is the port) — the raw
data on the evidence branch keeps the original values.

**Pinning policy**: each row binds a conclusion to the commit that produced
its evidence (linked in the row). Rows linked to pre-fix commits document the
review-response journey: every `relax/` change since the squash base
(`fa3ca97`) is a review fix listed in the PR, and every affected acceptance
dimension (reward consistency, failure injection, autoscaler cycle, training
continuity, training smoke) has been re-run from the final PR head on the
fixed code — the re-run verdicts (`*_r2`/`*_r3`/`_r5` directories) are the
ones backing the acceptance claims; `git diff --stat <linked-commit>..<pr head> -- relax/` on the archive branch lists exactly the review fixes. The
training-continuity row is additionally re-run from the final PR
head (`train_continuity_20260925`, final-head verdict below).

## Passing runs

| Run                                                          | Code commit                                                                                | Verdict                      | Key checks                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------ | ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Manual `1→2→1` under continuous scoring                      | [a2ca6cb](https://github.com/shanyulu/Relax/tree/a2ca6cb5b80fd20b372aa6e982805e345ce95dcd) | `E2E_PASS` (`verdicts.json`) | scale-out `CREATING→HEALTH_CHECKING→ACTIVE` in ~45 s on a probed free PG/GPU; scale-in `DRAINING→COMPLETED`; initial engine survived, removed engine was exactly the elastic one; elastic engine served 511 reqs; three-phase greedy short-generation prefixes identical; 4,163 load requests, 0 failures; Ray free GPUs `3→2→3`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| Autoscaler full cycle `LOW→HIGH→STEADY→LOW'`                 | [105c69b](https://github.com/shanyulu/Relax/tree/105c69b59ae3896cfca7dd0e01bf5b606f00c0a7) | `E2E_PASS` (`verdicts.json`) | auto scale-out decided inside HIGH (~16 s after onset); auto scale-in decided t≈132.5 s / completed t≈137.5 s, both inside STEADY — after scale-out the load was diluted across two engines (avg token usage ~1.8 % \< 5 % threshold), so this was a with-traffic scale-in, not an idle one; elastic engine served 516 reqs; 3,202 requests, 0 failures; final capacity == initial                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| Reward consistency, real dapo-genrm protocol                 | final head (`reward_consistency_20260925_r2`)                                              | `PASS` (`verdicts.json`)     | 50 fixed inputs (25 dataset positives + 25 corrupted negatives), exact production prompt/ICE/parser, thinking disabled, 800 engine-attributed replies across both engines: greedy verdicts identical across engines for every input; independent per-engine parse gate; zero truncated replies; exact elastic removal on scale-in. **Official sampling (temperature 0.1) verdicts are now also identical across engines: 0/50 instability** — the earlier 2/50 (4 %) flips traced to per-engine seeds (`args.seed + rank`) and were fixed by the replica-invariant seed (196d7e3); this re-run from the final PR head confirms the fix (the pre-fix discovery run stays on the evidence branch). Judge correctness vs ground truth 95 % / 95 % (initial / elastic; informational)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| Failure injection (drain / abort / kill)                     | final head (`failure_injection_20260925_r2`)                                               | `E2E_PASS` (`verdicts.json`) | every scenario starts from an observed in-flight victim (`inflight>=1` asserted, `ignore_eos` keeps decode alive); S1 empty drain completes instantly; S2 deadline-abort fails the registry `FAILED + cleanup_required` and holds the mutex, reconcile is accepted only after the parked drain fence (fail-closed by design, 607.2 s observed; re-verified on the final head with the fail-closed reconcile fix c03940b); S3 SIGKILL of the victim retries every in-flight request onto the initial engine with zero losses; cleanup green, GPUs returned (pins in the run directory)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| Training continuity through scaling windows (①, real recipe) | final head (`train_continuity_20260925_r4`)                                                | `E2E_PASS` (`verdicts.json`) | same real DAPO+GenRM recipe with actor TP1×DP1 leaving one GPU for the elastic engine; a sidecar monitor drives `scale_out`→ACTIVE and `scale_in`→COMPLETED (1 s drain) against the live service and asserts from two independent sources (job-log step timestamps + `/genrm/engines` counters): **three-window progress assertions** (before scaling / between operations / after scale-in, each independently non-empty — the review-driven rewrite; the 1 s scale-in window is shorter than one training iteration, so the honest claim is three-window coverage plus no stall >120 s across the whole span), the elastic engine really scored rewards (`served=1` observed on the elastic replica), training ran to completion afterwards (8/8 rollouts, set-equality), final capacity back to 1 with only the initial engine alive. Zero errors. Runs 1–5 are recorded driver-iteration evidence (disk-full, actor OOM, monitor-tail defects); final-head reruns r2 (actor first-step OOM from allocator fragmentation — fixed by expandable segments in the recipes) and r3 (tail `\b` regex never matched color-code-abutted `rollout N:` lines — fixed in the driver) are kept as defect records on the evidence branch; r4 is the passing verdict.                                                                         |
| Preregistered autoscaler r3 (v2 metric, frozen thresholds)   | this PR head (`autoscaler_prereg_v2_20260925_r3`)                                          | `E2E_PASS` (`verdicts.json`) | identical frozen configuration to r2; the only change is the preregistered v2 acceptance metric (count placement groups in a non-terminal state instead of the raw table length Ray 2.58 grows with `REMOVED` tombstones). **14/14 frozen sub-assertions pass**, including A7/B3 resources (non-terminal PG `1/1`, free GPUs `3.0/3.0`, memory within tolerance), the true-idle Round B scale-in on all three conditions with zero running requests, and the gated TUI double-screenshot; 3,214 load requests, 0 failures; cleanup green, GPUs returned. r2's recorded FAIL stands unchanged as the metric-defect evidence. **Final-head reruns**: r4 13/14 (B3 resource sample caught teardown memory lag — settled as observation timing, see known limits) and r5 **14/14** with the added `round_b_resources` sample log (`free 3.0/3.0`, `pgs 1/1`, mem delta 108 MiB), identical frozen thresholds throughout.                                                                                                                                                                                                                                                                                                                                                                                                                |
| Minimal training smoke (B2, real recipe)                     | c78e613 + final-head rerun (`b2_train_smoke_20260925_r2`)                                  | `PASS` (`verdicts.json`)     | native 4×4090 recipe via `ray-job.sh` + training venv runtime-env injection (zero-GPU probe first: megatron/TE/FA2/FA3/apex import on a real Ray worker); dapo-genrm protocol, step 1 trained with full metric set, checkpoints iters 0+1 saved; **GenRM judge call really happened** in the c78e613 run (`judge_response` landed on disk for a parseable answer); weight sync `update_weights_from_distributed` 200 OK ×9; zero errors, graceful shutdown, GPUs back to 4/4 free. Run 1 (512-token budget) truncated every response inside `<think>` and never reached the judge — kept as infra-only evidence; the 2048-token budget in c78e613 is what exercises the judge path. The 0.6B judge's noisy `<think>`-preamble verdicts are model capability, not pipeline defects. **Final-head rerun** (`_r2`, with the expandable-segments allocator fix): SUCCEEDED in 4 m 41 s, step 0 trained with the full metric set, weight sync 200 OK ×9, graceful shutdown; this run's 8 samples all truncated inside `<think>` (`answer_missing`) — whether a given sample reaches the judge is model-capability variance, and the judge invocation path itself is exercised at scale by the reward-consistency run (800 attributed judge calls); the r2 rerun's role is to verify the smoke pipeline on the final head, which it does. |

Re-render the charts from the raw summaries (checked out from the evidence
branch at `c56c821`):

```bash
python results/autoscaler_run_20260924_v3/plot_timeline.py
python results/train_continuity_20260925/plot_continuity.py
```

## Known limits of this evidence

- Reward consistency was established with the 0.6B judge, thinking disabled,
  greedy decoding, on 50 fixed inputs: greedy verdicts are identical across
  engines, and — after the replica-invariant seed fix (196d7e3), re-verified
  from the final head (`reward_consistency_20260925_r2`) — official sampling
  (temperature 0.1) verdicts are identical across engines too (0/50 flips).
  The pre-fix 2/50 flips were a seeding artifact, not a pipeline property.
  Judge correctness vs ground truth is 95 % / 95 % (initial / elastic): model
  capability at this scale, not a pipeline property.
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
- Final-head rerun r4 (`autoscaler_prereg_v2_20260925_r4`, identical frozen
  config) recorded **13/14**: `B3_resources_returned` failed at the single
  post-scale-in sample point even though end-of-run cleanup measured free
  GPUs back at `3.0` — settled by r5 as an observation-timing issue
  (engine-teardown memory release lag vs the driver's fixed 6 s sampling
  window), not a placement-group leak: r5 added the `round_b_resources`
  sample log and passed **14/14** with `free 3.0/3.0`, `pgs 1/1`, mem delta
  108 MiB (within the 500 MiB tolerance). The frozen thresholds were never
  touched across r2→r5.
- Single-Gateway adapter accounting; direct-client / cross-gateway drain is
  not claimed.
- Single-node, single-GPU elastic replicas only; multi-node TP/PP is
  rejected at elastic-op admission.
- ~~Idempotency records are retained for the component process lifetime~~
  Fixed by 4ba71eb: clean-terminal records now age out via a TTL and the
  operation history is bounded (`max_history=1024`, `history_truncated`
  flag); long-running training no longer grows the registry unboundedly.
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

| Run                                  | Result                                   | Lesson                                                                                                                                                                                                                                                                            |
| ------------------------------------ | ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `autoscaler_run_20260924` (v1)       | scale-out OK; scale-in never triggered   | cooldown / condition-window tuning; per-request data on the evidence branch                                                                                                                                                                                                       |
| `autoscaler_run_20260924_v2` (v2)    | scale-out OK; scale-in never triggered   | default `throughput_variance_threshold=0.1` never treated bursty short-request throughput as stable (measured variance 0.77 under low load); addressed with a per-service threshold of `1.0`                                                                                      |
| `failure_injection_20260925_v1`–`v3` | invalid test setup, not product verdicts | v1 had scenario sequencing flaws; v2 could not identify the container-side victim PID; v3 let long requests terminate early at EOS, so it never exercised the deadline-abort path. v4 adds `ignore_eos` and verifies victim `inflight>=1` before each scenario.                   |
| `autoscaler_prereg_20260925_r1`      | invalid driver launch                    | `ray.get()` cannot consume Ray Serve's `DeploymentResponse`; no preregistered assertion was reached. Fixed by awaiting `.result()` without changing the frozen policy.                                                                                                            |
| `autoscaler_prereg_20260925_r2`      | `FAIL` (A7, B3 only)                     | Both automatic cycles and all semantic assertions passed, but the PG-count return check failed after each scale-in. The exact failing verdict and normalized event timeline are committed on the evidence branch; full request logs/screenshots remain evidence-branch artifacts. |

## SHA256 (first 16 hex)

| File                                             | sha256             |
| ------------------------------------------------ | ------------------ |
| `autoscaler_prereg_v2_20260925_r5/verdicts.json` | `959e39a033da00db` |
| `autoscaler_preregistration_20260925.md`         | `9b72175b2c5e2102` |
| `autoscaler_preregistration_v2_20260925.md`      | `4363beb4f55481b4` |
| `autoscaler_run_20260924_v3/verdicts.json`       | `1303b57a9bef0cd1` |
| `b2_train_smoke_20260925/verdicts.json`          | `8d5b2944510ccf3b` |
| `b2_train_smoke_20260925_r2/verdicts.json`       | `9bbb5a84055dc40a` |
| `e2e_run_20260924/verdicts.json`                 | `c87ebec5df05ee7d` |
| `failure_injection_20260925_r2/verdicts.json`    | `c1c34ccdb0c13b5c` |
| `reward_consistency_20260925_r2/verdicts.json`   | `f639c6bc8edab47d` |
| `train_continuity_20260925_r4/verdicts.json`     | `c407c34266600439` |
