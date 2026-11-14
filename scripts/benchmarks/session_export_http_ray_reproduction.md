# Task14 HTTP/Ray functional check

Run this functional correctness gate in a **private Kaggle CPU notebook**. It
adds HTTP/Ray coverage to the reduced-copy implementation and measures zero
performance arms. Selected-source acceptance requires the 12 serializer controls
and all eight HTTP/Ray scenarios; this separate test recipe is not permission to
skip that coverage. It does not rerun or replace the historical v41 replay results.

Use the external source-only artifact `task14-http-ray-support-v56-source.zip`
(SHA256 `7962e2d004436ea888d0bc3e5b396ef49a5c24fa310a85c47440c564eafbc821`).
It contains five frozen Python support files and `support-manifest.json`.
Extract it outside the selected Relax checkout. The four runtime files have no
private cloud-owner, operation-directory or export-worker imports; the fifth
provides the 12 evidence-serializer controls. Keep the five files together.

Bind the actual candidate source to the run manifest before execution. This
candidate uses base `13bdd1b412c5942a995070ea44716feeb48e9df6`, pipeline SHA256
`2455c0ad053cc54025a8b618d1e03fa99854b54e13dd6ff75128bb92c23559f6`,
the added eight semantic regressions, and unchanged service SHA256
`44451ed7d74f1c7fa3a56c744f792332ffc458a6f64535c6c823aa221c7e5e5b`.
No thread export worker is part of this candidate. Record the complete source
inventory, actual imported pipeline, support hashes and provided Torch build
before and after the run. The helpers accept the existing `--source` argument;
the outer owner must enforce the candidate pipeline and source manifest.

The original v56 execution used pristine `bd8b059` with pipeline SHA256
`6eaa9a9dbdb4386b2f495ff091dab5d3c3c6bede7a56f79258d114f079eaa671`.
That pass establishes the frozen support's provenance. It does **not** establish
that pipeline `2455` passed; the selected candidate needs a fresh run.

Reuse the pinned CPU requirements and input preparation from
[the replay recipe](session_replay_reproduction.md), preserving the notebook's
provided Torch build. HTTP smoke needs only the pinned tokenizer files: prepare
them under `/tmp` without `--fetch-corpus`. No model weights or trajectories are
needed for these HTTP scenarios. After setting the three paths below to the
selected source, extracted support and prepared tokenizer, run the child commands
from that source checkout with a fresh output directory:

```bash
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$TASK14_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
TASK14_HTTP_OUTPUT=/kaggle/working/task14-http-functional-001
test ! -e "$TASK14_HTTP_OUTPUT"
mkdir "$TASK14_HTTP_OUTPUT"
python "$TASK14_HTTP_SUPPORT/test_evidence_projection.py" \
  --source "$TASK14_SOURCE" \
  --output "$TASK14_HTTP_OUTPUT/evidence-projection-controls"
python "$TASK14_HTTP_SUPPORT/smoke_session_export_http_ray.py" \
  --source "$TASK14_SOURCE" --tokenizer "$TASK14_TOKENIZER" \
  --json-out "$TASK14_HTTP_OUTPUT/http-ray-smoke.json"
```

Require zero exit codes, exactly 12 serializer controls, eight completed HTTP/Ray
scenarios, GPU count zero and the full cleanup receipts. Preserve first-error,
finally, source and `.projection.json` records on failure; do not automatically
rerun a failed case. Progress is written to
`http-ray-cleanup/scenario-progress.json`. The child supervisor has a 300-second
limit and reaps its owned subprocess tree. These files do not supply an
independent notebook watchdog or a Codex-chat wakeup. Background execution needs
the separately verified bounded owner and independent monitor; persistent files
alone are not evidence of automatic model follow-up.

The real components are loopback HTTP/SSE, local CPU Ray actors/Serve, RPC
cancellation, the original Session/Forest/compiler/tokenizer/export path and
training-content/ownership checks. Generation results, logprobs and reward are
deterministic fixtures. Each Session's Agent process is an in-memory fixture;
the separately retained native-process tests provide that coverage. This is not
SGLang model inference, GPU training, model TTFT/ITL or multi-node evidence.

The eight scenarios cover HTTP/training roundtrip, SSE disconnect plus explicit
drop, delivered RPC cancellation, conversion failure plus recovery, a custom
compiler, top-k content/ownership, concurrent owner drop and concurrent owner
shutdown. Cancellation and close gates are at asynchronous backend cleanup
**after** synchronous assembly. Already-created finish owners retain their
original payload; discarded groups reject later take. Different gate phases can
register a different outcome first, without changing the shared first-owner
publication strategy. Keep once-only publication/termination, resource/permit
and PID/creation-time/FD/port/directory cleanup checks strict.

A separate controlled synchronous-builder pause records health readiness before
an independent driver release. It is a scheduling diagnostic, not a natural
export latency benchmark. Thread executor/queued-reader contracts are N/A here.

## Mixed HTTP/Ray load

Use a fresh private Kaggle CPU runtime and the same pinned CPU requirements and
tokenizer preparation above. This run measures short HTTP chat completions
while a burst of long Session exports is active. All four arms share one VM.
Each arm warms up with one long export, measures four 8,192-token exports and
schedules 32 short HTTP requests at 2 ms intervals. The order is baseline,
candidate, candidate, baseline.

The comparison uses the same checkout for both versions. The baseline replaces
only the pipeline file with the blob from upstream `65d9356`, which also matches
the pipeline at `a155b62`. The candidate uses the pipeline from PR #537 at
`f8cc779`. Keep the original support archive outside both checkouts.

```bash
RUN_DIR=$(mktemp -d /tmp/task14-http-ray-mixed.XXXXXX)
CANDIDATE=f8cc779f0b5c98aa6261387d2c0a50a083ffc63a
BASELINE=65d9356d55107c7b9ba350f60b03bb0d8c778827
PIPELINE=relax/agentic/pipeline/__init__.py
git clone --no-checkout https://github.com/Timothy-kira/Relax.git "$RUN_DIR/candidate"
git -C "$RUN_DIR/candidate" checkout --detach "$CANDIDATE"
git -C "$RUN_DIR/candidate" fetch --depth 1 https://github.com/redai-studio/Relax.git "$BASELINE"
git -C "$RUN_DIR/candidate" worktree add --detach "$RUN_DIR/baseline" "$CANDIDATE"
git -C "$RUN_DIR/candidate" show "$BASELINE:$PIPELINE" > "$RUN_DIR/baseline/$PIPELINE"
sha256sum "$RUN_DIR/baseline/$PIPELINE" "$RUN_DIR/candidate/$PIPELINE"
python "$TASK14_MIXED_RUNNER" \
  --baseline "$RUN_DIR/baseline" --candidate "$RUN_DIR/candidate" \
  --tokenizer "$TASK14_TOKENIZER" --support "$TASK14_HTTP_ARCHIVE" \
  --output /kaggle/working/task14-http-ray-mixed-arms
```

Set `TASK14_TOKENIZER` to the pinned tokenizer scratch directory and
`TASK14_HTTP_ARCHIVE` to the path of the original v56 support archive. Set
`TASK14_MIXED_RUNNER` to the released
[`run_session_export_http_ray_mixed.py`](https://github.com/Timothy-kira/Relax/releases/download/task14-http-ray-v2-evidence-20261104/run_session_export_http_ray_mixed.py)
(SHA256 `46efbbd2306bd3745809694c1c564e01b8141cd807630fe517340be1b5214e1d`).
The expected pipeline SHA256 values are
`1d71188e874912229dfcb500275972fa361248ca3b13d49d9220078a95682f19`
for baseline and
`2455c0ad053cc54025a8b618d1e03fa99854b54e13dd6ff75128bb92c23559f6`
for candidate.

The short-request clock runs from scheduled arrival to completed HTTP
response and includes client scheduling delay. With 32 requests per arm,
nearest-rank P99 is the maximum observed request latency. The runner writes
per-request timestamps, overlap counts, export times, shard process CPU time
and training digests. The per-arm HTTP/Ray receipts record permit release,
owned process cleanup and port closure. Generation and the in-memory Agent
use deterministic fixtures, so the results describe this service-path
workload and its fixed export inputs.
