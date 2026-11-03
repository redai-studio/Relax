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
