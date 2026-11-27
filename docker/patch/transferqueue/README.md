# TransferQueue patches

The source is pinned to `9784ad0d8a5f0db93ff8326379ab2a6d06e640a0`.
Its archive SHA256 is
`5da896714784e4f3c03f54ef6454cc9893f6e143d447cf520e0c3bafd17aab31`.

The existing `9784ad0-assignment-transform.patch` now contains all fixes for
this version, applied once with zero fuzz:

- Preserve the existing sequence-length sampler assignment transform.
- Reuse one logical insert ID across client socket retries, returning the original
  indices after a lost reply. Reusing an ID with different arguments is rejected.
- Preserve completed tasks' drain status after partition cleanup, including across
  checkpoints. Unknown/future partitions and unfinished tasks stay incomplete.
- Return empty metadata immediately for empty token-budget polling requests, so
  the Controller request thread can acknowledge production notifications.
  Non-polling callers retain their bounded wait.

These incorporate the former source-branch
`9784ad0-insert-idempotency-drain-terminal.patch` and
`9784ad0-controller-notify-fairness.patch` into the existing version patch.

## Long-running controllers

There is no lifetime insert or partition-creation quota. Cleared insert requests
retain only their IDs with `None` tombstones; their metadata and signatures are
released. A late retry remains rejected even after a same-name partition is
recreated. Completed-task sets are retained per cleared partition and reset on
recreation. Both histories are checkpointed, and older checkpoints remain readable.

History is owned by the Controller and released when it exits. It grows with the
number of distinct logical inserts and cleared partitions; it is not silently
evicted because that would let an arbitrarily late retry allocate again. CPU
validation measured approximately 98 MiB per million retired 32-character IDs and
282 MiB per million retired partitions with one completed task (CPython 3.12;
allocator/RSS and active metadata can add overhead).

Consumers must back off between empty polls. The patch preserves the production
notification ACK contract and does not retry an ambiguous data write.

## Reproducible wheel

Both CUDA Docker recipes build an isolated wheel from the same `TRAIN_IMAGE`
and mount that wheel into the final stage; they do not patch installed packages.
To build without installing:

```bash
TQ_BUILD_PYTHON=/path/to/build-venv/bin/python \
  bash docker/scripts/build_transfer_queue.sh /tmp/tq-wheel
```

The environment needs `build`, `setuptools` and `wheel`. The script verifies
the archive, patch and baseline source hashes, applies the single patch, and builds
`0.1.10.dev0+relax.9784ad0.lifecycle.ack`. It verifies the wheel RECORD and
packaged source hashes, includes `_relax_build.json`, and emits a build receipt.
Optional second and third arguments provide the patch and a downloaded archive.

## Distribution scope and follow-up

This change distributes the fixes through Relax's training-image build. Fully
async training, including a Relax source checkout, requires an image containing
this patch; direct installation from the TransferQueue repository is outside
the current distribution scope.

Keep this patch until Relax pins a TransferQueue revision that includes the insert retry, terminal-state, and nonblocking polling fixes listed above.
