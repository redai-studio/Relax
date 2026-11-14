# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Recover private source scratch under an exclusive, crash-released lease."""

import shutil
import tempfile
from pathlib import Path

from relax.utils.logging_utils import get_logger

from .store import atomic_write, canonical, digest, writer_lock


logger = get_logger(__name__)


class ScratchWorkspace:
    def __init__(self, root, scope, *, protected=()):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.key = digest(scope.encode())
        self.prefix = f"candidate-{self.key}-"
        self.record = self.root / f"cleanup-{self.key}.json"
        self.lease = writer_lock(self.root / f"owner-{self.key}.lock")
        self.lease.__enter__()
        self.closed = False
        try:
            self.reconcile(protected)
        except BaseException:
            self.close()
            raise

    def reconcile(self, protected=()):
        protected = {Path(path).resolve() for path in protected}
        pending = []
        for path in sorted(self.root.glob(self.prefix + "*")):
            if path.resolve() in protected:
                continue
            if not self.remove(path):
                pending.append(path.name)
        atomic_write(self.record, canonical({"scope": self.key, "pending_cleanup": pending}))
        return pending

    def temporary(self):
        if self.closed:
            raise RuntimeError("Source scratch owner is closed")
        return Path(tempfile.mkdtemp(prefix=self.prefix, dir=self.root))

    def remove(self, path):
        path = Path(path)
        if path.parent.resolve() != self.root.resolve() or not path.name.startswith(self.prefix) or path.is_symlink():
            raise ValueError("Cleanup target is outside this source scratch owner")
        try:
            shutil.rmtree(path)
        except FileNotFoundError:
            pass
        except OSError:
            # The directory name is itself a durable retry record. Reconcile
            # also records every remaining name for operators and restarts.
            logger.warning("Source scratch cleanup pending: %s", path, exc_info=True)
            remaining = [p.name for p in self.root.glob(self.prefix + "*") if p.is_dir()]
            atomic_write(self.record, canonical({"scope": self.key, "pending_cleanup": remaining}))
            return False
        return True

    def close(self):
        if not self.closed:
            self.lease.__exit__(None, None, None)
            self.closed = True
