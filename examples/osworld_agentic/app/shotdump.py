"""Per-turn screenshot + trajectory dump (opt-in, sampled by task_id)."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path


_S = {"dir": None, "on": False}


def _sess_tag() -> str:
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == "--input-json" and i + 1 < len(argv):
            return Path(argv[i + 1]).parent.name.replace("relax-agentic-command-", "")
    return "sess"


def init(task_id: str, instruction: str = "") -> None:
    _S["on"] = False
    _S["dir"] = None
    base = os.environ.get("OSWORLD_SHOT_DUMP_DIR")
    if not base or not task_id:
        return
    every = int(os.environ.get("OSWORLD_SHOT_EVERY_N_TASKS") or 37)
    if every > 1:
        h = int(hashlib.md5(task_id.encode()).hexdigest(), 16)
        if h % every != 0:
            return
    d = Path(base) / task_id / (time.strftime("%m%d-%H%M%S") + "_" + _sess_tag())
    try:
        d.mkdir(parents=True, exist_ok=True)
        meta = {
            "task_id": task_id,
            "instruction": instruction,
            "run_tag": os.environ.get("OSWORLD_RUN_TAG", ""),
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    except Exception:
        return
    _S["dir"] = d
    _S["on"] = True


def shot(step_idx: int, data) -> None:
    if not _S["on"] or not data:
        return
    try:
        (_S["dir"] / ("%03d.png" % step_idx)).write_bytes(bytes(data))
    except Exception:
        pass


def turn(rec: dict) -> None:
    if not _S["on"]:
        return
    try:
        with (_S["dir"] / "traj.jsonl").open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass


def finish(rec: dict) -> None:
    if not _S["on"]:
        return
    try:
        p = _S["dir"] / "meta.json"
        cur = json.loads(p.read_text()) if p.exists() else {}
        cur.update(rec)
        cur["ended_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        p.write_text(json.dumps(cur, ensure_ascii=False, indent=2))
    except Exception:
        pass
