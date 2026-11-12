# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Exclusive eight-GPU admission and exact owned-process lifecycle helpers."""

import json
import os
import shutil
import subprocess
from pathlib import Path


REG = Path(os.environ["RT"]) / "claims"
REG.mkdir(exist_ok=True)


def identity(pid: int) -> dict:
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return dict(pid=pid, pgid=int(fields[2]), start_ticks=int(fields[19]))


def group_members(pgid: int) -> list[int]:
    members = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == pgid and fields[0] != "Z":
                members.append(int(path.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return members


def claim_paths(registry: Path, task: str, root: Path) -> list[Path]:
    paths = [registry / f"{task}_{root.name}_gpu{gpu}_claim.json" for gpu in range(8)]
    assert not any(path.exists() for path in paths), "Claim collision"
    return paths


def resources() -> dict:
    claims = [
        str(p) for p in REG.glob("*_claim.json") if not Path(json.loads(p.read_text())["release_file"]).is_file()
    ]
    assert not claims, "An owned claim remains active"
    for path in (os.environ["RT"], os.environ["ROOT"]):
        assert shutil.disk_usage(path).free >= 100_000_000_000
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.total,memory.used,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    gpus = [line.split(", ") for line in raw.strip().splitlines()]
    assert len(gpus) == 8 and [int(x[0]) for x in gpus] == list(range(8))
    assert all(int(x[2]) >= 80000 and int(x[3]) < 2000 and int(x[4]) < 85 for x in gpus)
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,used_memory", "--format=csv,noheader,nounits"], text=True
    ).strip()
    keeper = os.environ.get("RETOOL_KEEPER_PID")
    if apps:
        assert keeper, "GPU processes exist; never evict unknown work"
        parent = Path("/proc") / keeper
        assert b"gpu_keeper.py" in (parent / "cmdline").read_bytes()
        children = (parent / "task" / keeper / "children").read_text().split()
        fillers = []
        for pid in children:
            argv = (Path("/proc") / pid / "cmdline").read_bytes().split(b"\0")
            if argv[0] == b"nvidia-smi":
                continue
            assert len(argv) >= 7 and argv[1] == b"-c" and b"target_util" in argv[2]
            fillers.append(int(argv[3]))
        assert sorted(fillers) == list(range(8))
        rows = [line.split(", ") for line in apps.splitlines()]
        assert len(rows) == 8 and len({x[0] for x in rows}) == 8
        assert all(int(x[1]) < 2000 for x in rows)
    return dict(
        status="EIGHT_DEDICATED_GPUS_EXCLUSIVE_AVAILABLE",
        physical=gpus,
        live_unreleased_claims=claims,
        keeper_present=bool(apps),
        keeper_yield="original memory threshold only",
        no_NVML_host_PID_container_join=True,
    )
