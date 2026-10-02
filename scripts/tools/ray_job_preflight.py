# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Read-only, fail-closed admission check for shared-cluster experiments.

Run under the launcher's submission lock. This is a snapshot, not a lease: non-
cooperating submitters may still race it, so callers must never infer
permission to delete resources from a successful check.
"""

import argparse
import json
import socket
import subprocess
import urllib.parse
import urllib.request
from typing import Any, Dict, List


def assert_idle(snapshot: Dict[str, Any]) -> None:
    """Reject busy or incomplete observations without modifying the cluster."""
    jobs = snapshot.get("jobs")
    groups = snapshot.get("placement_groups")
    applications = snapshot.get("applications")
    processes = snapshot.get("gpu_processes")
    nodes = snapshot.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("missing node inventory")
    if any(not isinstance(node, dict) or node.get("state") not in {"ALIVE", "DEAD"} for node in nodes):
        raise ValueError("unknown node state")
    alive = [node for node in nodes if node["state"] == "ALIVE"]
    if len(nodes) >= 10000 or len(alive) != 1 or alive[0].get("local") is not True:
        raise ValueError("safe submission requires exactly one live local node")
    if not isinstance(jobs, list) or not isinstance(groups, list):
        raise ValueError("missing jobs or placement-group inventory")
    if not isinstance(applications, dict) or not isinstance(processes, list):
        raise ValueError("missing Serve or GPU-process inventory")
    if len(groups) >= 10000:
        raise ValueError("placement-group inventory may be truncated")
    if any(not isinstance(job, dict) or job.get("status") not in {"SUCCEEDED", "FAILED", "STOPPED"} for job in jobs):
        raise ValueError("non-terminal or unknown Ray job; refusing submission")
    if applications:
        raise ValueError("existing Serve applications; explicit ownership cleanup required")
    if any(not isinstance(group, dict) or group.get("state") != "REMOVED" for group in groups):
        raise ValueError("live or unknown placement group; refusing submission")
    if processes:
        raise ValueError("GPU processes already present; refusing submission")


def is_local_address(address: str) -> bool:
    """Binding succeeds only for an address belonging to this host."""
    try:
        family = socket.AF_INET6 if ":" in address else socket.AF_INET
        with socket.socket(family, socket.SOCK_DGRAM) as probe:
            probe.bind((address, 0))
        return True
    except OSError:
        return False


def collect_snapshot(address: str) -> Dict[str, Any]:
    """Query the supplied dashboard and local GPU processes, without
    cleanup."""
    from ray.job_submission import JobSubmissionClient
    from ray.util.state import list_nodes, list_placement_groups

    parsed = urllib.parse.urlparse(address)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("address must be an explicit HTTP(S) Ray dashboard")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("dashboard address must not contain credentials, query or fragment")
    jobs = JobSubmissionClient(address).list_jobs()
    nodes = list_nodes(address=address, detail=True, limit=10000, timeout=10, raise_on_missing_output=True)
    groups = list_placement_groups(address=address, detail=True, limit=10000, timeout=10, raise_on_missing_output=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(address.rstrip("/") + "/api/serve/applications/", timeout=10) as response:
        serve = json.load(response)
    gpu = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    processes: List[str] = [line.strip() for line in gpu.stdout.splitlines() if line.strip()]
    return {
        "nodes": [{"state": node.state, "local": is_local_address(node.node_ip)} for node in nodes],
        "jobs": [{"status": str(job.status)} for job in jobs],
        "placement_groups": [{"state": group.state} for group in groups],
        "applications": serve.get("applications"),
        "gpu_processes": processes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", required=True)
    args = parser.parse_args()
    try:
        assert_idle(collect_snapshot(args.address))
    except Exception as exc:
        parser.exit(75, f"Safe submission refused: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
