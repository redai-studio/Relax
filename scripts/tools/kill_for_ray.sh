#!/bin/bash


# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Kill ONLY stale Relax training / SGLang inference worker processes left behind
# when a driver dies abnormally (SIGKILL) and Ray fails to reap its actors.
#
# Match known worker signatures and explicitly exclude Ray infrastructure
# processes to protect the running cluster during cleanup.


set -uo pipefail


echo "=== Cleaning up residual Relax/SGLang worker processes (whitelist) ==="


# Worker signatures that are SAFE to kill. These are Relax training actors, the
# training entrypoint, and SGLang inference engine procs. None of these ever name
# a Ray daemon.
WORKER_RE='ray::MegatronTrainRayActor|ray::ServeReplica|relax\.entrypoints\.train|sglang::|sglang\.srt|sglang\.launch|python3? -m sglang'


# HARD guard: never touch any Ray infrastructure process, even if a signature above
# somehow overlaps. Anything matching this is skipped unconditionally.
GUARD_RE='ray start|raylet|gcs_server|plasma|dashboard|log_monitor|monitor\.py|ray::IDLE|ray::DashboardAgent|ray::ServeController|runtime_env_agent|ray\.util\.client\.server|kill_for_ray'


# Collect PIDs: match worker signature AND not a guarded infra proc AND not self/grep.
mapfile -t PIDS < <(
  ps -eo pid,cmd --no-headers \
    | grep -E "$WORKER_RE" \
    | grep -vE "$GUARD_RE" \
    | grep -vE '\bgrep\b' \
    | awk -v self="$$" '$1 != self {print $1}'
)


if [ "${#PIDS[@]}" -eq 0 ]; then
  echo "=== No residual worker processes on $(hostname) ==="
  exit 0
fi


echo "=== Killing ${#PIDS[@]} residual worker PID(s) on $(hostname): ${PIDS[*]} ==="
# SIGTERM first for a chance to release GPU cleanly, then SIGKILL the stubborn ones.
kill -15 "${PIDS[@]}" 2>/dev/null || true
sleep 2
kill -9 "${PIDS[@]}" 2>/dev/null || true
echo "=== Done on $(hostname) ==="
