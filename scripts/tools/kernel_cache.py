#!/usr/bin/env python3

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Prepare detached Relax kernel-cache agents on every GPU node."""

import argparse
import json

from relax.distributed.ray.kernel_cache import (
    KernelCacheConfig,
    prepare_kernel_cache_agents,
    prepare_local_kernel_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-dir", required=True)
    parser.add_argument("--local-dir", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--cache-key", required=True)
    parser.add_argument("--build-fingerprint", required=True)
    parser.add_argument("--sync-interval-sec", type=int, default=900)
    parser.add_argument("--lease-timeout-sec", type=int, default=600)
    parser.add_argument("--startup-timeout-sec", type=int, default=7200)
    parser.add_argument("--compression", choices=("none", "gzip"), default="none")
    parser.add_argument("--mode", choices=("attach", "cluster", "local"), default="cluster")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = KernelCacheConfig(
        shared_dir=args.shared_dir,
        local_dir=args.local_dir,
        session_id=args.session_id,
        cache_key=args.cache_key,
        build_fingerprint=args.build_fingerprint,
        sync_interval_sec=args.sync_interval_sec,
        lease_timeout_sec=args.lease_timeout_sec,
        startup_timeout_sec=args.startup_timeout_sec,
        compression=args.compression,
    )
    if args.mode == "local":
        result = prepare_local_kernel_cache(config)
    else:
        result = prepare_kernel_cache_agents(config, attach_only=args.mode == "attach")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
