#!/usr/bin/env python3
"""Keep MegatronTrainRayActor.train processes and clamp malformed Python
stacks.

Usage: python3 scripts/profile/fix_trace_stacks.py tmp/traces/*.trace.json.gz
Writes fixed/<name>.fixed-stacks.trace.json.gz beside each input.
By default writes fixed/merged.trace.json.gz beside the first input, grouped by CPU/GPU stream, then rank.
Use --merge PATH only to override the merged output path.
Uses only the Python standard library and streams large traces.

Fix and trim each rank independently, without merging (default: middle 4 forward calls):
  python3 scripts/profile/fix_trace_stacks.py tmp/traces/*.trace.json.gz --per-rank
Writes fixed/<name>.fixed-stacks.trimmed.trace.json.gz beside each input.
--output-dir overrides the destination; --overview retains the full-step overview instead.
Time/microbatch windows are selected independently for each rank.

Independently trim an existing merged trace (does not repair or merge again):
  python3 scripts/profile/fix_trace_stacks.py merged.trace.json.gz --trim-merged --microbatch-count 4
Or select a shared time window relative to the first timestamp, in milliseconds:
  python3 scripts/profile/fix_trace_stacks.py merged.trace.json.gz --trim-merged \
      --window-start-ms 1000 --window-duration-ms 2000
Writes merged.trimmed.trace.json.gz by default; --output overrides this path.
To trim and merge existing per-rank fixed files into a new file:
  python3 scripts/profile/fix_trace_stacks.py fixed/*.fixed-stacks.trace.json.gz \
      --trim-fixed --microbatch-count 4 --output fixed/merged.trimmed.trace.json.gz
Use --trim-fixed --overview instead for a full-step overview of all PP stages,
written to merged.overview.trace.json.gz by default.
From raw dumps, --export-views --output-dir PATH produces both final views.
--reuse-fixed optionally reuses newer fixed copies with identical headers.
"""

import argparse
import gzip
import json
import math
import os
import re
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterator


CHUNK = 1024 * 1024
DECODER = json.JSONDecoder()
SEPARATORS = re.compile(r"[\s,]*")
TRAIN_PROCESS = "MegatronTrainRayActor.train"


def trace_parts(source: Path) -> Iterator:
    """Yield the header, (event, raw JSON) pairs, and tail without loading the
    trace."""
    with gzip.open(source, "rt", encoding="utf-8") as inp:
        buf = inp.read(CHUNK)
        bracket = buf.index("[", buf.index('"traceEvents"'))
        yield buf[: bracket + 1]
        pos = bracket + 1
        while True:
            pos = SEPARATORS.match(buf, pos).end()
            if pos >= len(buf):
                buf = inp.read(CHUNK)
                pos = 0
                if not buf:
                    raise ValueError("Truncated traceEvents array")
                continue
            if buf[pos] == "]":
                tail = buf[pos:] + inp.read()
                json.loads('{"traceEvents":[]' + tail[1:])
                yield tail
                return
            try:
                event, end = DECODER.raw_decode(buf, pos)
            except json.JSONDecodeError:
                more = inp.read(CHUNK)
                if not more:
                    raise
                buf = buf[pos:] + more
                pos = 0
                continue
            raw = buf[pos:end]
            pos = end
            yield event, raw


def read_layout(source: Path) -> dict:
    processes, labels, threads = {}, {}, {}
    for part in trace_parts(source):
        if isinstance(part, str):
            continue
        event, _ = part
        if event.get("ph") != "M":
            continue
        pid = event.get("pid")
        args = event.get("args", {})
        if event.get("name") == "process_name":
            processes[pid] = args.get("name", "")
        elif event.get("name") == "process_labels":
            labels[pid] = args.get("labels", "")
        elif event.get("name") == "thread_name":
            threads[pid, event.get("tid")] = args.get("name", "")
    selected = {pid for pid, name in processes.items() if name in (TRAIN_PROCESS, f"ray::{TRAIN_PROCESS}")}
    if not selected:
        raise ValueError(f"No {TRAIN_PROCESS} process found in {source}")
    return {"processes": selected, "labels": labels, "threads": threads}


def fixed_chunks(source: Path, counts: dict, layout: dict = None) -> Iterator[str]:
    if layout is None:
        layout = read_layout(source)
    stacks, last_start = {}, {}
    first = True
    for part in trace_parts(source):
        if isinstance(part, str):
            yield part
            continue
        event, raw = part
        counts["events"] += 1
        if event.get("pid") not in layout["processes"]:
            continue
        if event.get("cat") == "python_function" and event.get("ph") == "X":
            args = event.get("args", {})
            key = (event.get("pid"), args.get("Python thread", event.get("tid")))
            ts = event["ts"]
            if ts < last_start.get(key, ts):
                raise ValueError(f"Python events are not chronological on thread {key}")
            last_start[key] = ts
            stack = stacks.setdefault(key, [])
            while stack and stack[-1] <= ts:
                stack.pop()
            end_time = ts + event["dur"]
            if stack and end_time > stack[-1]:
                duration = round(stack[-1] - ts, 3)
                # Chrome JSON timestamps are microseconds; tolerate float rounding.
                if event["dur"] - duration > 0.002:
                    event["dur"] = duration
                    raw = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
                    counts["corrected"] += 1
                end_time = stack[-1]
            stack.append(end_time)
        yield ("" if first else ",\n") + raw
        first = False


def write_gzip(output: Path, chunks: Iterator[str]) -> None:
    """Publish only a complete output, leaving existing files intact on
    failure."""
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=output.name + ".", suffix=".partial", dir=output.parent)
    os.close(fd)
    try:
        with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=3) as out:
            pieces = []
            size = 0
            for piece in chunks:
                pieces.append(piece)
                size += len(piece)
                if size >= CHUNK:
                    out.write("".join(pieces))
                    pieces.clear()
                    size = 0
            out.write("".join(pieces))
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def fix(source: Path, output: Path = None) -> Path:
    suffix = ".trace.json.gz"
    if not source.name.endswith(suffix):
        raise ValueError(f"Expected a {suffix} file: {source}")
    if source.name.endswith(".fixed-stacks" + suffix):
        raise ValueError(f"Already a repaired trace: {source}")
    output = output or source.parent / "fixed" / (source.name[: -len(suffix)] + ".fixed-stacks" + suffix)
    counts = {"events": 0, "corrected": 0}
    write_gzip(output, fixed_chunks(source, counts))
    print(f"{source}: {counts['events']:,} events, {counts['corrected']:,} durations fixed -> {output}")
    return output


def read_header(source: Path) -> dict:
    with gzip.open(source, "rt", encoding="utf-8") as inp:
        buf = inp.read(CHUNK)
    bracket = buf.index("[", buf.index('"traceEvents"'))
    header = json.loads(buf[: bracket + 1] + "]}")
    del header["traceEvents"]
    return header


def merged_chunks(sources: list) -> Iterator[str]:
    headers = [read_header(source) for source in sources]
    # Different profiler time bases must be aligned, never independently zeroed.
    if any(not isinstance(h.get("baseTimeNanoseconds"), int) for h in headers):
        raise ValueError("Merging requires baseTimeNanoseconds in every PyTorch trace header")
    base = min(h["baseTimeNanoseconds"] for h in headers)
    yield '{"displayTimeUnit":"ms","baseTimeNanoseconds":' + str(base) + ',"traceEvents":['
    first = True
    processes, threads, ids = {}, {}, {}
    track_order, thread_processes = {}, {}
    frames = {}
    source_metadata = []

    def mapped(mapping: dict, key: tuple) -> int:
        if key not in mapping:
            mapping[key] = len(mapping) + 1
        return mapping[key]

    def encode(event: dict) -> str:
        nonlocal first
        result = ("" if first else ",\n") + json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        first = False
        return result

    for index, (source, header) in enumerate(zip(sources, headers)):
        rank = header.get("distributedInfo", {}).get("rank", source.name)
        label = f"Rank {rank}"
        offset = (header["baseTimeNanoseconds"] - base) / 1000
        counts = {"events": 0, "corrected": 0}
        layout = read_layout(source)
        chunks = fixed_chunks(source, counts, layout)
        next(chunks)  # Header; remaining chunks are individual events followed by the JSON tail.
        rank_key = (0, int(rank)) if str(rank).isdigit() else (1, str(rank))
        track_names = {}
        tail_metadata = {}
        for chunk in chunks:
            if chunk.startswith("]"):
                tail_metadata = json.loads('{"traceEvents":[]' + chunk[1:])
                del tail_metadata["traceEvents"]
                break
            event = json.loads(chunk.lstrip(",\n"))
            original_pid = event.get("pid")
            original_tid = event.get("tid")
            if event.get("ph") == "M":
                # Rebuild layout metadata after remapping; old sort indices no longer apply.
                continue
            device = layout["labels"].get(original_pid, "CPU")
            thread_name = layout["threads"].get((original_pid, original_tid), str(original_tid))
            stream = re.search(r"\bstream\s+(\d+)", thread_name)
            if device.startswith("GPU") or stream:
                stream_id = int(stream[1]) if stream else original_tid
                group = f"GPU stream {stream_id}" if original_tid is not None else "GPU"
                group_key = (1, int(stream_id)) if str(stream_id).isdigit() else (2, str(stream_id))
            else:
                group, group_key = "CPU", (0, 0)
            event["pid"] = mapped(processes, (group_key, group))
            track_key = (index, original_pid, original_tid)
            if "tid" in event:
                event["tid"] = mapped(threads, track_key)
                thread_processes[track_key] = event["pid"]
                track_order[track_key] = (rank_key, str(original_pid), str(original_tid))
                track_names[track_key] = (event["pid"], f"{label} / {device} / {thread_name}")
            if "ts" in event and offset:
                event["ts"] += offset
            for field in ("id", "bind_id"):
                if field in event:
                    event[field] = mapped(ids, (index, "id", str(event[field])))
            if "id2" in event:
                event["id2"] = {
                    kind: hex(mapped(ids, (index, kind, str(value)))) for kind, value in event["id2"].items()
                }
            for field in ("sf", "esf"):
                if field in event:
                    event[field] = f"{index}:{event[field]}"
            yield encode(event)
        metadata = {**header, **tail_metadata}
        for key, frame in metadata.pop("stackFrames", {}).items():
            frame = dict(frame)
            if "parent" in frame:
                frame["parent"] = f"{index}:{frame['parent']}"
            frames[f"{index}:{key}"] = frame
        source_metadata.append({"source": str(source), "metadata": metadata})
        for track_key, (pid, name) in track_names.items():
            yield encode(
                {"ph": "M", "name": "thread_name", "pid": pid, "tid": threads[track_key], "args": {"name": name}}
            )
        print(f"Merged {label}: {counts['events']:,} events", flush=True)
    for order, ((group_key, name), pid) in enumerate(sorted(processes.items())):
        yield encode({"ph": "M", "name": "process_name", "pid": pid, "args": {"name": name}})
        yield encode({"ph": "M", "name": "process_sort_index", "pid": pid, "args": {"sort_index": order}})
    for order, track_key in enumerate(sorted(track_order, key=track_order.get)):
        # Thread IDs are globally unique, but Chrome metadata still requires the containing pid.
        yield encode(
            {
                "ph": "M",
                "name": "thread_sort_index",
                "pid": thread_processes[track_key],
                "tid": threads[track_key],
                "args": {"sort_index": order},
            }
        )
    yield '],"stackFrames":' + json.dumps(frames) + ',"mergedSources":' + json.dumps(source_metadata) + "}"


def trim_window(source: Path, options: argparse.Namespace) -> tuple:
    """Select one common timestamp window; forward calls anchor microbatch
    selection."""
    first_ts = None
    markers, ranks = {}, {}
    source_rank = read_header(source).get("distributedInfo", {}).get("rank")
    for part in trace_parts(source):
        if isinstance(part, str):
            continue
        event, _ = part
        if "ts" in event:
            first_ts = min(first_ts, event["ts"]) if first_ts is not None else event["ts"]
        if event.get("ph") == "M" and event.get("name") == "thread_name":
            match = re.search(r"Rank (\d+) /", event.get("args", {}).get("name", ""))
            if match:
                ranks[event.get("tid")] = int(match[1])
        name = event.get("name", "")
        if event.get("ph") == "X" and (
            name == "forward_step" or ("schedules.py" in name and name.endswith(": forward_step"))
        ):
            markers.setdefault(event.get("tid"), []).append((event["ts"], event["ts"] + event["dur"]))
    if first_ts is None:
        raise ValueError("Trace has no timestamped events")
    if options.microbatch_count is not None:
        if source_rank is not None:
            for tid in markers:
                ranks.setdefault(tid, int(source_rank))
        candidates = [tid for tid in markers if tid in ranks]
        if not candidates:
            raise ValueError("No forward_step markers on merged CPU rank tracks; use an explicit time window")
        tid = min(candidates, key=lambda tid: (ranks[tid], -len(markers[tid])))
        calls = sorted(set(markers[tid]))
        # Ignore nested duplicate annotations of the same forward call.
        calls = [call for i, call in enumerate(calls) if i == 0 or call[0] >= calls[i - 1][1]]
        count = options.microbatch_count
        start = options.microbatch_start
        if start is None:
            start = max((len(calls) - count) // 2, 0)
        if start + count > len(calls):
            raise ValueError(f"Requested forward calls [{start}, {start + count}), only {len(calls)} found")
        lo = calls[start][0]
        hi = calls[start + count][0] if start + count < len(calls) else calls[-1][1]
        print(f"Window anchored to rank {ranks[tid]} forward calls [{start}, {start + count})", flush=True)
    else:
        lo = first_ts + options.window_start_ms * 1000
        hi = lo + options.window_duration_ms * 1000
    return lo, hi, ranks


def overlaps(event: dict, lo: float, hi: float) -> bool:
    if "ts" not in event:
        return False
    ts = event["ts"]
    duration = event.get("dur", 0)
    return ts < hi and (ts + duration > lo if duration > 0 else ts >= lo)


def flow_key(event: dict) -> tuple:
    return event.get("scope"), event.get("id"), json.dumps(event.get("id2"), sort_keys=True)


def trim_merged(
    source: Path, output: Path, options: argparse.Namespace, window: tuple = None, origin: Path = None
) -> None:
    """Stream a cropped copy, preserving linked launches, flows and referenced
    stack frames."""
    overview = getattr(options, "overview", False)
    lo, hi, ranks = (
        (-math.inf, math.inf, {}) if overview else (window if window is not None else trim_window(source, options))
    )
    correlations, flows = set(), set()
    for part in () if overview else trace_parts(source):
        if isinstance(part, str):
            continue
        event, _ = part
        if not overlaps(event, lo, hi):
            continue
        correlation = event.get("args", {}).get("correlation")
        if isinstance(correlation, int) and correlation >= 0:
            correlations.add((ranks.get(event.get("tid")), correlation))
        if event.get("ph") in ("s", "t", "f"):
            flows.add(flow_key(event))

    def chunks() -> Iterator[str]:
        first = True
        kept, total = 0, 0
        referenced = set()
        for part in trace_parts(source):
            if isinstance(part, str):
                if not part.startswith("]"):
                    yield part
                    continue
                tail = json.loads('{"traceEvents":[]' + part[1:])
                del tail["traceEvents"]
                frames = tail.get("stackFrames", {})
                pending = list(referenced)
                while pending:
                    frame = frames.get(pending.pop(), {})
                    parent = frame.get("parent")
                    if parent is not None and str(parent) not in referenced:
                        referenced.add(str(parent))
                        pending.append(str(parent))
                tail["stackFrames"] = {key: value for key, value in frames.items() if key in referenced}
                tail["trimInfo"] = {
                    "source": str(origin or source),
                    "start_us": None if overview else lo,
                    "end_us": None if overview else hi,
                    "input_events": total,
                    "output_events": kept,
                    "note": "Full-step overview; Python, CPU operator and fwdbwd events and Call stack text removed."
                    if overview
                    else "Intersecting events retain full durations; linked launches/flows may be outside "
                    "window. Memory events cover this window only, not a full-step memory history.",
                }
                yield "]," + json.dumps(tail, ensure_ascii=False, separators=(",", ":"))[1:]
                print(f"Retained {kept:,}/{total:,} events -> {output}", flush=True)
                continue
            event, raw = part
            total += 1
            correlation = event.get("args", {}).get("correlation")
            linked = correlation is not None and (ranks.get(event.get("tid")), correlation) in correlations
            flow = event.get("ph") in ("s", "t", "f") and flow_key(event) in flows
            if not (event.get("ph") == "M" or overlaps(event, lo, hi) or linked or flow):
                continue
            if overview and event.get("cat") in ("python_function", "cpu_op", "fwdbwd"):
                continue
            if options.drop_python and event.get("cat") == "python_function":
                continue
            if (overview or options.drop_call_stacks) and "Call stack" in event.get("args", {}):
                del event["args"]["Call stack"]
                raw = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
            referenced.update(str(event[key]) for key in ("sf", "esf") if key in event)
            yield ("" if first else ",\n") + raw
            first = False
            kept += 1

    if overview:
        print(f"Full-step overview: {source}", flush=True)
    else:
        print(f"Shared window: {lo / 1000:.3f}–{hi / 1000:.3f} ms (trace timestamps)", flush=True)
    write_gzip(output, chunks())


def trim_fixed(sources: list, output: Path, options: argparse.Namespace) -> None:
    """Crop per-rank traces on the same absolute clock, then merge cropped
    copies."""
    headers = [read_header(source) for source in sources]
    if any(not isinstance(header.get("baseTimeNanoseconds"), int) for header in headers):
        raise ValueError("Trimming fixed traces requires baseTimeNanoseconds for every input")
    reference = min(range(len(sources)), key=lambda i: int(headers[i].get("distributedInfo", {}).get("rank", i)))
    if options.overview:
        lo, hi = -math.inf, math.inf
    else:
        lo, hi, _ = trim_window(sources[reference], options)
    base = headers[reference]["baseTimeNanoseconds"]
    # Scratch files live beside the output, on the same shared filesystem.
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="trim-fixed-", dir=output.parent) as temporary:
        cropped = []
        for index, (source, header) in enumerate(zip(sources, headers)):
            offset = (header["baseTimeNanoseconds"] - base) / 1000
            target = Path(temporary) / f"rank-{index}.trace.json.gz"
            # Each input contains one rank, so correlation IDs are local here.
            trim_merged(source, target, options, window=(lo - offset, hi - offset, {}))
            cropped.append(target)
        write_gzip(output, merged_chunks(cropped))
    print(f"Trimmed and merged trace -> {output}", flush=True)


def fix_trim_per_rank(sources: list[Path], options: argparse.Namespace) -> None:
    """Repair and crop each raw trace, publishing separate files without a
    merge."""
    suffix = ".overview.trace.json.gz" if options.overview else ".trimmed.trace.json.gz"
    outputs = [
        (options.output_dir or source.parent / "fixed")
        / (source.name.removesuffix(".trace.json.gz") + ".fixed-stacks" + suffix)
        for source in sources
    ]
    protected = {source.resolve() for source in sources}
    if len({output.resolve() for output in outputs}) != len(outputs):
        raise ValueError("Per-rank output names collide; use distinct input filenames")
    if any(output.exists() or output.resolve() in protected for output in outputs):
        raise ValueError("Per-rank outputs must be new files; choose a new --output-dir")
    for source, output in zip(sources, outputs):
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="per-rank-", dir=output.parent) as temporary:
            repaired = fix(source, Path(temporary) / "repaired.trace.json.gz")
            window = None
            if not options.overview:
                lo, hi, _ = trim_window(repaired, options)
                # One rank per input: CPU and GPU correlations share a local key.
                window = (lo, hi, {})
            trim_merged(repaired, output, options, window=window, origin=source)


def export_views(sources: list, directory: Path, options: argparse.Namespace) -> None:
    """Build both final merged views from raw dumps, keeping intermediates
    temporary."""
    overview = directory / "merged.overview.trace.json.gz"
    detail = directory / f"merged.microbatch{options.microbatch_count}.trace.json.gz"
    if overview.exists() or detail.exists():
        raise ValueError("Output views already exist; choose a new --output-dir")
    headers = [read_header(source) for source in sources]
    if any("rank" not in header.get("distributedInfo", {}) for header in headers):
        raise ValueError("--export-views requires per-rank raw dumps with distributedInfo.rank")
    if len({header["distributedInfo"]["rank"] for header in headers}) != len(headers):
        raise ValueError("Expected one input dump per rank")
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="export-views-", dir=directory) as temporary:
        repaired = []
        for index, (source, header) in enumerate(zip(sources, headers)):
            cached = source.parent / "fixed" / source.name.replace(".trace.json.gz", ".fixed-stacks.trace.json.gz")
            if (
                options.reuse_fixed
                and cached.is_file()
                and cached.stat().st_mtime_ns >= source.stat().st_mtime_ns
                and read_header(cached) == header
            ):
                print(f"Reusing repaired trace: {cached}", flush=True)
                repaired.append(cached)
            else:
                repaired.append(fix(source, Path(temporary) / f"rank-{index}.fixed-stacks.trace.json.gz"))
        detail_options = argparse.Namespace(**vars(options))
        detail_options.overview = False
        overview_options = argparse.Namespace(**vars(options))
        overview_options.overview = True
        # The two outputs are independent; separate processes also isolate their
        # streaming state and let the small overview finish before the detail.
        with ProcessPoolExecutor(max_workers=2) as pool:
            tasks = [
                pool.submit(trim_fixed, repaired, overview, overview_options),
                pool.submit(trim_fixed, repaired, detail, detail_options),
            ]
            for task in tasks:
                task.result()
    print(f"Both views complete:\n{overview}\n{detail}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path, nargs="+", help="PyTorch .trace.json.gz files")
    parser.add_argument(
        "--merge", type=Path, help="Merged output path (default: first input directory/fixed/merged.trace.json.gz)"
    )
    parser.add_argument("--trim-merged", action="store_true", help="Only trim one existing merged trace")
    parser.add_argument("--trim-fixed", action="store_true", help="Trim existing per-rank fixed traces and merge them")
    parser.add_argument(
        "--per-rank", action="store_true", help="Fix and trim each raw rank independently; never merge"
    )
    parser.add_argument(
        "--overview", action="store_true", help="Full-step overview: omit Python/CPU ops and stack text"
    )
    parser.add_argument("--export-views", action="store_true", help="From raw dumps, export overview and local detail")
    parser.add_argument("--output-dir", type=Path, help="Output directory for --export-views or --per-rank")
    parser.add_argument("--reuse-fixed", action="store_true", help="Reuse newer fixed copies with matching headers")
    parser.add_argument("--output", type=Path, help="New trimmed output path; input is never overwritten")
    parser.add_argument("--window-start-ms", type=float, default=0, help="Window offset from first timestamp")
    parser.add_argument("--window-duration-ms", type=float, help="Shared window duration")
    parser.add_argument(
        "--microbatch-count", type=int, help="Anchor window to N consecutive forward calls on first rank"
    )
    parser.add_argument("--microbatch-start", type=int, help="Zero-based forward call index (default: middle calls)")
    parser.add_argument("--drop-python", action="store_true", help="Remove Python timeline events in trimmed output")
    parser.add_argument("--drop-call-stacks", action="store_true", help="Remove repeated operator Call stack text")
    options = parser.parse_args()
    if options.export_views:
        if (
            options.trim_fixed
            or options.trim_merged
            or options.per_rank
            or options.overview
            or options.merge
            or options.output
            or options.window_duration_ms is not None
            or options.window_start_ms != 0
            or options.drop_python
            or options.drop_call_stacks
        ):
            parser.error("--export-views accepts raw inputs, --output-dir and microbatch options only")
        if not options.output_dir:
            parser.error("--export-views requires --output-dir")
        if len({p.resolve() for p in options.traces}) != len(options.traces):
            parser.error("Duplicate input files")
        if any(not p.name.endswith(".trace.json.gz") or ".fixed-stacks." in p.name for p in options.traces):
            parser.error("--export-views requires raw .trace.json.gz inputs")
        options.microbatch_count = options.microbatch_count if options.microbatch_count is not None else 4
        if options.microbatch_count <= 0 or (options.microbatch_start is not None and options.microbatch_start < 0):
            parser.error("Microbatch count must be positive and start must be non-negative")
        export_views(options.traces, options.output_dir, options)
        raise SystemExit(0)
    if options.reuse_fixed or (options.output_dir and not options.per_rank):
        parser.error("--reuse-fixed requires --export-views; --output-dir requires --export-views or --per-rank")
    if options.per_rank:
        if options.trim_fixed or options.trim_merged or options.merge or options.output:
            parser.error("--per-rank cannot use --trim-fixed, --trim-merged, --merge or --output")
        if any(not p.name.endswith(".trace.json.gz") or ".fixed-stacks." in p.name for p in options.traces):
            parser.error("--per-rank requires raw .trace.json.gz inputs")
        if not options.overview and options.microbatch_count is None and options.window_duration_ms is None:
            options.microbatch_count = 4
    trim_options = (
        options.output is not None
        or options.window_start_ms != 0
        or options.window_duration_ms is not None
        or options.microbatch_count is not None
        or options.microbatch_start is not None
        or options.drop_python
        or options.drop_call_stacks
        or options.overview
    )
    if options.trim_merged or options.trim_fixed or options.per_rank:
        if options.trim_merged and options.trim_fixed:
            parser.error("Choose either --trim-merged or --trim-fixed")
        if options.merge or (options.trim_merged and len(options.traces) != 1):
            parser.error("Trimming cannot use --merge; --trim-merged requires exactly one input")
        if len({p.resolve() for p in options.traces}) != len(options.traces):
            parser.error("Duplicate input files")
        if options.trim_fixed and any(not p.name.endswith(".fixed-stacks.trace.json.gz") for p in options.traces):
            parser.error("--trim-fixed requires per-rank .fixed-stacks.trace.json.gz inputs")
        if options.overview and (
            options.microbatch_count is not None
            or options.microbatch_start is not None
            or options.window_duration_ms is not None
            or options.window_start_ms != 0
        ):
            parser.error("--overview covers the full step and cannot use window/microbatch options")
        if not options.overview and (options.microbatch_count is None) == (options.window_duration_ms is None):
            parser.error("Choose exactly one of --microbatch-count and --window-duration-ms")
        if options.microbatch_count is not None and options.microbatch_count <= 0:
            parser.error("--microbatch-count must be positive")
        if options.microbatch_start is not None and (options.microbatch_count is None or options.microbatch_start < 0):
            parser.error("--microbatch-start requires --microbatch-count and must be non-negative")
        if options.microbatch_count is not None and options.window_start_ms != 0:
            parser.error("--window-start-ms cannot be combined with --microbatch-count")
        if (
            not math.isfinite(options.window_start_ms)
            or options.window_start_ms < 0
            or (
                options.window_duration_ms is not None
                and (not math.isfinite(options.window_duration_ms) or options.window_duration_ms <= 0)
            )
        ):
            parser.error("Window start must be non-negative and duration must be positive")
        if options.per_rank:
            fix_trim_per_rank(options.traces, options)
            raise SystemExit(0)
        source = options.traces[0]
        default_name = (
            "merged.trimmed.trace.json.gz"
            if options.trim_fixed
            else (source.name.removesuffix(".trace.json.gz") + ".trimmed.trace.json.gz")
        )
        if options.overview:
            default_name = (
                "merged.overview.trace.json.gz"
                if options.trim_fixed
                else (source.name.removesuffix(".trace.json.gz") + ".overview.trace.json.gz")
            )
        output = options.output or source.with_name(default_name)
        if output.resolve() in {p.resolve() for p in options.traces} or output.exists():
            parser.error("Trimmed output must be a new file and must not overwrite the input")
        if not output.name.endswith(".json.gz"):
            parser.error("Trimmed output must end in .json.gz")
        if options.trim_fixed:
            trim_fixed(options.traces, output, options)
        else:
            trim_merged(source, output, options)
        raise SystemExit(0)
    if trim_options:
        parser.error("Trimming options require --trim-merged, --trim-fixed or --per-rank")
    if len({p.resolve() for p in options.traces}) != len(options.traces):
        parser.error("Duplicate input files")
    output = options.merge or options.traces[0].parent / "fixed" / "merged.trace.json.gz"
    protected = {p.resolve() for p in options.traces}
    protected.update(
        (p.parent / "fixed" / p.name.replace(".trace.json.gz", ".fixed-stacks.trace.json.gz")).resolve()
        for p in options.traces
    )
    if output.resolve() in protected:
        parser.error("Merged output must not overwrite an input or per-rank repaired file")
    if not output.name.endswith(".json.gz"):
        parser.error("Merged output must end in .json.gz")
    repaired = [fix(trace) for trace in options.traces]
    write_gzip(output, merged_chunks(repaired))
    print(f"Merged trace -> {output}")
