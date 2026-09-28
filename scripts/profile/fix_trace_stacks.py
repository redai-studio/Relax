#!/usr/bin/env python3
"""Keep MegatronTrainRayActor.train processes and clamp malformed Python
stacks.

Usage: python3 scripts/profile/fix_trace_stacks.py tmp/traces/*.trace.json.gz
Writes fixed/<name>.fixed-stacks.trace.json.gz beside each input.
Always writes fixed/merged.trace.json.gz beside the first input, grouped by CPU/GPU stream, then rank.
Use --merge PATH only to override the merged output path.
Uses only the Python standard library and streams large traces.
"""

import argparse
import gzip
import json
import os
import re
import tempfile
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


def fix(source: Path) -> Path:
    suffix = ".trace.json.gz"
    if not source.name.endswith(suffix):
        raise ValueError(f"Expected a {suffix} file: {source}")
    if source.name.endswith(".fixed-stacks" + suffix):
        raise ValueError(f"Already a repaired trace: {source}")
    output = source.parent / "fixed" / (source.name[: -len(suffix)] + ".fixed-stacks" + suffix)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path, nargs="+", help="PyTorch .trace.json.gz files")
    parser.add_argument(
        "--merge", type=Path, help="Merged output path (default: first input directory/fixed/merged.trace.json.gz)"
    )
    options = parser.parse_args()
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
