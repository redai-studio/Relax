# Full-response repetition benchmark

Reproduce from the repository root (standard library only):

```bash
python -m relax.tools.benchmark_repetition --output benchmarks/repetition/local.json --repeats 7
```

The checked-in `windows-python313.json` contains all 24 measured combinations,
raw timings, configuration and environment. The benchmark runs the actual
`relax.utils.repetition` APIs without mocking compression or limiting windows.

## Recorded results

Recorded on Windows 11, AMD64 Family 26 Model 68 Stepping 0 (12 logical CPUs),
Python 3.13.9, zlib runtime 1.3.1. Seed: 20260918. Seven timing repetitions.
The table shows the high-entropy ASCII control using the detailed API:

| Characters | Windows | Median CPU / response | Median wall / response | Process peak working set | Separate scan traced peak |
| ---------: | ------: | --------------------: | ---------------------: | -----------------------: | ------------------------: |
|     10,000 |       1 |              0.053 ms |               0.068 ms |                24.06 MiB |             311,442 bytes |
|    100,000 |      19 |              1.645 ms |               1.690 ms |                25.69 MiB |             321,771 bytes |
|  1,000,000 |     199 |             18.229 ms |              18.701 ms |                37.19 MiB |             321,771 bytes |

These are measurements on this machine, not performance guarantees. CPU time
can be below wall time because scheduling and timer quantization affect short
calls. Each repetition batches complete detector calls for approximately 100 ms
and reports time per response; this mitigates Windows' coarse process CPU timer.
The JSON records the batch size, minimum and nearest-rank p95 as well as median.
With seven repetitions, nearest-rank p95 is the maximum observed batch average.

## Inputs and interpretation

- `control`: deterministic pseudo-random ASCII, no hit; both APIs must scan all
  windows. This is the worst-case path for the boolean API.
- `middle_repeat`: repeated middle half with random first/last quarters, exposing
  the old tail-only detector's limitation at 100k and 1m characters.
- `repeated`: repeated ASCII throughout, showing early exit for the boolean API
  versus the full diagnostic scan.
- `chinese_middle_repeat`: random Chinese first/last quarters and repeated
  Chinese middle half; verifies multi-byte input is measured in characters.

At 10k, the middle-repeat input has normal text in the same single window and
can remain below the threshold; this is intentional and is not a detection
failure. Offsets/windows use Unicode code points, whereas zlib compresses UTF-8.
Every detailed case asserts its scanned window count. No window cap is used.

## Memory methodology

Each case, length and API runs in a fresh subprocess. Inputs are generated and a
warmup runs before timed scans. Timings exclude imports, input generation and
memory tracing. Process peak memory is captured before the separate traced scan:

- Windows: `GetProcessMemoryInfo.PeakWorkingSetSize`, in bytes.
- Linux: `resource.getrusage(...).ru_maxrss`, converted from KiB to bytes.
- macOS: the same resource field is already in bytes.

The native process peak includes interpreter/imports, input construction,
warmup, result objects and native zlib memory. It is deliberately **not** claimed
to be incremental detector memory: random input construction itself can produce
a higher peak. The pre-timing high-water value is also saved for transparency.

`scan_tracemalloc_peak_bytes` measures one separate detector call after enabling
tracing, with input/imports excluded. It includes allocations visible to Python's
tracer but cannot account for every native allocation. Reading dump files and
serializing reports are outside this scanner benchmark.
