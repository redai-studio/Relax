# Test Quality and Verification (Relax Project)

## Risk, Oracle, and Execution

- For each test group, name the real regression, supported boundary, or failure it protects. Trace the expectation to the algorithm, user contract, prior bug, or independent reference; an expectation copied from the new implementation can repeat its error.
- Identify which production entry point and execution boundary actually run, and what fixtures or mocks replace. Follow the asserted output back through that path. Test names, assertion volume, coverage percentage, and a passing suite do not establish the claimed capability.
- A bug regression should make the original faulty version fail for the expected reason, rather than an unrelated import or fixture error. Run that comparison when practical; otherwise state the unverified claim and the scope of evidence available.

## Choosing the Boundary

| Evidence | What it can establish | Limit |
|----------|-----------------------|-------|
| Small numerical case with an independent oracle | Mask coordinates, reductions, gradients, or algorithmic definitions | Does not establish backend or multi-node execution |
| Production parser, loader, or protocol path with controlled I/O | Accepted inputs, metadata compatibility, ordering, and failure propagation | Replacing the core operation restricts the claim to the surrounding boundary |
| Controlled concurrency or collective-participation case | A concrete interleaving, ownership rule, or required group-wide call | Serial mocks do not reproduce a hardware deadlock or scheduling mode |
| Real backend, end-to-end run, or resume | Integration and preserved behavior in the exercised configuration | Does not replace focused edge cases or cover unexercised topology and precision |

Use the smallest test that establishes the risk and execution boundary. Combine these forms when their evidence is complementary; neither CPU tests nor end-to-end runs are universally sufficient.

## Mocks and Implementation Assertions

- A mock may isolate external I/O, a worker, or a wrapper contract. It must not implement the core capability whose correctness the test claims to prove. Assert the production result or required interaction, and describe the boundary honestly.
- Source strings, `inspect.getsource`, hashes, and helper call counts are useful only when the observed fact is itself a confirmed contract or directly proves a relevant mechanism. A metadata schema, artifact digest, or required protocol call may qualify; incidental spelling and refactor choices usually do not.
- A static check can complement behavior tests, but cannot establish numerical correctness, state restoration, or successful execution of code it never runs.

## Retain, Merge, or Remove

- Compare independent failure modes and execution boundaries, not test counts. Two tests using similar data may protect different contracts; differently named tests may assert the same weak fact.
- When removing or merging redundant tests, name the retained stronger coverage, what failure it detects, and the boundary it exercises. Tests of behavior that still exists need continued coverage; tests for an intentionally removed policy need not preserve that policy.
- Do not add tests for optional style preferences or every implementation detail. Keep a test when its protection justifies its maintenance and execution cost.

## Calibration Examples

- **Insufficient for resume correctness:** a fake loader restores model, optimizer, scheduler, and RNG, then the test compares next-step loss. The fixture implemented restoration. **Stronger evidence:** exercise the production save/load path and compare resumed state or the next update; if dependencies prevent that, keep any useful wrapper-contract test and state that resume is unverified.
- **Useful mock:** replace transport with a recording endpoint while the production serializer constructs and sends a request. This can verify the wire payload and required ordering, but makes no claim about the remote worker's computation.
- **Weak oracle:** expected token scores use the same mask shift helper as production. **Stronger evidence:** hand-computed scores with distinct values and unequal sequence lengths that expose a one-position shift; exercise the production metric path.
- **Distinct coverage:** a unit case protects pair adjacency and a backend case protects partial DP partitions. Retain both if they detect independent errors, even when both involve a tail batch.
