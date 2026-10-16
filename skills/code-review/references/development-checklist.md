# Development Principles and Necessity (Relax Project)

## Necessity

- Identify consequential new behavior, including parameters, modules, state, checks, defaults, normalization, and compatibility promises. Trace its purpose to a confirmed requirement, existing contract, or demonstrated use case, and inspect producers and consumers, including supported extension points. A caller proves reachability, not necessity.
- Check whether existing functionality already meets the need or the value can be derived from authoritative state. Name the observable behavior lost if the addition were removed, merged, or simplified.
- For an overdesign finding, identify the required behavior, added maintenance obligations, and a smaller alternative that preserves the acceptance criteria. Compare current benefit with added cost; a plausible future use or generic robustness claim alone does not justify the extra complexity.

## Product and Acceptance Boundaries

- Separate behavior supported for framework users from the procedure used to accept one implementation. Fixed evaluation sizes, step-zero files, predetermined batch plans, or exact numerical replay need a product requirement before becoming conditions of normal training, evaluation, or recovery.
- Useful acceptance evidence may belong in a PR, experiment report, or external artifact. Preserve actual input validation, pair integrity, DP partition correctness, and reference-model identity checks at initialization and resume; removing an experimental gate does not remove those guarantees.
- Ask what a check establishes beyond existing guarantees. Repeated full-state hashes or extra forward probes may justify a diagnostic experiment without justifying their cost on every save or training step. A hash can still be an appropriate persisted identity or integrity check at an owning boundary.

## CLI and User Documentation

- Treat a CLI as a user contract: concept, unit, counted object, default, interactions, existing interface reuse, and compatibility cost. Pairs, branches, samples, and tokens are distinct units. Reusing an internal training path does not make a different learning objective a subtype of that path's public concept.
- Resource budgets must have an explicit data policy. Batching or memory limits should not silently truncate, drop, or replace training examples; reuse the established oversize policy when it satisfies the need.
- Check guides against what users need to prepare, choose, run, recover, and interpret metrics. Keep real limitations and behavior aligned across languages; place acceptance logs and internal work history with their evidence. Assess documentation defects by user impact, not file type.
- Keep PR scope, interfaces, and verification claims aligned with the current implementation and tested revision.

## Development Principles

| Principle | What to investigate | Boundary |
|-----------|---------------------|----------|
| Simple and correct | Branches, wrappers, configuration, or dependencies that add no current behavior or clarity | Prefer the smallest clear design that satisfies the requirement, not the fewest lines |
| Follow established project patterns | Parallel registries, configuration objects, or management layers where an existing path already meets the need | Prefer the existing structure when it satisfies the contract; departures need an unmet requirement or demonstrated problem with that structure |
| Keep implementation choices internal | New flags, modes, or switches that transfer internal decisions to users and add documentation, testing, or compatibility obligations | Expose choices when supported use cases need distinct behavior; otherwise prefer a clear internal decision |
| Handle realistic failures | Fallbacks for states excluded by established contracts, defaults that conceal broken invariants, or new input-type and compatibility support without an established use case | Validate external input at boundaries and preserve required compatibility; additional accepted inputs need defined semantics, while real I/O and worker failures still need justified handling |
| Small, cohesive functions | Mixed responsibilities, unclear inputs/outputs, or dependencies that make a change hard to reason about | Split by responsibility; function length alone is not a finding |
| Reduce the effort needed to understand an operation | Helpers, interfaces, or layers that scatter one operation across more locations without clarifying responsibilities or reducing duplicated policy | Examine the complete call path; indirection is justified when it isolates real responsibilities or shared behavior, not merely when each function becomes shorter |
| Prefer pure functions, immutable data, and explicit ownership | Hidden mutation of caller-owned data, implicit I/O or dependencies, or shared state with unclear ownership | Keep side effects explicit; owned in-place tensor operations can be appropriate when contracts, autograd, and performance justify them |
| Refactor around real concepts or existing commonality | Repeated policy branches drifting apart, patch layers, or generic frameworks for hypothetical future needs | A specific implementation is valid; add extension points for current requirements, not hypothetical variants or duplicate counts, and keep refactoring within scope |
| One authoritative source per fact | Derived values stored and updated independently, with inconsistent update or invalidation paths | Caches and snapshots need an explicit owner, lifetime, and consistency contract; justified caching is not inherently duplication |

## Calibration Examples

- **Report:** two independently updated fields encode the same rollout state, and a cancellation path updates only one. **Do not report:** a snapshot deliberately captures state at a documented version boundary.
- **Report:** a fallback converts a violated batch contract into apparently valid training data. **Do not report:** boundary validation rejects malformed external input, or a bounded retry handles a transient worker failure.
- **Report:** repeated backend policy branches already disagree for a supported mode. **Do not report:** a short dispatch over a closed set of modes, or a cohesive function solely because it exceeds a line count.
- **Report:** evaluation rejects a supported dataset because its size differs from a fixed acceptance fixture. **Smaller alternative:** evaluate the supplied data, preserve complete pairs and correct tail partitions, and keep the fixed experiment with acceptance evidence. **Do not report:** a documented benchmark command intentionally selects that fixture.
- **Report:** a last-token position is passed through several layers although the consumer can derive it from its authoritative sequence length. **Smaller alternative:** derive it at the consumer. **Do not report:** a packed or transformed sequence needs offsets that length alone cannot recover.
- **Report:** a memory budget silently shortens supervised examples despite a default policy to keep them. **Smaller alternative:** batch within the budget where possible and apply the explicit oversize policy. **Do not report:** users select a documented truncation policy.
