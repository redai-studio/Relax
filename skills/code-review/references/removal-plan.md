# Removal & Iteration Plan (Relax Project)

## Scope and Evidence

- Limit dead-code cleanup to unused or redundant code introduced by the reviewed change, or existing code made unused by it (including imports, variables, and functions).
- Do not include pre-existing dead code in the cleanup or removal plan. If noticed, mention it separately as out of scope; do not expand the current change to remove it.
- Use the severity definitions in `SKILL.md`; scheduling a cleanup does not make it P0/P1.
- A disabled feature, absent direct caller, or passing test suite does not establish that a public API has no consumers. Respect repository approval requirements for deleting or renaming public APIs.
- For removed or merged tests, identify retained coverage that detects the relevant independent failures and actually executes their production boundary. Do not preserve an intentionally removed policy solely because its old tests encode it.

______________________________________________________________________

## Template

### Safe to Remove Now

| Field | Details |
|-------|---------|
| **Location** | `path/to/file.py:line` |
| **Type** | Unused function / Dead class / Deprecated module / Feature flag |
| **Rationale** | Why remove |
| **Impact** | None / Low — no active consumers |
| **Steps** | Minimal removal and any required migration; retain tests for behavior that still exists |

### Defer Removal

| Field | Details |
|-------|---------|
| **Location** | `path/to/file.py:line` |
| **Why defer** | Active consumers / needs migration |
| **Preconditions** | What must happen first |
| **Breaking changes** | API/contract changes |

______________________________________________________________________

## Checklist Before Removal

### Code Analysis

- [ ] Searched codebase for all references (`rg`, `grep`)
- [ ] Checked for dynamic usage (`getattr`, string-based references in YAML/JSON)
- [ ] Checked `__init__.py` exports and `__all__`

### Relax-Specific Checks

- [ ] Ray actor registrations
- [ ] Argument parser registrations in `relax/utils/arguments.py`
- [ ] Loss function / reward function registries
- [ ] Shell scripts in `scripts/`
- [ ] Config files in `configs/`
- [ ] Pickle/checkpoint serialization compatibility
