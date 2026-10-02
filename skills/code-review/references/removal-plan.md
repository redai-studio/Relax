# Removal & Iteration Plan (Relax Project)

## Scope and Evidence

- Limit removal proposals to code made obsolete by the reviewed change or needed to solve its concrete problem.
- Use the severity definitions in `SKILL.md`; scheduling a cleanup does not make it P0/P1.
- A disabled feature, absent direct caller, or passing test suite does not establish that a public API has no consumers. Respect repository approval requirements for deleting or renaming public APIs.

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
