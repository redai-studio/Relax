# Code Quality Checklist (Relax Project)

Contents: [Error handling](#error-handling), [resources](#resource-management), [performance](#performance), [boundaries](#boundary-conditions), [types](#type-safety), [readability](#code-readability).

Use these as investigation prompts. Establish the input contract and a reachable impact before recommending defensive code, abstraction, or optimization.

## Error Handling

### Anti-patterns to Flag

```python
# Bad: Swallowed exception
try:
    risky_operation()
except Exception:
    pass

# Bad: Bare except — catches KeyboardInterrupt, SystemExit
try:
    operation()
except:
    handle_error()

# Good: Catch specific, preserve chain
try:
    risky_operation()
except ValueError as e:
    logger.error(f"Operation failed: {e}")
    raise OperationError("Failed") from e
```

- Overly broad `except Exception` hiding real bugs
- I/O, Ray, or distributed failures that bypass the intended retry, propagation, or cleanup contract; check the owning layer before adding a local handler
- Detached tasks whose exceptions or lifetime are not managed by their owner; not every task must be awaited immediately
- Fallbacks that turn an invalid internal state into a success-shaped result, or repeated validation after a trusted boundary has already established the contract

______________________________________________________________________

## Resource Management

```python
# Bad: Resource leak on exception
f = open("file.txt")
data = f.read()
f.close()

# Good: Context manager
with open("file.txt") as f:
    data = f.read()
```

Key resources in Relax: file handles, locks (`threading.Lock`, `asyncio.Lock`), GPU memory, temporary files, network sockets.

______________________________________________________________________

## Performance

### Hot Path Issues

- Repeated work on a demonstrated hot path; establish its cost before introducing caching or new state
- String building or repeated setup with material cost at the actual input size

### Memory

- Unbounded collections growing without limit
- Large objects held past useful lifetime
- Loading entire large files — use streaming/iteration

### Caching

- A cache key or invalidation policy that omits changing inputs or model versions
- Unbounded cache retention; identify lifetime and memory cost
- TTL is only one policy: immutable inputs, versioned keys, explicit invalidation, or request-scoped lifetime can also establish correctness

______________________________________________________________________

## Boundary Conditions

### None / Empty Handling

```python
# Bad: Truthy check when 0, "", [] are valid values
if value:
    process(value)

# Good: Explicit None check
if value is not None:
    process(value)
```

- Division by zero: trace whether zero is reachable and what the operation should mean. Reject invalid external input or preserve an internal invariant failure; do not silently replace the denominator with `max(count, 1)` or an arbitrary epsilon
- Empty collection access when an empty input is permitted; do not add a guard if the producer already guarantees a nonempty value
- Off-by-one in slicing / ranges

______________________________________________________________________

## Type Safety

- Missing type hints on public functions
- Excessive `Any` usage
- Missing `Optional[]` for nullable values
- Type dispatch that duplicates or contradicts supported behavior; an explicit dispatch over a closed set of types can be clearer than polymorphism

______________________________________________________________________

## Code Readability

- **Magic numbers/strings** → extract named constants
- **Complex nested conditionals** → extract to named booleans
- **Deep nesting** → use early returns / `continue`
- **Mixed responsibilities** → identify a meaningful boundary; length alone does not justify splitting
- Public functions missing docstrings
