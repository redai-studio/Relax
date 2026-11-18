---
title: 'Example: turning an investigation into a reusable note'
date: '2026-10-02'
author:
  name: 渡晓
  github: SigureMo
co_authors:
  - name: 禹哲
    github: NINGBENZHE
---

::: info Example article
This article previews the blog layout. All scenarios and data are illustrative, not actual Relax incidents or performance results.
:::

After an investigation, the command that solved the problem is often the only thing we save. A few weeks later, we may no longer remember why we ran it, what it tested, or which possibilities we ruled out.

A useful note can start with those questions.

## Describe the symptom first

Suppose a task appears slower. Record what you can observe repeatedly: when the change started, which steps slowed down, and whether the same input reproduces it.

> Turning “it feels slower” into a question you can test is the first step of an investigation.

| Record | Example |
| --- | --- |
| Trigger | Repeated runs with the same inputs |
| Observation | The time taken by one step |
| Control | Keep inputs and the runtime environment unchanged |

## Test one hypothesis at a time

### Keep the raw observations

Save individual observations before summarizing them. This code only illustrates how to organize three measurements; it does not represent a real training task.

```python
from statistics import mean

# Example data, in seconds.
durations = [3.4, 3.1, 3.8]
summary = {
    "runs": len(durations),
    "mean_seconds": round(mean(durations), 2),
    "max_seconds": max(durations),
}
```

### Record what remains untested

A local experiment only answers the questions it covers. If you tested one input size, record that scope. If a change affected two conditions at once, preserve that uncertainty and plan another comparison.

## Leave a path for the next reader

End with the reproduction conditions, key observations, and next steps. Even without a final explanation, someone else can continue from the evidence you collected.

- Save the smallest input that reproduces the symptom.
- Link the relevant code or existing documentation.
- Separate observations from hypotheses that still need testing.

## Further reading

- [Organizing an experiment log in one table](./example-experiment-log.md)
- [From team notes to a community article](./example-sharing-notes.md)
- [Relax debugging guide](../guide/debugging.md)
