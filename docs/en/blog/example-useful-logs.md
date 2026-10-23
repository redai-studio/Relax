---
title: "Example: what makes a log useful for debugging?"
date: '2026-09-26'
author:
  name: 禹哲
  github: NINGBENZHE
---

::: info Example article
This article tests the blog layout. Scenarios and data are illustrative and do not represent actual project results.
:::

## Leave context with each event

These log lines only demonstrate a code block. They are not real output from any component.

```text
run=demo-01 event=input_ready items=8
run=demo-01 event=step_started step=1
run=demo-01 event=step_finished step=1
```

## Use consistent names

A consistent name for the same event makes searching and comparison more direct. A correlation identifier helps readers connect scattered records to one run.

## Check whether the question is answered

- Can the reader identify which run produced the event?
- Can they locate the step being observed?
- Is any context needed for interpretation still missing?

## Further reading

- [Turning an investigation into a reusable note](./example-debugging-notes.md)
- [Organizing an experiment log in one table](./example-experiment-log.md)
