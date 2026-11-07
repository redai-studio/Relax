---
title: "Example: start with a timeline when observing asynchronous tasks"
date: '2026-09-28'
author:
  name: 无幻
  github: yangruipis
---

::: info Example article
This article tests the blog layout. Scenarios and data are illustrative and do not represent actual project results.
:::

## Record the events first

The following timeline is a layout example. It does not describe the actual scheduling behavior of Relax.

```text
00:00  task A started
00:02  task B started
00:05  task A completed
00:07  task B completed
```

## Separate waiting from execution

Total elapsed time alone does not show whether time was spent waiting or executing. Start and end records for key events can provide clues for further investigation.

## Explain the observation boundary

> An ordering in time is not enough, by itself, to establish a causal relationship between two events.

Mark any gaps in the recorded stages instead of presenting an inferred sequence as confirmed.

## Further reading

- [Turning an investigation into a reusable note](./example-debugging-notes.md)
- [Organizing an experiment log in one table](./example-experiment-log.md)
