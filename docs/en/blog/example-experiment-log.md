---
title: 'Example: organizing an experiment log in one table'
date: '2026-10-01'
author:
  name: 禹哲
  github: NINGBENZHE
co_authors:
  - name: 月天
    github: Aurelius84
  - name: 伊优
    github: DrRyanHuang
---

::: info Example article
This article previews tables, equations, and callouts. The numbers below are fictional, not a Relax benchmark.
:::

An experiment log does not have to start out complicated. A table of conditions, results, and notes is often enough to recover the reasoning behind a decision.

## Fix the comparison conditions first

Before comparing two runs, write down the inputs, environment, and measurement scope. Looking only at the final numbers can hide differences such as one run covering fewer steps.

| Run | Condition | Time / seconds | Notes |
| --- | --- | ---: | --- |
| A | Example baseline | 120 | Fixed inputs |
| B | Example adjustment | 108 | Other conditions unchanged |
| C | Repeat of B | 111 | Check variation across runs |

## Make the calculation reviewable

The following equation only demonstrates how to express a relative reduction in elapsed time:

$$
\text{reduction} = \frac{t_{\mathrm{baseline}} - t_{\mathrm{candidate}}}{t_{\mathrm{baseline}}}
$$

Using the example values from A and B gives $(120 - 108) / 120 = 10\%$. This calculation describes those two illustrative records; it does not establish the same result for other inputs or environments.

::: tip Keep conditions next to the numbers
If a conclusion depends on specific conditions, put them near the table so readers do not have to search for them.
:::

## Reading long equations on a narrow screen

The fictional timing terms below exercise the layout of a long equation; they do not describe an actual training pipeline. On a narrow screen, scroll within the equation while the page keeps its original width.

$$
\overline{T}_{\mathrm{example}} = \frac{1}{N} \sum_{i=1}^{N} \left(t_{\mathrm{data\ loading}}^{(i)} + t_{\mathrm{request\ queue}}^{(i)} + t_{\mathrm{generation}}^{(i)} + t_{\mathrm{reward\ evaluation}}^{(i)} + t_{\mathrm{parameter\ update}}^{(i)} + t_{\mathrm{checkpoint\ save}}^{(i)} + t_{\mathrm{metric\ export}}^{(i)}\right)
$$

An inline expression can also be long, such as $t_{\mathrm{data\ loading}} + t_{\mathrm{request\ queue}} + t_{\mathrm{generation}} + t_{\mathrm{reward\ evaluation}} + t_{\mathrm{parameter\ update}} + t_{\mathrm{checkpoint\ save}} + t_{\mathrm{metric\ export}}$, without making the whole page wider. Text before and after it should still wrap normally.

## Record questions you cannot answer yet

These questions can guide the next set of observations:

1. Are the results stable across repeated runs?
2. Does the change persist at a different input size?
3. Does the adjustment affect the correctness of the results?

Keeping these questions visible makes follow-up discussion easier than turning a small set of observations into a general conclusion.

## Further reading

- [Turning an investigation into a reusable note](./example-debugging-notes.md)
- [From team notes to a community article](./example-sharing-notes.md)
- [Introduction to Relax](../guide/introduction.md)
