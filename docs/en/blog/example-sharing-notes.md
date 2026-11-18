---
title: 'Example: from team notes to a community article, explaining the context, the process, and the questions still open'
date: '2026-09-30'
author:
  name: 月天
  github: Aurelius84
co_authors:
  - name: 渡晓
    github: SigureMo
  - name: 禹哲
    github: NINGBENZHE
  - name: 伊优
    github: DrRyanHuang
  - name: 无幻
    github: yangruipis
---

::: info Example article
This sample previews long titles, the page outline, and internal links. It is not an official team publication.
:::

Readers of team notes often already know the background. Sharing a note with the community means filling in what we normally leave unstated: why the problem came up, what we tried, and which questions remain unanswered.

## Give the article a clear starting point

The first paragraph can answer three questions:

- What does this article discuss?
- What background does the reader need?
- What can they do after reading it?

> An article does not have to cover the whole system, but it should explain its scope.

## Make the narrative easy to follow

### Organize the process with headings

Separate preparation, observations, and conclusions so readers can jump to what they need. The page outline also provides a reading path through longer articles.

### Keep context around code

For example, to preview this repository's documentation site, run the existing command from the project root:

```bash
npm run docs:dev
```

Explaining where a command runs and what it does makes it easier to reuse than a command alone. Use inline code for paths or identifiers, such as `docs/en/blog/example-sharing-notes.md`.

## Read it once more before sharing

- [ ] Do the title and summary explain the topic?
- [ ] Are examples clearly distinguished from actual observations?
- [ ] Do the links lead to useful further reading?
- [ ] Do the English and Chinese versions cover the same content?

These checks help make the article a useful starting point for further discussion.

## Further reading

- [Turning an investigation into a reusable note](./example-debugging-notes.md)
- [Organizing an experiment log in one table](./example-experiment-log.md)
- [How to contribute](../guide/how-to-contribute.md)
