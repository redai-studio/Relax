---
title: "示例：观察异步任务时，先画出时间线"
date: '2026-09-28'
author:
  name: 无幻
  github: yangruipis
---

::: info 示例文章
本文用于测试博客排版，场景和数据均为示例，不代表实际项目结果。
:::

## 先记录事件

下面是一条用于说明排版的假想时间线，不对应 Relax 的实际调度行为。

```text
00:00  task A started
00:02  task B started
00:05  task A completed
00:07  task B completed
```

## 区分等待与执行

仅凭总耗时，很难判断时间花在等待还是执行上。为关键事件留下开始和结束记录，可以为后续调查提供线索。

## 写清楚观察边界

> 时间上的先后关系，不足以单独证明两个事件之间存在因果关系。

如果缺少某个阶段的记录，就标明这个空白，避免把推测当作已经确认的过程。

## 继续阅读

- [把一次问题排查写成别人能复用的笔记](./example-debugging-notes.md)
- [用一张表整理实验记录](./example-experiment-log.md)
