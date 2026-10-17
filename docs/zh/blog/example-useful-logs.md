---
title: "示例：什么样的日志更方便排查问题？"
date: '2026-09-26'
author:
  name: 禹哲
  github: NINGBENZHE
---

::: info 示例文章
本文用于测试博客排版，场景和数据均为示例，不代表实际项目结果。
:::

## 给事件留下上下文

下面的日志仅用于展示代码块，不是任何组件的真实输出。

```text
run=demo-01 event=input_ready items=8
run=demo-01 event=step_started step=1
run=demo-01 event=step_finished step=1
```

## 保持名称一致

同一个事件使用一致的名称，能让搜索和比较更直接。关联标识则帮助读者把分散的记录放回同一次运行。

## 检查是否回答了问题

- 能否判断事件属于哪次运行？
- 能否定位正在观察的步骤？
- 是否还有影响判断的上下文缺失？

## 继续阅读

- [把一次问题排查写成别人能复用的笔记](./example-debugging-notes.md)
- [用一张表整理实验记录](./example-experiment-log.md)
