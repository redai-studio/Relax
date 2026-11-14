---
title: '示例：用一张表整理实验记录'
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

::: info 示例文章
本文用于预览表格、公式和提示块。下列数字为虚构数据，不是 Relax 基准测试。
:::

实验记录不需要一开始就很复杂。一张包含条件、结果和备注的表格，通常就能帮我们找回当时的判断依据。

## 先固定比较条件

在比较两次运行前，写下输入、环境和测量范围。只看最终数字，很容易漏掉“这次少运行了一部分步骤”之类的差异。

| 运行 | 条件 | 耗时 / 秒 | 备注 |
| --- | --- | ---: | --- |
| A | 示例基线 | 120 | 使用固定输入 |
| B | 示例调整 | 108 | 其余条件保持一致 |
| C | 重复 B | 111 | 检查重复运行的波动 |

## 让计算过程可以复查

下面只演示相对耗时减少比例的写法：

$$
\text{reduction} = \frac{t_{\mathrm{baseline}} - t_{\mathrm{candidate}}}{t_{\mathrm{baseline}}}
$$

代入示例 A、B 的数字，结果为 $(120 - 108) / 120 = 10\%$。这个算式只描述这两条示例记录，不能说明其他输入或环境也会得到相同结果。

::: tip 把条件和数字放在一起
如果结论依赖特定条件，就把条件写在表格附近，让读者不必来回寻找。
:::

## 长公式在窄屏上的阅读

下面用一组虚构的耗时分项检查长公式排版，不代表实际训练流程。屏幕较窄时，可以在公式区域内横向滚动，页面本身应保持原来的宽度。

$$
\overline{T}_{\mathrm{example}} = \frac{1}{N} \sum_{i=1}^{N} \left(t_{\mathrm{data\ loading}}^{(i)} + t_{\mathrm{request\ queue}}^{(i)} + t_{\mathrm{generation}}^{(i)} + t_{\mathrm{reward\ evaluation}}^{(i)} + t_{\mathrm{parameter\ update}}^{(i)} + t_{\mathrm{checkpoint\ save}}^{(i)} + t_{\mathrm{metric\ export}}^{(i)}\right)
$$

行内也可能出现较长的表达式，例如 $t_{\mathrm{data\ loading}} + t_{\mathrm{request\ queue}} + t_{\mathrm{generation}} + t_{\mathrm{reward\ evaluation}} + t_{\mathrm{parameter\ update}} + t_{\mathrm{checkpoint\ save}} + t_{\mathrm{metric\ export}}$，它同样不应把整页撑宽。公式前后的文字仍应正常换行。

## 记录暂时回答不了的问题

下面这些问题可以作为下一轮记录的起点：

1. 重复运行后的结果是否稳定？
2. 更换输入规模后，变化是否仍然存在？
3. 改动有没有影响结果的正确性？

保留这些问题，比把一组局部数据写成普遍结论更有利于后续讨论。

## 继续阅读

- [把一次问题排查写成别人能复用的笔记](./example-debugging-notes.md)
- [从团队笔记到社区文章](./example-sharing-notes.md)
- [Relax 介绍](../guide/introduction.md)
