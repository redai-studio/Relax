# Doc Page Template

These English and Chinese templates provide a reusable starting point for Relax documentation. Publish English pages under `docs/en/` and Chinese pages under `docs/zh/` with aligned technical coverage and commands.

## Using the templates

Read two or three relevant repository guides first. For algorithm documentation, useful starting points are `docs/en/guide/ppo-training.md` and `docs/en/examples/algorithms.md`, along with their Chinese counterparts. Follow their terminology, level of detail, and way of introducing examples, without copying report content or inaccuracies.

Copy a template, then select and arrange sections for the reader's task. Configuration, operation, limitations, and troubleshooting are content needs, not mandatory headings. Architecture, features, API reference, additional examples, and best practices are optional. A short feature may need only an introduction and one usage section.

The text, table cells, diagrams, and code comments below are placeholders, not real Relax APIs or runnable instructions. Replace them with source-verified content and remove unused sections and placeholder guidance before publishing. Keep useful context around examples instead of turning every paragraph into a list of commands.

## English Template

````markdown
# Feature Name

Briefly introduce the feature and the user task it addresses.

## Overview

Explain what the feature does and what its main differences mean to the user.
Include enough context to understand the example that follows.

## Architecture

<!-- Optional: include when the reader needs to understand the components. -->

```text
┌─────────────────┐         ┌─────────────────┐
│   Component A   │ ──────> │   Component B   │
└─────────────────┘         └────────┬────────┘
                                     │
                            ┌────────▼────────┐
                            │   Component C   │
                            └─────────────────┘
```

### Component Responsibilities

| Component | Responsibility |
|---|---|
| Component A | What the user needs to know about its role |
| Component B | How it works with the other components |

## Features

<!-- Optional: include only capabilities relevant to this task.
     Explain meaningful differences, not a generic feature list. -->

## Quick Start

State the required environment, input files, paths, and resources. Put any
warning about destructive commands before the command, not after it.

```bash
# Replace with the smallest source-verified launch or setup command.
```

Explain the important choices and where users can find the output. Use numbered
steps only when order matters; a single command may need only one paragraph.

## Configuration

Introduce the settings users are likely to change. Distinguish example-script
defaults from parser defaults and explain relevant constraints near each setting.

| Parameter | Default | Description |
|---|---|---|
| `<real parameter>` | `<verified default>` | What it changes and when to use it |

## API Reference

<!-- Optional: include when users call a public API directly. -->

### Public API Name

```python
# Insert the actual signature and a minimal working call.
```

Explain the required inputs, useful defaults, and return value. Do not invent
method names or signatures to fit the template.

## Usage Examples

<!-- Optional: add scenarios that are not already covered by Quick Start. -->

### Common Scenario

Describe the goal and inputs before the example, then explain the key choices.

```python
# Insert a source-verified example for this scenario.
```

## Best Practices

<!-- Optional: explain relevant trade-offs and why the recommendation helps.
     Omit this section if it would only repeat the configuration notes. -->

## Troubleshooting

### User-visible symptom

Explain the likely cause and the concrete setting, input, or action to check.
Use stable causes and remedies, not historical run logs or acceptance evidence.

## Next Steps

<!-- Optional: link to a few relevant pages; adjust paths for the page location. -->

- [Configuration](./configuration.md) — Review the available training settings.
- [Customize Training](./customize-training.md) — Adapt a training script.
````

## Chinese Template

Use the same technical coverage and commands, but write natural Chinese rather than copying English sentence patterns. Diagram labels remain in English. Localize code comments as needed.

````markdown
# 功能名称

简短介绍功能，以及它帮助用户完成的任务。

## 概述

说明功能做什么、主要差异对用户有什么意义，并为下面的示例提供必要背景。

## 架构

<!-- 可选：仅在用户需要了解组件关系时保留。 -->

```text
┌─────────────────┐         ┌─────────────────┐
│   Component A   │ ──────> │   Component B   │
└─────────────────┘         └────────┬────────┘
                                     │
                            ┌────────▼────────┐
                            │   Component C   │
                            └─────────────────┘
```

### 组件职责

| 组件 | 职责 |
|---|---|
| Component A | 用户需要了解的作用 |
| Component B | 与其他组件如何配合 |

## 功能特性

<!-- 可选：只保留与当前任务相关的能力，解释有意义的差异，不堆砌功能列表。 -->

## 快速开始

说明所需环境、输入文件、路径和资源。命令有破坏性操作时，在执行命令前说明风险。

```bash
# 替换为经过源码核验的最小启动或准备命令。
```

解释关键选择，以及在哪里查看输出。只有执行顺序重要时才使用编号步骤；
单条命令通常只需要一段说明。

## 配置

介绍用户常调整的设置，区分示例脚本默认值和解析器默认值，并在设置附近说明限制。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `<实际参数>` | `<已核验默认值>` | 调整什么、何时使用 |

## API 参考

<!-- 可选：用户需要直接调用公开 API 时保留。 -->

### 公开 API 名称

```python
# 插入实际函数签名和最小可用调用。
```

说明必需输入、常用默认值和返回值。不要为了填充模板而编造方法名或签名。

## 使用示例

<!-- 可选：补充快速开始未覆盖的场景。 -->

### 常见场景

先说明目标和输入，再给出示例，并解释关键选择。

```python
# 插入经过源码核验的场景示例。
```

## 最佳实践

<!-- 可选：说明相关取舍及建议的理由；若只是重复配置说明，则删除此节。 -->

## 故障排除

### 用户可见的问题

说明可能原因，以及应检查的具体设置、输入或操作，不复制历史日志或验收记录。

## 下一步

<!-- 可选：链接到相关页面，并按当前页面位置调整路径。 -->

- [配置说明](./configuration.md) — 查看可用的训练参数。
- [自定义训练](./customize-training.md) — 调整训练脚本。
````

## Natural bilingual writing

Prefer concise sentences, clear actions, and consistent terminology. Aim for “80% of the way to ASD-STE100” as a clarity goal, not a rule that every sentence must be short, imperative, or isolated. Combine related ideas when it helps the reader follow the explanation.

Introduce what an example does and what it needs. After the code, explain the important choices rather than paraphrasing every flag. Keep routine constraints next to the affected settings; reserve warning containers for consequential risks.

For example, the same batch setting can read naturally in either language:

**English:**

> The recipe generates 8 responses for each of 4 prompts, giving 32 responses per rollout. Keep the global batch size at 32 to train on that rollout in one batch.

**Chinese:**

> 示例每次采样 4 个提示词，每个生成 8 个回答，共 32 个回答。将全局批量设为 32，即可用一个训练批次处理这次采样。

These numbers describe the REINFORCE++ recipe, not universal framework defaults. Check the current source before using them elsewhere. Read each language version independently: translate meaning, not syntax, while keeping settings and limitations aligned.

## Conventions

### Code Blocks

Specify the language, such as `python`, `bash`, `yaml`, or `text`. Published examples must use real commands, imports, and signatures verified against source. English code comments are in English; Chinese code comments are in Chinese.

The templates use four backticks around their Markdown so the inner three-backtick code blocks can be copied unchanged.

### VitePress Containers

Use containers sparingly: `tip` for a useful shortcut, `warning` for a consequential risk, and `danger` for critical hazards.

```markdown
::: tip Tip Title
Helpful shortcut.
:::

::: warning Warning Title
Explain the risk and the required precaution.
:::

::: danger Danger Title
Explain the critical hazard before the action.
:::
```

Chinese titles and explanations should be natural Chinese, for example `提示`, `警告`, and `危险`.

### Internal Links and Tables

Use relative links such as `[Configuration](./configuration.md)` or `[Algorithms](../examples/algorithms.md)`, and check them from the final page's location. Use standard Markdown tables when they make settings easier to compare.

### ASCII Diagrams

Use box-drawing characters (`┌ ┐ └ ┘ ─ │ ┬ ┴ ├ ┤ ┼`) and arrows (`▼ ▲ ► ◄`). Keep labels in English in both versions, and include diagrams only when they help the user.

## Keep verification separate

Do not add test counts, acceptance checklists, training reports, execution logs, machine traces, or verification history to the user page. Record build, link, render, and test evidence in internal notes or the PR. A successful test or experiment is not a general feature guarantee.
