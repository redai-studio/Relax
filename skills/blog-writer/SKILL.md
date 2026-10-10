---
name: blog-writer
description: Write, translate, update, and prepare bilingual Relax blog posts under docs/en/blog and docs/zh/blog. Covers verified frontmatter, rich summaries, image submodule revisions, and static-site validation. Repository READMEs and skill instructions do not require bilingual copies.
---

# Relax 博客撰写与发布

为 Relax 文档站撰写、翻译或更新博客。图片与发布规范以 [docs/README.md](../../docs/README.md) 的「Images」和「Publishing Blog Posts」为准；普通公开指南使用 [doc-writer](../doc-writer/SKILL.md)。遵循 [AGENTS.md 的 Markdown 写作格式](../../AGENTS.md#markdown-写作格式)，自然段和列表项正文保持一个源码行，不做 hard wrap。双语要求只适用于公开博客文章，不要求 README、SKILL.md、references、模板或 PR 文本同时提供中英文。

## 先确认内容与实现

1. 阅读 `docs/README.md`、`docs/.vitepress/theme/posts.data.mts` 和一组现有双语文章。涉及摘要、导航或图片时，按需阅读 `BlogIndex.vue`、`BlogPostNavigation.vue` 和 `BlogAuthor.vue`。不要根据其他博客的功能推断 Relax 的支持范围。
2. 确认文章主题、作者、发布日期与可公开的材料。从用户材料和当前源码核实技术描述，对外部资料使用原始来源。性能数字注明配置、测量方式和出处，区分计划、源码分析与实测结果。不要编造作者、测试结论或已经上线的能力。
3. 草稿先保留在 `docs/draft/` 或用户指定的位置。只有准备公开或用户要求发布时，才移入博客目录；博客目录中的 Markdown 会随文档站构建发布。

## 文件与双语规范

- 英文：`docs/en/blog/<slug>.md`；中文：`docs/zh/blog/<slug>.md`。
- `<slug>` 使用小写 kebab-case，如 `hello-world`；两种语言必须同名。不要改动 `index.md` 或 `page/`，它们用于列表和分页。新文章由加载器自动收录，无需逐篇修改导航或侧边栏。
- 同时提供中英文版本，保持技术内容、发布日期、作者顺序和置顶状态一致。标题、摘要、正文和图片替代文本自然翻译，代码、参数名、命令与引用来源保持一致。
- 正文不重复 H1；页面从 frontmatter 的 `title` 渲染标题，章节从 `##` 开始。使用具体标题、清晰的背景和可复现的例子；章节结构按文章内容组织，不套用指南模板。

## 元信息与摘要

以下是结构示例，发布前替换标题、日期、作者和摘要：

```markdown
---
title: 文章标题
date: '2026-10-02'
author: 作者昵称
---

开头段落可以包含 **强调**、链接和图片。

<!-- more -->

## 第一个章节

文章正文。
```

- `title`、`date`、`author` 必填；日期为加引号的有效 `YYYY-MM-DD` 字符串，如 `date: '2026-10-02'`。未加引号的日期会被拒绝，避免 YAML 在校验前自动转换并修正无效日期。页面描述自动取正文摘要的纯文本，无需重复填写。`description` 可选，仅在需要单独设置页面描述时使用，不影响列表摘要。
- `author` 支持姓名字符串，或 `{ name, github?, avatar? }`。`github` 使用真实的、不带 `@` 的 GitHub 用户名；不要把昵称当用户名。`avatar` 为 HTTP(S) 图片地址或站内 `/images/...` 路径，可覆盖 GitHub 头像。
- `co_authors` 可选，使用与 `author` 相同格式的列表；主作者在前，共同作者随后。不把 agent 自动加入文章作者，除非用户明确要求。
- `pinned` 可选，使用布尔值 `true` 或 `false`。置顶只影响列表顺序，上一篇与下一篇仍按同语言文章的发布日期排列。
- 用 `<!-- more -->` 标记富文本摘要的结尾。分隔符之前的 Markdown 在列表中渲染为静态 HTML，支持强调、链接、图片和公式；这部分在文章正文中仍然可见。Mermaid 图、Vue 组件和脚本放在分隔符之后。没有分隔符时使用正文的第一个普通段落，跳过标题、提示块、列表和引用块。
- 摘要中的文章链接可写成 `./other-post.md`；图片使用 `/images/...`。`cover`、`category` 等字段当前没有对应博客功能，不应把它们写成发布要求。

## 图片规范与子模块

先运行：

```bash
git submodule update --init docs/public/images
```

图片来自 [redai-studio/relax-images](https://github.com/redai-studio/relax-images)，挂载在 `docs/public/images/`。文章 Markdown 保留在 Relax。

- 文章与图片目录同名：`hello-world.md` 的图片存入 `docs/public/images/blog/hello-world/`。文件名使用小写 kebab-case。
- 中英文共用与语言无关的图片；图中文字需要翻译时使用 `-en`、`-zh` 后缀。图片提供有意义的替代文本，必要时用图注解释配置、数据来源或结论。
- 示意图优先 SVG，截图和照片优先 WebP。保证文字可读，压缩图片，使图片子模块的每个文件不超过 500 KiB。图片大小检查由图片仓库的 pre-commit hook 和 `Image Size Check` CI 执行，配置与本地检查步骤见该仓库的 README。不为压缩图片新增项目依赖，也不迁移已有图片，除非用户要求。
- Markdown 引用如 `![架构概览](/images/blog/hello-world/architecture.svg)`。不写 `docs/public`、`/Relax/` 或 GitHub raw URL。VitePress 负责部署前缀；若编写 Vue 组件，使用 `withBase()`。
- 图片与文章可同时开 PR，并互相链接依赖。先合入图片 PR，再把 Relax 子模块切到图片仓库中已合入的提交，将 `docs/public/images` 指针与文章一起暂存、构建验证后请求最终 review。不要让最终指针依赖仅存在于贡献者 fork 的提交；具体步骤见 [Article and image PRs](../../docs/README.md#article-and-image-prs)。
- 不使用 `git submodule update --remote` 作为构建步骤，构建必须使用主仓库记录的提交。仅推送图片仓库不会更新网站，还需要在 Relax 合入指针更新并触发部署。
- 遵循本次任务对提交、推送和发布的授权。加载 SKILL 不授予额外远端写入权限。图片提交尚未发布时，保留本地改动并明确说明，不把不可获取的指针当作发布完成。

## 验证与交付

1. 对照两种语言，核实元信息、链接、图片与技术结论；检查图片子模块的 `git status`，确认主仓库指向预期、可获取的图片提交。
2. 从仓库根目录运行：

   ```bash
   npm run docs:build
   pre-commit run --all-files
   npm run docs:preview
   ```

   完整构建包含 Python OpenAPI 生成、VitePress 和 chunk 名称修复，依赖见 `package.json` 与 `.github/workflows/deploy-docs.yml`。缺少运行环境时报告具体阻塞，不把只启动开发服务器当作静态构建验证。
3. 在生产预览中检查中英文列表与文章、摘要、作者、图片、上一篇/下一篇和目录。至少检查 320px 窄屏和 1280px 以上宽屏，没有横向溢出；图片路径须包含实际部署 base。直接打开文章地址并刷新，核实静态页面可用；超过 10 篇时检查分页。
4. 确认发布产物中没有 `images/.git`，并移除临时验证文章和图片。交付时列出文章和图片提交、验证结果、尚未完成的发布步骤；不替用户发布公告或回复 GitHub。
