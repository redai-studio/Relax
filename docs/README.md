# Relax Documentation

This directory contains the VitePress documentation site for Relax.

## Features

- 📚 Bilingual documentation (English & Chinese)
- 🎨 Beautiful VitePress theme with custom branding
- 🔍 Full-text search support
- 🖼️ **Image zoom functionality** - Click any image to view it in full size
- 📊 Mermaid diagram support
- 🌓 Dark mode support

## Development

```bash
# Initialize the image submodule / 初始化图片子模块
git submodule update --init docs/public/images

# Install dependencies
npm install

# Start dev server
npm run docs:dev

# Build for production
npm run docs:build

# Preview production build
npm run docs:preview
```

## Structure

```
docs/
├── .vitepress/
│   ├── config.mts          # VitePress configuration
│   └── theme/              # Custom theme
├── public/                 # Static assets
│   └── images/             # redai-studio/relax-images Git submodule
├── en/                     # English documentation
│   ├── guide/              # English guides
│   │   ├── introduction.md
│   │   ├── installation.md
│   │   ├── quick-start.md
│   │   └── ...
│   ├── api/                # API documentation
│   ├── examples/           # Example documentation
│   ├── blog/               # English blog articles and listing
│   └── index.md            # English homepage
├── zh/                     # Chinese documentation
│   ├── guide/              # Chinese guides
│   ├── api/                # API documentation
│   ├── examples/           # Example documentation
│   ├── blog/               # Chinese blog articles and listing
│   └── index.md            # Chinese homepage
├── draft/                  # Draft documentation
│   ├── design.md
│   ├── metrics_service_usage.md
│   └── ...
└── index.md                # Root homepage (defaults to Chinese)
```

## Adding New Pages

1. Create a new markdown file in the appropriate directory
2. Add the page to the sidebar in `.vitepress/config.mts`
3. Create the matching English and Chinese pages with the same structure and coverage

## Images / 图片

New documentation and blog images belong in
[redai-studio/relax-images](https://github.com/redai-studio/relax-images), mounted
at `docs/public/images/`. Article Markdown stays in this repository. Existing
assets such as `/logo.jpg` can stay where they are.

Run the development commands from the repository root. For an existing checkout,
initialize the submodule with `git submodule update --init docs/public/images`.
A fresh clone can use `git clone --recurse-submodules https://github.com/redai-studio/Relax.git`.
After pulling a Relax revision that changes the image pointer, run the submodule
update command again to use that revision's images.

Following [PFCCLab's naming convention](https://github.com/PFCCLab/blog/blob/main/CONTRIBUTING.md):

- Use lowercase kebab-case for article slugs, directories, and image filenames.
- An article `en/blog/hello-world.md` and its Chinese counterpart share the image
  directory `docs/public/images/blog/hello-world/`.
- Guide images use `docs/public/images/guide/<topic-slug>/`.
- Share language-independent images; use `-en` and `-zh` suffixes for translated images.
- Prefer SVG for diagrams and WebP for raster images. Compress raster images and
  aim for 300 KiB or less when practical, while keeping labels readable. This is
  an authoring recommendation, not a build limit; no automatic compression is configured.

Reference an image as `![Architecture overview](/images/blog/hello-world/architecture.svg)`.
Include meaningful alt text. Use the same `/images/...` path in custom avatar
frontmatter. Do not include `docs/public`, `/Relax/`, or a GitHub raw URL;
VitePress adds the deployment base. Vue components must use `withBase()`.

Publish image changes to the image repository first, then update the submodule
to the published commit and stage `docs/public/images` alongside the article.
The parent repository records an exact image commit. GitHub Pages checks out
that commit with `submodules: true`; it does not follow the latest image branch.
An image-only push needs a corresponding pointer update in Relax to be deployed.
The build removes the submodule's `.git` metadata from the published site.

新的文档和博客图片统一存入
[redai-studio/relax-images](https://github.com/redai-studio/relax-images)，
作为子模块挂载在 `docs/public/images/`，文章 Markdown 保留在当前仓库。
`/logo.jpg` 等已有资源可以继续保留在原处。

开发命令从仓库根目录运行。已有仓库使用 `git submodule update --init docs/public/images`
初始化子模块；首次克隆可使用 `git clone --recurse-submodules https://github.com/redai-studio/Relax.git`。
拉取 Relax 更新后，如果图片指针有变化，再运行一次子模块更新命令，使图片与当前提交一致。

沿用 [PFCCLab 的命名约定](https://github.com/PFCCLab/blog/blob/main/CONTRIBUTING.md)：

- 文章标识、目录名和图片名使用小写 kebab-case。
- `en/blog/hello-world.md` 及其中文版本共用 `docs/public/images/blog/hello-world/` 图片目录。
- 指南图片存入 `docs/public/images/guide/<主题标识>/`。
- 与语言无关的图片共用；翻译过图中文字的版本使用 `-en`、`-zh` 后缀。
- 示意图优先使用 SVG，位图优先使用 WebP。在保证文字可读的前提下压缩，尽量控制在
  300 KiB 以内。这是撰写建议，不是构建限制；当前没有自动压缩流程。

图片引用写成 `![架构概览](/images/blog/hello-world/architecture.svg)`，填写有意义的替代文本。
自定义头像也使用 `/images/...` 路径，不要写入 `docs/public`、`/Relax/` 或 GitHub raw 地址。
VitePress 会补上部署前缀，Vue 组件须使用 `withBase()`。

先将图片改动发布到图片仓库，再把子模块更新到已发布的提交，在主仓库中将
`docs/public/images` 指针与文章一起暂存。主仓库记录的是固定图片提交；GitHub Pages
通过 `submodules: true` 检出该提交，不会自动追踪图片分支。仅推送图片仓库不会触发部署，
还需要更新 Relax 中的子模块指针。构建会从发布产物中移除子模块的 `.git` 元数据。

## Publishing Blog Posts / 发布博客

The Blog navigation opens the article list. Its Markdown loader follows
[PFCCLab's implementation](https://github.com/PFCCLab/blog/tree/main/src/.vitepress/theme/loaders),
with a Relax-themed layout informed by the [Vue](https://blog.vuejs.org/),
[Hugging Face](https://huggingface.co/blog), and [PyTorch](https://pytorch.org/blog/) blogs.
Add matching Markdown files at `en/blog/<slug>.md` and `zh/blog/<slug>.md` using the
frontmatter below. Use a lowercase kebab-case slug, such as `hello-world`.
Translate the title, summary, and body in each version, with matching section
structure, publication date, and author order. For agent-assisted publishing, use
the [blog-writer skill](../skills/blog-writer/SKILL.md).
The list automatically displays each language's posts, pinned posts first and
then newest first within each group; no sidebar
or navigation edits are needed for individual posts.
Each language has 10 posts per page. Pagination appears when there are more than
10 posts; later pages use URLs such as `/en/blog/page/2.html`. Like PFCCLab, the
site generates separate HTML pages during the build, so direct links and refreshes work.

博客导航进入文章列表，Markdown 加载方式参考
[PFCCLab 源码](https://github.com/PFCCLab/blog/tree/main/src/.vitepress/theme/loaders)，
布局借鉴 [Vue](https://blog.vuejs.org/)、[Hugging Face](https://huggingface.co/blog) 和
[PyTorch](https://pytorch.org/blog/) 博客，配色与字体沿用 Relax 官网。
在 `en/blog/<slug>.md` 和 `zh/blog/<slug>.md` 添加同名的中英文文章，填写以下元数据，
文章标识使用小写 kebab-case，例如 `hello-world`。翻译标题、摘要和正文，保持章节结构、
发布日期与作者顺序一致。使用 agent 撰写时，加载 [blog-writer SKILL](../skills/blog-writer/SKILL.md)。
列表自动收录对应语言的文章，置顶文章优先，两组内部均按日期倒序，
无需逐篇修改导航或侧边栏。
每种语言每页显示 10 篇，超过 10 篇时显示分页；后续页面使用 `/zh/blog/page/2.html` 这样的地址。
与 PFCCLab 一样，分页会在构建时生成独立 HTML，支持直接打开链接和刷新。

```markdown
---
title: Your article title
date: '2026-10-02'
author:
  name: 渡晓
  github: SigureMo
co_authors:
  - name: 禹哲
    github: NINGBENZHE
  - name: 月天
    github: Aurelius84
  - name: Relax Team
    avatar: /logo.jpg
pinned: true
---

An introduction with **emphasis** and a [link](./example-debugging-notes.md).

<!-- more -->

The rest of the article.
```

`title`, `date`, and `author` are required; `co_authors` and `pinned` are optional.
Use `YYYY-MM-DD` for the publication date.
The page renders its heading from `title`; start the body directly, without
repeating an H1. Use H2 (`##`) for sections.
For `author`, set `name` to the display name and `github` to the GitHub username
without `@`. The article list and header display the GitHub avatar and link to
the profile. A plain author name is also supported for authors without GitHub.
Set `avatar` to override the GitHub avatar, with or without a GitHub account.
Use an HTTP(S) image URL or a path in `docs/public`, such as `/logo.jpg`;
store new avatar images in the image submodule and use `/images/blog/<slug>/avatar.webp`.
The site's deployment base is added automatically. Failed images fall back to the first character of the name.
Following PFCCLab's format, `co_authors` is a list with the same structure as
`author`. Authors appear in order: the primary author, then the co-authors.
Multi-author cards show avatars that overlap as space narrows, with the primary
author in front. Below 1280px, names and avatars appear below the article title
in equal-width columns that adapt to the available space. On wider screens,
authors form a vertical column to the left of the body, following PFCCLab's layout.
The article outline sits on the right, aligned with the first author below the title,
and stays visible while scrolling.
Set `pinned: true` to keep a post at the top of the list and show a pinned badge.
Links to the previous (older) and next (newer) posts stay within the same language;
pinning does not change this chronological order. On wide screens these links sit
below the authors in the left column; on smaller screens they appear after the article.
Both layouts wrap long titles without truncation.

To show a rich summary, place `<!-- more -->` after the opening paragraphs.
The list renders the Markdown before this marker, including emphasis, links, images, and formulas.
Relative article links resolve from the article, including on paginated lists.
For summary images, use an HTTP(S) URL or a site-root path in `docs/public`, such as `/logo.jpg`.
New article images follow the submodule conventions in [Images](#images-图片).
Keep Vue components and scripts after the marker: summaries render static HTML.
Without the marker, the list uses the first ordinary paragraph, skipping headings, notices, lists, and blockquotes.
The page description is generated from the summary's plain text, so there is no need to write it twice.
An optional frontmatter `description` overrides the page description only; it does not change the list summary.
The opening paragraphs remain visible in the full article.

Keep `index.md` as the listing page. Files in the blog directories are published
with the documentation site; add articles there when they are ready to be shared.
The `page/` directory is reserved for pagination templates.

`title`、`date`、`author` 为必填，`co_authors` 和 `pinned` 可省略。
发布日期使用 `YYYY-MM-DD` 格式。`index.md` 保留为列表页。
页面会根据 `title` 自动显示大标题，正文无需重复写一级标题；章节从二级标题（`##`）开始。
`author.name` 是显示昵称，`author.github` 是不带 `@` 的 GitHub 用户名。
列表和文章页会显示 GitHub 头像并链接到个人主页。没有 GitHub 账号的作者也可以直接填写姓名字符串。
设置 `avatar` 可以覆盖 GitHub 头像，也支持没有 GitHub 账号的作者。
可以填写 HTTP(S) 图片地址，或 `docs/public` 下的站内路径，例如 `/logo.jpg`；
部署前缀会自动补上。图片加载失败时显示名字的第一个字。
新头像图片存入图片子模块，路径使用 `/images/blog/<slug>/avatar.webp`。
与 PFCCLab 一样，`co_authors` 使用列表，每一项的格式与 `author` 相同。
显示顺序为主作者在前、共同作者随后。多作者卡片只显示头像，空间变窄时逐渐重叠，
主作者位于最前层；文章页在小于 1280px 时，用等宽网格在标题下方显示头像和姓名，
列数随可用宽度调整。宽屏时参考 PFCCLab，将作者纵向排列在正文左侧。
文章目录位于正文右侧，从标题下方开始，与第一位作者顶部对齐，并在滚动时保持可见。
设置 `pinned: true` 可将文章置于列表顶部，并显示置顶标记。
上一篇（较早发布）和下一篇（较晚发布）始终链接同语言文章，顺序不受置顶影响。
宽屏时导航位于左侧作者下方，较窄屏幕则放在正文底部；两种布局都会换行显示完整标题。

如需富文本摘要，在开头的段落之后添加 `<!-- more -->`。
列表会渲染分隔符之前的 Markdown，支持强调、链接、图片和公式。
相对文章链接始终以原文章为基准解析，在分页中也能正常跳转。
摘要图片使用 HTTP(S) 地址或 `docs/public` 下的站内路径，例如 `/logo.jpg`。
新文章图片遵循上文[图片规范](#images-图片)，存入图片子模块。
Vue 组件和脚本放在分隔符之后，摘要只渲染静态 HTML。
未添加分隔符时，列表使用正文的第一个普通段落，跳过标题、提示块、列表和引用块。
页面描述自动取摘要的纯文本，无需重复填写。
如需单独设置页面描述，可在 frontmatter 中填写 `description`，它不会改变列表摘要。
开头的摘要段落会保留在文章正文中。

博客目录下的文章会随文档站一起发布，在文章可以公开时再放入该目录。
`page/` 目录保留给分页模板使用。

## Image Zoom Feature

All images in the documentation support click-to-zoom functionality powered by `medium-zoom`.

**Usage:**

- Hover over any image to see the zoom cursor
- Click to enlarge the image
- Click again or press ESC to close

**Documentation:**

- [Quick Start Guide](QUICK_START_IMAGE_ZOOM.md)
- [Feature Details](IMAGE_ZOOM_FEATURE.md)
- [Implementation Summary](IMAGE_ZOOM_IMPLEMENTATION.md)
- [Testing Guide](TESTING_IMAGE_ZOOM.md)

## Deployment

The documentation can be deployed to:

- GitHub Pages
- Vercel
- Netlify
- Any static hosting service

See [VitePress deployment guide](https://vitepress.dev/guide/deploy) for details.
