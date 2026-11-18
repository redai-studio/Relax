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
# Initialize the image submodule
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
3. For published site pages, create matching English and Chinese versions with aligned technical coverage. Repository READMEs, skills, and internal notes do not need bilingual copies.

## Images

New documentation and blog images belong in [redai-studio/relax-images](https://github.com/redai-studio/relax-images), mounted at `docs/public/images/`. Article Markdown stays in Relax. Existing site assets such as `/rai-studio-logo.png` can stay where they are.

Run development commands from the repository root. For an existing checkout, initialize the submodule with `git submodule update --init docs/public/images`. A fresh clone can use `git clone --recurse-submodules https://github.com/redai-studio/Relax.git`. After pulling a Relax revision that changes the image pointer, run the submodule update command again to use that revision's images.

- Use lowercase kebab-case for article slugs, directories, and image filenames.
- An article `en/blog/hello-world.md` and its Chinese counterpart share `docs/public/images/blog/hello-world/`.
- Guide images use `docs/public/images/guide/<topic-slug>/`.
- Share language-independent images; use `-en` and `-zh` suffixes for translated images.
- Prefer SVG for diagrams and WebP for raster images. Keep labels readable and compress assets before submitting them. Each file in the image submodule must be at most 500 KiB; split large diagrams into focused images when needed.

Reference an image as `![Architecture overview](/images/blog/hello-world/architecture.svg)` and include meaningful alt text. Use the same `/images/...` path in custom avatar frontmatter. Do not include `docs/public`, `/Relax/`, or a GitHub raw URL; VitePress adds the deployment base. Vue components must use `withBase()`.

The size limit is enforced by the image repository's pre-commit hook and `Image Size Check` CI workflow. Follow its [local check instructions](https://github.com/redai-studio/relax-images#local-checks) before opening an image PR.

### Article and image PRs

1. Draft the paired article files in Relax and add any new images on a branch in `relax-images`. Preview locally with the submodule checked out to the image branch.
2. Open the image PR and a Relax draft PR. Link the PRs and state their dependency; they can be reviewed at the same time. A Relax draft that still records the old image revision will not include the new images when someone checks it out.
3. Merge the image PR first. Fetch the image repository, check out the merged commit in `docs/public/images`, and stage that exact pointer with the article files in Relax. Do not publish a pointer to a commit available only in a contributor's fork. An unmerged commit already pushed to the configured image repository can be previewed, but the final Relax PR should pin the merged commit.
4. Run the complete docs build, inspect the article, summary, and avatar images in the production preview, then mark the Relax PR ready. Merge Relax after the article and image pointer have been reviewed together.

Relax records an exact image commit. Checkout and deployment use that revision rather than following the image branch. An image-only push does not update the site; a corresponding pointer update in Relax must trigger a deployment. The build removes the submodule's `.git` metadata from the published files.

## Publishing Blog Posts

Add matching Markdown files at `en/blog/<slug>.md` and `zh/blog/<slug>.md` using a lowercase kebab-case slug such as `hello-world`. Translate the title, summary, body, and image alt text naturally while keeping technical coverage, publication date, author order, and pinning aligned. Use the [blog-writer skill](../skills/blog-writer/SKILL.md) for agent-assisted publishing. Follow the repository-wide [Markdown writing format](../AGENTS.md#markdown-写作格式): keep prose paragraphs and list-item bodies on one source line.

The loader automatically collects each language's posts, with pinned posts first and newest posts first within each group. Individual posts do not need sidebar or navigation edits. Each language has 10 posts per page; additional pages are generated as independent HTML at URLs such as `/en/blog/page/2.html`, so direct links and refreshes work.

The following example uses fictional authors.

```markdown
---
title: Your article title
date: '2026-10-02'
author:
  name: Alice
co_authors:
  - name: Bob
  - name: Carol
  - name: Dave
    avatar: /logo.jpg
pinned: true
---

An introduction with **emphasis** and a [link](./example-debugging-notes.md).

<!-- more -->

The rest of the article.
```

`title`, `date`, and `author` are required; `co_authors` and `pinned` are optional. Use a valid `YYYY-MM-DD` date in quotes, such as `date: '2026-10-02'`. Quotes keep YAML from converting and normalizing the date before validation; unquoted date values are rejected. The page renders its heading from `title`, so start the body without repeating an H1 and use H2 (`##`) for sections.

For `author`, set `name` to the display name and `github` to the GitHub username without `@`. A plain author name is supported for authors without GitHub. Set `avatar` to an HTTP(S) image URL or a site-root path to override the GitHub avatar; store new avatar images in the image submodule as `/images/blog/<slug>/avatar.webp`. Failed images fall back to the first character of the name. `co_authors` is a list with the same structure as `author`, displayed after the primary author.

Below 1280px, authors appear below the article title in equal-width columns that adapt to the available space. On wider screens, they form a vertical column to the left of the body. The outline sits on the right, aligned with the first author, and stays visible while scrolling. `pinned: true` keeps a post at the top of the list. Previous and next links remain within the same language and follow publication date, regardless of pinning; they appear below the authors on wide screens and after the article on smaller screens.

To show a rich summary, place `<!-- more -->` after the opening paragraphs. The list renders the Markdown before this marker, including emphasis, links, images, and formulas. Relative article links resolve from the original article even on paginated lists. New summary images follow the [image conventions](#images). Keep Mermaid diagrams, Vue components, and scripts after the marker because summaries render static HTML. Without the marker, the list uses the first ordinary paragraph, skipping headings, notices, lists, and blockquotes.

The page description is generated from the summary's plain text. An optional frontmatter `description` overrides only the page description, not the list summary. The opening paragraphs remain visible in the full article.

Keep `index.md` as the listing page and `page/` for pagination templates. Blog Markdown is published with the documentation site; keep drafts in `docs/draft/` until they are ready to be shared.

### Localization

The site uses VitePress's [built-in internationalization](https://vitepress.dev/guide/i18n): `locales.en` and `locales.zh` in `docs/.vitepress/config.mts` provide language-specific navigation and theme configuration. Custom blog components read the active locale through [`useData().lang`](https://vitepress.dev/reference/runtime-api#usedata), then select their blog labels and filter posts by language. These blog labels belong to the custom theme; VitePress does not supply their translations automatically.

### Build and deployment

`npm run docs:build` generates OpenAPI JSON with Python, builds VitePress, and runs the chunk filename repair script. During the VitePress build, the post loader validates article metadata and the pagination loader generates each list page. Static files from `docs/public/`, including the pinned images, are copied into `docs/.vitepress/dist`. The `buildEnd` hook removes `images/.git`, renames chunks containing `.md.`, and updates references to those chunks; `docs/fix-chunk-names.js` handles any remaining chunk names. These filename changes avoid the site's WAF blocking `.md.` URLs.

The PR's Pre-commit Checks job runs the repository hooks. It does not check the image submodule or run the complete docs build; run `npm run docs:build` locally before requesting final review. After merging into `main`, changes matching `.github/workflows/deploy-docs.yml` trigger the Pages workflow, which checks out the same pinned image commit, builds the site, uploads `docs/.vitepress/dist`, and deploys the artifact. `workflow_dispatch` can also trigger deployment.

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
