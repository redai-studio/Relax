import { createContentLoader, createMarkdownRenderer, type SiteConfig } from 'vitepress'
import { extractBlogExcerpt } from '../blog-excerpt'

export type Author = string | { name: string; github?: string; avatar?: string }

interface Post {
  title: string
  url: string
  date: string
  authors: Author[]
  excerpt?: string
  pinned: boolean
}

declare const data: Post[]
export { data }

const config = (globalThis as typeof globalThis & { VITEPRESS_CONFIG: SiteConfig }).VITEPRESS_CONFIG
const md = await createMarkdownRenderer(
  config.srcDir,
  config.markdown,
  config.site.base,
  config.logger
)

export default createContentLoader(['en/blog/**/*.md', 'zh/blog/**/*.md'], {
  excerpt(file) {
    file.excerpt = extractBlogExcerpt(md, file.content)
  },
  transform(pages): Post[] {
    return pages
      .filter(({ url }) => !/^\/(en|zh)\/blog\/(?:$|page\/)/.test(url))
      .map(({ url, frontmatter, excerpt }) => {
        const { title, author, date, description } = frontmatter
        if (frontmatter.pinned !== undefined && typeof frontmatter.pinned !== 'boolean') {
          throw new Error(`${url}: pinned must be a boolean`)
        }
        const coAuthors = frontmatter.co_authors ?? []
        if (!Array.isArray(coAuthors)) {
          throw new Error(`${url}: co_authors must be an array`)
        }
        const authors: Author[] = [author, ...coAuthors]
        for (const [field, value] of Object.entries({ title, date })) {
          if (typeof value !== 'string' || !value.trim()) {
            throw new Error(`${url}: blog frontmatter requires a non-empty ${field} string`)
          }
        }
        if (description !== undefined && (typeof description !== 'string' || !description.trim())) {
          throw new Error(`${url}: description must be a non-empty string when provided`)
        }
        for (const [index, entry] of authors.entries()) {
          const field = index === 0 ? 'author' : `co_authors[${index - 1}]`
          const authorName = typeof entry === 'string' ? entry : entry?.name
          if (typeof authorName !== 'string' || !authorName.trim()) {
            throw new Error(`${url}: ${field} requires a non-empty name`)
          }
          if (
            typeof entry !== 'string' &&
            entry.github !== undefined &&
            (typeof entry.github !== 'string' ||
              !/^[a-z\d](?:[a-z\d-]{0,37}[a-z\d])?$/i.test(entry.github))
          ) {
            throw new Error(`${url}: ${field}.github must be a GitHub username`)
          }
          if (
            typeof entry !== 'string' &&
            entry.avatar !== undefined &&
            (typeof entry.avatar !== 'string' || !/^(?:https?:\/\/|\/)[^\s]+$/.test(entry.avatar))
          ) {
            throw new Error(`${url}: ${field}.avatar must be an HTTP(S) URL or a site-root path`)
          }
        }
        const timestamp = Date.parse(`${date}T00:00:00Z`)
        if (
          !/^\d{4}-\d{2}-\d{2}$/.test(date) ||
          !Number.isFinite(timestamp) ||
          new Date(timestamp).toISOString().slice(0, 10) !== date
        ) {
          throw new Error(`${url}: blog date must be a valid YYYY-MM-DD date`)
        }

        return {
          title,
          url,
          date,
          authors,
          excerpt,
          pinned: frontmatter.pinned === true,
        }
      })
      .sort(
        (a, b) =>
          Number(b.pinned) - Number(a.pinned) ||
          b.date.localeCompare(a.date) ||
          a.url.localeCompare(b.url)
      )
  },
})
