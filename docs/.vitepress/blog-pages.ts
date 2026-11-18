import { readdirSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { sep } from 'node:path'
import { POSTS_PER_PAGE } from './theme/blog-pagination'

export function createBlogPagePaths(locale: 'en' | 'zh'): { params: { page: string } }[] {
  const directory = fileURLToPath(new URL(`../${locale}/blog/`, import.meta.url))
  const posts = readdirSync(directory, { recursive: true })
    .map((file) => file.split(sep).join('/'))
    .filter((file) => file.endsWith('.md') && file !== 'index.md' && !file.startsWith('page/'))
  const totalPages = Math.ceil(posts.length / POSTS_PER_PAGE)

  // Page one is the existing index; later pages are emitted as static HTML.
  return Array.from({ length: Math.max(0, totalPages - 1) }, (_, index) => ({
    params: { page: String(index + 2) },
  }))
}
