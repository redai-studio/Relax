import { statSync } from 'node:fs'
import { dirname, isAbsolute, relative, resolve, sep } from 'node:path'
import type { MarkdownRenderer } from 'vitepress'

interface SourceLinksOptions {
  repo: string
  branch: string
  repoRoot: string
  docsRoot: string
}

/** Link repository files outside the documentation source directory to GitHub. */
export default function sourceLinks(md: MarkdownRenderer, options: SourceLinksOptions): void {
  const { repo, branch } = options
  const repoRoot = resolve(options.repoRoot)
  const docsRoot = resolve(repoRoot, options.docsRoot)

  function toSourceUrl(href: string, documentPath?: string): string {
    if (!documentPath || !/^\.\.?\//.test(href)) return href

    const [, pathname, suffix] = href.match(/^([^?#]*)(.*)$/)!
    let targetPath: string
    try {
      targetPath = resolve(dirname(documentPath), decodeURIComponent(pathname))
    } catch {
      return href
    }

    const repoPath = relative(repoRoot, targetPath)
    const pathParts = repoPath.split(sep)
    if (isAbsolute(repoPath) || pathParts[0] === '..') return href
    if (targetPath === docsRoot || targetPath.startsWith(docsRoot + sep)) return href

    try {
      const targetStats = statSync(targetPath, { throwIfNoEntry: false })
      if (!targetStats) {
        console.warn(`[source-links] ${documentPath}: "${href}" points to missing target ${targetPath}`)
        return href
      }

      const kind = targetStats.isDirectory() ? 'tree' : 'blob'
      const urlPath = pathParts.map(encodeURIComponent).join('/')
      return `${repo.replace(/\/$/, '')}/${kind}/${encodeURIComponent(branch)}/${urlPath}${suffix}`
    } catch (error) {
      console.warn(`[source-links] ${documentPath}: cannot resolve "${href}": ${error}`)
      return href
    }
  }

  // Rewrite before VitePress normalizes local links into HTML page URLs.
  const renderLink = md.renderer.rules.link_open!
  md.renderer.rules.link_open = (tokens, index, options, env, self) => {
    const token = tokens[index]
    const href = token.attrGet('href')
    if (href) token.attrSet('href', toSourceUrl(href, env?.realPath ?? env?.path))
    return renderLink(tokens, index, options, env, self)
  }
}
