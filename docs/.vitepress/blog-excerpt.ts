import type { MarkdownRenderer } from 'vitepress'

type Token = ReturnType<MarkdownRenderer['parse']>[number]

function selectExcerpt(tokens: Token[]) {
  const marker = tokens.findIndex(
    (token) =>
      token.type === 'html_block' && token.level === 0 && token.content.trim() === '<!-- more -->'
  )
  if (marker !== -1) return { tokens: tokens.slice(0, marker), range: [0, tokens[marker].map![0]] }

  // Skip headings and paragraphs nested in notices, lists, or blockquotes.
  const paragraph = tokens.findIndex(
    (token) => token.type === 'paragraph_open' && token.level === 0
  )
  return {
    tokens: paragraph === -1 ? [] : tokens.slice(paragraph, paragraph + 3),
    range: tokens[paragraph]?.map,
  }
}

export function extractBlogExcerpt(md: MarkdownRenderer, source: string): string {
  const { range } = selectExcerpt(md.parse(source, {}))
  if (!range) return ''
  return source.replace(/\r\n?/g, '\n').split('\n').slice(range[0], range[1]).join('\n').trimEnd()
}

function plainText(md: MarkdownRenderer, tokens: Token[]): string {
  return tokens
    .map((token) => {
      if (token.children) return plainText(md, token.children)
      if (token.type === 'softbreak' || token.type === 'hardbreak') return ' '
      if (token.type === 'text') {
        return token.content.replace(/&(?:#x[\da-f]+|#\d+|[a-z][a-z\d]+);/gi, md.utils.unescapeAll)
      }
      if (
        ['code_inline', 'math_inline', 'math_block', 'code_block', 'fence'].includes(token.type)
      ) {
        return token.content
      }
      return ''
    })
    .join('')
}

export default function blogDescription(md: MarkdownRenderer) {
  md.core.ruler.push('blog-description', (state) => {
    const { relativePath, frontmatter } = state.env
    if (
      !/^((en|zh)\/blog\/)(?!index\.md$|page\/).+\.md$/.test(relativePath ?? '') ||
      !frontmatter ||
      frontmatter.description !== undefined ||
      frontmatter.head?.some(
        ([tag, attrs]: [string, Record<string, string>]) =>
          tag === 'meta' && attrs?.name === 'description'
      )
    ) {
      return
    }
    frontmatter.description = selectExcerpt(state.tokens)
      .tokens.map((token) => plainText(md, [token]))
      .join(' ')
      .replace(/\s+/g, ' ')
      .trim()
  })
}
