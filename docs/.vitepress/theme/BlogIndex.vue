<script setup lang="ts">
import { computed } from 'vue'
import { useData, withBase } from 'vitepress'
import { data as posts } from './posts.data.mts'
import BlogAuthors from './BlogAuthors.vue'
import BlogPagination from './BlogPagination.vue'
import { POSTS_PER_PAGE } from './blog-pagination'

const { lang, frontmatter, page } = useData()
const isChinese = computed(() => lang.value.startsWith('zh'))
const localizedPosts = computed(() =>
  posts.filter((post) => post.url.startsWith(isChinese.value ? '/zh/' : '/en/'))
)
const currentPage = computed(() => Number(page.value.params?.page ?? 1))
const totalPages = computed(() => Math.ceil(localizedPosts.value.length / POSTS_PER_PAGE))
const postsInPage = computed(() =>
  localizedPosts.value.slice((currentPage.value - 1) * POSTS_PER_PAGE, currentPage.value * POSTS_PER_PAGE)
)
// Excerpts also appear on paginated lists, so resolve links from the article URL.
const formatExcerpt = (html: string, postUrl: string) =>
  html.replace(/<(?:a|img)\b[^>]*>/g, (tag) =>
    tag.replace(/\s(href|src)="([^"]*)"/g, (attribute, name: string, value: string) => {
      if (!value || /^(?:[a-z][a-z\d+.-]*:|\/\/)/i.test(value)) return attribute
      if (value.startsWith('/')) {
        return name === 'src' ? ` ${name}="${withBase(value)}"` : attribute
      }
      const url = new URL(value, `https://blog.invalid${withBase(postUrl)}`)
      return ` ${name}="${url.pathname}${url.search}${url.hash}"`
    })
  )
const formatDate = (date: string) =>
  new Date(`${date}T00:00:00Z`).toLocaleDateString(isChinese.value ? 'zh-CN' : 'en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    timeZone: 'UTC',
  })
</script>

<template>
  <main class="blog-index">
    <header class="blog-hero">
      <p class="eyebrow">{{ isChinese ? '工程 · 研究 · 社区' : 'ENGINEERING · RESEARCH · COMMUNITY' }}</p>
      <h1>Relax <span>{{ isChinese ? '博客' : 'Blog' }}</span><span class="period">.</span></h1>
      <p class="intro">{{ frontmatter.description }}</p>
    </header>

    <section aria-labelledby="articles-heading">
      <div class="section-heading">
        <h2 id="articles-heading">{{ isChinese ? '最新文章' : 'Latest articles' }}</h2>
        <span v-if="localizedPosts.length" class="article-count">
          {{ String(localizedPosts.length).padStart(2, '0') }}
        </span>
        <span v-if="totalPages > 1" class="page-indicator">
          {{ isChinese ? `第 ${currentPage} / ${totalPages} 页` : `Page ${currentPage} of ${totalPages}` }}
        </span>
      </div>
      <div v-if="!localizedPosts.length" class="empty">
        <span class="empty-symbol" aria-hidden="true">[ &nbsp; ]</span>
        <h3>{{ isChinese ? '文章正在整理中' : 'Good things take a little writing.' }}</h3>
        <p>{{ isChinese ? '从设计思考到实践复盘，我们会在这里分享 Relax 背后的故事。' : 'Design notes, lessons learned, and the stories behind Relax. Coming soon.' }}</p>
        <a :href="withBase(isChinese ? '/zh/guide/introduction.html' : '/en/guide/introduction.html')">
          {{ isChinese ? '先了解 Relax' : 'Explore Relax' }} <span aria-hidden="true">↗</span>
        </a>
      </div>
      <div v-else class="posts">
        <article v-for="(post, index) in postsInPage" :key="post.url" class="post" :class="{ latest: currentPage === 1 && index === 0 }">
          <div class="metadata">
            <span v-if="post.pinned" class="latest-label">{{ isChinese ? '置顶' : 'PINNED' }}</span>
            <span v-else-if="currentPage === 1 && index === 0" class="latest-label">{{ isChinese ? '最新发布' : 'LATEST' }}</span>
            <time :datetime="post.date">{{ formatDate(post.date) }}</time>
          </div>
          <h3><a :href="withBase(post.url)">{{ post.title }}</a></h3>
          <div v-if="post.excerpt" class="summary vp-doc" v-html="formatExcerpt(post.excerpt, post.url)" />
          <div class="post-footer">
            <BlogAuthors class="authors" :authors="post.authors" compact />
            <a class="read-more" :href="withBase(post.url)" :aria-label="`${isChinese ? '阅读全文' : 'Read more'}: ${post.title}`">
              {{ isChinese ? '阅读全文' : 'Read more' }} <span aria-hidden="true">→</span>
            </a>
          </div>
        </article>
      </div>
      <BlogPagination :current-page="currentPage" :total-pages="totalPages" />
    </section>
  </main>
</template>

<style scoped>
.blog-index {
  max-width: 1120px;
  margin: 0 auto;
  padding: 80px 32px 96px;
}

.blog-hero {
  padding-bottom: 64px;
}

.eyebrow {
  margin-bottom: 24px;
  color: var(--vp-c-text-2);
  font-family: var(--vp-font-family-label);
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.12em;
}

.eyebrow::before {
  display: inline-block;
  width: 8px;
  height: 8px;
  margin-right: 12px;
  border-radius: 50%;
  background: var(--vp-c-brand-1);
  content: '';
}

h1 {
  font-family: var(--vp-font-family-headline);
  font-size: clamp(44px, 7vw, 80px);
  font-weight: 700;
  line-height: 1.15;
  letter-spacing: -0.045em;
}

h1 span {
  color: var(--vp-c-brand-1);
}

.period {
  margin-left: 4px;
}

.intro {
  max-width: 620px;
  margin-top: 24px;
  color: var(--vp-c-text-2);
  font-size: 18px;
  line-height: 1.8;
}

.section-heading {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 20px 0;
  border-top: 1px solid var(--vp-c-divider);
}

.section-heading h2 {
  font-size: 16px;
  font-weight: 600;
}

.article-count {
  color: var(--vp-c-text-3);
  font-family: var(--vp-font-family-mono);
  font-size: 12px;
}

.page-indicator {
  margin-left: auto;
  color: var(--vp-c-text-2);
  font-size: 13px;
}

.posts {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 20px;
}

.post {
  display: flex;
  flex-direction: column;
  min-width: 0;
  padding: 32px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 16px;
  background: var(--vp-c-bg-soft);
  transition: border-color 0.2s;
  overflow-wrap: anywhere;
}

.post:hover {
  border-color: var(--vp-c-brand-1);
}

.latest {
  grid-column: 1 / -1;
  padding: 40px;
  border-top: 2px solid var(--vp-c-brand-1);
}

.metadata {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 16px;
  color: var(--vp-c-text-2);
  font-family: var(--vp-font-family-label);
  font-size: 13px;
}

.latest-label {
  color: var(--vp-c-brand-1);
  font-weight: 600;
  letter-spacing: 0.06em;
}

.post h3 {
  margin-top: 20px;
  font-family: var(--vp-font-family-headline);
  font-size: 24px;
  font-weight: 600;
  line-height: 1.5;
}

.latest h3 {
  max-width: 800px;
  font-size: 32px;
}

.post h3 a {
  color: var(--vp-c-text-1);
}

a:hover,
.post h3 a:hover {
  color: var(--vp-c-brand-1);
}

.summary {
  min-width: 0;
  max-width: 800px;
  margin: 16px 0 28px;
  color: var(--vp-c-text-2);
  line-height: 1.8;
}

.summary :deep(> :first-child) {
  margin-top: 0;
}

.summary :deep(> :last-child) {
  margin-bottom: 0;
}

.summary :deep(img) {
  max-width: 100%;
  border-radius: 8px;
}

.post-footer {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  align-items: center;
  gap: 20px;
  margin-top: auto;
  font-size: 14px;
}

.authors {
  min-width: 0;
}

.read-more {
  white-space: nowrap;
}

.read-more,
.empty a {
  color: var(--vp-c-brand-1);
  font-weight: 600;
}

.empty {
  padding: 48px 32px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 16px;
  background: var(--vp-c-bg-soft);
  text-align: center;
}

.empty-symbol {
  color: var(--vp-c-brand-1);
  font-family: var(--vp-font-family-mono);
  font-size: 32px;
}

.empty h3 {
  margin-top: 20px;
  font-size: 22px;
  font-weight: 600;
}

.empty p {
  margin: 12px auto 24px;
  color: var(--vp-c-text-2);
  line-height: 1.8;
}

@media (max-width: 639px) {
  .blog-index {
    padding: 48px 24px 64px;
  }

  .blog-hero {
    padding-bottom: 40px;
  }

  .posts {
    grid-template-columns: minmax(0, 1fr);
  }

  .post,
  .empty {
    padding: 24px;
  }

  .latest h3 {
    font-size: 26px;
  }
}

@media (prefers-reduced-motion: reduce) {
  .post {
    transition: none;
  }
}
</style>
