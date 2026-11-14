<script setup lang="ts">
import { computed } from 'vue'
import { useData, withBase } from 'vitepress'
import BlogAuthors from './BlogAuthors.vue'
import BlogPostNavigation from './BlogPostNavigation.vue'

const { page, frontmatter, lang } = useData()
const isChinese = computed(() => lang.value.startsWith('zh'))
const isBlogPost = computed(
  () => /^(en|zh)\/blog\/.+\.md$/.test(page.value.relativePath) &&
    !/\/blog\/(?:index\.md$|page\/)/.test(page.value.relativePath)
)
const date = computed(() => new Date(frontmatter.value.date).toISOString().slice(0, 10))
const authors = computed(() => [frontmatter.value.author, ...(frontmatter.value.co_authors ?? [])])
</script>

<template>
  <template v-if="isBlogPost">
    <header class="blog-post-header">
      <h1>{{ frontmatter.title }}</h1>
      <div class="publication">
        <time :datetime="date">{{ date }}</time>
        <a class="back-link" :href="withBase(isChinese ? '/zh/blog/' : '/en/blog/')">
          <span aria-hidden="true">←</span> {{ isChinese ? '所有文章' : 'All articles' }}
        </a>
      </div>
    </header>
    <div class="blog-post-sidebar">
      <BlogAuthors class="blog-post-authors" :authors="authors" />
      <BlogPostNavigation sidebar />
    </div>
  </template>
</template>

<style scoped>
.blog-post-header {
  margin-bottom: 24px;
  font-size: 14px;
}

.blog-post-authors {
  padding-bottom: 28px;
  margin-bottom: 32px;
  border-bottom: 1px solid var(--vp-c-divider);
}

.blog-post-header h1 {
  margin: 0;
  color: var(--vp-c-text-1);
  font-family: var(--vp-font-family-headline);
  font-size: clamp(32px, 4vw, 48px);
  font-weight: 700;
  line-height: 1.25;
  letter-spacing: -0.035em;
  overflow-wrap: anywhere;
  text-wrap: balance;
}

.back-link {
  color: var(--vp-c-brand-1);
  font-weight: 600;
}

.publication {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  margin-top: 24px;
  color: var(--vp-c-text-2);
}

/* Keep the title above the author, article, and outline columns. */
@media (min-width: 1280px) {
  :global(#VPContent .VPDoc:has(.blog-post-header) > .container) {
    display: grid;
    grid-template-columns: 192px minmax(0, 1fr) 224px;
    column-gap: 40px;
    max-width: 1280px;
    padding-bottom: 128px;
  }

  :global(#VPContent .VPDoc:has(.blog-post-header) > .container > .content),
  :global(#VPContent .VPDoc:has(.blog-post-header) > .container > .content > .content-container) {
    display: contents;
  }

  :global(#VPContent .VPDoc:has(.blog-post-header) > .container > .aside) {
    grid-column: 3;
    grid-row: 2 / span 3;
    min-width: 0;
    max-width: none;
    padding: 0;
  }

  :global(#VPContent .VPDoc:has(.blog-post-header) .aside-container) {
    position: sticky;
    top: calc(var(--vp-nav-height) + 24px);
    width: auto;
    height: auto;
    max-height: calc(100dvh - var(--vp-nav-height) - 48px);
    padding-top: 0;
  }

  :global(#VPContent .VPDoc:has(.blog-post-header) .aside-curtain) {
    display: none;
  }

  :global(#VPContent .VPDoc:has(.blog-post-header) .aside-content) {
    min-height: 0;
    padding-bottom: 0;
  }

  :global(#VPContent .VPDoc:has(.blog-post-header) .VPDocAsideOutline > .content) {
    padding-left: 20px;
  }

  :global(#VPContent .VPDoc:has(.blog-post-header) .outline-title) {
    color: var(--vp-c-text-2);
    font-family: var(--vp-font-family-label);
    font-size: 12px;
    letter-spacing: 0.06em;
  }

  :global(#VPContent .VPDoc:has(.blog-post-header) .outline-link) {
    font-size: 13px;
  }

  .blog-post-header {
    grid-column: 1 / -1;
    grid-row: 1;
    margin-bottom: 32px;
    padding-bottom: 32px;
    border-bottom: 1px solid var(--vp-c-divider);
    text-align: center;
  }

  .publication {
    justify-content: center;
    gap: 24px;
  }

  .blog-post-sidebar {
    grid-column: 1;
    grid-row: 2;
    align-self: start;
    min-width: 0;
  }

  .blog-post-authors {
    grid-template-columns: minmax(0, 1fr);
    gap: 28px;
    margin: 0;
    padding: 0;
    border: 0;
  }

  :global(#VPContent .VPDoc:has(.blog-post-header) .main),
  :global(#VPContent .VPDoc:has(.blog-post-header) .VPDocFooter) {
    grid-column: 2;
    min-width: 0;
  }
}
</style>
