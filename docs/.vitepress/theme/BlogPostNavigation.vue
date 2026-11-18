<script setup lang="ts">
import { computed } from 'vue'
import { useData, withBase } from 'vitepress'
import { data as posts } from './posts.data.mts'

defineProps<{ sidebar?: boolean }>()

const { page, lang } = useData()
const isChinese = computed(() => lang.value.startsWith('zh'))
// Pinning changes the list only; article navigation remains chronological.
const chronologicalPosts = computed(() =>
  posts
    .filter((post) => post.url.startsWith(isChinese.value ? '/zh/' : '/en/'))
    .sort((a, b) => b.date.localeCompare(a.date) || a.url.localeCompare(b.url))
)
const currentIndex = computed(() => chronologicalPosts.value.findIndex(
  (post) => post.url === `/${page.value.relativePath.replace(/\.md$/, '.html')}`
))
const previousPost = computed(() => currentIndex.value < 0 ? undefined : chronologicalPosts.value[currentIndex.value + 1])
const nextPost = computed(() => currentIndex.value < 0 ? undefined : chronologicalPosts.value[currentIndex.value - 1])
</script>

<template>
  <nav v-if="previousPost || nextPost" class="blog-post-navigation" :class="{ sidebar }" :aria-label="isChinese ? '文章导航' : 'Article navigation'">
    <p class="navigation-label">{{ isChinese ? '继续阅读' : 'Keep reading' }}</p>
    <div class="links">
      <a v-if="previousPost" class="previous" :href="withBase(previousPost.url)" rel="prev">
        <span class="direction">
          <span>{{ isChinese ? '上一篇' : 'Previous article' }}</span>
          <span class="arrow" aria-hidden="true">←</span>
        </span>
        <span class="title">{{ previousPost.title }}</span>
      </a>
      <a v-if="nextPost" class="next" :href="withBase(nextPost.url)" rel="next">
        <span class="direction">
          <span>{{ isChinese ? '下一篇' : 'Next article' }}</span>
          <span class="arrow" aria-hidden="true">→</span>
        </span>
        <span class="title">{{ nextPost.title }}</span>
      </a>
    </div>
  </nav>
</template>

<style scoped>
.blog-post-navigation {
  margin-top: 32px;
  padding-top: 24px;
  border-top: 1px solid var(--vp-c-divider);
}

.blog-post-navigation.sidebar {
  display: none;
}

.navigation-label {
  margin: 0 0 16px;
  color: var(--vp-c-text-2);
  font-family: var(--vp-font-family-label);
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.06em;
}

.links {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
}

.links:has(> a:only-child) {
  grid-template-columns: minmax(0, 1fr);
}

a {
  display: flex;
  flex-direction: column;
  gap: 12px;
  min-width: 0;
  padding: 20px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  background: var(--vp-c-bg-soft);
  text-decoration: none;
  transition: border-color 0.2s;
}

a:hover,
a:focus-visible {
  border-color: var(--vp-c-brand-1);
}

a:focus-visible {
  outline: 2px solid var(--vp-c-brand-1);
  outline-offset: 4px;
}

.direction {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  color: var(--vp-c-text-2);
  font-size: 13px;
}

.arrow {
  color: var(--vp-c-brand-1);
  font-size: 18px;
  line-height: 1;
}

.title {
  color: var(--vp-c-text-1);
  font-size: 16px;
  font-weight: 500;
  line-height: 1.7;
  white-space: normal;
  overflow-wrap: anywhere;
}

a:hover .title,
a:focus-visible .title {
  color: var(--vp-c-brand-1);
}

@media (max-width: 639px) {
  .links {
    grid-template-columns: minmax(0, 1fr);
  }
}

@media (min-width: 1280px) {
  .blog-post-navigation {
    display: none;
  }

  .blog-post-navigation.sidebar {
    display: block;
  }

  .links {
    grid-template-columns: minmax(0, 1fr);
    gap: 20px;
  }

  a {
    gap: 8px;
    padding: 0 0 20px;
    border: 0;
    border-bottom: 1px solid var(--vp-c-divider);
    border-radius: 0;
    background: none;
  }

  a:last-child {
    padding-bottom: 0;
    border-bottom: 0;
  }

  .title {
    font-size: 14px;
    line-height: 1.8;
  }
}

@media (prefers-reduced-motion: reduce) {
  a {
    transition: none;
  }
}
</style>
