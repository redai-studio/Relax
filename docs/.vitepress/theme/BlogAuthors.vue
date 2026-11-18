<script setup lang="ts">
import { computed } from 'vue'
import { useData } from 'vitepress'
import type { Author } from './posts.data.mts'
import BlogAuthor from './BlogAuthor.vue'

const props = defineProps<{ authors: Author[]; compact?: boolean }>()
const avatarGroup = computed(() => props.compact && props.authors.length > 1)
const { lang } = useData()
</script>

<template>
  <ul
    class="blog-authors"
    :class="{ compact, 'avatar-group': avatarGroup }"
    :style="{ '--author-count': authors.length }"
    :aria-label="lang.startsWith('zh') ? '作者' : 'Authors'"
  >
    <li
      v-for="(author, index) in authors"
      :key="index"
      :style="avatarGroup ? { '--author-index': index } : undefined"
    >
      <BlogAuthor :author="author" :compact="compact" :avatar-only="avatarGroup" />
    </li>
  </ul>
</template>

<style scoped>
.blog-authors {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 176px), 1fr));
  gap: 20px 24px;
  width: 100%;
  min-width: 0;
  padding: 0;
  margin: 0;
  list-style: none;
}

.blog-authors li {
  min-width: 0;
  max-width: 100%;
}

.avatar-group {
  display: grid;
  /* Tracks shrink with the available space; avatars retain their full size. */
  grid-template-columns: repeat(calc(var(--author-count) - 1), minmax(0, 40px)) 32px;
  gap: 0;
  max-width: calc((var(--author-count) - 1) * 40px + 32px);
  isolation: isolate;
}

.avatar-group li {
  position: relative;
  z-index: calc(var(--author-count) - var(--author-index));
  width: 32px;
  max-width: none;
}

.avatar-group li:hover,
.avatar-group li:focus-within {
  z-index: calc(var(--author-count) + 1);
}
</style>
