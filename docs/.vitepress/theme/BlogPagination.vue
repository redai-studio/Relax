<script setup lang="ts">
import { computed } from 'vue'
import { useData, withBase } from 'vitepress'
import { getPageNumbers } from './blog-pagination'

const props = defineProps<{ currentPage: number; totalPages: number }>()
const { lang } = useData()
const isChinese = computed(() => lang.value.startsWith('zh'))
const pages = computed(() => getPageNumbers(props.currentPage, props.totalPages))
const pageLabel = (page: number) => isChinese.value ? `第 ${page} 页` : `Page ${page}`
const pageLink = (page: number) =>
  withBase(`/${isChinese.value ? 'zh' : 'en'}/blog/${page === 1 ? '' : `page/${page}.html`}`)
</script>

<template>
  <nav v-if="totalPages > 1" class="pagination" :aria-label="isChinese ? '博客分页' : 'Blog pagination'">
    <a v-if="currentPage > 1" class="previous" :href="pageLink(currentPage - 1)" rel="prev">
      <span aria-hidden="true">←</span> {{ isChinese ? '上一页' : 'Previous' }}
    </a>
    <span v-else class="previous disabled" aria-disabled="true">
      <span aria-hidden="true">←</span> {{ isChinese ? '上一页' : 'Previous' }}
    </span>
    <ul class="pages">
      <li v-for="(item, index) in pages" :key="index">
        <a
          v-if="typeof item === 'number'"
          class="page-number"
          :href="pageLink(item)"
          :aria-label="pageLabel(item)"
          :aria-current="item === currentPage ? 'page' : undefined"
        >{{ item }}</a>
        <span v-else class="ellipsis" aria-hidden="true">{{ item }}</span>
      </li>
    </ul>
    <a v-if="currentPage < totalPages" class="next" :href="pageLink(currentPage + 1)" rel="next">
      {{ isChinese ? '下一页' : 'Next' }} <span aria-hidden="true">→</span>
    </a>
    <span v-else class="next disabled" aria-disabled="true">
      {{ isChinese ? '下一页' : 'Next' }} <span aria-hidden="true">→</span>
    </span>
  </nav>
</template>

<style scoped>
.pagination {
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  align-items: center;
  gap: 16px;
  margin-top: 40px;
  padding-top: 24px;
  border-top: 1px solid var(--vp-c-divider);
  font-size: 14px;
}

.pages {
  display: flex;
  justify-content: center;
  gap: 4px;
  padding: 0;
  margin: 0;
  list-style: none;
}

.page-number,
.ellipsis {
  display: flex;
  align-items: center;
  justify-content: center;
  min-width: 36px;
  height: 36px;
  padding: 0 6px;
  border-radius: 8px;
  font-variant-numeric: tabular-nums;
}

.previous {
  justify-self: start;
}

.next {
  justify-self: end;
}

.previous,
.next {
  padding: 8px 0;
  white-space: nowrap;
}

a:hover,
.page-number[aria-current='page'] {
  color: var(--vp-c-brand-1);
}

.page-number:hover,
.page-number[aria-current='page'] {
  background: var(--vp-c-brand-soft);
}

.page-number[aria-current='page'] {
  font-weight: 600;
}

.disabled,
.ellipsis {
  color: var(--vp-c-text-3);
}

a:focus-visible {
  outline: 2px solid var(--vp-c-brand-1);
  outline-offset: 3px;
}

@media (max-width: 639px) {
  .pagination {
    grid-template-columns: 1fr 1fr;
    gap: 12px;
  }

  .pages {
    grid-column: 1 / -1;
    grid-row: 1;
  }

  .page-number,
  .ellipsis {
    min-width: 32px;
    height: 36px;
  }
}
</style>
