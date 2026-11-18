<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { withBase } from 'vitepress'
import type { Author } from './posts.data.mts'

const props = defineProps<{ author: Author; compact?: boolean; avatarOnly?: boolean }>()
const name = computed(() => typeof props.author === 'string' ? props.author : props.author.name)
const github = computed(() => typeof props.author === 'string' ? undefined : props.author.github)
const label = computed(() => github.value ? `${name.value} (@${github.value})` : name.value)
const githubUrl = computed(() => github.value ? `https://github.com/${encodeURIComponent(github.value)}` : undefined)
const avatarUrl = computed(() => {
  const customAvatar = typeof props.author === 'string' ? undefined : props.author.avatar
  return customAvatar ? withBase(customAvatar) : githubUrl.value ? `${githubUrl.value}.png?size=80` : undefined
})
const avatarFailed = ref(false)
watch(avatarUrl, () => { avatarFailed.value = false })
</script>

<template>
  <component
    :is="githubUrl ? 'a' : 'span'"
    class="blog-author"
    :class="{ compact, 'avatar-only': avatarOnly }"
    :aria-label="avatarOnly ? label : undefined"
    :title="label"
    :href="githubUrl"
    :target="githubUrl ? '_blank' : undefined"
    :rel="githubUrl ? 'noopener noreferrer' : undefined"
  >
    <img
      v-if="avatarUrl && !avatarFailed"
      class="avatar VPImage"
      :src="avatarUrl"
      alt=""
      width="40"
      height="40"
      loading="lazy"
      decoding="async"
      @error="avatarFailed = true"
    />
    <span v-else class="avatar fallback" aria-hidden="true">{{ [...name][0] }}</span>
    <span class="details">
      <span class="name">{{ name }}</span>
      <span v-if="github" class="github">@{{ github }}</span>
    </span>
  </component>
</template>

<style scoped>
.blog-author {
  display: flex;
  align-items: center;
  gap: 10px;
  width: 100%;
  min-width: 0;
  max-width: 100%;
  text-decoration: none;
}

.blog-author .avatar {
  flex-shrink: 0;
  width: 40px;
  height: 40px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 50%;
  object-fit: cover;
  cursor: inherit;
}

.blog-author .avatar:hover {
  transform: none;
}

.compact .avatar {
  width: 32px;
  height: 32px;
}

.avatar-only {
  width: 32px;
  max-width: none;
  border-radius: 50%;
}

.avatar-only .avatar {
  border: 2px solid var(--vp-c-bg-soft);
}

.avatar-only .details {
  display: none;
}

.blog-author:focus-visible {
  outline: 2px solid var(--vp-c-brand-1);
  outline-offset: 3px;
}

.fallback {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background: var(--vp-c-brand-soft);
  color: var(--vp-c-brand-1);
}

.details {
  display: flex;
  flex-direction: column;
  min-width: 0;
  line-height: 1.5;
  overflow: hidden;
}

.name,
.github {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.name {
  color: var(--vp-c-text-1);
  font-size: 14px;
  font-weight: 500;
}

.github {
  color: var(--vp-c-text-2);
  font-family: var(--vp-font-family-label);
  font-size: 12px;
  font-weight: 400;
}

a.blog-author:hover .name,
a.blog-author:hover .github {
  color: var(--vp-c-brand-1);
}
</style>
