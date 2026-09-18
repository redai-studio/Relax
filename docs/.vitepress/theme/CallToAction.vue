<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { useData, useRoute, withBase } from 'vitepress'

const { lang } = useData()
const route = useRoute()
const isZh = computed(() => lang.value === 'zh-CN')

const isHome = ref(false)

function checkIsHome() {
  const path = route.path
  const p = path.replace(/index\.html$/, '').replace(/\/+$/, '')
  isHome.value = p === '' || p.endsWith('/en') || p.endsWith('/zh')
}

onMounted(() => checkIsHome())
watch(() => route.path, () => checkIsHome())

const headline = computed(() => isZh.value ? '准备好构建未来了吗？' : 'Ready to build the future?')
const subtitle = computed(() =>
  isZh.value
    ? '加入我们，一起用 Relax 探索强化学习的无限可能。'
    : 'Get started with Relax and explore the future of reinforcement learning together.'
)
const primaryLabel = computed(() => isZh.value ? '参与贡献' : 'Contribute Now')
const secondaryLabel = computed(() => isZh.value ? '讨论区' : 'Discussion')

const primaryLink = 'https://github.com/redai-studio/Relax/blob/main/CONTRIBUTING.md'
const secondaryLink = 'https://github.com/redai-studio/Relax/discussions'
</script>

<template>
  <section v-if="isHome" class="cta-section">
    <div class="cta-container">
      <!-- Subtle gradient overlay -->
      <div class="cta-gradient-overlay" />

      <h2 class="cta-headline">{{ headline }}</h2>
      <p class="cta-subtitle">{{ subtitle }}</p>

      <div class="cta-buttons">
        <a :href="primaryLink" class="cta-btn cta-btn--primary">{{ primaryLabel }}</a>
        <a :href="secondaryLink" class="cta-btn cta-btn--secondary">{{ secondaryLabel }}</a>
      </div>
    </div>
  </section>
</template>

<style scoped>
.cta-section {
  position: relative;
  z-index: 1;
  max-width: 1400px;
  margin: 0 auto;
  padding: 0 24px 96px;
}

.cta-container {
  position: relative;
  background: rgba(19, 19, 22, 0.8);
  padding: 48px;
  border-radius: 12px;
  border: 1px solid rgba(72, 71, 75, 0.1);
  text-align: center;
  overflow: hidden;
}

@media (min-width: 640px) {
  .cta-container {
    padding: 64px;
  }
}

.cta-gradient-overlay {
  position: absolute;
  inset: 0;
  background: linear-gradient(135deg, rgba(232, 56, 74, 0.05) 0%, transparent 60%);
  pointer-events: none;
}

.cta-headline {
  position: relative;
  font-family: "Space Grotesk", sans-serif;
  font-size: 2.2rem;
  font-weight: 700;
  color: #f0edf1;
  margin-bottom: 20px;
}

@media (min-width: 640px) {
  .cta-headline {
    font-size: 2.8rem;
  }
}

@media (min-width: 960px) {
  .cta-headline {
    font-size: 3rem;
  }
}

.cta-subtitle {
  position: relative;
  font-family: "Manrope", sans-serif;
  font-size: 1.125rem;
  color: #acaaae;
  max-width: 560px;
  margin: 0 auto 40px;
  line-height: 1.6;
}

.cta-buttons {
  position: relative;
  display: flex;
  flex-direction: column;
  gap: 16px;
  justify-content: center;
  align-items: center;
}

@media (min-width: 480px) {
  .cta-buttons {
    flex-direction: row;
  }
}

.cta-btn {
  display: inline-block;
  padding: 14px 40px;
  font-family: "Space Grotesk", sans-serif;
  font-size: 0.95rem;
  font-weight: 700;
  border-radius: 6px;
  text-decoration: none;
  transition: all 0.2s ease;
}

.cta-btn:hover {
  transform: scale(1.05);
}

.cta-btn:active {
  transform: scale(0.95);
}

.cta-btn--primary {
  background: #e8384a;
  color: #ffffff;
}

.cta-btn--primary:hover {
  filter: brightness(1.1);
  box-shadow: 0 0 24px rgba(232, 56, 74, 0.3);
}

.cta-btn--secondary {
  background: transparent;
  color: #f0edf1;
  border: 1px solid rgba(72, 71, 75, 0.6);
}

.cta-btn--secondary:hover {
  background: rgba(37, 37, 42, 0.6);
  border-color: rgba(72, 71, 75, 0.9);
}
</style>
