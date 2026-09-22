<script setup lang="ts">
import { computed } from 'vue'
import { useData, withBase } from 'vitepress'

const { lang } = useData()
const isZh = computed(() => lang.value === 'zh-CN')

interface Feature {
  icon: string
  title: string
  titleZh: string
  description: string
  descriptionZh: string
  colorClass: 'primary' | 'tertiary'
  link?: string
  linkZh?: string
}

const features: Feature[] = [
  {
    icon: 'bot',
    title: 'Agentic Rollout',
    titleZh: 'Agentic Rollout',
    description: 'Connect your existing agent app to Relax training',
    descriptionZh: '将已有 Agent 接入 Relax 训练',
    colorClass: 'primary',
    link: '/en/guide/agentic-rollout',
    linkZh: '/zh/guide/agentic-rollout',
  },
  {
    icon: 'zap',
    title: 'High Performance',
    titleZh: '高性能计算',
    description: 'Support Megatron for training and SGLang for inference engine with optimized distributed computing',
    descriptionZh: '支持 Megatron 训练和 SGLang 推理引擎，优化的分布式计算能力',
    colorClass: 'tertiary',
  },
  {
    icon: 'layers',
    title: 'Multi-Modal Support',
    titleZh: '多模态支持',
    description: 'Seamlessly integrate text, vision, and audio modalities for comprehensive AI training',
    descriptionZh: '无缝集成文本、视觉和音频模态，实现全面的 AI 训练',
    colorClass: 'primary',
  },
  {
    icon: 'git-branch',
    title: 'Flexible Architecture',
    titleZh: '灵活架构',
    description: 'Service-oriented design with Controller, Service Layer, and Implementation Layer for easy customization',
    descriptionZh: '面向服务的设计，包含控制层、服务层和实现层，易于定制',
    colorClass: 'tertiary',
  },
  {
    icon: 'bar-chart',
    title: 'Advanced Monitoring',
    titleZh: '高级监控',
    description: 'Built-in metrics service, health check manager, and notification system for production-ready deployments',
    descriptionZh: '内置 Metrics 服务、健康检查管理器和通知系统，满足生产环境需求',
    colorClass: 'primary',
  },
  {
    icon: 'smile',
    title: 'Easy to Use',
    titleZh: '易于使用',
    description: 'Quick start examples and comprehensive documentation to get you up and running in minutes',
    descriptionZh: '快速入门示例和全面的文档，让您在几分钟内上手',
    colorClass: 'tertiary',
  },
]

function featureHref(feature: Feature): string | undefined {
  const link = isZh.value ? feature.linkZh : feature.link
  return link ? withBase(link) : undefined
}
</script>

<template>
  <section class="feature-grid-section">
    <div class="feature-grid">
      <component
        :is="feature.link ? 'a' : 'div'"
        v-for="(feature, index) in features"
        :key="index"
        class="feature-card"
        :href="featureHref(feature)"
      >
        <div :class="['feature-icon', `feature-icon--${feature.colorClass}`]">
          <!-- Bot icon -->
          <svg v-if="feature.icon === 'bot'" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 8V4H8" /><rect width="16" height="12" x="4" y="8" rx="2" />
            <path d="M2 14h2" /><path d="M20 14h2" /><path d="M15 13v2" /><path d="M9 13v2" />
          </svg>
          <!-- Zap icon -->
          <svg v-else-if="feature.icon === 'zap'" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
            <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
          </svg>
          <!-- Layers icon -->
          <svg v-else-if="feature.icon === 'layers'" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
            <polygon points="12 2 2 7 12 12 22 7 12 2" />
            <polyline points="2 17 12 22 22 17" />
            <polyline points="2 12 12 17 22 12" />
          </svg>
          <!-- GitBranch icon -->
          <svg v-else-if="feature.icon === 'git-branch'" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
            <line x1="6" y1="3" x2="6" y2="15" /><circle cx="18" cy="6" r="3" /><circle cx="6" cy="18" r="3" />
            <path d="M18 9a9 9 0 0 1-9 9" />
          </svg>
          <!-- BarChart icon -->
          <svg v-else-if="feature.icon === 'bar-chart'" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
            <line x1="12" y1="20" x2="12" y2="10" /><line x1="18" y1="20" x2="18" y2="4" /><line x1="6" y1="20" x2="6" y2="16" />
          </svg>
          <!-- Smile icon -->
          <svg v-else-if="feature.icon === 'smile'" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="10" /><path d="M8 14s1.5 2 4 2 4-2 4-2" />
            <line x1="9" y1="9" x2="9.01" y2="9" /><line x1="15" y1="9" x2="15.01" y2="9" />
          </svg>
        </div>
        <h3 class="feature-title">{{ isZh ? feature.titleZh : feature.title }}</h3>
        <p class="feature-desc">{{ isZh ? feature.descriptionZh : feature.description }}</p>
      </component>
    </div>
  </section>
</template>

<style scoped>
.feature-grid-section {
  position: relative;
  z-index: 1;
  max-width: 1152px;
  margin: 0 auto;
  padding: 0 32px 80px;
}

.feature-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 20px;
}

@media (min-width: 640px) {
  .feature-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}

@media (min-width: 960px) {
  .feature-grid {
    grid-template-columns: repeat(3, 1fr);
  }
}

.feature-card {
  background: rgba(19, 19, 22, 0.6);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  padding: 32px;
  border-radius: 12px;
  border: 1px solid rgba(72, 71, 75, 0.1);
  display: flex;
  flex-direction: column;
  transition: all 0.3s ease;
  color: inherit;
  text-decoration: none;
}

.feature-card[href]:focus-visible {
  outline: 2px solid #e8384a;
  outline-offset: 4px;
}

.feature-card:hover {
  background: rgba(31, 31, 35, 0.8);
  transform: translateY(-5px);
}

.feature-icon {
  width: 40px;
  height: 40px;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 24px;
}

.feature-icon svg {
  width: 20px;
  height: 20px;
}

.feature-icon--primary {
  background: rgba(232, 56, 74, 0.1);
  color: #e8384a;
}

.feature-icon--tertiary {
  background: rgba(189, 162, 255, 0.1);
  color: #bda2ff;
}

.feature-title {
  font-family: "Space Grotesk", sans-serif;
  font-size: 1.2rem;
  font-weight: 700;
  margin-bottom: 12px;
  color: #f0edf1;
}

.feature-desc {
  font-family: "Manrope", sans-serif;
  font-size: 0.875rem;
  line-height: 1.6;
  color: #acaaae;
  margin: 0;
}
</style>
