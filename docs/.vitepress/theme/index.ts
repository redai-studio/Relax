// .vitepress/theme/index.ts
import { h, onMounted, watch, nextTick } from 'vue'
import type { Theme } from 'vitepress'
import DefaultTheme from 'vitepress/theme'
import mediumZoom from 'medium-zoom'
import { useRoute } from 'vitepress'
import AsciiBackground from './AsciiBackground.vue'
import AsciiHero from './AsciiHero.vue'
import AsciiFireworks from './AsciiFireworks.vue'
import FeatureShowcase from './FeatureShowcase.vue'
import FeatureGrid from './FeatureGrid.vue'
import CallToAction from './CallToAction.vue'
import MarqueeStrip from './MarqueeStrip.vue'
import SwaggerUI from './SwaggerUI.vue'
import BlogPostHeader from './BlogPostHeader.vue'
import BlogPostNavigation from './BlogPostNavigation.vue'
import './custom.css'

export default {
  extends: DefaultTheme,
  Layout: () => {
    return h(DefaultTheme.Layout, null, {
      // Inject ASCII art background canvas behind all content
      'layout-top': () => h(AsciiBackground),
      'doc-before': () => h(BlogPostHeader),
      'doc-after': () => h(BlogPostNavigation),
      // Inject typographic hero title before the default hero info
      'home-hero-info-before': () => h(AsciiHero),
      // Inject custom feature grid before the default VitePress features
      'home-features-before': () => h(FeatureGrid),
      // Inject feature showcase section after features, before footer
      'home-features-after': () => [h(FeatureShowcase), h(MarqueeStrip), h(CallToAction)],
      // Inject ASCII fireworks animation at the bottom of the page
      'layout-bottom': () => h(AsciiFireworks),
    })
  },
  setup() {
    const route = useRoute()
    const initZoom = () => {
      mediumZoom('.main img:not(.VPImage)', {
        background: 'var(--vp-c-bg)',
        margin: 24,
      })
    }
    onMounted(() => {
      initZoom()
    })
    watch(
      () => route.path,
      () => nextTick(() => initZoom())
    )
  },
  enhanceApp({ app, router, siteData }) {
    app.component('SwaggerUI', SwaggerUI)
  },
} satisfies Theme
