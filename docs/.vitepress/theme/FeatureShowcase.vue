<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch, nextTick } from 'vue'
import { useRoute, useData } from 'vitepress'

const { lang } = useData()
const isZh = computed(() => lang.value === 'zh-CN')

const isHome = ref(false)
const route = useRoute()
const sectionRef = ref<HTMLElement | null>(null)
const isVisible = ref(false)

// --- Scroll-based progressive overlay ---
// scrollProgress: 0 = section not yet visible, 1 = fully scrolled into view
const scrollProgress = ref(0)

function checkIsHome() {
  const path = route.path
  const p = path.replace(/index\.html$/, '').replace(/\/+$/, '')
  isHome.value = p === '' || p.endsWith('/en') || p.endsWith('/zh')
}

// --- IntersectionObserver for initial visibility detection ---
let observer: IntersectionObserver | null = null

function setupObserver() {
  if (!sectionRef.value) return
  observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        isVisible.value = entry.isIntersecting
      }
    },
    { threshold: 0.05 }
  )
  observer.observe(sectionRef.value)
}

// --- Scroll listener for progressive overlay darkening ---
let rafId: number | null = null

function updateScrollProgress() {
  if (!sectionRef.value) return

  const rect = sectionRef.value.getBoundingClientRect()
  const viewportH = window.innerHeight

  // Start darkening well before the section enters —
  // when the section top is 200px below the viewport bottom, progress starts.
  // Fully dark when section top has scrolled to 30% from the top of viewport.
  const triggerOffset = 200
  const rawProgress = (viewportH + triggerOffset - rect.top) / (viewportH * 0.7 + triggerOffset)
  scrollProgress.value = Math.max(0, Math.min(1, rawProgress))
}

function onScroll() {
  if (rafId !== null) return
  rafId = requestAnimationFrame(() => {
    rafId = null
    updateScrollProgress()
  })
}

onMounted(() => {
  checkIsHome()
  nextTick(() => {
    setupObserver()
    updateScrollProgress()
  })
  window.addEventListener('scroll', onScroll, { passive: true })
  initTabs()
})

onUnmounted(() => {
  if (observer) {
    observer.disconnect()
    observer = null
  }
  if (rafId !== null) {
    cancelAnimationFrame(rafId)
    rafId = null
  }
  window.removeEventListener('scroll', onScroll)
  if (timerId !== null) {
    clearTimeout(timerId)
    timerId = null
  }
})

watch(() => route.path, () => {
  checkIsHome()
})

// Toggle class on <html> so global CSS can dim the ASCII canvas smoothly.
// Also set a CSS custom property for progressive dimming of the canvas.
watch(isVisible, (val) => {
  document.documentElement.classList.toggle('feature-showcase-active', val)
})

watch(scrollProgress, (val) => {
  document.documentElement.style.setProperty('--showcase-scroll-progress', String(val))
})


// --- Tab switching logic (ported from feature.html) ---
const DURATION = 10000
let currentIndex = 0
let timerId: ReturnType<typeof setTimeout> | null = null

const tabRefs = ref<HTMLElement[]>([])
const panelRefs = ref<HTMLElement[]>([])
const displayContainerRef = ref<HTMLElement | null>(null)

// --- Pause on hover state ---
const isPaused = ref(false)
let remainingTime = DURATION          // ms remaining when paused
let timerStartedAt = 0                // timestamp when current timer started

function switchTab(index: number) {
  const tabs = tabRefs.value
  const panels = panelRefs.value
  if (!tabs.length || !panels.length) return

  // Remove old state
  tabs[currentIndex]?.classList.remove('active')
  panels[currentIndex]?.classList.remove('active')

  currentIndex = index

  // Add new state
  tabs[currentIndex]?.classList.add('active')
  panels[currentIndex]?.classList.add('active')

  restartPanelAnimation()

  // Progress bar
  startProgressBar(tabs[currentIndex])

  // Auto-switch timer
  scheduleNextTab(DURATION)
}

/** Schedule the next tab switch after `delay` ms */
function scheduleNextTab(delay: number) {
  if (timerId !== null) clearTimeout(timerId)
  remainingTime = delay
  timerStartedAt = Date.now()
  timerId = setTimeout(() => {
    const nextIndex = (currentIndex + 1) % tabRefs.value.length
    switchTab(nextIndex)
  }, delay)
}

function startProgressBar(tabElement: HTMLElement) {
  const bar = tabElement.querySelector('.progress-bar-fill') as HTMLElement
  if (!bar) return
  bar.style.transition = 'none'
  bar.style.width = '0%'
  void bar.offsetWidth
  bar.style.transition = `width ${DURATION}ms linear`
  bar.style.width = '100%'
}

function initTabs() {
  nextTick(() => {
    if (tabRefs.value.length && panelRefs.value.length) {
      switchTab(0)
    }
  })
}

function onTabHover(index: number) {
  switchTab(index)
}

/** Pause timer & animations when mouse enters display panel */
function onDisplayEnter() {
  if (isPaused.value) return
  isPaused.value = true

  // Calculate how much time is left
  const elapsed = Date.now() - timerStartedAt
  remainingTime = Math.max(remainingTime - elapsed, 0)

  // Clear the running timer
  if (timerId !== null) {
    clearTimeout(timerId)
    timerId = null
  }

  // Pause the progress bar CSS transition
  const activeTab = tabRefs.value[currentIndex]
  if (activeTab) {
    const bar = activeTab.querySelector('.progress-bar-fill') as HTMLElement
    if (bar) {
      const computed = getComputedStyle(bar).width
      bar.style.transition = 'none'
      bar.style.width = computed          // freeze at current position
    }
  }

  // Pause right-side panel CSS animations
  const container = displayContainerRef.value
  if (container) {
    container.style.setProperty('--panel-play-state', 'paused')
  }
}

/** Resume timer & animations when mouse leaves display panel */
function onDisplayLeave() {
  if (!isPaused.value) return
  isPaused.value = false

  // Resume the progress bar from where it stopped
  const activeTab = tabRefs.value[currentIndex]
  if (activeTab) {
    const bar = activeTab.querySelector('.progress-bar-fill') as HTMLElement
    if (bar) {
      void bar.offsetWidth
      bar.style.transition = `width ${remainingTime}ms linear`
      bar.style.width = '100%'
    }
  }

  // Resume right-side panel CSS animations
  const container = displayContainerRef.value
  if (container) {
    container.style.setProperty('--panel-play-state', 'running')
  }

  // Restart timeout for remaining time
  scheduleNextTab(remainingTime)
}

function setTabRef(el: any, index: number) {
  if (el) tabRefs.value[index] = el
}

function setPanelRef(el: any, index: number) {
  if (el) panelRefs.value[index] = el
}

function restartPanelAnimation() {
  const container = displayContainerRef.value
  if (!container) return
  container.classList.remove('panel-active-anim')
  void container.offsetWidth
  container.classList.add('panel-active-anim')
}

// --- Panel 1: current Agentic rollout integration path ---
const agenticTab = ref<'apps' | 'contexts' | 'request'>('apps')

function switchAgenticTab(tab: 'apps' | 'contexts' | 'request') {
  agenticTab.value = tab
  nextTick(() => {
    restartPanelAnimation()
    displayContainerRef.value?.style.setProperty('--panel-play-state', 'running')
  })
}
</script>

<template>
  <section
    v-show="isHome"
    ref="sectionRef"
    class="feature-showcase-section"
    :class="{ 'is-visible': isVisible }"
  >
    <!-- Solid color background that fades in -->
    <div class="showcase-bg-solid" />

    <div class="showcase-inner">
      <main class="showcase-main-content">
        <!-- Left: Tab controls -->
        <section class="showcase-text-section">
          <h2 class="showcase-title">
            <template v-if="isZh">技术精深，<br/><span class="showcase-title-brand">但体验极简。</span></template>
            <template v-else>Technical Sophistication<br/><span class="showcase-title-brand">Without Complexity.</span></template>
          </h2>

          <div class="showcase-feature-list">
            <!-- Tab 1: Server-based -->
            <div
              class="showcase-feature-item active"
              :ref="(el) => setTabRef(el, 0)"
              @mouseenter="onTabHover(0)"
            >
              <div class="progress-container"><div class="progress-bar-fill" /></div>
              <div class="feature-header">
                <div class="step-number">1</div>
                <h3 class="feature-title">Server-Based Architecture</h3>
              </div>
              <p class="feature-desc">Layered service design — Orchestration layer coordinates training loop, Components layer deploys Actor/Rollout/GenRM as Ray Serve services, Engine &amp; Backends layer runs Megatron and SGLang. Data flows through TransferQueue, weights sync via Checkpoint Engine.</p>
            </div>

            <!-- Tab 2: Fully Async -->
            <div
              class="showcase-feature-item"
              :ref="(el) => setTabRef(el, 1)"
              @mouseenter="onTabHover(1)"
            >
              <div class="progress-container"><div class="progress-bar-fill" /></div>
              <div class="feature-header">
                <div class="step-number">2</div>
                <h3 class="feature-title">Fully Async Pipeline</h3>
              </div>
              <p class="feature-desc">Rollout, Reference, Forward, and Training stages run as interleaved pipelines — achieving near-zero GPU idle time with perfect overlap across batches.</p>
            </div>

            <!-- Tab 3: Agentic -->
            <div
              class="showcase-feature-item"
              :ref="(el) => setTabRef(el, 2)"
              @mouseenter="onTabHover(2)"
            >
              <div class="progress-container"><div class="progress-bar-fill" /></div>
              <div class="feature-header">
                <div class="step-number">3</div>
                <h3 class="feature-title">Agentic Rollout</h3>
              </div>
              <p class="feature-desc">Connect an existing agent app through Chat Completions. Relax manages the process, records multi-turn contexts, and sends selected contexts to training.</p>
            </div>

            <!-- Tab 4: Elastic Rollout -->
            <div
              class="showcase-feature-item"
              :ref="(el) => setTabRef(el, 3)"
              @mouseenter="onTabHover(3)"
            >
              <div class="progress-container"><div class="progress-bar-fill" /></div>
              <div class="feature-header">
                <div class="step-number">4</div>
                <h3 class="feature-title">Elastic Rollout Scaling</h3>
              </div>
              <p class="feature-desc">Scale inference engines up or down via REST API — without restarting training. Supports same-cluster scaling and cross-cluster federation with P2P weight sync.</p>
            </div>
          </div>
        </section>

        <!-- Right: Display panels -->
        <section
          class="showcase-display-section"
          ref="displayContainerRef"
          @mouseenter="onDisplayEnter"
          @mouseleave="onDisplayLeave"
        >
          <div class="window-header">
            <div class="mac-btn close" />
            <div class="mac-btn minimize" />
            <div class="mac-btn maximize" />
            <span class="window-title">Relax Framework</span>
          </div>

          <!-- Panel 3: Agentic visual showcase -->
          <div class="display-panel" :ref="(el) => setPanelRef(el, 2)" id="showcase-panel-1">
            <div class="agentic-visual-stage">
              <!-- Existing agent applications remain outside Relax. -->
              <div v-show="agenticTab === 'apps'" class="agentic-visual agentic-apps-view">
                <svg viewBox="0 0 600 420" role="img" aria-labelledby="agent-apps-title agent-apps-desc">
                  <title id="agent-apps-title">Existing agent applications connect to Relax</title>
                  <desc id="agent-apps-desc">Search, coding, and multimodal agent applications remain separate from Relax and call one Chat Completions endpoint, which records SessionForest state and sends selected contexts to training.</desc>
                  <defs>
                    <marker id="agent-app-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
                      <path d="M 0 0 L 10 5 L 0 10 z" fill="#7da2ff" />
                    </marker>
                  </defs>

                  <path class="agent-app-flow" d="M208 118 H376" marker-end="url(#agent-app-arrow)" />
                  <path class="agent-app-flow" d="M208 208 H280 Q288 208 288 200 V134 Q288 126 296 126 H376" marker-end="url(#agent-app-arrow)" />
                  <path class="agent-app-flow" d="M208 298 H304 Q312 298 312 290 V150 Q312 142 320 142 H376" marker-end="url(#agent-app-arrow)" />
                  <path class="agent-app-internal-flow" d="M460 160 V188" marker-end="url(#agent-app-arrow)" />
                  <path class="agent-app-internal-flow" d="M460 244 V284" marker-end="url(#agent-app-arrow)" />

                  <path class="agent-app-flow-active agent-app-flow-active-1" d="M208 118 H376" aria-hidden="true" />
                  <path class="agent-app-flow-active agent-app-flow-active-2" d="M208 208 H280 Q288 208 288 200 V134 Q288 126 296 126 H376" aria-hidden="true" />
                  <path class="agent-app-flow-active agent-app-flow-active-3" d="M208 298 H304 Q312 298 312 290 V150 Q312 142 320 142 H376" aria-hidden="true" />
                  <path class="agent-app-internal-active agent-app-flow-active-4" d="M460 160 V188" aria-hidden="true" />
                  <path class="agent-app-internal-active agent-app-flow-active-5" d="M460 244 V284" aria-hidden="true" />

                  <rect class="agent-app-zone" x="24" y="44" width="220" height="320" rx="10" />
                  <text class="agent-app-zone-label" x="44" y="72">EXISTING AGENT APPS</text>

                  <rect class="agent-app-zone agent-app-relax-zone" x="344" y="44" width="232" height="320" rx="10" />
                  <text class="agent-app-zone-label" x="364" y="72">RELAX</text>

                  <rect class="agent-app-node" x="56" y="90" width="152" height="56" rx="8" />
                  <text class="agent-app-node-title" x="132" y="116" text-anchor="middle">Search Agent</text>
                  <text class="agent-app-node-sub" x="132" y="134" text-anchor="middle">existing application</text>

                  <rect class="agent-app-node" x="56" y="180" width="152" height="56" rx="8" />
                  <text class="agent-app-node-title" x="132" y="206" text-anchor="middle">Coding Agent</text>
                  <text class="agent-app-node-sub" x="132" y="224" text-anchor="middle">existing application</text>

                  <rect class="agent-app-node" x="56" y="270" width="152" height="56" rx="8" />
                  <text class="agent-app-node-title" x="132" y="296" text-anchor="middle">Multimodal Agent</text>
                  <text class="agent-app-node-sub" x="132" y="314" text-anchor="middle">existing application</text>

                  <rect class="agent-app-node agent-app-api-node" x="376" y="96" width="168" height="64" rx="8" />
                  <text class="agent-app-node-title" x="460" y="124" text-anchor="middle">Chat Completions</text>
                  <text class="agent-app-node-sub" x="460" y="144" text-anchor="middle">one Relax endpoint</text>

                  <rect class="agent-app-node agent-app-state-node" x="376" y="188" width="168" height="56" rx="8" />
                  <text class="agent-app-node-title" x="460" y="214" text-anchor="middle">SessionForest</text>
                  <text class="agent-app-node-sub" x="460" y="232" text-anchor="middle">committed contexts</text>

                  <rect class="agent-app-node agent-app-train-node" x="376" y="284" width="168" height="56" rx="8" />
                  <text class="agent-app-node-title" x="460" y="310" text-anchor="middle">Training</text>
                  <text class="agent-app-node-sub" x="460" y="328" text-anchor="middle">selected contexts</text>

                  <text class="agent-app-callout" x="300" y="400" text-anchor="middle">Change the endpoint · keep the agent app</text>
                </svg>
              </div>

              <!-- SessionForest branches become selected training contexts. -->
              <div v-show="agenticTab === 'contexts'" class="agentic-visual agentic-contexts-view">
                <svg viewBox="0 0 600 420" role="img" aria-labelledby="agent-contexts-title agent-contexts-desc">
                  <title id="agent-contexts-title">Select training contexts from SessionForest</title>
                  <desc id="agent-contexts-desc">A SessionForest contains main, searcher, and alternative branches. Main and searcher are selected and sent to training while the alternative branch remains omitted.</desc>
                  <defs>
                    <marker id="context-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
                      <path d="M 0 0 L 10 5 L 0 10 z" fill="#7da2ff" />
                    </marker>
                  </defs>

                  <path class="context-tree-edge" d="M144 210 H164 Q172 210 172 202 V112 Q172 104 180 104 H208" />
                  <path class="context-tree-edge" d="M144 210 H208" />
                  <path class="context-tree-edge" d="M144 210 H164 Q172 210 172 218 V304 Q172 312 180 312 H208" />
                  <path class="context-export-edge" d="M352 104 H444" marker-end="url(#context-arrow)" />
                  <path class="context-export-edge" d="M352 210 H396 Q404 210 404 218 V244 Q404 252 412 252 H444" marker-end="url(#context-arrow)" />

                  <path class="context-export-active context-export-active-1" d="M352 104 H444" aria-hidden="true" />
                  <path class="context-export-active context-export-active-2" d="M352 210 H396 Q404 210 404 218 V244 Q404 252 412 252 H444" aria-hidden="true" />

                  <rect class="context-zone" x="24" y="40" width="360" height="332" rx="10" />
                  <text class="context-zone-label" x="44" y="68">SESSIONFOREST</text>
                  <rect class="context-zone context-training-zone" x="420" y="40" width="156" height="332" rx="10" />
                  <text class="context-zone-label" x="440" y="68">TRAINING CONTEXTS</text>

                  <rect class="context-node context-root-node" x="48" y="182" width="96" height="56" rx="8" />
                  <text class="context-node-title" x="96" y="208" text-anchor="middle">Session</text>
                  <text class="context-node-sub" x="96" y="226" text-anchor="middle">one process</text>

                  <rect class="context-node context-selected-node context-main-node" x="208" y="76" width="144" height="56" rx="8" />
                  <text class="context-node-title" x="280" y="102" text-anchor="middle">main</text>
                  <text class="context-node-sub" x="280" y="120" text-anchor="middle">selected</text>

                  <rect class="context-node context-selected-node context-searcher-node" x="208" y="182" width="144" height="56" rx="8" />
                  <text class="context-node-title" x="280" y="208" text-anchor="middle">searcher_0</text>
                  <text class="context-node-sub" x="280" y="226" text-anchor="middle">selected</text>

                  <rect class="context-node context-omitted-node" x="208" y="284" width="144" height="56" rx="8" />
                  <text class="context-node-title" x="280" y="310" text-anchor="middle">alternative</text>
                  <text class="context-node-sub" x="280" y="328" text-anchor="middle">omitted</text>

                  <rect class="context-tray-slot context-tray-main" x="444" y="84" width="108" height="72" rx="8" />
                  <text class="context-node-title" x="498" y="114" text-anchor="middle">main</text>
                  <text class="context-credit" x="498" y="138" text-anchor="middle">credit 1.0</text>

                  <rect class="context-tray-slot context-tray-searcher" x="444" y="216" width="108" height="72" rx="8" />
                  <text class="context-node-title" x="498" y="246" text-anchor="middle">searcher_0</text>
                  <text class="context-credit" x="498" y="270" text-anchor="middle">credit 0.6</text>

                  <text class="agent-app-callout" x="300" y="400" text-anchor="middle">Export selected contexts · omit the rest</text>
                </svg>
              </div>

              <!-- The Agent App sees one request and one response. -->
              <div v-show="agenticTab === 'request'" class="agentic-visual agentic-request-view">
                <svg viewBox="0 0 600 420" role="img" aria-labelledby="agent-request-title agent-request-desc">
                  <title id="agent-request-title">Agentic requests with and without a rollout step interruption</title>
                  <desc id="agent-request-desc">Without a rollout step interruption, an Agent App request passes through Relax to SGLang. In partial rollout or fully async mode, a step close makes Relax hold the request; the next step open resumes it toward SGLang. The close, hold, and resume remain invisible to the Agent App. Both responses return through Relax.</desc>
                  <defs>
                    <marker id="request-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
                      <path d="M 0 0 L 10 5 L 0 10 z" fill="#7da2ff" />
                    </marker>
                    <marker id="response-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
                      <path d="M 0 0 L 10 5 L 0 10 z" fill="#e8384a" />
                    </marker>
                  </defs>

                  <!-- Scenario 1: forward in the current rollout step. -->
                  <rect class="request-scenario request-scenario-now" x="28" y="44" width="544" height="146" rx="10" />
                  <text class="request-scenario-title" x="50" y="72">NO INTERRUPTION</text>
                  <text class="request-scenario-sub" x="50" y="90">request continues in the current rollout step</text>

                  <rect class="request-node request-agent-node" x="64" y="110" width="108" height="52" rx="8" />
                  <text class="request-node-title" x="118" y="134" text-anchor="middle">Agent App</text>
                  <text class="request-node-sub" x="118" y="151" text-anchor="middle">waiting</text>
                  <rect class="request-node request-relax-node" x="246" y="110" width="108" height="52" rx="8" />
                  <text class="request-node-title" x="300" y="142" text-anchor="middle">Relax</text>
                  <rect class="request-node request-sglang-node" x="428" y="110" width="108" height="52" rx="8" />
                  <text class="request-node-title" x="482" y="134" text-anchor="middle">SGLang</text>
                  <text class="request-node-sub" x="482" y="151" text-anchor="middle">generate</text>

                  <path class="request-flow" d="M172 126 H246" marker-end="url(#request-arrow)" />
                  <path class="request-flow" d="M354 126 H428" marker-end="url(#request-arrow)" />
                  <path class="request-response-flow" d="M428 148 H354" marker-end="url(#response-arrow)" />
                  <path class="request-response-flow" d="M246 148 H172" marker-end="url(#response-arrow)" />
                  <text class="request-path-label" x="187" y="118">REQUEST</text>
                  <text class="request-path-label request-response-label" x="187" y="166">RESPONSE</text>

                  <!-- Scenario 2: hold and continue in a later rollout step. -->
                  <rect class="request-scenario request-scenario-later" x="28" y="208" width="544" height="166" rx="10" />
                  <text class="request-scenario-title" x="50" y="236">STEP CLOSE → STEP OPEN</text>
                  <text class="request-scenario-sub" x="50" y="254">partial rollout · fully async</text>

                  <rect class="request-node request-agent-node" x="64" y="278" width="108" height="52" rx="8" />
                  <text class="request-node-title" x="118" y="302" text-anchor="middle">Agent App</text>
                  <text class="request-node-sub" x="118" y="319" text-anchor="middle">still waiting</text>
                  <rect class="request-node request-relax-node" x="244" y="278" width="108" height="52" rx="8" />
                  <text class="request-node-title" x="298" y="310" text-anchor="middle">Relax</text>
                  <rect class="request-hold-chip" x="370" y="282" width="44" height="24" rx="12" />
                  <text class="request-state-static" x="392" y="298" text-anchor="middle">HOLD→RESUME</text>
                  <text class="request-state-active request-hold-label" x="392" y="298" text-anchor="middle" aria-hidden="true">HOLD</text>
                  <text class="request-state-active request-resume-label" x="392" y="298" text-anchor="middle" aria-hidden="true">RESUME</text>
                  <text class="request-event-static" x="422" y="270" text-anchor="middle">STEP CLOSE → STEP OPEN</text>
                  <text class="request-event-active request-event-close" x="422" y="270" text-anchor="middle" aria-hidden="true">STEP CLOSE</text>
                  <text class="request-event-active request-event-open" x="422" y="270" text-anchor="middle" aria-hidden="true">STEP OPEN</text>
                  <rect class="request-node request-sglang-node" x="448" y="278" width="108" height="52" rx="8" />
                  <text class="request-node-title" x="502" y="302" text-anchor="middle">SGLang</text>
                  <text class="request-node-sub" x="502" y="319" text-anchor="middle">generate</text>

                  <path class="request-flow" d="M172 294 H244" marker-end="url(#request-arrow)" />
                  <path class="request-flow" d="M352 294 H370" marker-end="url(#request-arrow)" />
                  <path class="request-flow" d="M414 294 H448" marker-end="url(#request-arrow)" />
                  <path class="request-response-flow" d="M448 322 H352" marker-end="url(#response-arrow)" />
                  <path class="request-response-flow" d="M244 322 H172" marker-end="url(#response-arrow)" />
                  <text class="request-path-label" x="181" y="286">REQUEST</text>
                  <text class="request-path-label request-response-label" x="181" y="340">RESPONSE</text>

                  <!-- Decorative motion overlays; static meaning remains complete. -->
                  <path class="request-flow-active request-active-now-agent" d="M172 126 H246" marker-end="url(#request-arrow)" aria-hidden="true" />
                  <path class="request-flow-active request-active-now-engine" d="M354 126 H428" marker-end="url(#request-arrow)" aria-hidden="true" />
                  <path class="request-flow-active request-active-now-return-engine" d="M428 148 H354" marker-end="url(#response-arrow)" aria-hidden="true" />
                  <path class="request-flow-active request-active-now-return-agent" d="M246 148 H172" marker-end="url(#response-arrow)" aria-hidden="true" />
                  <path class="request-flow-active request-active-later-agent" d="M172 294 H244" marker-end="url(#request-arrow)" aria-hidden="true" />
                  <path class="request-flow-active request-active-later-hold" d="M352 294 H370" marker-end="url(#request-arrow)" aria-hidden="true" />
                  <path class="request-flow-active request-active-later-out" d="M414 294 H448" marker-end="url(#request-arrow)" aria-hidden="true" />
                  <path class="request-flow-active request-active-later-return-engine" d="M448 322 H352" marker-end="url(#response-arrow)" aria-hidden="true" />
                  <path class="request-flow-active request-active-later-return-agent" d="M244 322 H172" marker-end="url(#response-arrow)" aria-hidden="true" />

                  <text class="agent-app-callout" x="300" y="402" text-anchor="middle">The Agent App always sees one request and one response</text>
                </svg>
              </div>
            </div>

            <div class="agentic-tab-bar">
              <button class="agentic-tab-btn" :class="{ active: agenticTab === 'apps' }" @click="switchAgenticTab('apps')">agent apps</button>
              <button class="agentic-tab-btn" :class="{ active: agenticTab === 'contexts' }" @click="switchAgenticTab('contexts')">contexts</button>
              <button class="agentic-tab-btn" :class="{ active: agenticTab === 'request' }" @click="switchAgenticTab('request')">cross-step</button>
            </div>
          </div>

          <!-- Panel 1: Server-Based Architecture SVG -->
          <div class="display-panel active" :ref="(el) => setPanelRef(el, 0)" id="showcase-panel-2">
            <svg class="arch-svg" viewBox="0 0 600 420" preserveAspectRatio="xMidYMid meet">
              <defs>
                <marker id="arrow-blue" viewBox="0 0 10 10" refX="9" refY="5"
                  markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                  <path d="M 0 0 L 10 5 L 0 10 z" fill="#4AADFF" />
                </marker>
                <marker id="arrow-green" viewBox="0 0 10 10" refX="9" refY="5"
                  markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                  <path d="M 0 0 L 10 5 L 0 10 z" fill="#34D399" />
                </marker>
              </defs>

              <!-- ===== Layer 1: Orchestration (center=300) ===== -->
              <text class="layer-label" x="300" y="20">Orchestration Layer</text>
              <rect class="node-controller" x="180" y="30" width="240" height="40" rx="8" />
              <text class="node-text-main" x="300" y="54">Controller</text>

              <!-- ===== Layer 2: Components (5 nodes, symmetric around 300) ===== -->
              <text class="layer-label" x="300" y="100">Components Layer</text>
              <!-- x centers: 86, 192, 300, 408, 514  — symmetric pairs around 300 -->
              <rect class="node-service" x="42" y="110" width="88" height="36" rx="6" />
              <text class="node-text-sm" x="86" y="132">Actor</text>

              <rect class="node-service" x="148" y="110" width="88" height="36" rx="6" />
              <text class="node-text-sm" x="192" y="132">Rollout</text>

              <rect class="node-service" x="256" y="110" width="88" height="36" rx="6" />
              <text class="node-text-sm" x="300" y="132">Ref</text>

              <rect class="node-service" x="362" y="110" width="88" height="36" rx="6" />
              <text class="node-text-sm" x="406" y="132">GenRM</text>

              <rect class="node-service" x="468" y="110" width="88" height="36" rx="6" />
              <text class="node-text-sm" x="512" y="132">ActorFwd</text>

              <!-- ===== Layer 3: Engine & Backends (symmetric around 300) ===== -->
              <text class="layer-label" x="300" y="182">Engine &amp; Backends Layer</text>

              <!-- Transfer Queue (centered, above engines) -->
              <rect class="node-tq" x="170" y="195" width="260" height="32" rx="6" />
              <text class="node-text-sm tq-text" x="300" y="215">Transfer Queue</text>

              <!-- Megatron (left of center) -->
              <rect class="node-impl" x="110" y="248" width="160" height="40" rx="6" />
              <text class="node-text-sm" x="190" y="272">Megatron</text>

              <!-- SGLang (right of center, symmetric) -->
              <rect class="node-impl" x="330" y="248" width="160" height="40" rx="6" />
              <text class="node-text-sm" x="410" y="272">SGLang</text>

              <!-- Checkpoint Engine (centered, below engines) -->
              <rect class="node-ckpt" x="170" y="310" width="260" height="32" rx="6" />
              <text class="node-text-sm ckpt-text" x="300" y="330">Checkpoint Engine</text>

              <!-- ===== Data Flow (Blue): SGLang → TransferQueue → Megatron ===== -->
              <!-- SGLang top-right → up → TransferQueue right edge -->
              <path class="flow-data" d="M 450 248 L 450 211 L 430 211"
                marker-end="url(#arrow-blue)" />
              <!-- Transfer Queue left edge → down → Megatron top (symmetric with green) -->
              <path class="flow-data" d="M 170 211 L 150 211 L 150 248"
                marker-end="url(#arrow-blue)" />

              <!-- ===== Weight Flow (Green): Megatron → Checkpoint → SGLang ===== -->
              <!-- Megatron bottom-left → down → Checkpoint left edge -->
              <path class="flow-weight" d="M 150 288 L 150 326 L 170 326"
                marker-end="url(#arrow-green)" />
              <!-- Checkpoint right edge → up → SGLang bottom-right -->
              <path class="flow-weight" d="M 430 326 L 450 326 L 450 288"
                marker-end="url(#arrow-green)" />

              <!-- ===== Legend ===== -->
              <line x1="148" y1="385" x2="190" y2="385" class="flow-data" style="animation: none;" />
              <text class="legend-text" x="198" y="389">Data Flow</text>
              <line x1="328" y1="385" x2="370" y2="385" class="flow-weight" style="animation: none;" />
              <text class="legend-text" x="378" y="389">Weight Flow</text>
            </svg>
          </div>

          <!-- Panel 2: Fully Async Pipeline (Gantt-style, one complete pass) -->
          <div class="display-panel" :ref="(el) => setPanelRef(el, 1)" id="showcase-panel-3">
            <div class="pipeline-container">
              <div class="pipeline-group-label">Rollout</div>
              <div class="pipeline-lane"><span class="pl-label">GPU0</span><div class="pipe-row">
                <div class="pipe-block batch-a pa-r-b0" style="left:0%;width:27%;">B0</div>
                <div class="pipe-block batch-b pa-r-b1" style="left:28%;width:27%;">B1</div>
                <div class="pipe-block batch-c pa-r-b2" style="left:56%;width:43%;">B2</div>
              </div></div>
              <div class="pipeline-lane"><span class="pl-label">GPU1</span><div class="pipe-row">
                <div class="pipe-block batch-a pa-r-b0" style="left:0%;width:27%;">B0</div>
                <div class="pipe-block batch-b pa-r-b1" style="left:28%;width:27%;">B1</div>
                <div class="pipe-block batch-c pa-r-b2" style="left:56%;width:43%;">B2</div>
              </div></div>
              <div class="pipeline-lane"><span class="pl-label">GPU2</span><div class="pipe-row">
                <div class="pipe-block batch-a pa-r-b0" style="left:0%;width:27%;">B0</div>
                <div class="pipe-block batch-b pa-r-b1" style="left:28%;width:27%;">B1</div>
                <div class="pipe-block batch-c pa-r-b2" style="left:56%;width:43%;">B2</div>
              </div></div>
              <div class="pipeline-lane"><span class="pl-label">GPU3</span><div class="pipe-row">
                <div class="pipe-block batch-a pa-r-b0" style="left:0%;width:27%;">B0</div>
                <div class="pipe-block batch-b pa-r-b1" style="left:28%;width:27%;">B1</div>
                <div class="pipe-block batch-c pa-r-b2" style="left:56%;width:43%;">B2</div>
              </div></div>

              <div class="pipeline-separator"></div>
              <div class="pipeline-group-label">Reference</div>
              <div class="pipeline-lane"><span class="pl-label">GPU4</span><div class="pipe-row">
                <div class="pipe-block batch-a pa-ref-b0" style="left:28%;width:17%;">B0</div>
                <div class="pipe-block batch-b pa-ref-b1" style="left:56%;width:17%;">B1</div>
              </div></div>

              <div class="pipeline-separator"></div>
              <div class="pipeline-group-label">Forward</div>
              <div class="pipeline-lane"><span class="pl-label">GPU5</span><div class="pipe-row">
                <div class="pipe-block batch-a pa-fwd-b0" style="left:28%;width:17%;">B0</div>
                <div class="pipe-block batch-b pa-fwd-b1" style="left:56%;width:17%;">B1</div>
              </div></div>

              <div class="pipeline-separator"></div>
              <div class="pipeline-group-label">Train</div>
              <div class="pipeline-lane"><span class="pl-label">GPU6</span><div class="pipe-row">
                <div class="pipe-block batch-a pa-t-b0" style="left:46%;width:27%;">B0</div>
                <div class="pipe-block batch-b pa-t-b1" style="left:73%;width:27%;">B1</div>
              </div></div>
              <div class="pipeline-lane"><span class="pl-label">GPU7</span><div class="pipe-row">
                <div class="pipe-block batch-a pa-t-b0" style="left:46%;width:27%;">B0</div>
                <div class="pipe-block batch-b pa-t-b1" style="left:73%;width:27%;">B1</div>
              </div></div>
            </div>
          </div>

          <!-- Panel 4: Elastic Rollout Scaling -->
          <div class="display-panel" :ref="(el) => setPanelRef(el, 3)" id="showcase-panel-4">
            <svg class="elastic-svg" viewBox="0 0 600 440" preserveAspectRatio="xMidYMid meet">
              <defs>
                <marker id="arrow-cyan" viewBox="0 0 10 10" refX="9" refY="5"
                  markerWidth="5" markerHeight="5" orient="auto-start-reverse">
                  <path d="M 0 0 L 10 5 L 0 10 z" fill="#22d3ee" />
                </marker>
              </defs>

              <!-- ===== Section title ===== -->
              <text class="elastic-section-label" x="300" y="22">Elastic Rollout Scaling</text>

              <!-- ===== REST API trigger ===== -->
              <rect class="elastic-api-box" x="210" y="34" width="180" height="30" rx="6" />
              <text class="elastic-api-text" x="300" y="53">POST /scale_out</text>

              <!-- Arrow from API to Router -->
              <line x1="300" y1="64" x2="300" y2="86" stroke="#6B7280" stroke-width="1" stroke-dasharray="3 2" />

              <!-- ===== Router node ===== -->
              <rect class="elastic-router" x="220" y="88" width="160" height="36" rx="8" />
              <text class="elastic-router-text" x="300" y="110">SGLang Router</text>

              <!-- ===== Initial engines (always present, 4 engines) ===== -->
              <text class="elastic-group-label" x="155" y="152">Initial Engines</text>

              <!-- Engine 0 -->
              <rect class="elastic-engine elastic-engine-initial el-eng-0" x="50" y="162" width="100" height="56" rx="6" />
              <circle class="elastic-status-dot el-dot-active" cx="66" cy="176" r="4" />
              <text class="elastic-engine-id" x="80" y="179">Engine 0</text>
              <text class="elastic-engine-label" x="100" y="201">ACTIVE</text>

              <!-- Engine 1 -->
              <rect class="elastic-engine elastic-engine-initial el-eng-1" x="160" y="162" width="100" height="56" rx="6" />
              <circle class="elastic-status-dot el-dot-active" cx="176" cy="176" r="4" />
              <text class="elastic-engine-id" x="190" y="179">Engine 1</text>
              <text class="elastic-engine-label" x="210" y="201">ACTIVE</text>

              <!-- Engine 2 -->
              <rect class="elastic-engine elastic-engine-initial el-eng-2" x="270" y="162" width="100" height="56" rx="6" />
              <circle class="elastic-status-dot el-dot-active" cx="286" cy="176" r="4" />
              <text class="elastic-engine-id" x="300" y="179">Engine 2</text>
              <text class="elastic-engine-label" x="320" y="201">ACTIVE</text>

              <!-- Engine 3 -->
              <rect class="elastic-engine elastic-engine-initial el-eng-3" x="380" y="162" width="100" height="56" rx="6" />
              <circle class="elastic-status-dot el-dot-active" cx="396" cy="176" r="4" />
              <text class="elastic-engine-id" x="410" y="179">Engine 3</text>
              <text class="elastic-engine-label" x="430" y="201">ACTIVE</text>

              <!-- Lock icon on initial engines -->
              <text class="elastic-lock" x="136" y="207">&#128274;</text>
              <text class="elastic-lock" x="246" y="207">&#128274;</text>
              <text class="elastic-lock" x="356" y="207">&#128274;</text>
              <text class="elastic-lock" x="466" y="207">&#128274;</text>

              <!-- Router → Engine connection lines -->
              <line class="elastic-route-line" x1="260" y1="124" x2="100" y2="162" />
              <line class="elastic-route-line" x1="280" y1="124" x2="210" y2="162" />
              <line class="elastic-route-line" x1="320" y1="124" x2="320" y2="162" />
              <line class="elastic-route-line" x1="340" y1="124" x2="430" y2="162" />

              <!-- ===== Scaled-out engines (appear with animation) ===== -->
              <text class="elastic-group-label elastic-scaledout-label" x="155" y="248">Scaled-Out Engines</text>

              <!-- Engine 4 (scale-out, animated) -->
              <g class="elastic-scaleout-group el-scaleout-4">
                <rect class="elastic-engine elastic-engine-new" x="105" y="258" width="100" height="56" rx="6" />
                <circle class="elastic-status-dot el-dot-syncing" cx="121" cy="272" r="4" />
                <text class="elastic-engine-id" x="135" y="275">Engine 4</text>
                <text class="elastic-engine-label el-label-syncing" x="155" y="297">SYNCING</text>
              </g>

              <!-- Engine 5 (scale-out, animated) -->
              <g class="elastic-scaleout-group el-scaleout-5">
                <rect class="elastic-engine elastic-engine-new" x="215" y="258" width="100" height="56" rx="6" />
                <circle class="elastic-status-dot el-dot-syncing" cx="231" cy="272" r="4" />
                <text class="elastic-engine-id" x="245" y="275">Engine 5</text>
                <text class="elastic-engine-label el-label-syncing" x="265" y="297">SYNCING</text>
              </g>

              <!-- Engine 6 (external, animated) -->
              <g class="elastic-scaleout-group el-scaleout-6">
                <rect class="elastic-engine elastic-engine-ext" x="325" y="258" width="100" height="56" rx="6" />
                <circle class="elastic-status-dot el-dot-syncing" cx="341" cy="272" r="4" />
                <text class="elastic-engine-id" x="355" y="275">Engine 6</text>
                <text class="elastic-engine-label el-label-ext" x="375" y="297">EXTERNAL</text>
              </g>

              <!-- Router → Scaled-out engine lines (appear with scale-out) -->
              <line class="elastic-route-line-new el-route-new-4" x1="260" y1="124" x2="155" y2="258" />
              <line class="elastic-route-line-new el-route-new-5" x1="300" y1="124" x2="265" y2="258" />
              <line class="elastic-route-line-new el-route-new-6" x1="340" y1="124" x2="375" y2="258" />

              <!-- ===== P2P Weight Sync arrows ===== -->
              <g class="elastic-p2p-group">
                <path class="elastic-p2p-arrow" d="M 100 218 C 100 238, 155 238, 155 258" />
                <path class="elastic-p2p-arrow" d="M 210 218 C 210 238, 265 238, 265 258" />
                <path class="elastic-p2p-arrow" d="M 380 218 C 380 238, 375 238, 375 258" />
                <text class="elastic-p2p-label" x="470" y="240">P2P Direct Sync</text>
                <text class="elastic-p2p-label" x="470" y="254">/ OR /</text>
                <text class="elastic-p2p-label" x="470" y="268">NCCL Broadcast</text>
              </g>

              <!-- ===== State Machine flow at bottom ===== -->
              <g class="elastic-states">
                <text class="elastic-state-title" x="300" y="345">Scale-Out State Machine</text>

                <!-- State boxes -->
                <rect class="elastic-state-box el-state-1" x="20" y="356" width="70" height="24" rx="4" />
                <text class="elastic-state-text" x="55" y="372">PENDING</text>

                <rect class="elastic-state-box el-state-2" x="110" y="356" width="76" height="24" rx="4" />
                <text class="elastic-state-text" x="148" y="372">CREATING</text>

                <rect class="elastic-state-box el-state-3" x="206" y="356" width="88" height="24" rx="4" />
                <text class="elastic-state-text" x="250" y="372">HEALTH_CHK</text>

                <rect class="elastic-state-box el-state-4" x="314" y="356" width="78" height="24" rx="4" />
                <text class="elastic-state-text" x="353" y="372">WT_SYNC</text>

                <rect class="elastic-state-box el-state-5" x="412" y="356" width="60" height="24" rx="4" />
                <text class="elastic-state-text" x="442" y="372">READY</text>

                <rect class="elastic-state-box elastic-state-active el-state-6" x="492" y="356" width="68" height="24" rx="4" />
                <text class="elastic-state-text-active" x="526" y="372">ACTIVE</text>

                <!-- Arrows between states -->
                <line class="elastic-state-arrow el-arrow-1" x1="90" y1="368" x2="108" y2="368" />
                <line class="elastic-state-arrow el-arrow-2" x1="186" y1="368" x2="204" y2="368" />
                <line class="elastic-state-arrow el-arrow-3" x1="294" y1="368" x2="312" y2="368" />
                <line class="elastic-state-arrow el-arrow-4" x1="392" y1="368" x2="410" y2="368" />
                <line class="elastic-state-arrow el-arrow-5" x1="472" y1="368" x2="490" y2="368" />
              </g>

              <!-- ===== Legend ===== -->
              <rect class="elastic-engine-initial" x="30" y="402" width="14" height="10" rx="2" style="opacity:0.6" />
              <text class="elastic-legend-text" x="50" y="411">Initial (protected)</text>

              <rect class="elastic-engine-new" x="170" y="402" width="14" height="10" rx="2" style="opacity:0.6" />
              <text class="elastic-legend-text" x="190" y="411">Ray Native</text>

              <rect class="elastic-engine-ext" x="280" y="402" width="14" height="10" rx="2" style="opacity:0.6" />
              <text class="elastic-legend-text" x="300" y="411">External</text>

              <circle class="el-dot-active" cx="387" cy="407" r="4" />
              <text class="elastic-legend-text" x="396" y="411">Active</text>

              <circle class="el-dot-syncing" cx="452" cy="407" r="4" />
              <text class="elastic-legend-text" x="461" y="411">Syncing</text>

              <line x1="510" y1="407" x2="535" y2="407" class="elastic-p2p-arrow" style="animation:none;" />
              <text class="elastic-legend-text" x="542" y="411">Weight</text>
            </svg>
          </div>
        </section>
      </main>
    </div>
  </section>
</template>

<style scoped>
/* ============================================================
   Section wrapper
   ============================================================ */
.feature-showcase-section {
  position: relative;
  width: 100%;
  z-index: 2;
  overflow: visible;
}

.showcase-bg-solid {
  position: absolute;
  inset: -80px 0;          /* extend above & below to soften edges */
  background: linear-gradient(
    to bottom,
    rgba(14, 14, 17, 0)    0%,
    rgba(14, 14, 17, 0.6)  12%,
    rgba(14, 14, 17, 0.85) 30%,
    rgba(14, 14, 17, 0.85) 70%,
    rgba(14, 14, 17, 0.6)  88%,
    rgba(14, 14, 17, 0)    100%
  );
  opacity: 0;
  transition: opacity 0.8s cubic-bezier(0.4, 0, 0.2, 1);
  z-index: 0;
  pointer-events: none;
}

.feature-showcase-section.is-visible .showcase-bg-solid {
  opacity: 1;
}

.showcase-inner {
  position: relative;
  z-index: 1;
  max-width: 1400px;
  margin: 0 auto;
  padding: 80px 60px;
  opacity: 0;
  transform: translateY(40px);
  transition: opacity 0.8s cubic-bezier(0.4, 0, 0.2, 1) 0.15s,
              transform 0.8s cubic-bezier(0.4, 0, 0.2, 1) 0.15s;
}

.feature-showcase-section.is-visible .showcase-inner {
  opacity: 1;
  transform: translateY(0);
}

/* ============================================================
   Main layout — 2-column grid
   ============================================================ */
.showcase-main-content {
  display: grid;
  grid-template-columns: 5fr 6fr;
  gap: 48px;
  align-items: stretch;
  width: 100%;
}

@media (max-width: 960px) {
  .showcase-main-content {
    grid-template-columns: 1fr;
    gap: 40px;
  }
  .showcase-inner {
    padding: 60px 24px;
  }
}

/* ============================================================
   Left: text & tab controls
   ============================================================ */
.showcase-text-section {
  display: flex;
  flex-direction: column;
  justify-content: center;
}

.showcase-title {
  font-family: "Space Grotesk", "Manrope", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-size: 2.5rem;
  font-weight: 700;
  line-height: 1.2;
  letter-spacing: -1px;
  margin-bottom: 40px;
  color: #ffffff;
}

.showcase-title-brand {
  color: #e8384a;
}

.showcase-feature-list {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.showcase-feature-item {
  padding: 24px;
  border-radius: 12px;
  cursor: pointer;
  transition: background-color 0.2s, border-color 0.2s;
  position: relative;
  border: 1px solid transparent;
}

.showcase-feature-item:hover {
  background-color: rgba(255, 255, 255, 0.04);
  border-color: rgba(255, 255, 255, 0.06);
}

.showcase-feature-item.active {
  background-color: rgba(255, 255, 255, 0.06);
  border-color: rgba(255, 255, 255, 0.08);
}

/* Progress bar */
.progress-container {
  width: 100%;
  height: 4px;
  background-color: rgba(255, 255, 255, 0.1);
  border-radius: 2px;
  margin-bottom: 20px;
  overflow: hidden;
  display: none;
}

.showcase-feature-item.active .progress-container {
  display: block;
}

.progress-bar-fill {
  height: 100%;
  background-color: #e8384a;
  width: 0%;
}

.feature-header {
  display: flex;
  align-items: center;
  gap: 16px;
}

.step-number {
  width: 32px;
  height: 32px;
  background-color: rgba(255, 255, 255, 0.08);
  color: #9CA3AF;
  display: flex;
  justify-content: center;
  align-items: center;
  border-radius: 6px;
  font-weight: 500;
  flex-shrink: 0;
}

.showcase-feature-item.active .step-number {
  color: #e8384a;
}

.feature-title {
  font-size: 1.25rem;
  font-weight: 500;
  color: #9CA3AF;
}

.showcase-feature-item.active .feature-title {
  color: #fff;
}

.feature-desc {
  margin-top: 12px;
  color: #9CA3AF;
  line-height: 1.6;
  display: none;
  padding-left: 48px;
}

.showcase-feature-item.active .feature-desc {
  display: block;
}

/* ============================================================
   Right: display panels (macOS-style window)
   ============================================================ */
.showcase-display-section {
  position: relative;
  border-radius: 12px;
  background-color: rgba(255, 255, 255, 0.04);
  border: 1px solid rgba(255, 255, 255, 0.08);
  box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.4);
  overflow: hidden;
  display: flex;
  flex-direction: column;
  min-height: 520px;
  height: 100%;
}

.window-header {
  height: 38px;
  background-color: rgba(255, 255, 255, 0.06);
  display: flex;
  align-items: center;
  padding: 0 16px;
  gap: 8px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.05);
  flex-shrink: 0;
  z-index: 10;
}

.window-title {
  color: #666;
  font-size: 12px;
  margin-left: auto;
  margin-right: auto;
  font-family: "Manrope", -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

.mac-btn { width: 12px; height: 12px; border-radius: 50%; }
.close    { background-color: #FF5F56; }
.minimize { background-color: #FFBD2E; }
.maximize { background-color: #27C93F; }

/* Panels */
.display-panel {
  position: absolute;
  top: 38px;
  left: 0;
  right: 0;
  bottom: 0;
  opacity: 0;
  visibility: hidden;
  transition: opacity 0.4s ease;
  overflow: hidden;
}

.display-panel.active {
  opacity: 1;
  visibility: visible;
  z-index: 5;
}

/* ============================================================
   Panel 1: Agentic visual showcase
   ============================================================ */
#showcase-panel-1 {
  background: linear-gradient(135deg, #131316 0%, #1f1f23 100%);
  display: flex;
  flex-direction: column;
}

.agentic-visual-stage {
  flex: 1;
  min-height: 0;
  padding: 8px 12px 0;
}

.agentic-visual {
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
}

.agentic-visual svg {
  width: 100%;
  height: 100%;
}

.agent-app-zone,
.context-zone {
  fill: rgba(255, 255, 255, 0.025);
  stroke: rgba(255, 255, 255, 0.14);
  stroke-width: 1;
}

.agent-app-relax-zone,
.context-training-zone {
  fill: rgba(232, 56, 74, 0.035);
  stroke: rgba(232, 56, 74, 0.38);
}

.agent-app-zone-label,
.context-zone-label {
  fill: #8b8c93;
  font-family: "Manrope", sans-serif;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 1.4px;
}

.agent-app-node,
.context-node {
  fill: #1f1f23;
  stroke: rgba(255, 255, 255, 0.22);
  stroke-width: 1;
}

.agent-app-api-node,
.context-selected-node {
  fill: rgba(232, 56, 74, 0.12);
  stroke: #e8384a;
  stroke-width: 1.4;
}

.agent-app-node-title,
.context-node-title {
  fill: #f0edf1;
  font-family: "Manrope", sans-serif;
  font-size: 14px;
  font-weight: 600;
}

.agent-app-node-sub,
.context-node-sub {
  fill: #8b8c93;
  font-family: "Manrope", sans-serif;
  font-size: 10px;
}

.agent-app-flow,
.agent-app-internal-flow {
  fill: none;
  stroke: #7da2ff;
  stroke-width: 1.4;
}

.agent-app-internal-flow {
  stroke: #9ca3af;
}

.agent-app-flow-active,
.agent-app-internal-active,
.context-export-active {
  fill: none;
  stroke: #7da2ff;
  stroke-width: 2.4;
  stroke-linecap: round;
  opacity: 0;
}

.agent-app-internal-active {
  stroke: #e8384a;
}

.agent-app-callout {
  fill: #b7b5bb;
  font-family: "Manrope", sans-serif;
  font-size: 12px;
  font-weight: 500;
}

.agent-app-state-node,
.context-root-node {
  fill: rgba(255, 255, 255, 0.045);
}

.agent-app-train-node {
  fill: rgba(125, 162, 255, 0.08);
  stroke: rgba(125, 162, 255, 0.65);
}

.context-tree-edge,
.context-export-edge {
  fill: none;
  stroke: #6f7179;
  stroke-width: 1.2;
}

.context-export-edge {
  stroke: #7da2ff;
}

.context-omitted-node {
  fill: rgba(255, 255, 255, 0.015);
  stroke: rgba(255, 255, 255, 0.24);
  stroke-dasharray: 5 4;
  opacity: 0.55;
}

.context-tray-slot {
  fill: rgba(125, 162, 255, 0.08);
  stroke: rgba(125, 162, 255, 0.65);
  stroke-width: 1.2;
}

.context-credit {
  fill: #8b8c93;
  font-family: "Manrope", sans-serif;
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.6px;
}

.request-scenario {
  fill: rgba(255, 255, 255, 0.018);
  stroke: rgba(255, 255, 255, 0.13);
}

.request-scenario-later {
  fill: rgba(232, 56, 74, 0.025);
  stroke: rgba(232, 56, 74, 0.28);
}

.request-scenario-title,
.request-scenario-sub,
.request-path-label,
.request-event-static,
.request-event-active {
  fill: #8b8c93;
  font-family: "Manrope", sans-serif;
  font-size: 9px;
  font-weight: 700;
  letter-spacing: 1px;
}

.request-scenario-title {
  fill: #d2d5dc;
}

.request-scenario-later + .request-scenario-title {
  fill: #e98d97;
}

.request-scenario-sub {
  font-size: 9px;
  font-weight: 500;
  letter-spacing: 0.4px;
}

.request-node {
  fill: rgba(255, 255, 255, 0.025);
  stroke: rgba(255, 255, 255, 0.22);
  stroke-width: 1;
}

.request-relax-node {
  fill: rgba(232, 56, 74, 0.055);
  stroke: rgba(232, 56, 74, 0.48);
}

.request-sglang-node {
  fill: rgba(125, 162, 255, 0.07);
  stroke: rgba(125, 162, 255, 0.55);
}

.request-node-title {
  fill: #f0edf1;
  font-family: "Manrope", sans-serif;
  font-size: 12px;
  font-weight: 600;
}

.request-node-sub {
  fill: #8b8c93;
  font-family: "Manrope", sans-serif;
  font-size: 9px;
}

.request-event-static,
.request-event-active {
  fill: #e98d97;
  font-size: 8px;
}

.request-event-active {
  opacity: 0;
}

.request-hold-chip {
  fill: rgba(232, 56, 74, 0.1);
  stroke: rgba(232, 56, 74, 0.62);
}

.request-hold-label,
.request-resume-label,
.request-state-static {
  fill: #e98d97;
  font-family: "Manrope", sans-serif;
  font-size: 8px;
  font-weight: 700;
  letter-spacing: 0.8px;
}

.request-state-static {
  font-size: 6px;
  letter-spacing: 0;
}

.request-state-active {
  opacity: 0;
}

.showcase-display-section.panel-active-anim #showcase-panel-1 .agentic-request-view .request-event-static,
.showcase-display-section.panel-active-anim #showcase-panel-1 .agentic-request-view .request-state-static {
  opacity: 0;
}

.request-flow,
.request-response-flow {
  fill: none;
  stroke: #7da2ff;
  stroke-width: 1.4;
  stroke-dasharray: 5 4;
}

.request-response-flow {
  stroke: #e8384a;
}

.request-response-label {
  fill: #e98d97;
}

.request-flow-active {
  fill: none;
  stroke: #7da2ff;
  stroke-width: 1.4;
  stroke-dasharray: 5 4;
  stroke-linecap: round;
  opacity: 0;
}

.request-active-now-return-engine,
.request-active-now-return-agent,
.request-active-later-return-engine,
.request-active-later-return-agent {
  stroke: #e8384a;
}

/* Bottom Agentic visual tab bar */
.agentic-tab-bar {
  display: flex;
  justify-content: center;
  gap: 12px;
  padding: 12px 16px;
  border-top: 1px solid rgba(255, 255, 255, 0.06);
  background: rgba(255, 255, 255, 0.02);
  flex-shrink: 0;
}

.agentic-tab-btn {
  background: rgba(255, 255, 255, 0.06);
  border: 1px solid rgba(255, 255, 255, 0.08);
  color: #888;
  padding: 6px 20px;
  border-radius: 20px;
  font-size: 13px;
  font-family: "Manrope", -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  cursor: pointer;
  transition: all 0.2s ease;
}

.agentic-tab-btn:hover {
  background: rgba(255, 255, 255, 0.1);
  color: #ccc;
}

.agentic-tab-btn.active {
  background: #e8384a;
  color: #ffffff;
  font-weight: 500;
  border-color: transparent;
}

/* ============================================================
   Panel 2: Architecture SVG
   ============================================================ */
#showcase-panel-2 {
  background-color: #1a1a1a;
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 12px;
}

.arch-svg { width: 100%; height: 100%; }

.layer-label {
  fill: #6B7280;
  font-family: "Manrope", -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  font-size: 10px;
  font-weight: 600;
  text-anchor: middle;
  text-transform: uppercase;
  letter-spacing: 1.5px;
}

.node-controller {
  fill: #1e293b;
  stroke: #3b82f6;
  stroke-width: 1.5;
}

.node-text-main {
  fill: #e2e8f0;
  font-family: "Manrope", -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  font-size: 15px;
  font-weight: 600;
  text-anchor: middle;
  dominant-baseline: middle;
}

.node-service {
  fill: #1e293b;
  stroke: #64748b;
  stroke-width: 1;
}

.node-text-sm {
  fill: #cbd5e1;
  font-family: "Manrope", -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  font-size: 12px;
  font-weight: 500;
  text-anchor: middle;
  dominant-baseline: middle;
}

.node-impl {
  fill: #172030;
  stroke: #475569;
  stroke-width: 1;
  stroke-dasharray: 4 2;
}

.node-tq {
  fill: rgba(74, 173, 255, 0.08);
  stroke: #4AADFF;
  stroke-width: 1;
}

.tq-text { fill: #7dc4ff; }

.node-ckpt {
  fill: rgba(52, 211, 153, 0.08);
  stroke: #34D399;
  stroke-width: 1;
}

.ckpt-text { fill: #6ee7b7; }

.flow-data {
  fill: none;
  stroke: #4AADFF;
  stroke-width: 2;
  stroke-dasharray: 6 4;
  animation: flowDataAnim 1.2s linear infinite;
}

.flow-weight {
  fill: none;
  stroke: #34D399;
  stroke-width: 2;
  stroke-dasharray: 6 4;
  animation: flowWeightAnim 1.2s linear infinite;
}

.legend-text {
  fill: #9CA3AF;
  font-family: "Manrope", -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  font-size: 11px;
  dominant-baseline: middle;
}

@keyframes flowDataAnim {
  from { stroke-dashoffset: 20; }
  to   { stroke-dashoffset: 0; }
}

@keyframes flowWeightAnim {
  from { stroke-dashoffset: 0; }
  to   { stroke-dashoffset: 20; }
}

/* ============================================================
   Panel 3: Fully Async Pipeline (Gantt-style)
   ============================================================ */
#showcase-panel-3 {
  background-color: #1a1a1a;
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 20px 24px;
}

.pipeline-container {
  width: 100%;
  display: flex;
  flex-direction: column;
  gap: 3px;
}

.pipeline-group-label {
  color: #6B7280;
  font-size: 10px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 1.2px;
  margin-top: 2px;
  margin-bottom: 1px;
  padding-left: 50px;
}

.pipeline-separator {
  height: 1px;
  background: rgba(255, 255, 255, 0.06);
  margin: 4px 0;
}

.pipeline-lane {
  display: flex;
  align-items: center;
  height: 28px;
}

.pl-label {
  width: 44px;
  flex-shrink: 0;
  color: #6B7280;
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  font-size: 10px;
  font-weight: 500;
  text-align: right;
  padding-right: 6px;
}

.pipe-row {
  flex: 1;
  height: 20px;
  background: rgba(255, 255, 255, 0.04);
  border-radius: 4px;
  overflow: hidden;
  position: relative;
  border: 1px solid rgba(255, 255, 255, 0.06);
}

.pipe-block {
  position: absolute;
  top: 2px;
  bottom: 2px;
  border-radius: 3px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 9px;
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  font-weight: 600;
  color: rgba(255, 255, 255, 0.92);
  letter-spacing: 0.3px;
  overflow: hidden;
}

/* Flat batch colors — blue / teal-emerald / violet */
.batch-a {
  background: #3b82f6;
  box-shadow: 0 0 8px rgba(59, 130, 246, 0.25);
}
.batch-b {
  background: #14b8a6;
  box-shadow: 0 0 8px rgba(20, 184, 166, 0.25);
}
.batch-c {
  background: #8b5cf6;
  box-shadow: 0 0 8px rgba(139, 92, 246, 0.25);
}

/* ============================================================
   Panel 4: Elastic Rollout Scaling SVG
   ============================================================ */
#showcase-panel-4 {
  background-color: #1a1a1a;
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 8px 12px;
}

.elastic-svg { width: 100%; height: 100%; }

.elastic-section-label {
  fill: #6B7280;
  font-family: "Manrope", -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  font-size: 10px;
  font-weight: 600;
  text-anchor: middle;
  text-transform: uppercase;
  letter-spacing: 1.5px;
}

.elastic-api-box {
  fill: rgba(139, 92, 246, 0.12);
  stroke: #8b5cf6;
  stroke-width: 1;
  stroke-dasharray: 4 2;
}

.elastic-api-text {
  fill: #a78bfa;
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  font-size: 12px;
  font-weight: 600;
  text-anchor: middle;
  dominant-baseline: middle;
}

.elastic-router {
  fill: #1e293b;
  stroke: #22d3ee;
  stroke-width: 1.5;
}

.elastic-router-text {
  fill: #67e8f9;
  font-family: "Manrope", -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  font-size: 13px;
  font-weight: 600;
  text-anchor: middle;
  dominant-baseline: middle;
}

.elastic-group-label {
  fill: #6B7280;
  font-family: "Manrope", -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  font-size: 9px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 1.2px;
}

.elastic-engine {
  stroke-width: 1;
}

.elastic-engine-initial {
  fill: #1e293b;
  stroke: #3b82f6;
}

.elastic-engine-new {
  fill: rgba(16, 185, 129, 0.1);
  stroke: #10b981;
}

.elastic-engine-ext {
  fill: rgba(251, 191, 36, 0.1);
  stroke: #fbbf24;
}

.elastic-status-dot {
  /* base — overridden by specific classes */
}

.el-dot-active {
  fill: #22c55e;
}

.el-dot-syncing {
  fill: #fbbf24;
}

.elastic-engine-id {
  fill: #e2e8f0;
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  font-size: 10px;
  font-weight: 600;
  dominant-baseline: middle;
}

.elastic-engine-label {
  fill: #22c55e;
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  font-size: 8px;
  font-weight: 500;
  text-anchor: middle;
  dominant-baseline: middle;
  text-transform: uppercase;
  letter-spacing: 0.8px;
}

.el-label-syncing {
  fill: #fbbf24;
}

.el-label-ext {
  fill: #fbbf24;
}

.elastic-lock {
  font-size: 9px;
  dominant-baseline: middle;
}

.elastic-route-line {
  stroke: rgba(34, 211, 238, 0.25);
  stroke-width: 1;
}

.elastic-route-line-new {
  stroke: rgba(16, 185, 129, 0.3);
  stroke-width: 1;
  stroke-dasharray: 4 3;
}

.elastic-scaleout-group {
  opacity: 0;
}

.elastic-scaledout-label {
  opacity: 0;
}

.elastic-p2p-group {
  opacity: 0;
}

.elastic-p2p-arrow {
  fill: none;
  stroke: #f97316;
  stroke-width: 1.5;
  stroke-dasharray: 4 3;
}

.elastic-p2p-label {
  fill: #fb923c;
  font-family: "Manrope", -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  font-size: 9px;
  font-weight: 500;
}

.elastic-state-title {
  fill: #6B7280;
  font-family: "Manrope", -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  font-size: 9px;
  font-weight: 600;
  text-anchor: middle;
  text-transform: uppercase;
  letter-spacing: 1.2px;
}

.elastic-state-box {
  fill: rgba(255, 255, 255, 0.04);
  stroke: #4B5563;
  stroke-width: 1;
}

.elastic-state-active {
  fill: rgba(34, 197, 94, 0.15);
  stroke: #22c55e;
}

.elastic-state-text {
  fill: #9CA3AF;
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  font-size: 8px;
  font-weight: 500;
  text-anchor: middle;
  dominant-baseline: middle;
}

.elastic-state-text-active {
  fill: #22c55e;
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  font-size: 8px;
  font-weight: 700;
  text-anchor: middle;
  dominant-baseline: middle;
}

.elastic-state-arrow {
  stroke: #4B5563;
  stroke-width: 1;
  marker-end: url(#arrow-cyan);
}

.elastic-legend-text {
  fill: #9CA3AF;
  font-family: "Manrope", -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  font-size: 9px;
  dominant-baseline: middle;
}
</style>

<!-- Global (unscoped) styles for pipeline entrance animation -->
<style>
/* CSS variable for pause-on-hover — toggled by JS via inline style */
.showcase-display-section {
  --panel-play-state: running;
}

/* Pause ALL descendant animations when hovered */
.showcase-display-section[style*="paused"] *,
.showcase-display-section[style*="paused"] *::after,
.showcase-display-section[style*="paused"] *::before {
  animation-play-state: paused !important;
}

/* Agentic visual tabs: complete static diagrams with decorative motion overlays. */
.showcase-display-section.panel-active-anim #showcase-panel-1 .agent-apps-view .agent-app-flow-active,
.showcase-display-section.panel-active-anim #showcase-panel-1 .agent-apps-view .agent-app-internal-active,
.showcase-display-section.panel-active-anim #showcase-panel-1 .agentic-contexts-view .context-export-active {
  stroke-dasharray: 18 260;
  stroke-dashoffset: 278;
  animation: agenticPathFlow 4.8s linear infinite;
}

.showcase-display-section.panel-active-anim #showcase-panel-1 .agent-apps-view .agent-app-flow-active-2 {
  animation-delay: 1.2s;
}

.showcase-display-section.panel-active-anim #showcase-panel-1 .agent-apps-view .agent-app-flow-active-3 {
  animation-delay: 2.4s;
}

.showcase-display-section.panel-active-anim #showcase-panel-1 .agent-apps-view .agent-app-flow-active-4 {
  animation-delay: 1.8s;
}

.showcase-display-section.panel-active-anim #showcase-panel-1 .agent-apps-view .agent-app-flow-active-5 {
  animation-delay: 2.8s;
}

.showcase-display-section.panel-active-anim #showcase-panel-1 .agent-apps-view .agent-app-api-node,
.showcase-display-section.panel-active-anim #showcase-panel-1 .agentic-contexts-view .context-selected-node,
.showcase-display-section.panel-active-anim #showcase-panel-1 .agentic-contexts-view .context-tray-slot {
  animation: agenticNodeFocus 4.8s ease-in-out infinite;
}

.showcase-display-section.panel-active-anim #showcase-panel-1 .agentic-contexts-view .context-searcher-node,
.showcase-display-section.panel-active-anim #showcase-panel-1 .agentic-contexts-view .context-tray-searcher {
  animation-delay: 1.2s;
}

.showcase-display-section.panel-active-anim #showcase-panel-1 .agentic-contexts-view .context-export-active-2 {
  animation-delay: 1.2s;
}

.showcase-display-section.panel-active-anim #showcase-panel-1 .agentic-request-view .request-flow,
.showcase-display-section.panel-active-anim #showcase-panel-1 .agentic-request-view .request-response-flow {
  opacity: 0;
}

.showcase-display-section.panel-active-anim #showcase-panel-1 .agentic-request-view .request-flow-active {
  animation-name: var(--request-reveal), requestDashFlow;
  animation-duration: 7.6s, 0.8s;
  animation-timing-function: linear, linear;
  animation-iteration-count: infinite, infinite;
  animation-fill-mode: both, none;
}

.request-active-now-agent { --request-reveal: requestReveal06; }
.request-active-now-engine { --request-reveal: requestReveal14; }
.request-active-now-return-engine { --request-reveal: requestReveal22; }
.request-active-now-return-agent { --request-reveal: requestReveal30; }
.request-active-later-agent { --request-reveal: requestReveal38; }
.request-active-later-hold { --request-reveal: requestReveal46; }
.request-active-later-out { --request-reveal: requestReveal62; }
.request-active-later-return-engine { --request-reveal: requestReveal74; }
.request-active-later-return-agent { --request-reveal: requestReveal82; }

@keyframes requestDashFlow {
  from { stroke-dashoffset: 18; }
  to { stroke-dashoffset: 0; }
}

.showcase-display-section.panel-active-anim #showcase-panel-1 .agentic-request-view .request-event-close {
  animation: requestCloseState 7.6s linear infinite both;
}

.showcase-display-section.panel-active-anim #showcase-panel-1 .agentic-request-view .request-hold-chip {
  animation: requestReveal50 7.6s linear infinite both;
}

.showcase-display-section.panel-active-anim #showcase-panel-1 .agentic-request-view .request-hold-label {
  animation: requestHoldState 7.6s linear infinite both;
}

.showcase-display-section.panel-active-anim #showcase-panel-1 .agentic-request-view .request-event-open,
.showcase-display-section.panel-active-anim #showcase-panel-1 .agentic-request-view .request-resume-label {
  animation: requestReveal62 7.6s linear infinite both;
}

@keyframes agenticPathFlow {
  0% {
    opacity: 0;
    stroke-dashoffset: 278;
  }
  12%, 78% {
    opacity: 1;
  }
  100% {
    opacity: 0;
    stroke-dashoffset: 0;
  }
}

@keyframes agenticNodeFocus {
  0%, 100% {
    stroke-width: 1.4;
  }
  40%, 65% {
    stroke-width: 2.4;
  }
}

@keyframes requestReveal06 { 0%, 5% { opacity: 0; } 6%, 96% { opacity: 1; } 98%, 100% { opacity: 0; } }
@keyframes requestReveal14 { 0%, 13% { opacity: 0; } 14%, 96% { opacity: 1; } 98%, 100% { opacity: 0; } }
@keyframes requestReveal22 { 0%, 21% { opacity: 0; } 22%, 96% { opacity: 1; } 98%, 100% { opacity: 0; } }
@keyframes requestReveal30 { 0%, 29% { opacity: 0; } 30%, 96% { opacity: 1; } 98%, 100% { opacity: 0; } }
@keyframes requestReveal38 { 0%, 37% { opacity: 0; } 38%, 96% { opacity: 1; } 98%, 100% { opacity: 0; } }
@keyframes requestReveal46 { 0%, 45% { opacity: 0; } 46%, 96% { opacity: 1; } 98%, 100% { opacity: 0; } }
@keyframes requestReveal50 { 0%, 49% { opacity: 0; } 50%, 96% { opacity: 1; } 98%, 100% { opacity: 0; } }
@keyframes requestReveal62 { 0%, 61% { opacity: 0; } 62%, 96% { opacity: 1; } 98%, 100% { opacity: 0; } }
@keyframes requestReveal74 { 0%, 73% { opacity: 0; } 74%, 96% { opacity: 1; } 98%, 100% { opacity: 0; } }
@keyframes requestReveal82 { 0%, 81% { opacity: 0; } 82%, 96% { opacity: 1; } 98%, 100% { opacity: 0; } }

@keyframes requestCloseState {
  0%, 45% { opacity: 0; }
  46%, 61% { opacity: 1; }
  62%, 100% { opacity: 0; }
}

@keyframes requestHoldState {
  0%, 49% { opacity: 0; }
  50%, 61% { opacity: 1; }
  62%, 100% { opacity: 0; }
}

@media (prefers-reduced-motion: reduce) {
  #showcase-panel-1 *,
  #showcase-panel-1 *::before,
  #showcase-panel-1 *::after {
    animation: none !important;
    transition: none !important;
  }

  #showcase-panel-1 .agent-app-flow-active,
  #showcase-panel-1 .agent-app-internal-active,
  #showcase-panel-1 .context-export-active,
  #showcase-panel-1 .request-flow-active,
  #showcase-panel-1 .request-event-active,
  #showcase-panel-1 .request-state-active {
    display: none;
  }

  #showcase-panel-1 .request-event-static,
  #showcase-panel-1 .request-state-static {
    opacity: 1 !important;
  }

  #showcase-panel-1 .request-hold-chip {
    opacity: 1 !important;
  }

  #showcase-panel-1 .request-flow,
  #showcase-panel-1 .request-response-flow {
    opacity: 1;
  }
}

/*
 * Pipeline animation: slow liquid fill effect (~10s total)
 *
 * Each batch flows CONTINUOUSLY through stages:
 *   B0: Rollout(0.3→2.6) → Ref(2.6→4.9) → Fwd(2.6→4.9) → Train(5.2→7.5)
 *   B1: Rollout(2.6→4.9) → Ref(4.9→7.2) → Fwd(5.5→7.8) → Train(7.8→10.1)
 *   B2: Rollout(4.9→9.8)
 *
 * Within a stage, same-batch blocks across all lanes start SIMULTANEOUSLY.
 */

.showcase-display-section.panel-active-anim #showcase-panel-3 .pipe-block {
  animation: pipeFill 2.3s linear both;
}

/* ---- B0 continuous flow: Rollout → Ref → Fwd → Train ---- */
.showcase-display-section.panel-active-anim #showcase-panel-3 .pa-r-b0   { animation-delay: 0.3s; }
.showcase-display-section.panel-active-anim #showcase-panel-3 .pa-ref-b0 { animation-delay: 2.6s; }
.showcase-display-section.panel-active-anim #showcase-panel-3 .pa-fwd-b0 { animation-delay: 2.6s; }
.showcase-display-section.panel-active-anim #showcase-panel-3 .pa-t-b0   { animation-delay: 5.2s; }

/* ---- B1 continuous flow: Rollout → Ref → Fwd → Train ---- */
.showcase-display-section.panel-active-anim #showcase-panel-3 .pa-r-b1   { animation-delay: 2.6s; }
.showcase-display-section.panel-active-anim #showcase-panel-3 .pa-ref-b1 { animation-delay: 5.5s; }
.showcase-display-section.panel-active-anim #showcase-panel-3 .pa-fwd-b1 { animation-delay: 5.5s; }
.showcase-display-section.panel-active-anim #showcase-panel-3 .pa-t-b1   { animation-delay: 7.8s; }

/* ---- B2 continuous flow: Rollout → Ref → Fwd ---- */
/* All last blocks must finish at the same time (9.8s).
   duration = 9.8 - delay  so right edges arrive together. */
.showcase-display-section.panel-active-anim #showcase-panel-3 .pa-r-b2   { animation-delay: 4.9s; animation-duration: 4.9s; }

/* Slow liquid fill: water gradually fills the pipe from left to right */
@keyframes pipeFill {
  0% {
    opacity: 0.4;
    clip-path: inset(0 100% 0 0);
  }
  5% {
    opacity: 1;
  }
  100% {
    opacity: 1;
    clip-path: inset(0 0% 0 0);
  }
}

/* Subtle shimmer — gentle light sweep for flat style */
.showcase-display-section.panel-active-anim #showcase-panel-3 .pipe-block::after {
  content: '';
  position: absolute;
  top: 0;
  left: -80%;
  width: 40%;
  height: 100%;
  background: linear-gradient(
    90deg,
    transparent 0%,
    rgba(255, 255, 255, 0.1) 50%,
    transparent 100%
  );
  border-radius: inherit;
  animation: pipeShimmer 4s ease-in-out infinite;
  animation-delay: 4s;
  z-index: 2;
}

@keyframes pipeShimmer {
  0%   { left: -40%; }
  50%  { left: 110%; }
  100% { left: 110%; }
}

/* Syntax token colors — relax-web palette */
#showcase-panel-1 .ck { color: #bda2ff !important; font-weight: 700 !important; }  /* keyword: tertiary (import, class, def, return, if) */
#showcase-panel-1 .cc { color: #e8384a !important; font-weight: 700 !important; }  /* class/type name: primary */
#showcase-panel-1 .cf { color: #f0edf1 !important; }                                /* function name: on-surface */
#showcase-panel-1 .cp { color: #acaaae !important; }                                 /* parameter: on-surface-variant */
#showcase-panel-1 .cs { color: #acaaae !important; }                                 /* string/comment: on-surface-variant */
#showcase-panel-1 .cn { color: #d4323e !important; }                                 /* number: primary-container */
#showcase-panel-1 .cm { color: #6b6672 !important; font-style: italic !important; }   /* comment: muted italic */

/* ============================================================
   Panel 4: Elastic Rollout — entrance animations (global)
   ============================================================

   Timeline:
   0.0s – Panel appears (initial engines visible by default)
   0.5s – "Scaled-Out Engines" label fades in
   1.0s – Engine 4 scales in
   1.4s – Engine 5 scales in
   1.8s – Engine 6 (external) scales in
   2.2s – Route lines to new engines appear
   2.8s – P2P weight sync arrows animate
   3.5s – State machine boxes light up sequentially
   5.0s – Syncing dots turn green (engines become ACTIVE)
*/

/* --- Scaled-Out label fade in --- */
.showcase-display-section.panel-active-anim #showcase-panel-4 .elastic-scaledout-label {
  animation: elasticFadeIn 0.6s ease-out 0.5s both;
}

/* --- Engine groups scale in staggered --- */
.showcase-display-section.panel-active-anim #showcase-panel-4 .el-scaleout-4 {
  animation: elasticEngineAppear 0.7s cubic-bezier(0.34, 1.56, 0.64, 1) 1.0s both;
}
.showcase-display-section.panel-active-anim #showcase-panel-4 .el-scaleout-5 {
  animation: elasticEngineAppear 0.7s cubic-bezier(0.34, 1.56, 0.64, 1) 1.4s both;
}
.showcase-display-section.panel-active-anim #showcase-panel-4 .el-scaleout-6 {
  animation: elasticEngineAppear 0.7s cubic-bezier(0.34, 1.56, 0.64, 1) 1.8s both;
}

/* --- Route lines to new engines --- */
.showcase-display-section.panel-active-anim #showcase-panel-4 .el-route-new-4 {
  animation: elasticLineDraw 0.5s ease-out 2.0s both;
}
.showcase-display-section.panel-active-anim #showcase-panel-4 .el-route-new-5 {
  animation: elasticLineDraw 0.5s ease-out 2.2s both;
}
.showcase-display-section.panel-active-anim #showcase-panel-4 .el-route-new-6 {
  animation: elasticLineDraw 0.5s ease-out 2.4s both;
}

/* --- P2P weight sync group --- */
.showcase-display-section.panel-active-anim #showcase-panel-4 .elastic-p2p-group {
  animation: elasticFadeIn 0.6s ease-out 2.8s both;
}

/* --- P2P arrow flow animation (continuous after appear) --- */
.showcase-display-section.panel-active-anim #showcase-panel-4 .elastic-p2p-arrow {
  animation: elasticP2PFlow 1.2s linear infinite;
  animation-delay: 2.8s;
}

/* --- State machine boxes light up sequentially --- */
.showcase-display-section.panel-active-anim #showcase-panel-4 .el-state-1 {
  animation: elasticStateLight 0.4s ease-out 3.5s both;
}
.showcase-display-section.panel-active-anim #showcase-panel-4 .el-arrow-1 {
  animation: elasticFadeIn 0.3s ease-out 3.7s both;
}
.showcase-display-section.panel-active-anim #showcase-panel-4 .el-state-2 {
  animation: elasticStateLight 0.4s ease-out 3.9s both;
}
.showcase-display-section.panel-active-anim #showcase-panel-4 .el-arrow-2 {
  animation: elasticFadeIn 0.3s ease-out 4.1s both;
}
.showcase-display-section.panel-active-anim #showcase-panel-4 .el-state-3 {
  animation: elasticStateLight 0.4s ease-out 4.3s both;
}
.showcase-display-section.panel-active-anim #showcase-panel-4 .el-arrow-3 {
  animation: elasticFadeIn 0.3s ease-out 4.5s both;
}
.showcase-display-section.panel-active-anim #showcase-panel-4 .el-state-4 {
  animation: elasticStateLight 0.4s ease-out 4.7s both;
}
.showcase-display-section.panel-active-anim #showcase-panel-4 .el-arrow-4 {
  animation: elasticFadeIn 0.3s ease-out 4.9s both;
}
.showcase-display-section.panel-active-anim #showcase-panel-4 .el-state-5 {
  animation: elasticStateLight 0.4s ease-out 5.1s both;
}
.showcase-display-section.panel-active-anim #showcase-panel-4 .el-arrow-5 {
  animation: elasticFadeIn 0.3s ease-out 5.3s both;
}
.showcase-display-section.panel-active-anim #showcase-panel-4 .el-state-6 {
  animation: elasticStateLight 0.4s ease-out 5.5s both;
}

/* --- Syncing dots pulse then turn green --- */
.showcase-display-section.panel-active-anim #showcase-panel-4 .el-dot-syncing {
  animation: elasticDotPulse 0.8s ease-in-out infinite 1.5s;
}

/* --- Route line initial state: hidden --- */
#showcase-panel-4 .elastic-route-line-new {
  opacity: 0;
}

/* --- State arrows initial state: hidden --- */
#showcase-panel-4 .elastic-state-arrow {
  opacity: 0;
}

/* ===== Keyframes ===== */

@keyframes elasticFadeIn {
  from { opacity: 0; }
  to   { opacity: 1; }
}

@keyframes elasticEngineAppear {
  0% {
    opacity: 0;
    transform: scale(0.5) translateY(10px);
  }
  100% {
    opacity: 1;
    transform: scale(1) translateY(0);
  }
}

@keyframes elasticLineDraw {
  from {
    opacity: 0;
    stroke-dashoffset: 100;
  }
  to {
    opacity: 1;
    stroke-dashoffset: 0;
  }
}

@keyframes elasticP2PFlow {
  from { stroke-dashoffset: 14; }
  to   { stroke-dashoffset: 0; }
}

@keyframes elasticStateLight {
  0% {
    stroke: #4B5563;
    fill: rgba(255, 255, 255, 0.04);
  }
  50% {
    stroke: #22d3ee;
    fill: rgba(34, 211, 238, 0.15);
  }
  100% {
    stroke: #6B7280;
    fill: rgba(255, 255, 255, 0.06);
  }
}

@keyframes elasticDotPulse {
  0%, 100% {
    opacity: 1;
    r: 4;
  }
  50% {
    opacity: 0.5;
    r: 3;
  }
}
</style>
