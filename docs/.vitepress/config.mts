import taskLists from 'markdown-it-task-lists'
import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  vite: {
    build: {
      chunkSizeWarningLimit: 1500,
      rollupOptions: {
        output: {
          // Sanitize chunk filenames: replace `.md.` / trailing `.md` to avoid WAF 403
          chunkFileNames: (chunkInfo) => {
            const name = (chunkInfo.name || 'chunk')
              .replace(/\.md\./g, '-')
              .replace(/\.md$/g, '')
            return `assets/${name}.[hash].js`
          },
          assetFileNames: (assetInfo) => {
            const name = (assetInfo.name || 'asset')
              .replace(/\.md\./g, '-')
              .replace(/\.md\b/g, '')
            return `assets/${name}.[hash][extname]`
          }
        }
      }
    }
  },

  /**
   * buildEnd hook: runs after VitePress finishes generating all output files,
   * including the ".lean.js" page-data files that bypass Rollup's chunkFileNames.
   * Renames any remaining *.md.* files and patches all references in the bundle.
   */
  async buildEnd() {
    // Use dynamic import to access Node built-ins inside the ESM config
    const { existsSync, readdirSync, renameSync, readFileSync, writeFileSync } = await import('node:fs')
    const { join, dirname, basename } = await import('node:path')
    const { fileURLToPath } = await import('node:url')

    const configDir = dirname(fileURLToPath(import.meta.url))
    const distDir = join(configDir, 'dist')
    if (!existsSync(distDir)) return

    // ---------- Phase 1: Rename physical files containing ".md." ----------
    const targets: string[] = []
    const walk = (dir: string) => {
      for (const entry of readdirSync(dir, { withFileTypes: true })) {
        const full = join(dir, entry.name)
        if (entry.isDirectory()) walk(full)
        else if (entry.name.includes('.md.')) targets.push(full)
      }
    }
    walk(distDir)

    const renameMap: Record<string, string> = {}
    for (const file of targets) {
      const base = basename(file)
      const newBase = base.replace(/\.md\./g, '-')
      if (base !== newBase) renameMap[base] = newBase
    }

    for (const file of targets) {
      const base = basename(file)
      const newBase = renameMap[base]
      if (!newBase) continue
      renameSync(file, join(dirname(file), newBase))
    }

    // ---------- Phase 2: Patch static references to old filenames ----------
    const textFiles: string[] = []
    const walkText = (dir: string) => {
      for (const entry of readdirSync(dir, { withFileTypes: true })) {
        const full = join(dir, entry.name)
        if (entry.isDirectory()) walkText(full)
        else if (/\.(html|js|json|css)$/.test(entry.name)) textFiles.push(full)
      }
    }
    walkText(distDir)

    for (const file of textFiles) {
      let content = readFileSync(file, 'utf-8')
      let changed = false
      for (const [oldName, newName] of Object.entries(renameMap)) {
        const escaped = oldName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
        const re = new RegExp(escaped, 'g')
        if (re.test(content)) {
          content = content.replace(new RegExp(escaped, 'g'), newName)
          changed = true
        }
      }
      if (changed) writeFileSync(file, content, 'utf-8')
    }

    // ---------- Phase 3: Fix VitePress client-side dynamic URL construction ----------
    // VitePress generates URLs at runtime via __VP_HASH_MAP__:
    //   key = "index.md", hash = "BnBcK5_k" → URL = "assets/index.md.BnBcK5_k.js"
    // URLs containing ".md." are blocked by WAF (403). We need to:
    // (a) Strip ".md" suffix from hash-map keys
    // (b) Patch the framework JS that appends `.md` to route paths before lookup
    // (c) Patch the URL template that joins key + hash with `.` separator

    // (a) Fix __VP_HASH_MAP__ in all HTML files
    const htmlFiles: string[] = []
    const walkHtml = (dir: string) => {
      for (const entry of readdirSync(dir, { withFileTypes: true })) {
        const full = join(dir, entry.name)
        if (entry.isDirectory()) walkHtml(full)
        else if (entry.name.endsWith('.html')) htmlFiles.push(full)
      }
    }
    walkHtml(distDir)

    for (const file of htmlFiles) {
      let content = readFileSync(file, 'utf-8')
      const patched = content.replace(
        /window\.__VP_HASH_MAP__\s*=\s*JSON\.parse\("([^"]*)"\)/g,
        (_match: string, jsonStr: string) => {
          const unescaped = jsonStr.replace(/\\"/g, '"').replace(/\\\\/g, '\\')
          const map: Record<string, string> = JSON.parse(unescaped)
          const fixed: Record<string, string> = {}
          for (const [key, value] of Object.entries(map)) {
            fixed[key.replace(/\.md$/, '')] = value
          }
          const fixedStr = JSON.stringify(JSON.stringify(fixed))
          return `window.__VP_HASH_MAP__=JSON.parse(${fixedStr})`
        }
      )
      if (patched !== content) writeFileSync(file, patched, 'utf-8')
    }

    // (b) & (c) Patch framework JS to stop appending ".md" and use "-" separator
    // In the minified framework JS, VitePress has this pathToFile logic:
    //   t = gi(path.replace(/\//g,"_") || "index") + ".md"
    //   ...
    //   t = `${n}assets/${t}.${s}.js`
    //   ...
    //   t = `./${name}.md.js`
    // After patching:
    //   - Remove the `+ ".md"` so keys no longer end with .md
    //   - Change `${t}.${s}.js` to `${t}-${s}.js` to match renamed files
    //   - Change `.md.js` to `.js` for the inline data fallback path
    //   - Fix the _index.md fallback logic to use keys without .md
    for (const file of textFiles) {
      if (!basename(file).startsWith('framework.') || !file.endsWith('.js')) continue
      let content = readFileSync(file, 'utf-8')
      let changed = false

      // Remove: ||"index")+".md"  →  ||"index")
      // This is the line: t=gi(t.slice(n.length).replace(/\//g,"_")||"index")+".md"
      const oldAppend = '||"index")+".md"'
      if (content.includes(oldAppend)) {
        content = content.replace(oldAppend, '||"index")')
        changed = true
      }

      // Fix fallback: _index.md → _index, and .md → (empty)
      // Original: t.endsWith("_index.md")?t.slice(0,-9)+".md":t.slice(0,-3)+"_index.md"
      // New:      t.endsWith("_index")?t.slice(0,-6):t+"_index"
      const oldFallback = 't.endsWith("_index.md")?t.slice(0,-9)+".md":t.slice(0,-3)+"_index.md"'
      const newFallback = 't.endsWith("_index")?t.slice(0,-6):t+"_index"'
      if (content.includes(oldFallback)) {
        content = content.replace(oldFallback, newFallback)
        changed = true
      }

      // Fix URL template: assets/${t}.${s}.js → assets/${t}-${s}.js
      const oldTemplate = 'assets/${t}.${s}.js'
      const newTemplate = 'assets/${t}-${s}.js'
      if (content.includes(oldTemplate)) {
        content = content.replace(oldTemplate, newTemplate)
        changed = true
      }

      // Fix inline data fallback: .md.js → .js
      // Original: `./${gi(t.slice(1).replace(/\//g,"_"))}.md.js`
      // We need to remove .md from this pattern
      const oldInline = '.md.js`'
      if (content.includes(oldInline)) {
        content = content.replaceAll(oldInline, '.js`')
        changed = true
      }

      // Fix 404 fallback path computation that also appends .md
      // Original: .replace(/(\.html)?$/,".md").replace(/^\//,"")
      // This is used for relativePath in 404 handling — not a file request,
      // but let's keep it consistent. Actually this is internal data, keep it.

      if (changed) writeFileSync(file, content, 'utf-8')
    }

    // Also patch hashmap.json if it exists (used for client-side navigation fallback)
    const hashmapPath = join(distDir, 'hashmap.json')
    if (existsSync(hashmapPath)) {
      const raw = readFileSync(hashmapPath, 'utf-8')
      const map: Record<string, string> = JSON.parse(raw)
      const fixed: Record<string, string> = {}
      for (const [key, value] of Object.entries(map)) {
        fixed[key.replace(/\.md$/, '')] = value
      }
      writeFileSync(hashmapPath, JSON.stringify(fixed), 'utf-8')
    }
  },

  title: "Relax",
  description: "Towards Async, Omni-Modal RL at Scale, Just Relax.",

  // Base path for deployment under a sub-directory
  base: '/Relax/',
  
  // Ignore dead links for source code references and placeholder pages
  ignoreDeadLinks: true,

  // Enable LaTeX math rendering and GitHub-style task lists
  markdown: {
    math: true,
    config(md) {
      md.use(taskLists)
    }
  },
  
  // 多语言配置
  locales: {
    en: {
      label: 'English',
      lang: 'en',
      link: '/en/',
      themeConfig: {
        nav: [
          { text: 'Home', link: '/en/' },
          { text: 'Guide', link: '/en/guide/introduction' },
          { text: 'API', link: '/en/api/overview' },
          { text: 'Examples', link: '/en/examples/deepeyes' },
          {
            text: 'Resources',
            items: [
              { text: 'GitHub', link: 'https://github.com/redai-studio/Relax' },
              { text: 'Paper', link: 'https://arxiv.org/abs/2604.11554' }
            ]
          }
        ],
        sidebar: {
          '/en/guide/': [
            {
              text: 'Getting Started',
              items: [
                { text: 'Introduction', link: '/en/guide/introduction' },
                { text: 'Installation', link: '/en/guide/installation' },
                { text: 'Quick Start', link: '/en/guide/quick-start' },
                { text: 'Customize Training', link: '/en/guide/customize-training' },
                { text: 'SFT Training', link: '/en/guide/sft-training' },
                { text: 'MTP Training', link: '/en/guide/mtp-rl-training' },
                { text: 'PPO Training', link: '/en/guide/ppo-training' },
                { text: 'REINFORCE++', link: '/en/guide/reinforce-plus-plus' },
                { text: 'REINFORCE++ Report', link: '/en/guide/reinforce-plus-plus-training-report' },
                { text: 'Model Checkpoint Conversion', link: '/en/guide/model-conversion' },
                { text: 'Configuration', link: '/en/guide/configuration' }
              ]
            },
            {
              text: 'Core Concepts',
              items: [
                { text: 'Architecture', link: '/en/guide/architecture' },
                { text: 'Dataset Design', link: '/en/guide/dataset-design' },
                { text: 'Distributed Checkpoint', link: '/en/guide/distributed-checkpoint' },
                { text: 'Health Check Manager', link: '/en/guide/health-check-manager' }
              ]
            },
            {
              text: 'Advanced',
              items: [
                { text: 'Fully Async Training', link: '/en/guide/fully-async-training' },
                { text: 'Agentic Rollout', link: '/en/guide/agentic-rollout' },
                { text: 'Agentic KV Scheduling', link: '/en/guide/agentic-kv-scheduling' },
                { text: 'Hybrid Training Mode', link: '/en/guide/hybrid-training' },
                { text: 'Elastic Rollout Scaling', link: '/en/guide/elastic-rollout' },
                { text: 'Dynamic Context Parallelism', link: '/en/guide/dynamic-context-parallel' },
                { text: 'Metrics Service', link: '/en/guide/metrics-service-detailed' },
                { text: 'Notification System', link: '/en/guide/notification-system' },
                { text: 'Update Weights Pipeline', link: '/en/guide/update-weights-pipeline' },
                { text: 'Low-Rank Adaptation (LoRA) Training', link: '/en/guide/low-rank-adaptation-training' }
              ]
            },
            {
              text: 'Best Practices',
              items: [
                { text: 'Performance Tuning', link: '/en/guide/performance-tuning' },
                { text: 'Accelerated S3 Model Loading', link: '/en/guide/s3-model-loading' },
                { text: 'OOM Troubleshooting', link: '/en/guide/oom-troubleshooting' },
                { text: 'External Model Integration', link: '/en/guide/external-model-integration' }
              ]
            },
            {
              text: 'Development',
              items: [
                { text: 'How to Contribute', link: '/en/guide/how-to-contribute' },
                { text: 'Debugging Guide', link: '/en/guide/debugging' },
                { text: 'Config Doctor', link: '/en/guide/config-doctor' },
                { text: 'Trajectory Replay', link: '/en/guide/trajectory-replay' },
                { text: 'Rollout Result Viewer', link: '/en/guide/rollout-result-viewer' }
              ]
            }
          ],
          '/en/api/': [
            {
              text: 'API Reference',
              items: [
                { text: 'Overview', link: '/en/api/overview' }
              ]
            },
            {
              text: 'Service HTTP APIs',
              items: [
                { text: 'Actor', link: '/en/api/actor' },
                { text: 'Rollout', link: '/en/api/rollout' },
                { text: 'GenRM', link: '/en/api/genrm' },
                { text: 'ActorFwd', link: '/en/api/actor-fwd' }
              ]
            }
          ],
          '/en/examples/': [
            {
              text: 'Examples',
              items: [
                { text: 'DeepEyes', link: '/en/examples/deepeyes' },
                { text: 'On-Policy Distillation', link: '/en/examples/on-policy-distillation' },
                { text: 'Generative Reward Model', link: '/en/examples/generative-reward-model' },
                { text: 'Low-Precision Training', link: '/en/examples/low-precision-training' },
                { text: 'Elastic Rollout', link: '/en/examples/elastic-rollout' },
                { text: 'Algorithms', link: '/en/examples/algorithms' }
              ]
            }
          ]
        },
        footer: {
          message: 'Released under the Apache 2.0 License.',
          copyright: 'Copyright © 2026 Relax Team'
        }
      }
    },
    zh: {
      label: '简体中文',
      lang: 'zh-CN',
      link: '/zh/',
      themeConfig: {
        nav: [
          { text: '首页', link: '/zh/' },
          { text: '指南', link: '/zh/guide/introduction' },
          { text: 'API', link: '/zh/api/overview' },
          { text: '示例', link: '/zh/examples/deepeyes' },
          {
            text: '资源',
            items: [
              { text: 'GitHub', link: 'https://github.com/redai-studio/Relax' },
              { text: '论文', link: 'https://arxiv.org/abs/2604.11554' }
            ]
          }
        ],
        sidebar: {
          '/zh/guide/': [
            {
              text: '快速开始',
              items: [
                { text: '介绍', link: '/zh/guide/introduction' },
                { text: '安装', link: '/zh/guide/installation' },
                { text: '快速上手', link: '/zh/guide/quick-start' },
                { text: '自定义训练', link: '/zh/guide/customize-training' },
                { text: 'SFT 训练', link: '/zh/guide/sft-training' },
                { text: 'MTP 训练', link: '/zh/guide/mtp-rl-training' },
                { text: 'PPO 训练', link: '/zh/guide/ppo-training' },
                { text: 'REINFORCE++', link: '/zh/guide/reinforce-plus-plus' },
                { text: 'REINFORCE++ 训练与数值验证报告', link: '/zh/guide/reinforce-plus-plus-training-report' },
                { text: '模型 Checkpoint 转换', link: '/zh/guide/model-conversion' },
                { text: '配置说明', link: '/zh/guide/configuration' }
              ]
            },
            {
              text: '核心概念',
              items: [
                { text: '架构设计', link: '/zh/guide/architecture' },
                { text: '数据集设计', link: '/zh/guide/dataset-design' },
                { text: 'Distributed Checkpoint', link: '/zh/guide/distributed-checkpoint' },
                { text: '健康检查管理器', link: '/zh/guide/health-check-manager' }
              ]
            },
            {
              text: '进阶指南',
              items: [
                { text: '全异步训练流水线', link: '/zh/guide/fully-async-training' },
                { text: 'Agentic Rollout', link: '/zh/guide/agentic-rollout' },
                { text: 'Agentic KV 调度', link: '/zh/guide/agentic-kv-scheduling' },
                { text: 'Hybrid 混合训练模式', link: '/zh/guide/hybrid-training' },
                { text: '弹性 Rollout 扩缩容', link: '/zh/guide/elastic-rollout' },
                { text: 'Dynamic Context Parallelism', link: '/zh/guide/dynamic-context-parallel' },
                { text: 'Metrics 服务', link: '/zh/guide/metrics-service-detailed' },
                { text: '通知系统', link: '/zh/guide/notification-system' },
                { text: '权重更新流水线优化', link: '/zh/guide/update-weights-pipeline' },
                { text: '低秩适配（LoRA）训练', link: '/zh/guide/low-rank-adaptation-training' }
              ]
            },
            {
              text: '最佳实践',
              items: [
                { text: '性能调优', link: '/zh/guide/performance-tuning' },
                { text: 'S3 模型加载加速', link: '/zh/guide/s3-model-loading' },
                { text: 'OOM 排查', link: '/zh/guide/oom-troubleshooting' },
                { text: '外部模型接入', link: '/zh/guide/external-model-integration' }
              ]
            },
            {
              text: '开发指南',
              items: [
                { text: '如何贡献', link: '/zh/guide/how-to-contribute' },
                { text: '调试指南', link: '/zh/guide/debugging' },
                { text: '配置检查与 Dry-run', link: '/zh/guide/config-doctor' },
                { text: '轨迹重放', link: '/zh/guide/trajectory-replay' },
                { text: 'Rollout 结果可视化', link: '/zh/guide/rollout-result-viewer' }
              ]
            }
          ],
          '/zh/api/': [
            {
              text: 'API 参考',
              items: [
                { text: '概览', link: '/zh/api/overview' }
              ]
            },
            {
              text: '服务 HTTP API',
              items: [
                { text: 'Actor', link: '/zh/api/actor' },
                { text: 'Rollout', link: '/zh/api/rollout' },
                { text: 'GenRM', link: '/zh/api/genrm' },
                { text: 'ActorFwd', link: '/zh/api/actor-fwd' }
              ]
            }
          ],
          '/zh/examples/': [
            {
              text: '示例',
              items: [
                { text: 'DeepEyes', link: '/zh/examples/deepeyes' },
                { text: '在线策略蒸馏', link: '/zh/examples/on-policy-distillation' },
                { text: '生成式奖励模型', link: '/zh/examples/generative-reward-model' },
                { text: '低精度训练', link: '/zh/examples/low-precision-training' },
                { text: '弹性 Rollout', link: '/zh/examples/elastic-rollout' },
                { text: '算法参考', link: '/zh/examples/algorithms' }
              ]
            }
          ]
        },
        footer: {
          message: '基于 Apache 2.0 许可发布',
          copyright: 'Copyright © 2026 Relax 团队'
        },
        docFooter: {
          prev: '上一页',
          next: '下一页'
        },
        outline: {
          level: [2, 4],
          label: '页面导航'
        },
        lastUpdated: {
          text: '最后更新于',
          formatOptions: {
            dateStyle: 'short',
            timeStyle: 'medium'
          }
        },
        langMenuLabel: '多语言',
        returnToTopLabel: '回到顶部',
        sidebarMenuLabel: '菜单',
        darkModeSwitchLabel: '主题',
        lightModeSwitchTitle: '切换到浅色模式',
        darkModeSwitchTitle: '切换到深色模式'
      }
    }
  },

  themeConfig: {
    logo: '/rednote-logo.png',
    socialLinks: [
      { icon: 'github', link: 'https://github.com/redai-studio/Relax' }
    ],
    search: {
      provider: 'local'
    },
    outline: {
      level: [2, 4]
    }
  },

  head: [
    ['link', { rel: 'icon', type: 'image/png', sizes: '32x32', href: '/Relax/favicon-32.png' }],
    ['link', { rel: 'icon', type: 'image/png', sizes: '16x16', href: '/Relax/favicon-16.png' }],
    ['link', { rel: 'preconnect', href: 'https://fonts.googleapis.com' }],
    ['link', { rel: 'preconnect', href: 'https://fonts.gstatic.com', crossorigin: '' }],
    ['link', { href: 'https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Manrope:wght@300;400;500;600;700&family=Inter:wght@400;500;600&display=swap', rel: 'stylesheet' }]
  ],

  lastUpdated: true,
  cleanUrls: false,

  // Default to dark mode to match ASCII art background
  appearance: 'dark'
})
