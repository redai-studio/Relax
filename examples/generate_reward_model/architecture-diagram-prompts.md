# GenRM 部署模式架构图 —— 图像生成 Prompt

单张 3 格图，覆盖 [`README.md`](./README.md) 里的三种 GenRM 部署模式（角色 ×
阶段 × 装载/卸载状态）。给 Nano Banana / Gemini 2.5 Flash Image 之类图像
模型直接使用。

## Prompt（复制粘贴整段）

```
生成一张 4K 高清（3840×2160，横版 16:9，无损 PNG）的技术架构信息图。
标题「Relax GenRM Deployment Modes」，白底，扁平化 whitepaper 风格，
无 3D、无渐变、无阴影。

画布横向分成 1×3 三格，三格共用同一套视觉语言。每一格用泳道图（swim-lane）
表达：

- 纵向 3 行泳道，从上到下：Actor / Rollout / GenRM（行左侧写角色名）
- 横向从左到右是时间轴，用竖虚线分隔不同 phase，每段顶端标注 phase 名字
- 每个泳道 × phase 交叉处是一个状态色块：
  · 绿色实心色块 #22c55e = 该角色此阶段「装载在 GPU 上、活跃」
  · 灰色虚线空心框 #94a3b8 = 该角色此阶段「已卸载、休眠」
- 色块中间叠加小字，写此阶段该角色占用的 GPU 编号（例：「GPU 0..15」）
- Phase 之间需要显式 sleep/wake 切换时，用蓝色 #2563eb 粗箭头连接，
  箭头上方写事件（例：「offload Rollout → onload GenRM」）
- 同一 phase 内部的 HTTP 数据流用细灰色单向箭头 + 小字（例：「HTTP score」）

三格分别绘制以下三个模式，位置和内容严格按下面来：

──────── 左 —— Colocate / Split (8 GPU) ────────
两个 phase：Inference | Training

Inference 阶段：
  Actor   行：灰虚线框「asleep」
  Rollout 行：绿实色块「GPU 0..3」
  GenRM   行：绿实色块「GPU 4..7」
  Rollout 与 GenRM 两行之间画细灰双向箭头「inline HTTP score」

Training 阶段：
  Actor   行：绿实色块「GPU 0..7」
  Rollout 行：灰虚线框「offloaded」
  GenRM   行：灰虚线框「offloaded」

Phase 之间不需要蓝箭头。
说明文字：Rollout 与 GenRM 分片并行，inline reward。

──────── 中 —— Colocate / Shared Defer-swap (16 GPU) ────────
三个 phase：Phase A "Rollout" | Phase B "Score" | Phase C "Train"

Phase A：
  Actor   行：灰虚线框「asleep」
  Rollout 行：绿实色块「GPU 0..15, mem_fraction=0.85」
  GenRM   行：灰虚线框「asleep」

A→B 之间：蓝色粗箭头，箭头上两行文字
  「offload Rollout」
  「onload GenRM」

Phase B：
  Actor   行：灰虚线框「asleep」
  Rollout 行：灰虚线框「asleep」
  GenRM   行：绿实色块「GPU 0..15, batch score」

B→C 之间：蓝色粗箭头，箭头上两行文字
  「offload GenRM」
  「wake Actor」

Phase C：
  Actor   行：绿实色块「GPU 0..15, train + weight sync」
  Rollout 行：灰虚线框「asleep」
  GenRM   行：灰虚线框「asleep」

**关键强调**：这一格中每个 phase 有且只有一个绿色行，其余两个必须是灰虚线框。
视觉上形成三阶段「接力式」传递。
说明文字：串行 sleep-wake，独占全 16 GPU。

──────── 右 —— Fully Async (dedicated pools) ────────
只有一列「always-on」，不分 phase。

  Actor   行：绿实色块「pool A: 2 GPU (dedicated)」
  Rollout 行：绿实色块「pool B: 3 GPU (dedicated)」
  GenRM   行：绿实色块「pool C: 1 GPU (dedicated)」

三行之间画细灰单向箭头组成数据流：
  Rollout → GenRM 「HTTP score」
  Rollout → Actor 「samples → TransferQueue」
  GenRM   → Actor 「rewards → TransferQueue」

不需要 sleep/wake 蓝箭头（永不休眠）。
说明文字：角色独占 GPU 池，完全并行。

──────── 图例（画布最上方或最下方居中一条横条） ────────
  [绿实色块] Onload / Active
  [灰虚线框] Offload / Asleep
  [蓝粗箭头] Sleep-wake transition
  [细灰箭头] HTTP / Data flow

──────── 全局样式约束 ────────
- 三格大小相同、内边距一致、泳道行高一致，视觉上完全对齐
- 字体：无衬线（Inter / Helvetica），层级 标题 > phase 名 > 角色名 >
  GPU 编号 > 箭头说明
- 只用四种颜色：绿 #22c55e、灰 #94a3b8、蓝 #2563eb、文字 #0f172a，白底
- 全图除标题和图例外不要出现英文以外的字符
- 所有文字在 100% 显示比例下必须清晰可读，笔画锐利，不模糊
```

## 使用要点

- **一次生成即可拿到完整对比图**，不用跑 3 次。
- 若模型仍然把某一格画糊，只挑那一格的描述块另起 prompt 单独重画，然后
  用图像编辑工具替换到主图对应位置。
- 关键失败模式：
  - Defer-swap 那格所有行都画成绿的 → prompt 里已用**关键强调**明示，仍
    失败则复述一遍「every phase in bottom-left panel has exactly one
    green row, the other two rows are gray dashed outlines」。
  - Fully Async 那格误加 phase 分隔 → prompt 里已写「只有一列，不分
    phase」。
