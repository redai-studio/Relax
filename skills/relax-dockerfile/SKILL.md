---
name: relax-dockerfile
description: 修改或审查 Relax 的 Dockerfile、镜像构建脚本及依赖补丁时使用。指导原生扩展分阶段编译、构建缓存、镜像层数和平台验证。
---

# Relax 镜像构建

先从 `Makefile` 和 CI 镜像入口确认受影响的 Dockerfile、target、平台与 `TRAIN_IMAGE` 来源。区分 BuildKit 的步骤缓存和按内容哈希复用的整镜像。新增或移动 `COPY`、bind mount、编译参数时，核对相关输入是否也进入调用方的镜像哈希。

## 分离原生编译与运行时补丁

- 找出原生扩展实际依赖的源码、源码补丁、Python、Torch、CUDA、编译器和系统 ABI。能独立编译的扩展从同一 `TRAIN_IMAGE` 派生专用 wheel stage，把耗时编译放在高频修改的 requirements 和框架运行时补丁前。确认 stage 的父镜像和挂载也独立于这些输入。参考 [Hopper FA3 #665](https://github.com/redai-studio/Relax/pull/665)。
- 只让需要该扩展的目标依赖 wheel stage。最终阶段通过 `RUN --mount=type=bind,from=...` 安装，保留原有依赖约束和运行时校验。原生源码校验留在构建阶段，框架适配、已安装包和 cubin 校验在最终镜像执行。
- 保留架构、源码 pin 及 ABI 校验。划定缓存复用边界时，以编译实际读取的输入为准，编译确有依赖的补丁仍须参与构建。用日志确认昂贵步骤出现 `CACHED`，整镜像直接命中属于另一种复用。

## 控制最终镜像层数

- 兼容传统 Docker 镜像存储的 **125 层**上限，并给下游封装和后续依赖留余量。统计实际产物的 `.RootFS.Layers`，包含基础镜像和下游派生层。Dockerfile 的 `RUN/COPY/ADD` 计数只用于估算改动量。参考 [CUDA 13 冷拉取与减层 #528](https://github.com/redai-studio/Relax/pull/528)。
- 临时 patch、wheel 或辅助脚本优先用 build-time bind mount。下载、安装、清理放在同一层，避免文件留在较早的层。独立 builder 的中间层通过 bind 消费时，不计入最终镜像继承链。
- 接近上限或改动层结构时，记录各受影响产物的前后层数，并从干净的镜像存储验证冷拉取，包括 dev 和下游实际交付的镜像。验证不了时标明哪些是静态计数，哪些产物尚未实测。

## 管理依赖 patch

- 新建 `.patch` 前先核对上游 pin、现有 patch 和应用顺序。同一依赖、同一版本的修复优先合入已有 patch。Megatron 和 SGLang 沿用 `docker/patch/latest/` 的链接，更新实际目标。上游版本变了或确实无法共用时再新增 patch，并说明原因。
- 保留历史版本的 Megatron 和 SGLang patch。仅在确认内容完整迁移、所有引用更新后删除被替代的增量 patch。不同依赖和不兼容版本独立维护。架构或运行配置确有差异时，沿用已有主 patch 分段机制和应用顺序。
- patch 内容保留在 `.patch` 文件中。镜像内完成应用，输出镜像保留原有功能。合并 patch 时，列出旧内容的新位置并检查 Dockerfile、脚本、测试、文档和校验清单。用固定版本源码按实际镜像顺序验证应用结果，对比涉及的各配置。
- 按受影响平台验证，并记录实际完成的范围。区分 Dockerfile 静态检查、镜像构建、受控缓存复用、冷拉取及 GPU 运行验证。
