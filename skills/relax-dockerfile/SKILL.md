---
name: relax-dockerfile
description: 修改或审查 Relax 的 Dockerfile、镜像构建脚本及依赖补丁时使用。指导原生扩展分阶段编译、构建缓存、镜像层数和平台验证。
---

# Relax 镜像构建

先从 `Makefile` 和 CI 镜像入口确认受影响的 Dockerfile、target、平台与 `TRAIN_IMAGE` 来源。区分 BuildKit 的步骤缓存和按内容哈希复用的整镜像。新增或移动 `COPY`、bind mount、编译参数时，核对相关输入是否也进入调用方的镜像哈希。

## 把高频变更放在后面

Dockerfile 里的框架运行时 patch 通常改得比工具链和原生依赖频繁。修改 patch 会让复制或应用它的步骤失去缓存，并传到依赖这一层的后续步骤。满足依赖顺序的前提下，把这类 patch 尽量放在最终镜像的最后几层，让稳定依赖的安装和耗时编译更容易复用缓存。

- 固定版本源码的获取可以单独缓存。无关的运行时 patch 放在原生扩展和内核编译之后，应用完再执行相关的轻量校验。编译实际读取的源码 patch 留在其编译步骤之前。
- 昂贵的原生扩展优先拆成独立 wheel stage，从最终目标使用的同一 `TRAIN_IMAGE` 派生，沿用 Python、libc、Torch、CUDA 和编译器 ABI，保留原来的源码 pin、架构与编译参数。只让需要这个扩展的目标消费该 stage。
- wheel stage 只读取编译所需的源码、patch 与环境约束。应用层依赖安装、框架适配和运行时校验留给最终阶段。需要清除训练环境的 `LD_PRELOAD` 时，将作用域限在编译进程树。
- 最终阶段通过 `RUN --mount=type=bind,from=...` 安装 wheel。父镜像可能存在同版本旧包时，用 `--force-reinstall` 确保安装本次产物。运行依赖已由镜像锁定时，用 `--no-deps` 避免重新解析。保留原生扩展所需的导入和 cubin 校验。

参考 [#665 Hopper FA3 wheel 独立编译](https://github.com/redai-studio/Relax/pull/665)。

## 控制最终镜像层数

- 传统 Docker 镜像存储的层链上限为 **125 层**。计算层预算时计入基础镜像，并给 QS 等下游封装和后续依赖留余量。对 dev 和下游交付镜像，分别用 `docker image inspect --format '{{len .RootFS.Layers}}' <image>` 检查实际层数。Dockerfile 指令数可用来估算改动。
- 临时 patch、脚本和 wheel 按需用只读 bind mount，优先逐文件挂载，减少无关文件对缓存的影响。单独 `COPY` 后再删除，文件字节仍留在旧层中。仅通过 bind mount 消费的 builder 层在最终镜像的继承链之外。
- 把用途相同的下载、安装和清理合在同一个 `RUN`。为固定源码获取、昂贵的内核编译和易变的校验脚本保留有用的缓存边界。使用带锁的编译缓存时，把持锁范围限制在实际编译的 `RUN`，按架构隔离缓存目录。

参考 [#528 CUDA 13 运行时镜像减层与冷拉取](https://github.com/redai-studio/Relax/pull/528)。

## 管理依赖 patch

- 尽量不新建 `.patch` 文件。修改前先检查目标依赖的版本、现有 patch 和 Dockerfile 的应用顺序；同一依赖、同一版本的修复优先合入已有 patch，避免按功能不断叠加增量 patch。
- Megatron 和 SGLang 的相关修复统一收敛到对应版本的主 patch。沿用 `docker/patch/latest/` 的现有软链接并修改其实际目标文件，不另建 CUDA 专用目录或重复副本。升级依赖版本或确实无法共用时才新增 patch，并说明不能合并的原因。
- 保留历史版本的 Megatron、SGLang patch。区分历史版本 patch 与已合入当前主 patch 的增量文件；仅在确认内容完整迁移、所有引用已更新后，删除被替代的增量文件。
- 其他依赖也优先按依赖和版本合并 patch；不能为了减少文件数而混合不同依赖或不兼容版本。确有运行配置差异时，优先沿用主 patch 内的分段应用机制，保持各配置原有的生效范围和顺序。
- patch 代码保留在 `.patch` 文件中，不搬入 Dockerfile 的内联脚本或字符串替换。Dockerfile 尽量小改，只调整必要的复制、应用和构建步骤；现有镜像能力应保留在输出镜像内，不能改为要求用户在容器启动后手动补丁。
- 合并时列明旧文件内容的新位置，检查 Dockerfile、脚本、测试、文档和校验清单中的引用。对固定版本源码验证 patch 能按实际构建顺序干净应用，并核对各受影响配置合并前后的源码结果；不能仅凭 patch 文本拼接成功判断无冲突。未完成的镜像构建或 GPU 验证须明确说明。
