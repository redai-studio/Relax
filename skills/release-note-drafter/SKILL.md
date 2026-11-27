---
name: release-note-drafter
description: 为 Relax 起草或更新中英文对照的 GitHub Release Notes（发布说明）。扫描相邻版本 tag 之间的全部提交，提炼新特性、核心依赖升级和关键修复，并向用户确认本次发布的镜像地址。
---

# 起草 Relax Release Notes

交付可直接粘贴到 GitHub Release 的 Markdown 草稿，默认包含完整的英文版和中文版。两版独立可读，章节和条目一一对应。使用同一组事实核对版本号、适用条件、命令和链接。文档有对应译本时可链接各自语言的页面。用户指定其他语言或排版时按其要求书写。用用户的语言询问发布信息。

## 确定本版包含哪些提交

1. 使用本地 `gh` 核对官方仓库的版本 tag。下面的查询会翻页，且不依赖当前 checkout 的 `origin` 指向哪个 fork。

   ```bash
   gh api repos/redai-studio/Relax/tags --paginate --jq '.[] | [.name, .commit.sha] | @tsv'
   ```

2. 使用用户指定的目标 tag，未指定时取最新正式版本。默认基线为目标 tag 历史上可达的上一正式版本 tag。跳过归档 tag 和预发布 tag，维护分支按其发布线选择。用户明确指定比较区间时沿用该区间。若目标 tag 尚未创建，向用户询问预览用的 commit 或分支，在草稿旁注明实际 SHA。
3. 核对本地 tag 解引用后的 commit 与官方 SHA 一致，并确保区间历史完整。缺少对象时从官方仓库补齐。常规发布将 `previous`、`current` 设为核实过的两个 tag 名称，得到扫描区间。

   ```bash
   base=$(git rev-parse --verify "refs/tags/${previous}^{commit}")
   target=$(git rev-parse --verify "refs/tags/${current}^{commit}")
   git merge-base --is-ancestor "$base" "$target" || exit 1
   range="$base..$target"
   ```

   此区间含目标 tag、排除基线及其祖先。预览未打 tag 的版本时，`target` 使用用户确定的 commit SHA。首版经核实没有基线时，直接令 `range="$target"`，扫描目标可达的全部历史。

4. 保存完整清单，核对行数等于提交数；按清单逐批读到末尾。

   ```bash
   inventory=$(mktemp)
   git log --topo-order --reverse --format='%H %P %s' "$range" -- > "$inventory"
   git rev-list --count "$range"
   wc -l < "$inventory"
   ```

   保留分支合入的提交、直接提交和无前缀提交。按日期、`--first-parent` 或 PR 列表筛选都有可能漏项。

## 核实本版发布的镜像

确定目标版本后，尽早向用户询问本次公开发布的镜像变体和每个变体的准确镜像地址。找到候选地址时逐项列出，请用户确认。用户已明确确认本版部分地址时，只追问缺项。等待答复期间继续整理提交。

例如起草 v0.2.0 时，直接问：

> v0.2.0 的 `cu12`、`cu13-hopper`、`cu13-blackwell` 三个镜像，各自的完整拉取地址是什么？可以贴三条 `docker pull` 命令。如果某个镜像尚未发布，请标明。

其他版本先问本次有哪些变体。Dockerfile 的基础镜像、nightly 标签和上一版地址都不能证明本版产物已发布。只为用户确认的地址生成拉取命令；若用户给的是镜像页面 URL，可直接链接该页面。待确认项列在草稿外，用户确认本次不发镜像时省略该节。

## 从提交提炼用户可见的变化

逐条判断影响，必要时用 `git show <sha>` 查看改动，用 `gh pr view <number> --repo redai-studio/Relax` 查看 PR 的使用场景和验证记录。把同一能力的实现、修正和文档合并为一个条目。遇到回退、重做或多次升级，用两个 tag 的最终差异核实实际交付的行为。

| 内容 | 写给用户的信息 | 核实依据 |
| --- | --- | --- |
| 新特性 | 哪类用户可以在什么模型、硬件或训练模式下做什么；有可运行的 recipe 时给出入口。 | 目标 tag 下的实现、示例或指南。 |
| 核心依赖升级 | 写明旧版本 → 新版本，以及受影响的运行环境或升级动作。CUDA 12 和 CUDA 13 镜像的依赖可能不同，按实际变体注明。 | 对比两端的 `docker/Dockerfile*`、依赖清单、构建参数和补丁。依赖按 commit 固定时保留实际 revision。 |
| 关键问题修复 | 写出用户原先在什么条件下会遇到什么故障，以及升级后恢复的行为。优先考虑训练正确性、卡死、checkpoint 恢复和兼容性问题。 | 对应修复 diff、复现说明或回归验证。 |

每条特性或修复写完描述，句末直接附上引入该改动的公开 PR。链接文字用 `#编号`，多个 PR 用逗号分隔，例如「Qwen3.8 新增 LoRA 训练和 Blackwell GRPO 配方。 [#655](https://github.com/redai-studio/Relax/pull/655), [#646](https://github.com/redai-studio/Relax/pull/646)」。中英文对应条目使用相同的 PR 和顺序。提交标题未注明 PR 时，用提交 SHA 查 GitHub 上关联的 PR；确实没有公开 PR 时保留描述即可。

文档、示例、源码和完整变更链接都固定到本次扫描的目标版本。已打 tag 时使用核实过的 tag，未打 tag 的预览使用已核实且可在官方仓库访问的完整目标 commit SHA。性能数字附上测量条件和来源。日常重构、格式化和 CI 维护通常留在完整变更链接中。

## 组织草稿

- 以版本号和一句发布摘要开篇，`Highlights` 优先写对训练用户影响最大的变化。
- 在 `## English` 和 `## 中文` 下，按本版实际交付的核心特性分组。每组直接用 H3 命名，例如 `### Models and supervised training` 与 `### 模型与监督训练`。核心依赖升级和关键修复也各用独立的 H3，省略空节和重复条目。
- 当已有用户升级时需要调整配置、命令或数据，在靠前的 `### Upgrade Guide` 和 `### 升级指南` 中说明受影响的用户和具体迁移步骤。新配方的运行条件写在对应特性下。其他需要发布者判断的事实（如尚未公开的已知限制）集中向用户询问。
- 在完整变更链接前加入 `Acknowledgements` 和「致谢」。结合本版的具体工作表达感谢，也写出问题反馈、复现验证、评审或经验分享等社区贡献。按本次发布区间的提交和已合入 PR 核实作者与共同作者，按 GitHub 账号去重并剔除已确认的自动化账号。两版共用按用户名排序的折叠名单，标题写为 `Contributors (N)` 和「贡献者（N）」。
- `Container Images` 按已确认的变体逐行给出平台和拉取命令或链接。结尾附官方 `https://github.com/redai-studio/Relax/compare/<previous>...<current>` 完整变更链接。首版链接目标版本的提交历史。

参考 Relax v0.1.0 的 Highlights 和版本化配方链接，以及 v0.2.0 草稿的双语特性分组、升级指南和镜像表。[Relax v0.1.0 发布说明](https://github.com/redai-studio/Relax/releases/tag/v0.1.0) · [Relax v0.2.0 双语草稿（展开 PR 描述中的示例）](https://github.com/redai-studio/Relax/pull/798)
