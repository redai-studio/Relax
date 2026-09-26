---
name: sglang-upgrade
description: Upgrade the sglang version used by Relax's training Docker image. Use when bumping/upgrading sglang (changing the base image tag), rebasing docker/patch/latest/sglang.patch onto a new sglang release, or porting Relax's sglang customizations forward to a new version. Covers the version-determination mechanism, using THUDM/slime as the patch-rebase reference, classifying private patches into upstreamed-vs-still-needed, and validating the rebased patch against the exact target source.
---

# 升级 Relax 的 sglang 版本

把 Relax 训练镜像依赖的 sglang 升级到新版本。最大的工作量永远是 **rebase `docker/patch/latest/sglang.patch`**;核心方法是 **以 THUDM/slime 已 rebase 好的同版本 patch 为骨架,只 port Relax 真正私有且未被上游吸收的改动**。

升级时建议维护一份 `docs/draft/sglang-<ver>-upgrade-plan.md` 决策日志(见第 8 节),
记录本次保留/丢弃的私有改动与验证清单,可作为下次升级的模板。

## 0. 先搞清楚版本是怎么定的(关键前提)

sglang 版本 **不在 `requirements.txt` / `pyproject.toml` 固定**,完全由 Docker 构建决定:

- `docker/Dockerfile:4` 的 `ARG BASE_IMAGE=lmsysorg/sglang:<tag>` —— **版本由此唯一确定**。
- 历史上可能有 `git clone -b <branch> ... && rsync` 的**源码 overlay**(例如曾用 `update-transformers-v5` 给老镜像 backport transformers 5.x)。新镜像原生支持后应**删除 overlay**。
- `docker/patch/latest/sglang.patch`(真实文件,非软链)通过 `git apply --3way` 打入。
- 运行时 `--sglang-*` 参数(`relax/utils/arguments.py`)。

注意:`rsync` overlay 只替换源码,不动镜像里 setuptools_scm 固化的 `.dist-info`,所以 `pip show sglang` 可能与实际源码版本不符。

## 1. 前置确认

```bash
# 目标镜像是否存在(注意 -cuXXX 后缀 = CUDA 版本,如 -cu129 = CUDA 12.9)
docker pull lmsysorg/sglang:<NEW_TAG>            # 或查 hub.docker.com
# 目标版本对 transformers 等的要求(决定 overlay 能否删、requirements 怎么改)
git show <NEW_TAG>:python/pyproject.toml | grep -iE "transformers==|huggingface_hub"
```

## 2. 找到 slime 的对应升级作参考(省 90% 工作量)

Relax 的 `docker/patch` 派生自 [THUDM/slime](https://github.com/THUDM/slime),slime 按版本维护 `docker/patch/v<X>/sglang.patch` 并已 rebase。本地仓库:`/root/data/slime`。

```bash
cd /root/data/slime
# 找哪个 slime tag 的 base sglang == 目标版本
for t in $(git tag | grep '^v0\.'); do
  echo "$t: $(git show $t:docker/Dockerfile 2>/dev/null | grep SGLANG_IMAGE_TAG= | head -1)"
done
# 看那个 tag 的 Dockerfile diff(基线、torch_memory_saver、Megatron、PyJWT 等怎么改的)
git diff <slime_old> <slime_new> -- docker/Dockerfile
# 取出 slime 已 rebase 的目标版本 patch 作骨架
git show <slime_new>:docker/patch/<NEW_TAG>/sglang.patch > /tmp/slime_skeleton.patch
```

判断 Relax 当前 patch 最接近哪个 slime 版本(差异最小的即 fork 起点),用
`scripts/classify_patch.sh` 对比(见第 4 节)。

## 3. 准备目标源码树(让 patch 可验证,别盲改)

```bash
cd /root/data/sglang                      # 用户的 sglang 仓库
git fetch --depth 1 origin tag <NEW_TAG>  # 需代理
git worktree add --detach /tmp/sgl_base <NEW_TAG>
# 验证 slime 骨架能干净应用(exit 0 即可;"lacks blob/Falling back" 是浅克隆正常现象)
cd /tmp/sgl_base && git apply --check /tmp/slime_skeleton.patch; echo "exit=$?"
```

## 4. 把 Relax 私有改动分类(A/B/C)

```bash
bash <skill_dir>/scripts/classify_patch.sh \
  /root/data/Relax/docker/patch/latest/sglang.patch \
  /tmp/slime_skeleton.patch        # 用 fork 起点那版 slime patch 对比更准
```

输出三类:
- **A — 改动内容与 slime 完全一致**:slime 骨架已含,**无需处理**。
- **B — Relax 在共有文件上有额外改动**:需 port Relax 增量。
- **C — Relax 独有文件**:需 port(注意有些只是上游文件改名/移动,如 `rotary_embedding.py`→`rotary_embedding/base.py`)。

## 5. 逐项判断:已上游 vs 仍需 port(本步最省事也最关键)

很多 Relax 私有补丁(NSA、DeepSeek V3.2、BF16 MoE 等活跃开发功能)在新版本里 **已被上游原生实现**。对每个 B/C 项,先 grep 目标源码确认是否已存在,**已存在就丢弃**:

```bash
cd /tmp/sgl_base
grep -rn "<关键符号/函数名>" python/sglang/srt/<file>
```

经验法则(本次实证):
- 模型层 NSA/topk/eagle/rope 等 → 多半已上游,**丢弃**。
- BF16 DeepEP MoE → 查 `moe_runner/deep_gemm.py` 是否有 `_run_bf16_*`,有则**丢弃**自定义 kernel。
- 权重热更新链路(`post_process_weights`、`_import_static_state` inference_mode、disagg 队列释放)→ slime 骨架通常已含,**复用**。
- 真·Relax 私有(其他框架不会有的):`SafeUnpickler` 的 `"Relax."` 白名单、colocate 相关的多节点/设备处理等 → **必须 port**。

**原则**:拿不准是否仍需要的,先按"保留 Relax 私有"port,**记录到决策日志**,等实跑实验验证后再决定删除(见第 8 节)。被上游重构掉的代码(如某函数已删)别硬移植——适配到新结构或丢弃。

## 6. Port 私有改动并生成 patch

直接在已打骨架的 `/tmp/sgl_base` 上用 Edit 改文件(对照 Relax 旧 patch 的 hunk,适配新源码结构),然后:

```bash
cd /tmp/sgl_base
# 语法检查每个改过的文件
python -c "import ast; ast.parse(open('python/sglang/srt/<file>').read())"
# 先应用骨架(若还没实应用),再叠加你的 Edit;最后整体 diff
git add -A && git diff --cached <NEW_TAG> -- python/ > /tmp/relax_new.patch
```

**验证(必做)**:在全新 worktree 上 `git apply --check`,exit 0 才算过:
```bash
cd /root/data/sglang && git worktree add --detach /tmp/sgl_verify <NEW_TAG>
cd /tmp/sgl_verify && git apply --check /tmp/relax_new.patch; echo "exit=$?"
cp /tmp/relax_new.patch /root/data/Relax/docker/patch/latest/sglang.patch
```

完事清理:`cd /root/data/sglang && git worktree remove --force /tmp/sgl_base /tmp/sgl_verify; git worktree prune`

## 7. Dockerfile / requirements 改动(对照 slime 的 Dockerfile diff)

`docker/Dockerfile`:
- `:4` `BASE_IMAGE` → 新 tag。
- 删除过时的源码 overlay(如 `update-transformers-v5` rsync 段)。
- CUDA 耦合项:若 `-cuXXX` 变了,复核 tilelang wheel 源、flash-attn/apex/TE 重编、cudnn 版本(注:slime 在 cu129 base 上 tilelang 仍用 cu128)。
- `torch_memory_saver`:跟进 slime 的 `TMS_CUDA_MAJOR` **自动探测**写法 `$(python -c 'import torch; print(torch.version.cuda.split(".")[0])')`;commit 取 redai-studio fork 的 HEAD(先确认比 slime 用的新,别降级)。
- `docker/Dockerfile.npu` 的 `git clone -b <tag>` 同步。

`requirements.txt`(属 CLAUDE.md "Ask First",改前确认):
- `transformers`:**不固定版本**(与 slime 一致),由基线镜像提供;否则 `pip install -r` 会把镜像自带的版本降级造成冲突。
- 复核 `sglang-router`、`huggingface_hub` 与新版兼容。
- `pip install --ignore-installed PyJWT`:slime 加的防御行(规避 distutils 装的 PyJWT 无法卸载报错)。**先不加,构建真撞到再补**。

## 8. 记录决策 + 验证清单

把以下写入 `docs/draft/sglang-<ver>-upgrade-plan.md`:
- **保留的 Relax 私有改动**(及适配说明)
- **丢弃的项 + 依据**(标注"已上游/已覆盖,待实跑验证")
- **必须实跑验证的项**(优先级):BF16 MoE 走上游路径、权重同步/colocate 切换、各私有功能模型、transformers 新版兼容。

完成标准:`git apply --check` exit 0 + 改动文件语法通过。镜像构建与实跑需 GPU 环境,属后续验证。

## 常见坑

- patch 行号(`@@`)偏移会污染"是否改过"的判断 —— 比对时只比真实 `+`/`-` 行(`classify_patch.sh` 已处理)。
- slime 用自建 `slimerl/sglang` 基线(含 slime 私有改动,如带 `'slime'` 断言的 router);Relax 用官方 `lmsysorg/sglang`,**slime 的私有项别照搬**。
- **"与 slime 一致"要看 slime *实际怎么用*,不只看 patch 里*支持*什么**。slime 的 patch 常同时提供多条控制入口(如 NSA indexer rope 的 `INDEXER_ROPE_NEOX_STYLE` env 和 `--disable-indexer-rope-neox-style` CLI flag),但 slime 的启动脚本(`scripts/run-*.sh`)只用其中一条。改 Relax 的控制方式前,先 `git grep` slime 脚本确认它真正用哪条,否则会"对齐"到一个 slime 自己都不用的入口。
- patch 里存在某功能 ≠ 该功能在 Relax 需要/启用。很多是惰性保留(默认关、Relax 没用到),如 `is_slime_profiling_enabled`(`SLIME_ENABLE_PROFILING` 默认 False)、`update_weight_delta_*`——沿用即可,别误删也别误以为生效。
- 删私有项前先确认 **Relax 源码/脚本是否真的引用**(`grep -rn relax/ scripts/`);没人用的(如 `disable_draft_cuda_graph`)才能放心跟 slime 一起删。
- 只在用户要求时才 `git commit`;依赖类改动先确认。
