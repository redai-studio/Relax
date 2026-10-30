# SGLang CUDA 12.9 / Hopper 构建流程

**先构建可复用的 CUDA / Torch 基础层，再针对目标 SGLang tag 构建 kernel、原生依赖和
Python / Rust 包，最后组装 SGLang 基镜像。Relax 通过 `BASE_IMAGE` 使用它。**

这里整理官方 **v0.5.20、v0.5.21** 的分层流程。附录包含构建所需 Dockerfile、适配脚本和
GPU 检查脚本，不需要另一个构建仓库。后续版本沿用此流程，但需要重新核对依赖、源码
接口和补丁；不能只修改版本号就认定兼容。

**已发布的 SGLang 基镜像：`ghcr.io/redai-studio/relaxrl:sglang-v0.5.21-cu129`**，
Relax 的 CUDA 12 Dockerfile 默认使用它，可直接拉取，无需先从源码编译。
Relax 公开运行镜像为 **`ghcr.io/redai-studio/relaxrl:latest`**。本文源码构建仍使用
`sglang:v0.5.21-cu129` 等本地输出名，通过 `BASE_IMAGE` 可切换为自行构建的镜像。
训练 wheel 的导出和复用见[第 6 节](#6-导出与复用训练-wheel-缓存)。

## 构建环境与版本

| 组件                | 本流程选择                              |
| ------------------- | --------------------------------------- |
| CPU / 系统          | Linux x86_64、Ubuntu 24.04              |
| GPU                 | Hopper；SM90 / SM90a                    |
| CUDA                | 12.9.1 devel 镜像                       |
| Python              | 3.12                                    |
| Torch / torchvision | 2.13.0 cu129 / 0.28.0 cu129             |
| NCCL                | 2.30.7                                  |
| cuDNN               | 由 Torch 约束固定；当前组合为 9.20.0.48 |
| Rust                | SGLang package 阶段使用 1.92.0          |

需要 Docker BuildKit、Git、Bash 和宿主机 Python 3；只有验证阶段需要 GPU 和 NVIDIA
Container Toolkit。构建会下载源码和 wheel，并编译 CUDA / Rust，需要充足的 CPU、内存和
磁盘空间。不要通过清理 Docker 缓存来重试失败的构建。

Torch 从[官方 cu129 索引](https://download.pytorch.org/whl/cu129/torch/)安装。
这套流程固定 Torch 2.13，不自动升级到 Torch 2.14 / CUDA 13。
基础层显式安装 NCCL 2.30.7，并调整已安装 Torch 的 NCCL 依赖声明和对应 RECORD；
**不修改 Torch 的 Python 代码或二进制**。该依赖调整在 `pin_nccl.py` 中独立列出，
所有原生扩展都必须针对这个基础层构建。

## 分层产物

| 阶段                               | 产物                                              | 升级 SGLang 时                           |
| ---------------------------------- | ------------------------------------------------- | ---------------------------------------- |
| `base`                             | CUDA、Python、Torch、NCCL、编译工具链镜像         | ABI 不变时复用                           |
| `deps`                             | 根据目标 pyproject 解析的 Python 依赖镜像         | 重新解析                                 |
| `kernel`                           | 目标源码中的 `sglang-kernel` wheel，含 Hopper FA3 | 按目标源码重建；FA3 可严格校验后单独复用 |
| `deepgemm` / `deepep` / `flashmla` | 固定源码版本的原生扩展 wheel                      | 固定值和 ABI 不变时复用                  |
| `package`                          | SGLang wheel，包含其 Rust 扩展                    | 按目标源码重建                           |
| `extensions` / `runtime`           | 组合依赖、源码、原生扩展的最终 SGLang 镜像        | 每个 SGLang 版本重新组装                 |

最终镜像保留 `/sgl-workspace/sglang`、独立的 `.git` 目录、editable 安装，以及 `git`、
`wget` 和工具链，便于下游应用源码补丁。runtime 直接复用 package 中的 Rust `.so`，
不会为了 editable 安装再编译一次相同扩展。

## 1. 准备构建文件与官方源码

在 Relax 仓库根目录执行。以下命令把本文附录中的文件提取到 `build/cu129-tools`：

````bash
python3 - <<'PY'
import re
from pathlib import Path

document = Path("CUDA-129.md").read_text()
destination = Path("build/cu129-tools")
pattern = r"<!-- cu129-file: ([\w.-]+) -->\s*\n```[^\n]*\n(.*?)\n```"
files = re.findall(pattern, document, re.S)
assert files, "没有找到附录构建文件"
destination.mkdir(parents=True, exist_ok=True)
for name, content in files:
    (destination / name).write_text(content + "\n")
print(f"已写入 {len(files)} 个文件：{destination.resolve()}")
PY

SGLANG_VERSION=0.5.21  # 首次构建 0.5.20 时改为 0.5.20
mkdir -p build/sources
git clone --depth=1 --branch "v${SGLANG_VERSION}" --single-branch \
  https://github.com/sgl-project/sglang.git "build/sources/sglang-${SGLANG_VERSION}"

export SOURCE_DIR="$(realpath "build/sources/sglang-${SGLANG_VERSION}")"
export BUILD_DIR="$PWD/build/sglang-${SGLANG_VERSION}-cu129"
export PACKAGE_VERSION="$SGLANG_VERSION"
export BASE_IMAGE=sglang-cu129:torch2.13.0-nccl2.30.7-py312-r1
export IMAGE="sglang:v${SGLANG_VERSION}-cu129"
export TORCH_INDEX_URL=https://download.pytorch.org/whl/cu129
export PYPI_INDEX_URL=https://pypi.org/simple

git -C "$SOURCE_DIR" describe --tags --exact-match
git -C "$SOURCE_DIR" rev-parse HEAD
test -d "$SOURCE_DIR/.git"
````

必须使用官方目标 tag，而不是名字相似的开发分支。包版本由精确 tag 推导，显式
`PACKAGE_VERSION` 与 tag 不一致时会失败。源码需要独立 `.git` **目录**，不能把指向
宿主机其他位置的 worktree `.git` 文件直接复制进镜像。

| 官方 tag | 源码 commit                                |
| -------- | ------------------------------------------ |
| v0.5.20  | `94602c9c2b7cbdb8efd5c52802dac6a1c180089e` |
| v0.5.21  | `e00930c5489053f26d86b179cee0d087f846acbb` |

## 2. 首次构建基础层

```bash
bash build/cu129-tools/build.sh base
```

得到 `$BASE_IMAGE` 和 `${BASE_IMAGE}-compat`。后续版本继续使用同一基础层时跳过此步；
已经有满足版本和 ABI 要求的基础层，也可以直接将 `BASE_IMAGE` 设为它的镜像名。
不要把已经包含其他 SGLang 开发版本的成品误当成纯 CUDA / Torch 基础层。

`compat` 变体提供 CUDA forward compatibility 库路径和保守的 NCCL 配置。
能否在较旧驱动上运行仍取决于 NVIDIA 的驱动、GPU 和兼容库支持，编译成功不等于运行通过。

## 3. 构建目标 SGLang 镜像

```bash
for stage in deps kernel deepgemm deepep flashmla package extensions runtime; do
  bash build/cu129-tools/build.sh "$stage" || exit 1
done
```

- 最终镜像：`$IMAGE`。
- 每个阶段的日志：`$BUILD_DIR/logs/<stage>.log`。
- 导出的 SGLang / kernel / 原生依赖 wheel：`$BUILD_DIR/wheelhouse/`。

失败后重跑对应阶段即可；修好前置阶段后，再运行依赖它的后续阶段。`all` 会依次执行
基础层、所有构建和 GPU 验证；复用基础层时用上面的阶段列表即可。

CUDA 12.9 的适配集中在附录脚本中：

- 从目标 pyproject 生成 CUDA 12 依赖，使用 cu12 的 FlashInfer、NVSHMEM、CUTLASS DSL。
- kernel 仅编译 SM90 / SM90a，并一起裁掉 FlashMLA API 中的 SM100 分派，避免未定义符号。
- DeepEP 实现和打包目录分别固定源码 commit；打包代码不替换选定版本的实现。
- cuDNN frontend 1.31.0 的适配先校验上游 wheel SHA256，仅修改版本和 CUDA 依赖 metadata，
  保留 payload 并重新生成 RECORD。
- package 构建完整 Rust 扩展；runtime 复用这些 `.so`，再安装目标版本的 editable 源码。
- torchaudio 在 runtime 最后一层选择 `+cu129` wheel，避免从默认索引装入 CUDA 13 变体。

## 4. 复用与升级

升级时重新准备目标官方 tag，设置新的 `SOURCE_DIR`、`BUILD_DIR`、`PACKAGE_VERSION`
和 `IMAGE`，继续使用同一 `BASE_IMAGE`。

先检查目标版本的 `python/pyproject.toml`、kernel CMake 和官方构建工作流，再决定是否更新
`sources.env` 中的原生扩展 commit / 包版本。v0.5.20 和 v0.5.21 使用附录中的固定值；
后续版本不应直接沿用而不核对。

重跑第 3 步。未变化的原生组件会命中 Docker 缓存；SGLang wheel、Rust 扩展和 runtime
跟随新源码更新。Torch / CUDA / NCCL / Python 或编译 ABI 变化时，先重建基础层和所有原生
扩展，再导出新的 Relax 训练 wheel 缓存。

### 单独复用 FA3

有同 ABI 的既有 kernel builder 镜像时，可在 kernel 阶段显式指定它：

```bash
PREBUILT_FA3_IMAGE=sglang-cu129:kernel-94602c9c2b7c-torch213 \
  bash build/cu129-tools/build.sh kernel
```

这里只复用 `flash_ops.abi3.so`。`reuse_fa3.py` 比较 FA3 / CUTLASS 固定源码、接口、构建块和
完整 Torch / CUDA / NCCL 约束；全部相同才会装入目标版本的新 kernel wheel，并重建 RECORD。
其余 kernel 和版本 metadata 来自目标源码。校验失败时去掉 `PREBUILT_FA3_IMAGE`，完整重编。

## 5. 验证与交给 Relax

```bash
GPU_DEVICE=0 bash build/cu129-tools/build.sh verify
```

验证包含 Triton、RMSNorm、FA3、FlashInfer、DeepGEMM、FlashMLA GPU 运算，以及离线 tiny
Llama 生成；FA3 / FlashInfer 的生成 token 与 Transformers 参考结果逐个比较。

下面的 Relax 示例使用已发布的 **0.5.21 CUDA 12.9 / Hopper 基镜像**，
与当前 Dockerfile 的源码补丁和 Rust 绑定一致。
0.5.20 的 SGLang 基镜像可按前文独立构建；接入 Relax 时需使用与其配套的 Dockerfile 和 patch。
在 Relax 根目录执行：

```bash
docker pull ghcr.io/redai-studio/relaxrl:sglang-v0.5.21-cu129

TRAIN_IMAGE=ghcr.io/redai-studio/relaxrl:train-sgl0521-cu129
DEV_IMAGE=ghcr.io/redai-studio/relaxrl:dev-sgl0521-cu129

docker build --progress=plain -f docker/Dockerfile --target train \
  --build-arg BASE_IMAGE=ghcr.io/redai-studio/relaxrl:sglang-v0.5.21-cu129 -t "$TRAIN_IMAGE" .

docker build --progress=plain -f docker/Dockerfile --target relax \
  --build-arg BASE_IMAGE=ghcr.io/redai-studio/relaxrl:sglang-v0.5.21-cu129 \
  --build-arg TRAIN_IMAGE="$TRAIN_IMAGE" -t "$DEV_IMAGE" .
```

Relax 的 SGLang patch 必须与目标版本一致，例如 v0.5.20 使用 `v0.5.20.patch`，
v0.5.21 使用 `v0.5.21.patch`；构建前核对 `PATCH_VERSION` 和对应 patch 链接。
升级后仍需联合验证 rollout、前后向训练、权重同步、显存恢复和 checkpoint 保存。
训练 wheel 缓存只跳过训练扩展的编译，不代替这些验证。

## 6. 导出与复用训练 wheel 缓存

**先完整构建一次 `train` 镜像，再从该镜像导出缓存。后续构建传入
`PREBUILD_CACHE_TAR_URL` 即可安装缓存 wheel；不传则按 Dockerfile 原有流程编译。**
本节只适用于 CUDA 12.9 / Hopper，不用于 CUDA 13 镜像。

### 首次构建与导出

在仓库根目录执行。`BASE_IMAGE` 替换为自己准备的 SGLang CUDA 12.9 / Hopper
镜像；下面的本地镜像名与 `CUDA-129.md` 中的示例一致。

```bash
BASE_IMAGE=ghcr.io/redai-studio/relaxrl:sglang-v0.5.21-cu129
TRAIN_IMAGE=ghcr.io/redai-studio/relaxrl:train-sgl0521-cu129

docker build --progress=plain -f docker/Dockerfile --target train \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  -t "$TRAIN_IMAGE" .

bash docker/scripts/export_train_cache.sh \
  "$TRAIN_IMAGE" ./build-cache/cu129-hopper-train-wheels.tar

tar -tf ./build-cache/cu129-hopper-train-wheels.tar
```

导出脚本创建临时容器，通过 `docker cp` 调用
[`export_train_wheels.py`](docker/scripts/export_train_wheels.py)，导出后校验 ABI 和 wheel
SHA256，再生成 tar。无需 GPU，也无需宿主机安装 Python 包；输出文件已存在时会报错，
临时容器和目录在退出时清理。

tar 根目录包含 `manifest.json` 和以下 11 个包的 wheel：

| 包                                                                          | 用途                                 |
| --------------------------------------------------------------------------- | ------------------------------------ |
| `flash-attn`、`flash-attn-3`                                                | FA2 / Hopper FA3                     |
| `fast-hadamard-transform`                                                   | Hadamard 变换                        |
| `flash-mla`                                                                 | MLA 算子                             |
| `transformer-engine`、`transformer-engine-torch`、`transformer-engine-cu12` | TE Python 接口、Torch 扩展和 CUDA 库 |
| `torch-memory-saver`                                                        | 显存暂停、恢复                       |
| `apex`                                                                      | 融合优化器等训练扩展                 |
| `fake-int4-quant-cuda`                                                      | INT4 QAT CUDA 扩展                   |
| `nvidia-modelopt`                                                           | 量化组件                             |

使用 **train 阶段**作为导出源。dev 阶段可能根据运行时 requirements 调整包版本，
例如覆盖 ModelOpt；导出 dev 环境会把这些覆盖后的版本也写入缓存。

### 使用缓存构建 train / dev

将 tar 放到构建机器可下载的 HTTP(S) 地址。存储方式自行选择；下面的地址仅为占位示例。

```bash
BASE_IMAGE=ghcr.io/redai-studio/relaxrl:sglang-v0.5.21-cu129
TRAIN_IMAGE=ghcr.io/redai-studio/relaxrl:train-sgl0521-cu129
DEV_IMAGE=ghcr.io/redai-studio/relaxrl:dev-sgl0521-cu129
CACHE_URL=https://downloads.example.com/cache/cu129-hopper-train-wheels.tar

docker build --progress=plain -f docker/Dockerfile --target train \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  --build-arg PREBUILD_CACHE_TAR_URL="$CACHE_URL" \
  -t "$TRAIN_IMAGE" .

docker build --progress=plain -f docker/Dockerfile --target relax \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  --build-arg TRAIN_IMAGE="$TRAIN_IMAGE" \
  -t "$DEV_IMAGE" .
```

上述 train/dev tag 是本地构建的输出示例，不表示已经发布。公开镜像的日常使用入口为
`ghcr.io/redai-studio/relaxrl:latest`；复现特定构建时应固定实际发布的 tag 或 digest。

第二步直接使用第一步的 train 成品，不再重复构建训练扩展。若不通过 `TRAIN_IMAGE`
指定成品、而让 dev 构建内部 train 阶段，则也要传入 `PREBUILD_CACHE_TAR_URL`。
CI 或 Make 包装命令需要把该值显式转发为 Docker build argument；仅设置同名 shell
环境变量不会自动传给 Dockerfile。

缓存下载失败、ABI 不匹配或 wheel 校验失败会中止构建，不会静默改为源码编译。
需要恢复源码编译时，去掉 `--build-arg PREBUILD_CACHE_TAR_URL=...` 后重新运行即可。

### 复用范围与升级

`manifest.json` 记录 Torch 完整版本、CUDA、NCCL、Python 主次版本、CPU 架构、
C++11 ABI，以及各 wheel 的版本和 SHA256。构建时要求这些 ABI 字段与目标环境一致。

- 当前缓存面向 CUDA 12.9、Python 3.12、Hopper。换 Torch / CUDA / NCCL / Python
  或 GPU 架构时，重新编译并导出，不能沿用旧缓存。
- SGLang 从 0.5.20 升级到 0.5.21 时，只要训练扩展的源码、版本、编译选项和上述 ABI
  均未改变，就可以复用同一份训练 wheel tar。SGLang 自身的 kernel、Rust 扩展和源码补丁
  按目标版本单独处理。
- ABI 校验不检查 GPU SM、扩展源码或编译选项。修改 FA、FlashMLA、INT4 等扩展的源码或
  Dockerfile 中的编译版本后，必须重新导出，不能仅凭 ABI 校验通过就复用旧包。
- tar 只保存清单中的 wheel。其余 Python 依赖仍需安装；ModelOpt 等组件首次运行时的
  JIT 缓存、dev 阶段的 Router / HybridEP / SGLang Rust 编译产物不包含在内。
- 使用不可变的缓存文件名。更新内容时换一个 URL，避免 Docker 对相同 URL 复用旧层。

### 构建后的检查

```bash
docker run --rm --entrypoint python "$TRAIN_IMAGE" -m pip check
docker run --rm --entrypoint python "$DEV_IMAGE" -m pip check
docker run --rm --gpus all --entrypoint python "$DEV_IMAGE" -c \
  'import torch, sglang; print(torch.__version__, torch.version.cuda, sglang.__version__); print(torch.cuda.get_device_name(0))'
```

导入成功不能替代 GPU 验证。首次导出或更换 ABI 后，至少运行 FA2/FA3 前后向、
Hadamard、FlashMLA、TE、Apex 更新、INT4、ModelOpt 量化，以及 Memory Saver 的实际
暂停/恢复，再验证一次 Relax rollout、训练、权重同步和 checkpoint 保存。

## 附录：完整构建文件

下列文件由第 1 步提取到 `build/cu129-tools`。适配脚本对预期源码结构做检查；目标版本
改变了结构时应检查报错并更新适配，不能放宽检查后继续发布。

<details>
<summary><code>sources.env</code></summary>

<!-- cu129-file: sources.env -->

```bash
# Source commits and package versions for the official SGLang v0.5.20 tag.
# Update each pair against the selected SGLang revision's dependencies.
DEEPGEMM_REF=7fec51c2eeb29b755a27d86b851c9996b6b5dca5
DEEPGEMM_VERSION=0.2.0

DEEPEP_REF=8fe7c01e2bc1026be37eea496e5b33969b945ea8
DEEPEP_VERSION=0.1.2
DEEPEP_PACKAGING_REF=fde9a813271be9b4e66251a8dbf8f26fd274b4f4

FLASHMLA_REF=5ddc685e5ed5bee5934657ada3513678b012ef66
FLASHMLA_VERSION=1.0.0
```

</details>

<details>
<summary><code>build.sh</code></summary>

<!-- cu129-file: build.sh -->

```bash
#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_dir="${SOURCE_DIR:-$(cd "${script_dir}/../.." && pwd)}"
build_dir="${BUILD_DIR:-$(pwd)/build/sglang-cu129}"
stage="${1:-all}"
if [[ ! -f "${source_dir}/python/pyproject.toml" ]]; then
    printf 'Set SOURCE_DIR to an SGLang checkout (missing python/pyproject.toml): %s\n' "${source_dir}" >&2
    exit 1
fi
source "${script_dir}/sources.env"
mkdir -p "${build_dir}"/{logs,sources,wheelhouse}

commit="$(git -C "${source_dir}" rev-parse HEAD)"
short_commit="${commit:0:12}"
source_tag="$(git -C "${source_dir}" describe --tags --exact-match 2>/dev/null || true)"
package_version="${PACKAGE_VERSION:-${source_tag#v}}"
if [[ -z "${package_version}" ]]; then
    echo "Use a release tag checkout, or explicitly set PACKAGE_VERSION for a development build." >&2
    exit 1
fi
if [[ -n "${source_tag}" && "${package_version%%+*}" != "${source_tag#v}" ]]; then
    printf 'Version mismatch: source tag=%s, PACKAGE_VERSION=%s\n' "${source_tag}" "${package_version}" >&2
    exit 1
fi
printf 'Source: %s @ %s; package version: %s\n' "${source_tag:-development}" "${commit}" "${package_version}"
base_image="${BASE_IMAGE:-sglang-cu129:torch2.13.0-nccl2.30.7-py312-r1}"
dependency_key="$({ sha256sum "${source_dir}/python/pyproject.toml" "${script_dir}/adapt.py" | awk '{print $1}'; printf '%s\n' "${base_image}"; } | sha256sum | cut -c1-12)"
deps_image="${DEPS_IMAGE:-sglang-cu129:deps-${dependency_key}}"
kernel_image="sglang-cu129:kernel-${short_commit}-torch213"
deepgemm_image="sglang-cu129:deepgemm-${DEEPGEMM_REF:0:12}-torch213"
deepep_image="sglang-cu129:deepep-${DEEPEP_REF:0:12}-torch213-nccl2307"
flashmla_image="sglang-cu129:flashmla-${FLASHMLA_REF:0:12}-torch213"
extensions_image="${EXTENSIONS_IMAGE:-sglang-cu129:extensions-${short_commit}-torch213}"
package_image="sglang-cu129:package-${short_commit}-torch213"
runtime_image="${IMAGE:-sglang-cu129:v${package_version}-torch213-hopper-compat}"

build() {
    local name="$1"
    shift
    printf 'Building %s; log: %s/logs/%s.log\n' "${name}" "${build_dir}" "${name}"
    docker build --progress=plain --build-context "cu129-build=${script_dir}" \
        --build-arg http_proxy --build-arg https_proxy \
        --build-arg no_proxy --build-arg TORCH_INDEX_URL --build-arg PYPI_INDEX_URL \
        "$@" "${source_dir}" >"${build_dir}/logs/${name}.log" 2>&1 || {
        tail -80 "${build_dir}/logs/${name}.log" >&2
        return 1
    }
}

fetch_source() {
    local name="$1" ref="$2" path="${build_dir}/sources/$1" repo="${3:-$1}"
    if [[ ! -d "${path}/.git" ]]; then
        git init "${path}" >/dev/null
        git -C "${path}" remote add origin "https://github.com/sgl-project/${repo}.git"
    fi
    if [[ "$(git -C "${path}" rev-parse HEAD 2>/dev/null || true)" != "${ref}" ]]; then
        git -C "${path}" fetch --depth=1 origin "${ref}"
        git -C "${path}" checkout --detach FETCH_HEAD
    fi
    git -C "${path}" submodule update --init --recursive --depth=1
}

export_wheels() {
    local image="$1" name="$2" container
    container="$(docker create "${image}" /bin/true)"
    docker cp "${container}:/wheelhouse/." "${build_dir}/wheelhouse/"
    docker rm "${container}" >/dev/null
    printf '%s\n' "${image}" >"${build_dir}/${name}.image"
}

build_stage() {
    case "$1" in
        base)
            build base -f "${script_dir}/Dockerfile.base" --target torch \
                --build-arg TORCH_BUILD="${TORCH_BUILD:-2.13.0+cu129}" -t "${base_image}"
            build base-compat -f "${script_dir}/Dockerfile.base" --target compat \
                --build-arg TORCH_BUILD="${TORCH_BUILD:-2.13.0+cu129}" -t "${base_image}-compat"
            ;;
        deps)
            build deps -f "${script_dir}/Dockerfile.deps" --build-arg BASE_IMAGE="${base_image}" -t "${deps_image}"
            ;;
        kernel)
            local kernel_target=builder
            local -a kernel_context=()
            if [[ -n "${PREBUILT_FA3_IMAGE:-}" ]]; then
                kernel_target=cached-fa3
                kernel_context+=(--build-arg BUILD_FA3=OFF --build-context "prebuilt-fa3=docker-image://${PREBUILT_FA3_IMAGE}")
            fi
            build kernel -f "${script_dir}/Dockerfile.kernel" --target "${kernel_target}" \
                "${kernel_context[@]}" \
                --build-arg BASE_IMAGE="${base_image}" --build-arg KERNEL_CACHE_KEY="${short_commit}-torch213" -t "${kernel_image}"
            export_wheels "${kernel_image}" kernel
            ;;
        package)
            build package -f "${script_dir}/Dockerfile.package" --target builder \
                --build-arg BASE_IMAGE="${deps_image}" --build-arg SGLANG_VERSION="${package_version}" -t "${package_image}"
            export_wheels "${package_image}" package
            ;;
        deepgemm|deepep|flashmla)
            local component="$1" repo ref image version_arg
            local -a extra_context=()
            case "${component}" in
                deepgemm)
                    repo=DeepGEMM; ref="${DEEPGEMM_REF}"; image="${deepgemm_image}"
                    version_arg="DEEPGEMM_VERSION=${DEEPGEMM_VERSION}"
                    ;;
                deepep)
                    repo=DeepEP; ref="${DEEPEP_REF}"; image="${deepep_image}"
                    version_arg="DEEPEP_VERSION=${DEEPEP_VERSION}"
                    ;;
                flashmla)
                    repo=FlashMLA; ref="${FLASHMLA_REF}"; image="${flashmla_image}"
                    version_arg="FLASHMLA_VERSION=${FLASHMLA_VERSION}"
                    ;;
            esac
            fetch_source "${repo}" "${ref}"
            if [[ "${component}" == deepep ]]; then
                fetch_source DeepEP-packaging "${DEEPEP_PACKAGING_REF}" DeepEP
                extra_context+=(--build-context "deepep-packaging-source=${build_dir}/sources/DeepEP-packaging")
            fi
            build "${component}" -f "${script_dir}/Dockerfile.native" --target "${component}" \
                --build-arg BASE_IMAGE="${base_image}" \
                --build-arg "${version_arg}" \
                "${extra_context[@]}" \
                --build-context "${component}-source=${build_dir}/sources/${repo}" -t "${image}"
            export_wheels "${image}" "${component}"
            ;;
        extensions|runtime)
            local target image
            if [[ "$1" == extensions ]]; then
                target=extensions; image="${extensions_image}"
            else
                target=compat; image="${runtime_image}"
            fi
            build "$1" -f "${script_dir}/Dockerfile.runtime" --target "${target}" \
                --build-arg BASE_IMAGE="${deps_image}" \
                --build-arg SOURCE_COMMIT="${commit}" --build-arg SGLANG_VERSION="${package_version}" \
                --build-context "kernel-image=docker-image://${kernel_image}" \
                --build-context "deepgemm-image=docker-image://${deepgemm_image}" \
                --build-context "deepep-image=docker-image://${deepep_image}" \
                --build-context "flashmla-image=docker-image://${flashmla_image}" \
                --build-context "sglang-image=docker-image://${package_image}" -t "${image}"
            printf '%s\n' "${image}" >"${build_dir}/$1.image"
            ;;
        verify)
            docker run --rm --gpus "device=${GPU_DEVICE:-0}" --shm-size=2g \
                --env LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
                "${runtime_image}" python -u /opt/cu129/verify.py 2>&1 | tee "${build_dir}/logs/verify.log"
            for backend in fa3 flashinfer; do
                docker run --rm --gpus "device=${GPU_DEVICE:-0}" --shm-size=2g \
                    --env LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
                    "${runtime_image}" python -u /opt/cu129/smoke_inference.py \
                    --attention-backend "${backend}" 2>&1 | tee "${build_dir}/logs/inference-${backend}.log"
            done
            ;;
        *) echo "Usage: $0 [all|base|deps|kernel|deepgemm|deepep|flashmla|package|extensions|runtime|verify]" >&2; return 2 ;;
    esac
}

if [[ "${stage}" == all ]]; then
    for part in base deps kernel deepgemm deepep flashmla package extensions runtime verify; do
        build_stage "${part}"
    done
else
    build_stage "${stage}"
fi
```

</details>

<details>
<summary><code>Dockerfile.base</code></summary>

<!-- cu129-file: Dockerfile.base -->

```dockerfile
# syntax=docker/dockerfile:1
ARG CUDA_IMAGE=nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04@sha256:a2e1e2360c85298ac47ec2543b406ab1e8cec42e31ee47e4d32140ebc82e1067
FROM ${CUDA_IMAGE} AS toolchain

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/root/.cargo/bin:/usr/local/cuda/bin:${PATH} \
    TORCH_CUDA_ARCH_LIST=9.0a \
    CMAKE_BUILD_PARALLEL_LEVEL=16 \
    MAX_JOBS=16

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
        python3.12-dev python3.12-venv build-essential git curl ca-certificates \
        pkg-config ccache libssl-dev libffi-dev libelf-dev libdw-dev \
        libibverbs-dev librdmacm-dev libfabric-dev libnuma-dev \
        libopenmpi-dev openmpi-bin ibverbs-providers rdma-core \
        protobuf-compiler libprotobuf-dev ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && python3.12 -m venv /opt/venv \
    && ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --index-url https://pypi.org/simple \
        'pip>=25' 'setuptools>=77' wheel \
        'cmake>=3.31,<4' ninja scikit-build-core build packaging \
        setuptools-rust setuptools-scm numpy uv

ARG RUST_VERSION=1.90.0
RUN curl --fail --silent --show-error --retry 3 https://sh.rustup.rs -o /tmp/rustup.sh \
    && sh /tmp/rustup.sh -y --profile minimal --default-toolchain ${RUST_VERSION} \
    && rm /tmp/rustup.sh

FROM toolchain AS torch
ARG TORCH_VERSION=2.13.0
ARG TORCHVISION_VERSION=0.28.0
ARG TORCH_BUILD=${TORCH_VERSION}+cu129
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu129
ARG PYPI_INDEX_URL=https://pypi.org/simple
COPY --from=cu129-build pin_nccl.py /opt/cu129/pin_nccl.py
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python \
        torch==${TORCH_BUILD} torchvision==${TORCHVISION_VERSION}+cu129 \
        --index-url ${TORCH_INDEX_URL} \
    && python /opt/cu129/pin_nccl.py \
    && python -m pip install --no-deps --index-url ${PYPI_INDEX_URL} nvidia-nccl-cu12==2.30.7 \
    && python -m pip check \
    && mkdir -p /opt/cu129 \
    && python -m pip freeze | grep -E '^(torch|torchvision|triton|nvidia-)' > /opt/cu129/torch-constraints.txt \
    && python -c 'import torch; assert torch.version.cuda == "12.9"; print(torch.__version__, torch.version.cuda)'

LABEL org.opencontainers.image.description="CUDA 12.9.1 / Python 3.12 / PyTorch cu129 build base for Hopper; independent of SGLang"

# Kept separate because newer hosts can use the torch image without this layer.
FROM torch AS compat
ENV LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
    NCCL_NVLS_ENABLE=0 \
    NCCL_CUMEM_HOST_ENABLE=0
```

</details>

<details>
<summary><code>pin_nccl.py</code></summary>

<!-- cu129-file: pin_nccl.py -->

```python
"""Pin the installed Torch NCCL dependency without changing its binaries."""

import base64
import csv
import hashlib
import importlib.metadata as md
import io
import re
from pathlib import Path


dist = md.distribution("torch")
metadata = next(
    Path(dist.locate_file(p))
    for p in dist.files
    if str(p).endswith(".dist-info/METADATA")
)
text, count = re.subn(
    r"(?m)^(Requires-Dist: nvidia-nccl-cu12)[^;\n]*(;[^\n]*)?$",
    lambda match: match[1] + "==2.30.7" + (match[2] or ""),
    metadata.read_text(),
)
assert count == 1, f"Expected one Torch NCCL requirement, found {count}"
metadata.write_text(text)
payload = metadata.read_bytes()
sha = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).decode().rstrip("=")
record = metadata.with_name("RECORD")
rows = list(csv.reader(io.StringIO(record.read_text())))
matched = 0
for row in rows:
    if Path(dist.locate_file(row[0])) == metadata:
        row[1:] = ["sha256=" + sha, str(len(payload))]
        matched += 1
assert matched == 1, "Torch metadata is missing from RECORD"
with record.open("w", newline="") as stream:
    csv.writer(stream).writerows(rows)
print("Pinned Torch metadata to NCCL 2.30.7; Torch payload is unchanged")
```

</details>

<details>
<summary><code>Dockerfile.deps</code></summary>

<!-- cu129-file: Dockerfile.deps -->

```dockerfile
# syntax=docker/dockerfile:1
ARG BASE_IMAGE=sglang-cu129:torch2.13.0-py312-r1
FROM ${BASE_IMAGE} AS cudnn-frontend
ARG PYPI_INDEX_URL=https://pypi.org/simple
COPY --from=cu129-build adapt_cudnn.py /opt/cu129/adapt_cudnn.py
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip download --no-deps --index-url ${PYPI_INDEX_URL} \
        'nvidia-cudnn-frontend===1.31.0' --dest /tmp/frontend \
    && python /opt/cu129/adapt_cudnn.py /tmp/frontend /wheelhouse

FROM ${BASE_IMAGE}
ARG PYPI_INDEX_URL=https://pypi.org/simple
COPY --from=cudnn-frontend /wheelhouse /opt/cu129/wheelhouse/
COPY --from=cu129-build adapt.py /opt/cu129/adapt.py
COPY python/pyproject.toml /tmp/spec/python/pyproject.toml
RUN python /opt/cu129/adapt.py /tmp/spec \
    && cp /tmp/spec/requirements-cu129.txt /opt/cu129/requirements.txt \
    && cp /tmp/spec/extensions-cu129.json /opt/cu129/extensions.json
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python --index-url ${PYPI_INDEX_URL} \
        --find-links /opt/cu129/wheelhouse \
        --constraint /opt/cu129/torch-constraints.txt \
        --requirement /opt/cu129/requirements.txt \
    && python -m pip freeze > /opt/cu129/dependencies.lock \
    && python -m pip check \
    && python -c 'import importlib.metadata as m; cu13 = [d.metadata["Name"] for d in m.distributions() if d.metadata["Name"].lower().endswith(("-cu13", "_cu13"))]; assert not cu13, cu13'
```

</details>

<details>
<summary><code>Dockerfile.kernel</code></summary>

<!-- cu129-file: Dockerfile.kernel -->

```dockerfile
# syntax=docker/dockerfile:1
ARG BASE_IMAGE=sglang-cu129:torch2.13.0-py312-r1
FROM ${BASE_IMAGE} AS builder
COPY --from=cu129-build adapt.py flashmla-hopper.patch /opt/cu129/
COPY python/pyproject.toml /build/sglang/python/pyproject.toml
COPY python/sglang/kernels/aot /build/sglang/python/sglang/kernels/aot
RUN python /opt/cu129/adapt.py /build/sglang --kernel
WORKDIR /build/sglang/python/sglang/kernels/aot
ARG BUILD_FA3=ON
ENV CMAKE_ARGS="-DENABLE_BELOW_SM90=OFF -DSGL_KERNEL_ENABLE_FA3=${BUILD_FA3} -DSGL_KERNEL_COMPILE_THREADS=2" \
    CCACHE_DIR=/root/.cache/ccache \
    CMAKE_BUILD_PARALLEL_LEVEL=96
ARG KERNEL_CACHE_KEY=cu129
# SM90a also applies to common_ops; retain it when reusing a prebuilt FA3.
RUN if [ "${BUILD_FA3}" = OFF ]; then \
      sed -i 's/if ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "12.4" AND SGL_KERNEL_ENABLE_FA3)/if ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "12.4")/' CMakeLists.txt; \
    fi
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,id=sglang-cu129-kernel-${KERNEL_CACHE_KEY},target=/build/kernel \
    python -m pip wheel . --config-settings=build-dir=/build/kernel --no-build-isolation --no-deps --wheel-dir /wheelhouse

FROM builder AS cached-fa3
COPY --from=prebuilt-fa3 /build/sglang/python/sglang/kernels/aot/ /tmp/fa3-cache/source/
COPY --from=prebuilt-fa3 /opt/cu129/torch-constraints.txt /tmp/fa3-cache/torch-constraints.txt
COPY --from=prebuilt-fa3 /wheelhouse/ /tmp/fa3-cache/wheelhouse/
COPY --from=cu129-build reuse_fa3.py /opt/cu129/reuse_fa3.py
RUN python /opt/cu129/reuse_fa3.py

FROM scratch AS wheels
COPY --from=builder /wheelhouse /
```

</details>

<details>
<summary><code>Dockerfile.native</code></summary>

<!-- cu129-file: Dockerfile.native -->

```dockerfile
# syntax=docker/dockerfile:1
ARG BASE_IMAGE=sglang-cu129:torch2.13.0-py312-r1
FROM ${BASE_IMAGE} AS native-base
ARG PYPI_INDEX_URL=https://pypi.org/simple
COPY --from=cu129-build adapt_native.py /opt/cu129/adapt_native.py
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python --index-url ${PYPI_INDEX_URL} \
        --constraint /opt/cu129/torch-constraints.txt \
        apache-tvm-ffi==0.1.11 cuda-python==12.9.0 nvshmem4py-cu12==0.4.1

FROM native-base AS deepgemm
COPY --from=deepgemm-source / /build/DeepGEMM
WORKDIR /build/DeepGEMM
ARG DEEPGEMM_VERSION
RUN printf '%s+cu129' "${DEEPGEMM_VERSION}" > sgl_deep_gemm/VERSION \
    && bash build_sgl_deep_gemm.sh \
    && mkdir /wheelhouse && cp dist/*.whl /wheelhouse/

FROM native-base AS deepep
COPY --from=deepep-source / /build/DeepEP
COPY --from=deepep-packaging-source /sgl_deep_ep /build/DeepEP/sgl_deep_ep
WORKDIR /build/DeepEP
ARG DEEPEP_VERSION
RUN --mount=type=cache,target=/root/.cache/ccache \
    python /opt/cu129/adapt_native.py deepep /build/DeepEP \
    && SGL_DEEP_EP_VERSION=${DEEPEP_VERSION} bash sgl_deep_ep/build_sgl_deep_ep.sh /wheelhouse 12.9 x86_64

FROM native-base AS flashmla
COPY --from=flashmla-source / /build/FlashMLA
WORKDIR /build/FlashMLA
ARG FLASHMLA_VERSION
ENV NVCC_THREADS=2
RUN --mount=type=cache,target=/root/.cache/ccache \
    python /opt/cu129/adapt_native.py flashmla /build/FlashMLA \
    && printf '%s+cu129' "${FLASHMLA_VERSION}" > flash_mla/VERSION \
    && FLASH_MLA_SKIP_REG_SPILL_CHECK=1 python -m pip wheel . --no-build-isolation --no-deps --wheel-dir /wheelhouse
```

</details>

<details>
<summary><code>Dockerfile.package</code></summary>

<!-- cu129-file: Dockerfile.package -->

```dockerfile
# syntax=docker/dockerfile:1
ARG BASE_IMAGE
FROM ${BASE_IMAGE} AS rust-toolchain
ARG RUST_VERSION=1.92.0
RUN rustup toolchain install ${RUST_VERSION} --profile minimal \
    && rustup default ${RUST_VERSION}

FROM rust-toolchain AS builder
COPY python /build/sglang/python
COPY rust /build/sglang/rust
COPY proto /build/sglang/proto
COPY --from=cu129-build adapt.py /opt/cu129/adapt.py
ARG SGLANG_VERSION
ENV CARGO_BUILD_JOBS=32
RUN python /opt/cu129/adapt.py /build/sglang
RUN --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/root/.cargo/git \
    --mount=type=cache,id=sglang-cu129-rust-torch213,target=/build/sglang/rust/target \
    --mount=type=cache,id=sglang-cu129-radix-torch213,target=/build/sglang/rust/sglang-radix-tree/target \
    SETUPTOOLS_SCM_PRETEND_VERSION=${SGLANG_VERSION} \
    python -m pip wheel --verbose /build/sglang/python --no-build-isolation --no-deps --wheel-dir /wheelhouse
```

</details>

<details>
<summary><code>Dockerfile.runtime</code></summary>

<!-- cu129-file: Dockerfile.runtime -->

```dockerfile
# syntax=docker/dockerfile:1
ARG BASE_IMAGE
FROM ${BASE_IMAGE} AS extensions
COPY --from=kernel-image /wheelhouse /tmp/wheelhouse/
COPY --from=deepgemm-image /wheelhouse /tmp/wheelhouse/
COPY --from=deepep-image /wheelhouse /tmp/wheelhouse/
COPY --from=flashmla-image /wheelhouse /tmp/wheelhouse/
RUN python -m pip install --no-deps /tmp/wheelhouse/*.whl \
    && mkdir -p /opt/cu129/wheelhouse \
    && mv /tmp/wheelhouse/*.whl /opt/cu129/wheelhouse/ \
    && python -m pip freeze > /opt/cu129/extensions.lock

FROM extensions AS sglang
COPY --from=sglang-image /wheelhouse /opt/cu129/wheelhouse/
ARG SOURCE_COMMIT
ARG SGLANG_VERSION
RUN apt-get update && apt-get install -y --no-install-recommends wget \
    && rm -rf /var/lib/apt/lists/*
# Keep a real, independent Git repository so downstream images can apply source patches.
COPY . /sgl-workspace/sglang/
COPY --from=cu129-build adapt.py /opt/cu129/adapt.py
RUN test -d /sgl-workspace/sglang/.git \
    && test "$(git -C /sgl-workspace/sglang rev-parse HEAD)" = "${SOURCE_COMMIT}" \
    && git -C /sgl-workspace/sglang reset --hard "${SOURCE_COMMIT}" \
    && python /opt/cu129/adapt.py /sgl-workspace/sglang
# Reuse every native extension from the package stage; editable installation must
# not compile the same Rust sources a second time at a different source path.
RUN python - <<'PY'
from pathlib import Path
from zipfile import ZipFile

wheel = next(Path("/opt/cu129/wheelhouse").glob("sglang-*.whl"))
with ZipFile(wheel) as archive:
    extensions = [name for name in archive.namelist() if name.startswith("sglang/") and name.endswith(".so")]
    assert extensions, f"No native extensions in {wheel}"
    for name in extensions:
        archive.extract(name, "/sgl-workspace/sglang/python")
    print("Reused compiled extensions:", extensions)
PY
RUN SETUPTOOLS_SCM_PRETEND_VERSION=${SGLANG_VERSION} SGLANG_BUILD_RUST_EXTS=none \
    python -m pip install --no-build-isolation --no-deps -e /sgl-workspace/sglang/python \
    && python -c 'import importlib.metadata as m, sys; assert m.version("sglang") == sys.argv[1]' "${SGLANG_VERSION}" \
    && python -m pip check \
    && printf '%s\n' "${SOURCE_COMMIT}" > /opt/cu129/sglang-commit \
    && python -m pip freeze > /opt/cu129/runtime.lock
COPY --from=cu129-build verify.py /opt/cu129/verify.py
COPY --from=cu129-build smoke_inference.py /opt/cu129/smoke_inference.py
COPY --from=cu129-build sources.env /opt/cu129/sources.env
WORKDIR /sgl-workspace
LABEL org.opencontainers.image.source="https://github.com/sgl-project/sglang" \
    org.opencontainers.image.revision=${SOURCE_COMMIT} \
    org.opencontainers.image.version=${SGLANG_VERSION}

FROM sglang AS compat
ENV LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
    NCCL_NVLS_ENABLE=0 \
    NCCL_CUMEM_HOST_ENABLE=0 \
    SGLANG_USE_SGL_FA3_KERNEL=1

# Keep the compiled layers cached; select the CUDA 12 audio wheel at the end.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu129
RUN TORCHAUDIO_VERSION=$(python -c 'import importlib.metadata as m; print(m.version("torchaudio").split("+")[0])') \
    && python -m pip install --no-deps --index-url ${TORCH_INDEX_URL} \
        "torchaudio==${TORCHAUDIO_VERSION}+cu129" \
    && python -c 'import torch, torchaudio; assert torchaudio.__version__.endswith("+cu129")'
```

</details>

<details>
<summary><code>adapt.py</code></summary>

<!-- cu129-file: adapt.py -->

```python
"""Adapt copied build inputs; leave the upstream checkout untouched."""

import argparse
import json
import re
import shutil
import tomllib
from pathlib import Path


def adapt_metadata(root: Path):
    changes = []
    for path in (
        root / "python/pyproject.toml",
        root / "python/sglang/kernels/aot/pyproject.toml",
    ):
        if not path.exists():
            continue
        original = path.read_text()
        updated = re.sub(r"torch==2\.\d+\.\d+", "torch==2.13.0", original)
        updated = updated.replace("cuda-python>=13.0", "cuda-python==12.9.0")
        updated = updated.replace("[cu13]", "[cu12]")
        # An exact public version avoids unrelated local builds on package mirrors.
        updated = re.sub(r"(flashinfer_python\[cu12\])==", r"\1===", updated)
        # CUTLASS DSL's default distribution is its CUDA 12 variant.
        updated = updated.replace("nvidia-cutlass-dsl[cu12]", "nvidia-cutlass-dsl")
        updated = updated.replace("nvshmem4py-cu13", "nvshmem4py-cu12==0.4.1")
        if updated != original:
            path.write_text(updated)
            changes.append(str(path.relative_to(root)))
    spec = tomllib.loads((root / "python/pyproject.toml").read_text())
    native = {"sglang-kernel", "sgl-deep-gemm", "sgl-deep-ep", "sgl-flash-mla"}
    requirements = []
    extensions = {}
    for requirement in spec["project"]["dependencies"]:
        name = re.split(r"[<>=!~;\[ ]", requirement)[0].replace("_", "-").lower()
        if name in native:
            extensions[name] = requirement
        else:
            requirements.append(requirement)
    # Keep frontend >=1.31 for FlashInfer, using its CUDA 12 metadata variant.
    requirements.append("nvidia-cudnn-frontend==1.31.0+cu12")
    (root / "requirements-cu129.txt").write_text("\n".join(requirements) + "\n")
    (root / "extensions-cu129.json").write_text(json.dumps(extensions, indent=2) + "\n")
    print(
        json.dumps(
            {"metadata_changes": changes, "native_extensions": extensions}, indent=2
        )
    )


def adapt_kernel(root: Path):
    path = root / "python/sglang/kernels/aot/CMakeLists.txt"
    source = path.read_text()
    # The upstream CMake file supplies gencode flags directly, bypassing
    # TORCH_CUDA_ARCH_LIST. Keep SM90/SM90a and omit the SM100 library.
    source, count = re.subn(
        r'if \("\$\{CUDA_VERSION\}" VERSION_GREATER_EQUAL "12\.8" OR SGL_KERNEL_ENABLE_SM100A\)',
        "if (FALSE) # cu129 Hopper build: omit Blackwell gencodes",
        source,
    )
    if count != 2:
        raise RuntimeError(
            f"Upstream architecture conditions changed: expected 2, found {count}"
        )
    source = re.sub(
        r"# =+ Common SM100\+ Build =+ #.*?(?=find_package\(Python3)",
        "",
        source,
        flags=re.S,
    )
    source = re.sub(
        r"^target_link_libraries\(common_ops_sm100_build[^\n]*\)\n",
        "",
        source,
        flags=re.M,
    )
    source = re.sub(
        r"target_compile_definitions\(common_ops_sm100_build PRIVATE.*?\)\n",
        "",
        source,
        flags=re.S,
    )
    source = re.sub(
        r"^install\(TARGETS common_ops_sm100_build[^\n]*\)\n", "", source, flags=re.M
    )
    if "common_ops_sm100_build" in source:
        raise RuntimeError(
            "Upstream SM100 target changed; review the Hopper adaptation"
        )
    path.write_text(source)
    path = root / "python/sglang/kernels/aot/cmake/flashmla.cmake"
    source = path.read_text()
    source, count = (
        source.replace(
            "if(${CUDA_VERSION} VERSION_GREATER 12.8)",
            "if(FALSE) # cu129 Hopper build: omit Blackwell FlashMLA",
        ),
        source.count("if(${CUDA_VERSION} VERSION_GREATER 12.8)"),
    )
    if count != 1:
        raise RuntimeError("Upstream FlashMLA architecture condition changed")
    # The v0.5.20 FlashMLA API also references SM100 kernels unconditionally.
    # Guard those dispatches along with omitting the SM100 translation units.
    anchor = "FetchContent_Populate(repo-flashmla)"
    if source.count(anchor) != 1:
        raise RuntimeError("Upstream FlashMLA fetch declaration changed")
    source = source.replace(
        anchor,
        """FetchContent_GetProperties(repo-flashmla)
if(NOT repo-flashmla_POPULATED)
    FetchContent_Populate(repo-flashmla)
endif()
file(READ "${repo-flashmla_SOURCE_DIR}/csrc/api/sparse_prefill.cpp" FLASHMLA_API)
if(NOT FLASHMLA_API MATCHES "#ifdef FLASHMLA_ENABLE_SM100")
    execute_process(
        COMMAND patch --batch --forward --fuzz=0 -p1 -i "${CMAKE_CURRENT_LIST_DIR}/flashmla-hopper.patch"
        WORKING_DIRECTORY "${repo-flashmla_SOURCE_DIR}"
        COMMAND_ERROR_IS_FATAL ANY
    )
endif()""",
    )
    shutil.copyfile(Path(__file__).with_name("flashmla-hopper.patch"), path.with_name("flashmla-hopper.patch"))
    path.write_text(source)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--kernel", action="store_true")
    args = parser.parse_args()
    adapt_metadata(args.root)
    if args.kernel:
        adapt_kernel(args.root)
```

</details>

<details>
<summary><code>adapt_native.py</code></summary>

<!-- cu129-file: adapt_native.py -->

```python
"""CUDA 12 packaging changes for the pinned upstream extension sources."""

import argparse
from pathlib import Path


def adapt_deepep(root: Path):
    path = root / "sgl_deep_ep/build_sgl_deep_ep.sh"
    source = path.read_text()
    source = source.replace(
        "    13.0|13.0.*)", "    12.9|12.9.*) CUDA_TAG=cu129 ;;\n    13.0|13.0.*)"
    )
    source = source.replace("EXPECTED_CUDA_MAJOR = 13", "EXPECTED_CUDA_MAJOR = 12")
    source = source.replace("${CUDA_HOME}/include/cccl", "${CUDA_HOME}/include")
    source = source.replace(
        'export TORCH_CUDA_ARCH_LIST="9.0;10.0;10.3"',
        'export TORCH_CUDA_ARCH_LIST="9.0a"',
    )
    path.write_text(source)


def adapt_flashmla(root: Path):
    # The current FlashMLA CUDA 13 kernels implement only Blackwell. Retain
    # their schemas and compile the frozen Hopper implementations on CUDA 12.
    path = root / "setup.py"
    source = path.read_text()
    source = source.replace(
        'CUDA_ARCHS = ["sm_100a", "sm_103a"]', 'CUDA_ARCHS = ["sm_90a"]'
    )
    source = source.replace(
        "major > 13 or (major == 13 and minor >= 1)",
        "major > 12 or (major == 12 and minor >= 9)",
    )
    source = source.replace('"--compress-mode=size",', "")
    source = source.replace('"torch>=2.14.1"', '"torch==2.13.0"')
    source = source.replace("*API_SOURCES,", '"csrc/api/api.cpp",')
    source = source.replace("*CUDA_KERNEL_SOURCES,", "")
    source = source.replace("*OUR_OP_SOURCES,", "")
    path.write_text(source)
    path = root / "setup_legacy_sources.py"
    source = path.read_text().replace(
        'LEGACY_CUDA_ARCHS = ["sm_90a", "sm_100a", "sm_103a"]',
        'LEGACY_CUDA_ARCHS = ["sm_90a"]',
    )
    path.write_text(source)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("component", choices=["deepep", "flashmla"])
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    {"deepep": adapt_deepep, "flashmla": adapt_flashmla}[args.component](args.root)
```

</details>

<details>
<summary><code>adapt_cudnn.py</code></summary>

<!-- cu129-file: adapt_cudnn.py -->

```python
"""Keep frontend payloads intact and select CUTLASS's CUDA 12 libraries."""

import copy
import hashlib
import sys
from pathlib import Path

from wheel.wheelfile import WheelFile

source = next(Path(sys.argv[1]).glob("*.whl"))
assert hashlib.sha256(source.read_bytes()).hexdigest() == (
    "3171f102254136418f24b4470dfa5419f34b03d9d792a4b773cf65c7a37e6bf8"
)
target = Path(sys.argv[2])
target.mkdir(parents=True, exist_ok=True)
old_info = "nvidia_cudnn_frontend-1.31.0.dist-info"
new_info = "nvidia_cudnn_frontend-1.31.0+cu12.dist-info"
target /= source.name.replace("-1.31.0-", "-1.31.0+cu12-")
preserved = 0
with WheelFile(source) as original, WheelFile(target, "w") as adapted:
    for entry in original.infolist():
        if entry.filename == f"{old_info}/RECORD":
            continue
        data = original.read(entry)
        if entry.filename == f"{old_info}/METADATA":
            assert data.count(b"nvidia-cutlass-dsl[cu13]") == 1
            data = data.replace(b"nvidia-cutlass-dsl[cu13]", b"nvidia-cutlass-dsl")
            data = data.replace(b"Version: 1.31.0\n", b"Version: 1.31.0+cu12\n")
        elif not entry.filename.startswith(old_info + "/"):
            preserved += 1
        output_entry = copy.copy(entry)
        output_entry.filename = entry.filename.replace(old_info + "/", new_info + "/")
        adapted.writestr(output_entry, data)
print(f"Preserved {preserved} frontend payload files; adapted metadata: {target}")
```

</details>

<details>
<summary><code>flashmla-hopper.patch</code></summary>

<!-- cu129-file: flashmla-hopper.patch -->

```diff
--- a/csrc/api/sparse_prefill.cpp
+++ b/csrc/api/sparse_prefill.cpp
@@ -43,6 +43,7 @@
     }
 };
 
+#ifdef FLASHMLA_ENABLE_SM100
 class Fwd_Sm100_Head64_Impl : public FwdImplBase {
     DECLARE_SUPPORTED_FEATURES(
         FwdFeatures::HEAD_64,
@@ -91,6 +92,8 @@
         sm100::prefill::sparse_fwd_for_small_topk::head128::run_sparse_fwd_for_small_topk_phase1_kernel<SparseAttnFwdMode::Prefill, 512>(params);
     }
 };
+
+#endif
 
 std::vector<at::Tensor> sparse_attn_prefill_interface(
     const at::Tensor &q,
@@ -207,7 +210,9 @@
     if (is_sm90a) {
         Fwd_Sm90_Impl fwd_impl;
         fwd_impl.run(params, required_features);
-    } else if (is_sm100f) {
+    }
+#ifdef FLASHMLA_ENABLE_SM100
+    else if (is_sm100f) {
         if (h_q == 64) {
             Fwd_Sm100_Head64_Impl fwd_impl;
             fwd_impl.run(params, required_features);
@@ -229,7 +234,9 @@
         } else {
             TORCH_CHECK(false, "Unsupported h_q: ", h_q);
         }
-    } else {
+    }
+#endif
+    else {
         TORCH_CHECK(false, "Unsupported architecture");
     }
 
--- a/csrc/api/sparse_decode.cpp
+++ b/csrc/api/sparse_decode.cpp
@@ -86,6 +86,7 @@
     }
 };
 
+#ifdef FLASHMLA_ENABLE_SM100
 class Decode_Sm100_Head64_Impl : public DecodeImplBase {
     DECLARE_SUPPORTED_FEATURES(
         DecodeFeatures::HEAD_64,
@@ -221,6 +222,8 @@
 };
 
 
+#endif
+
 std::tuple<at::Tensor, at::Tensor, std::optional<at::Tensor>, std::optional<at::Tensor>>
 sparse_attn_decode_interface(
     const at::Tensor &q,   // [b, s_q, h_q, d_qk]
@@ -413,6 +416,7 @@
     }
 
     DecodeImplBase* impl;
+#ifdef FLASHMLA_ENABLE_SM100
     if (arch.is_sm100f()) {
         if (h_q == 64) {
             impl = new Decode_Sm100_Head64_Impl();
@@ -427,7 +431,9 @@
         } else {
             TORCH_CHECK(false, "Unsupported h_q: ", h_q);
         }
-    } else if (arch.is_sm90a()) {
+    } else
+#endif
+    if (arch.is_sm90a()) {
         impl = new Decode_Sm90_Impl();
     } else {
         TORCH_CHECK(false, "Unsupported architecture for sparse decode fwd");
```

</details>

<details>
<summary><code>reuse_fa3.py</code></summary>

<!-- cu129-file: reuse_fa3.py -->

```python
"""Reuse only FA3 when its sources, interface and Torch ABI match exactly."""

import re
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile

current = Path("/build/sglang/python/sglang/kernels/aot")
cached = Path("/tmp/fa3-cache/source")
for relative in ("csrc/flash_extension.cc", "python/sgl_kernel/flash_attn.py"):
    assert (current / relative).read_bytes() == (cached / relative).read_bytes(), relative
assert Path("/opt/cu129/torch-constraints.txt").read_bytes() == Path(
    "/tmp/fa3-cache/torch-constraints.txt"
).read_bytes(), "Torch/CUDA/NCCL constraints differ"

current_cmake = (current / "CMakeLists.txt").read_text()
cached_cmake = (cached / "CMakeLists.txt").read_text()
for dependency in ("repo-cutlass", "repo-flash-attention"):
    pattern = rf"FetchContent_Declare\(\s*{dependency}\s+.*?\)"
    assert re.search(pattern, current_cmake, re.S)[0] == re.search(
        pattern, cached_cmake, re.S
    )[0], dependency
pattern = r"# =+ Optional Install: FA3 .*?(?=# =+ InfLLM)"
assert re.search(pattern, current_cmake, re.S)[0] == re.search(
    pattern, cached_cmake, re.S
)[0], "FA3 build options differ"

wheel = next(Path("/wheelhouse").glob("sglang_kernel-*.whl"))
previous = next(Path("/tmp/fa3-cache/wheelhouse").glob("sglang_kernel-*.whl"))
with tempfile.TemporaryDirectory() as directory:
    subprocess.run([sys.executable, "-m", "wheel", "unpack", str(wheel), "-d", directory], check=True)
    unpacked = next(Path(directory).iterdir())
    relative = "sgl_kernel/flash_ops.abi3.so"
    with ZipFile(previous) as archive:
        (unpacked / relative).write_bytes(archive.read(relative))
    subprocess.run([sys.executable, "-m", "wheel", "pack", str(unpacked), "-d", "/wheelhouse"], check=True)
print("Reused FA3 only; all other kernels and wheel metadata come from the target SGLang source.")
```

</details>

<details>
<summary><code>verify.py</code></summary>

<!-- cu129-file: verify.py -->

```python
"""Exercise CUDA, Triton, SGLang kernels and attention against torch references."""

import importlib.metadata
import json

import torch
import torch.nn.functional as functional
import triton
import triton.language as tl


@triton.jit
def add_kernel(x, y, output, n: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    tl.store(
        output + offsets, tl.load(x + offsets, mask) + tl.load(y + offsets, mask), mask
    )


def verify_deep_gemm_and_mla():
    import deep_gemm
    import flash_mla

    from sgl_kernel.flash_mla import (
        flash_mla_with_kvcache,
        get_mla_metadata,
    )

    a = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
    b = torch.randn_like(a)
    output = torch.empty(128, 128, device="cuda", dtype=torch.bfloat16)
    deep_gemm.bf16_gemm_nt(a, b, output)
    torch.testing.assert_close(output, a @ b.T, atol=0.125, rtol=0.02)

    assert flash_mla.has_legacy()
    q = torch.randn(1, 1, 64, 576, device="cuda", dtype=torch.bfloat16)
    cache = torch.randn(1, 64, 1, 576, device="cuda", dtype=torch.bfloat16)
    block_table = torch.zeros(1, 1, device="cuda", dtype=torch.int32)
    lengths = torch.tensor([64], device="cuda", dtype=torch.int32)
    metadata, splits = get_mla_metadata()
    output, _ = flash_mla_with_kvcache(
        q, cache, block_table, lengths, 512, metadata, splits
    )
    keys = cache[0, :, 0].float()
    reference = ((q[0, 0].float() @ keys.T) / (576**0.5)).softmax(-1) @ keys[:, :512]
    torch.testing.assert_close(output[0, 0].float(), reference, atol=0.04, rtol=0.02)


def verify():
    assert torch.version.cuda == "12.9", torch.version.cuda
    assert torch.cuda.get_device_capability() == (9, 0)
    cu13_packages = [
        dist.metadata["Name"]
        for dist in importlib.metadata.distributions()
        if dist.metadata["Name"].lower().endswith(("-cu13", "_cu13"))
    ]
    assert not cu13_packages, cu13_packages
    torch.manual_seed(123)
    x = torch.randn(2048, device="cuda")
    y = torch.randn_like(x)
    output = torch.empty_like(x)
    add_kernel[(triton.cdiv(x.numel(), 256),)](x, y, output, x.numel(), 256)
    torch.testing.assert_close(output, x + y)

    import sgl_kernel

    x = torch.randn(32, 512, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(512, device="cuda", dtype=torch.bfloat16)
    actual = sgl_kernel.rmsnorm(x, weight, 1e-6)
    reference = (
        x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
    ).to(x.dtype) * weight
    torch.testing.assert_close(actual, reference, atol=0.04, rtol=0.02)

    import flashinfer

    q, k, v = [
        torch.randn(16, 4, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3)
    ]
    reference = (
        functional.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0),
            k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0),
            is_causal=True,
        )
        .squeeze(0)
        .transpose(0, 1)
    )
    actual = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True)
    torch.testing.assert_close(actual, reference, atol=0.03, rtol=0.02)

    from sgl_kernel.flash_attn import flash_attn_varlen_func

    lengths = torch.tensor([0, 16], device="cuda", dtype=torch.int32)
    actual = flash_attn_varlen_func(q, k, v, lengths, lengths, 16, 16, causal=True)
    if isinstance(actual, tuple):
        actual = actual[0]
    torch.testing.assert_close(actual, reference, atol=0.03, rtol=0.02)

    verify_deep_gemm_and_mla()
    # These backend imports also resolve their native .so dependencies.
    for module in (
        "deep_ep",
        "deep_gemm",
        "sglang.srt.layers.attention.flashattention_backend",
        "sglang.srt.server_args",
    ):
        importlib.import_module(module)

    torch.cuda.synchronize()
    report = {
        "gpu": torch.cuda.get_device_name(),
        "cuda": torch.version.cuda,
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "triton",
                "sglang",
                "sglang-kernel",
                "flashinfer-python",
                "sgl-deep-gemm",
                "sgl-deep-ep",
                "sgl-flash-mla",
            )
        },
        "passed": [
            "triton_jit",
            "rmsnorm",
            "flashinfer_prefill",
            "fa3_prefill",
            "deepgemm_bf16_gemm",
            "flashmla_dense_decode",
            "native_imports",
            "sglang_backend_imports",
        ],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    verify()
```

</details>

<details>
<summary><code>smoke_inference.py</code></summary>

<!-- cu129-file: smoke_inference.py -->

```python
"""Offline end-to-end inference using a tiny, deterministically generated Llama."""

import argparse
import json
import tempfile

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast


def main(attention_backend):
    torch.manual_seed(42)
    torch.set_num_threads(2)
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        bos_token_id=1,
        eos_token_id=None,
    )
    model = LlamaForCausalLM(config).to(torch.bfloat16).eval()
    model.generation_config.eos_token_id = None
    prompt_ids = [4, 5, 6]
    with torch.inference_mode():
        reference = model.generate(
            torch.tensor([prompt_ids]),
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=0,
        )[0, len(prompt_ids) :].tolist()

    with tempfile.TemporaryDirectory(prefix="cu129-llama-") as path:
        model.save_pretrained(path)
        tokenizer = Tokenizer(
            WordLevel({f"t{i}": i for i in range(128)}, unk_token="t0")
        )
        tokenizer.pre_tokenizer = Whitespace()
        PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            unk_token="t0",
            bos_token="t1",
            pad_token="t0",
        ).save_pretrained(path)

        from sglang import Engine

        engine = Engine(
            model_path=path,
            attention_backend=attention_backend,
            dtype="bfloat16",
            disable_cuda_graph=True,
            mem_fraction_static=0.1,
            context_length=256,
            max_total_tokens=256,
            max_running_requests=1,
        )
        try:
            result = engine.generate(
                input_ids=prompt_ids,
                sampling_params={
                    "temperature": 0,
                    "max_new_tokens": 8,
                    "ignore_eos": True,
                },
                return_logprob=True,
            )
            actual = [item[1] for item in result["meta_info"]["output_token_logprobs"]]
            assert actual == reference, {"actual": actual, "reference": reference}
            print(
                json.dumps(
                    {
                        "passed": "offline_llama_greedy_generation",
                        "attention_backend": attention_backend,
                        "token_ids": actual,
                    }
                )
            )
        finally:
            engine.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attention-backend", choices=("fa3", "flashinfer"), default="fa3"
    )
    main(parser.parse_args().attention_backend)
```

</details>
