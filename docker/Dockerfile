# syntax=docker/dockerfile:1.7
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ARG BASE_IMAGE=mirror.ccs.tencentyun.com/lmsysorg/sglang:v0.5.17-cu129
ARG TRAIN_IMAGE=train
ARG HYBRID_EP_COMMIT=eb9cee7de5a24193bf09500668d3a619d3d3f3fb
FROM ${BASE_IMAGE} as base

ENV NVIDIA_PYTORCH_VERSION=25.06 \
    PYTORCH_BUILD_VERSION=2.9.0 \
    TARGETARCH=amd64 \
    PYVER=3.12 \
    PYSFX=3 \
    L4T=0 \
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    TZ=Asia/Shanghai \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    OPENCV_VERSION=4.7.0 \
    DEBIAN_FRONTEND=noninteractive

RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y --no-install-recommends \
    build-essential patchelf tmux net-tools htop nano iputils-ping \
    autoconf automake libatlas-base-dev libgoogle-glog-dev libbz2-dev libc-ares2 \
    libleveldb-dev liblmdb-dev libprotobuf-dev libsnappy-dev libtool nasm protobuf-compiler pkg-config unzip sox \
    libsndfile1 libpng-dev libhdf5-dev gfortran rapidjson-dev ninja-build libedit-dev rsync && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y locales tzdata && \
    locale-gen "$LANG" && update-locale LANG="$LANG" LC_ALL="$LC_ALL" && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Optional build-time package mirror; keep the base pip configuration by default.
ARG PIP_INDEX_URL

FROM base as sglang

WORKDIR /sgl-workspace/sglang

RUN pip install nvidia-cudnn-cu12==9.16.0.29

FROM base as train

# Build for the base Torch ABI without relying on a guessed GitHub wheel URL.
RUN FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=64 pip -v install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir && \
    pip install --no-cache-dir flash-linear-attention==0.4.2 && \
    pip install --no-cache-dir tilelang -f https://tile-ai.github.io/whl/nightly/cu128/

# DeepSeek-V4 DSA deps, both listed under mcore's `no_pypi_wheels` so built from pinned source:
#   - fast-hadamard-transform: required by the Lightning Indexer (dsa.rotate_activation asserts on it)
#   - FlashMLA: fused DSA sparse attention; without it the CSA backward is quadratic in seq len
#     (mcore only allows the fusion on SM90 when dsa_indexer_loss_coeff == 0)
ARG FLASH_MLA_COMMIT=b7643bd54521f563b839b98289b5cd048c062ba2
RUN MAX_JOBS=64 pip install --no-cache-dir --no-build-isolation --no-deps \
        git+https://github.com/Dao-AILab/fast-hadamard-transform.git@f134af63deb2df17e1171a9ec1ea4a7d8604d5ca && \
    git clone --recurse-submodules https://github.com/deepseek-ai/FlashMLA.git /opt/FlashMLA && \
    cd /opt/FlashMLA && git checkout ${FLASH_MLA_COMMIT} && \
    git submodule update --init --recursive && \
    MAX_JOBS=64 pip install --no-cache-dir --no-build-isolation --no-deps . && \
    cd / && python -c "import flash_mla; assert hasattr(flash_mla, 'flash_mla_sparse_fwd'), \
        'flash_mla built but flash_mla_sparse_fwd missing'" && \
    rm -rf /opt/FlashMLA

# FA3 (Hopper flash-attention), built from source. This commit's _flash_attn_forward carries
# window_size_left/right to match TE 2.14.1 (docs/draft/sglang-0.5.12-upgrade-plan.md §8.4).
# The `cp` exposes flash_attn_3.flash_attn_interface, which is what TE imports. BUT this commit's
# setup.py (py_modules=["flash_attn_interface"]) ALSO installs a *top-level* flash_attn_interface.py
# (into the egg root). That top-level module and the copied flash_attn_3/flash_attn_interface.py
# both register the torch custom op flash_attn_3::_flash_attn_forward; a second import in the same
# process _destroy()s the Library TE bound to, so a later dispatch reads an empty schema
# (use-after-free) and every actor forward crashes with `schema_.has_value() INTERNAL ASSERT
# FAILED` (§8.5). Strip the stray top-level module (site-packages root + egg root). Keep
# flash_attn_config.py: flash_attn_3/flash_attn_interface.py does `from flash_attn_config import
# CONFIG` at forward time, so deleting it would break the forward with ModuleNotFoundError.
RUN MAX_JOBS=64 \
    cd /opt && git clone https://github.com/Dao-AILab/flash-attention.git && \
    cd flash-attention/ && \
    git checkout 0f82fead0b2d59f6390c62a1d74e927159bc5a7b && \
    git submodule update --init && \
    cd hopper/ && \
    MAX_JOBS=96 python setup.py install && \
    export python_path=`python -c "import site; print(site.getsitepackages()[0])"` && \
    mkdir -p $python_path/flash_attn_3 && \
    cp flash_attn_interface.py $python_path/flash_attn_3/flash_attn_interface.py && \
    find $python_path -maxdepth 1 -name flash_attn_interface.py -delete && \
    find $python_path -maxdepth 2 -path '*.egg/flash_attn_interface.py' -delete && \
    cd / && python -c "import importlib.util as u; assert u.find_spec('flash_attn_interface') is None, 'stray top-level flash_attn_interface still importable'; import flash_attn_config" && \
    rm -rf /opt/flash-attention/

RUN NVTE_PYTORCH_FORCE_BUILD=TRUE pip -v install --no-cache-dir --no-build-isolation "transformer_engine[pytorch]==2.14.1" && \
    TMS_CUDA_MAJOR=$(python -c 'import torch; print(torch.version.cuda.split(".")[0])') pip install git+https://github.com/redai-studio/torch_memory_saver.git@afc13785c50119048e2dd8ac497cc9e29ec75bd4 --no-cache-dir --force-reinstall && \
    pip install "nvidia-modelopt[torch]>=0.37.0" --no-build-isolation --no-cache-dir && \
    pip install "numpy<2" nvidia-cudnn-cu12==9.16.0.29 --no-cache-dir && \
    NVCC_APPEND_FLAGS="--threads 32" \
    pip -v install --disable-pip-version-check --no-cache-dir \
    --no-build-isolation \
    --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" \
    git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4

WORKDIR /root

# Build the INT4 fake-quant CUDA extension required by INT4 QAT rollout weight sync.
COPY relax/backends/megatron/kernels/int4_qat /tmp/int4_qat
RUN cd /tmp/int4_qat && \
    pip install . --no-build-isolation --no-cache-dir && \
    rm -rf /tmp/int4_qat

ARG TRAIN_IMAGE=train
# Build the CUDA 12 HybridEP wheel against the same Torch ABI as the final image.
FROM ${TRAIN_IMAGE} as hybridep_builder
COPY docker/patch/deepep/eb9cee7-hybridep-namespace.patch /tmp/hybridep-namespace.patch
ARG HYBRID_EP_COMMIT
ARG HYBRID_EP_CUDA_ARCH_LIST=9.0
RUN pip install --no-deps "nvidia-nvshmem-cu12==3.4.5" "pynvml==13.0.1" && \
    git clone --no-checkout https://github.com/deepseek-ai/DeepEP.git /tmp/HybridEP && \
    cd /tmp/HybridEP && git checkout --detach "$HYBRID_EP_COMMIT" && \
    test "$(git rev-parse HEAD)" = "$HYBRID_EP_COMMIT" && \
    patch --batch --forward --fuzz=0 -p1 < /tmp/hybridep-namespace.patch && \
    CUDA_VISIBLE_DEVICES="" TORCH_CUDA_ARCH_LIST="$HYBRID_EP_CUDA_ARCH_LIST" HYBRID_EP_MULTINODE=0 \
        CPATH="/usr/local/cuda/include/cccl${CPATH:+:$CPATH}" \
        LIBRARY_PATH="/usr/local/cuda/lib64/stubs${LIBRARY_PATH:+:$LIBRARY_PATH}" \
        MAX_JOBS=16 pip wheel --no-build-isolation --no-deps --wheel-dir /opt/ep-wheels .

# Build the patched router against the same Python/libc as the final image.
FROM ${TRAIN_IMAGE} as router_builder

ARG ROUTER_RUST_VERSION=1.97.1
# Override these build args when CI requires an approved Rust mirror.
ARG RUSTUP_DIST_SERVER=https://static.rust-lang.org
ARG RUSTUP_UPDATE_ROOT=https://static.rust-lang.org/rustup
ARG ROUTER_RUSTUP_INIT_URL=https://sh.rustup.rs
ARG ROUTER_BUILD_JOBS=16
ENV PATH=/root/.cargo/bin:${PATH} \
    CARGO_BUILD_JOBS=${ROUTER_BUILD_JOBS} \
    CARGO_PROFILE_RELEASE_LTO=false \
    CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl git libssl-dev pkg-config protobuf-compiler && \
    rm -rf /var/lib/apt/lists/* && \
    curl --proto '=https' --tlsv1.2 --fail --location --retry 3 --retry-all-errors \
        --connect-timeout 30 --max-time 300 "$ROUTER_RUSTUP_INIT_URL" -o /tmp/rustup-init.sh && \
    (for attempt in 1 2 3; do \
        if sh /tmp/rustup-init.sh -y --profile minimal --default-toolchain "${ROUTER_RUST_VERSION}"; then \
            exit 0; \
        fi; \
        if [ "$attempt" -eq 3 ]; then exit 1; fi; \
        sleep 5; \
    done) && \
    python -m pip install --no-cache-dir "maturin==1.9.6"
COPY docker/patch/sglang-router /tmp/router-patch
COPY docker/scripts/build_sglang_router.sh /tmp/build_sglang_router.sh
RUN bash /tmp/build_sglang_router.sh /tmp/router-patch /opt/router-wheels

FROM ${TRAIN_IMAGE} as relax

WORKDIR /root

ARG PATCH_VERSION=latest
ARG ENABLE_SGLANG_PATCH=1
ARG HYBRID_EP_COMMIT
ENV RELAX_HYBRID_EP_COMMIT=${HYBRID_EP_COMMIT}

ARG MEGATRON_BRIDGE_COMMIT=6862ee170a2569607e8ebc5dde91f119779093ca
ENV MEGATRON_BRIDGE_COMMIT=${MEGATRON_BRIDGE_COMMIT} \
    PYTHONPATH=/root/Megatron-LM/

RUN rm -rf /root/Megatron-LM && git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git && \
    cd /root/Megatron-Bridge/ && git checkout ${MEGATRON_BRIDGE_COMMIT} && \
    git submodule update --init --recursive && ./scripts/switch_mcore.sh dev && \
    mkdir /root/Megatron-LM &&\
    cp -r /root/Megatron-Bridge/src/megatron /root/Megatron-LM/ && \
    rsync -avP /root/Megatron-Bridge/3rdparty/Megatron-LM/megatron/ /root/Megatron-LM/megatron/ && \
    rm -rf /root/Megatron-Bridge

RUN wget -q https://github.com/apptainer/apptainer/releases/download/v1.4.1/apptainer_1.4.1_amd64.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends ./apptainer_1.4.1_amd64.deb \
    && rm apptainer_1.4.1_amd64.deb \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt

RUN --mount=type=bind,source=docker/patch/transferqueue/9784ad0-assignment-transform.patch,target=/tmp/tq-assignment.patch \
    pip install --no-cache-dir "ray[default,serve]==2.56.1" && \
    pip install --ignore-installed PyJWT && \
    pip install -r /tmp/requirements.txt --no-cache-dir && \
    pip install --no-cache-dir "compressed_tensors>=0.13.0" tensordict==0.10.0 pyvers==0.1.0 'nvidia-modelopt[hf]==0.44.0' --no-deps && \
    pip install "transferqueue @ git+https://github.com/redai-studio/TransferQueue.git@9784ad0d8a5f0db93ff8326379ab2a6d06e640a0" --no-deps && \
    TQ_SITE="$(python -c 'import importlib.util; from pathlib import Path; print(Path(importlib.util.find_spec("transfer_queue").origin).parent.parent)')" && \
    patch --batch --forward --fuzz=0 -d "$TQ_SITE" -p1 < /tmp/tq-assignment.patch

# MCore's pinned revision requires the NVRx 0.6 async checkpoint API.
# Build the fixed v0.6.0 source for bases older than the wheel's glibc 2.39.
# The final stage also covers externally supplied TRAIN_IMAGE builds.
RUN pip install --no-cache-dir \
    "nvidia-resiliency-ext @ git+https://github.com/NVIDIA/nvidia-resiliency-ext.git@6c5f2a13c7688d92a7ac7ee6e464721eb8b7345d"

# Pin both FLA distributions in the final stage, including external TRAIN_IMAGE builds.
RUN pip install --no-cache-dir --force-reinstall --no-deps \
    "fla-core==0.5.2" "flash-linear-attention==0.5.2"

# Preserve small KDA gate gradients in FP32 dot products inside the backward kernel.
COPY docker/patch/fla/0.5.2-kda-backward-precision.patch /tmp/fla-kda-backward-precision.patch
RUN cd "$(python -c 'import importlib.util; from pathlib import Path; print(Path(importlib.util.find_spec("fla").origin).parent.parent)')" && \
    patch --batch --forward -p1 < /tmp/fla-kda-backward-precision.patch

# Install all convolution entries; each training process selects its mode at import.
ENV RELAX_FLA_CONV_NB_MODE=auto
COPY docker/patch/fla/0.5.2-qwen38-output-config.patch /tmp/fla-output-config.patch
COPY docker/patch/fla/0.5.2-conv-static-entries.patch /tmp/fla-conv-entries.patch
RUN cd "$(python -c 'import importlib.util; from pathlib import Path; print(Path(importlib.util.find_spec("fla").origin).parent.parent)')" && \
    patch --batch --forward --fuzz=0 -p1 < /tmp/fla-output-config.patch && \
    patch --batch --forward --fuzz=0 -p1 < /tmp/fla-conv-entries.patch && \
    python -m compileall -q fla/modules/conv/triton && \
    rm /tmp/fla-output-config.patch /tmp/fla-conv-entries.patch

# Install the R3 + PD DP-aware router after requirements, including external TRAIN_IMAGE builds.
COPY --from=router_builder /opt/router-wheels /tmp/router-wheels
RUN python -m pip install --no-cache-dir --force-reinstall --no-deps /tmp/router-wheels/*.whl && \
    python -c "import json, importlib.resources as r, sglang_router; from sglang_router import sglang_router_rs; \
        b = json.loads(r.files('sglang_router').joinpath('_relax_build.json').read_text()); \
        assert sglang_router.__version__ == b['version'] == '0.3.2+slime.dpaware'; \
        assert b['source_commit'] == '9daabcd801c861088d93b8ce8a2a831490ce4bee'; print(b)" && \
    rm -rf /tmp/router-wheels


COPY docker/patch/${PATCH_VERSION}/megatron.patch /root/Megatron-LM/
COPY docker/patch/megatron/sync-save-bounded-staging.patch /root/Megatron-LM/
COPY docker/patch/megatron/20261016-qwen38-models.patch /root/Megatron-LM/
COPY docker/patch/megatron/20261020-qwen35-optimized.patch /root/Megatron-LM/
COPY docker/patch/megatron/20261020-model-operators.patch /root/Megatron-LM/
RUN cd Megatron-LM && \
    patch -p1 < /root/Megatron-LM/megatron.patch || (cat megatron/bridge/models/qwen_vl/modelling_qwen3_vl/text_model.py.rej && exit 1 )&& \
    if grep -R -n '^<<<<<<< ' .; then \
    echo "Patch failed to apply cleanly. Please resolve conflicts." && \
    exit 1; \
    fi && \
    patch --batch --forward -p1 < sync-save-bounded-staging.patch && \
    patch --batch --forward --fuzz=0 -p1 < 20261016-qwen38-models.patch && \
    patch --batch --forward --fuzz=0 -p1 < 20261020-qwen35-optimized.patch && \
    patch --batch --forward --fuzz=0 -p1 < 20261020-model-operators.patch && \
    sha256sum 20261020-model-operators.patch > /etc/relax-model-operators-patch.sha256 && \
    sha256sum 20261016-qwen38-models.patch 20261020-qwen35-optimized.patch > /etc/relax-qwen-model-patches.sha256 && \
    python -m compileall -q megatron && \
    rm megatron.patch sync-save-bounded-staging.patch 20261016-qwen38-models.patch 20261020-qwen35-optimized.patch && \
    apt update && apt install -y jq && ln -s /usr/local/lib/python3.12/dist-packages/torch_memory_saver_hook_mode_preload_cu12.abi3.so \
    /usr/local/lib/python3.12/dist-packages/torch_memory_saver_hook_mode_preload.abi3.so

COPY docker/patch/${PATCH_VERSION}/sglang.patch /sgl-workspace/sglang/
COPY docker/patch/latest/sglang_per_pos_topk.patch /sgl-workspace/sglang/
RUN if [ "$ENABLE_SGLANG_PATCH" = "1" ]; then \
    cd /sgl-workspace/sglang && git add . && \
    git update-index --refresh && \
    git apply sglang.patch --3way && \
    if grep -R -n '^<<<<<<< ' .; then \
    echo "Patch failed to apply cleanly. Please resolve conflicts." && \
    exit 1; \
    fi && \
    git apply --check sglang_per_pos_topk.patch && \
    git apply sglang_per_pos_topk.patch && \
    rm sglang.patch sglang_per_pos_topk.patch; \
fi

# Shared MCore model sources are installed above, but this CUDA 12 image retains
# its SGLang 0.5.17/TE 2.14 runtime. The optimized 280G Blackwell TP4 recipe and Qwen3.8
# SGLang 0.5.20 rollout require Dockerfile.cu13; Torch stays on CUDA 12.

# Shared experiment fixes belong in the default CUDA 12 build as well. Keep its
# NumPy<2, TE 2.14 and CUDA 12 libraries; do not install the 280G Blackwell cu13/v2 stack.
COPY docker/scripts/install_legacy_fla.py \
    docker/scripts/validate_hybridep_install.py \
    docker/scripts/record_qwen35_runtime.py \
    docker/scripts/validate_final_runtime.py \
    docker/scripts/check_gpu_imports.py \
    docker/scripts/repair_te_wheel_metadata.py /opt/relax/
COPY docker/patch/fla/0.4.2-legacy-runtime.patch /tmp/fla042.patch
RUN python /opt/relax/install_legacy_fla.py --target /opt/relax/compat/fla-0.4.2 \
        --patch /tmp/fla042.patch --manifest /etc/relax-fla042-runtime.json && \
    rm /tmp/fla042.patch
COPY --from=hybridep_builder /opt/ep-wheels /tmp/ep-wheels
COPY docker/cu12-runtime-requirements.txt /opt/relax/cu12-runtime-requirements.txt
COPY docker/patch/transformer_engine/2.14.1-cu12-wheel-tag.patch /opt/relax/te-wheel-tag.patch
# Resolve late so existing compiled layers remain cached. NIXL packages both
# backends, but dispatches to cu12 based on Torch; this does not change CUDA.
RUN python -c 'import importlib.metadata as m; print("\n".join("{}=={}".format(d.metadata["Name"], d.version) for d in m.distributions() if d.metadata["Name"].lower().replace("_", "-") in ("torch", "torchvision", "torchaudio", "triton", "transformer-engine", "transformer-engine-cu12", "transformer-engine-torch") or d.metadata["Name"].lower().startswith("nvidia-")))' > /tmp/cu12-abi-constraints.txt && \
    pip install --no-cache-dir -c /tmp/cu12-abi-constraints.txt -r /opt/relax/cu12-runtime-requirements.txt && \
    python /opt/relax/repair_te_wheel_metadata.py /opt/relax/te-wheel-tag.patch && \
    pip install --no-deps "nvidia-nvshmem-cu12==3.4.5" "pynvml==13.0.1" && \
    pip install --force-reinstall --no-deps /tmp/ep-wheels/*.whl && \
    rm -r /tmp/ep-wheels && \
    python /opt/relax/validate_hybridep_install.py && \
    python /opt/relax/validate_final_runtime.py --profile cu12
