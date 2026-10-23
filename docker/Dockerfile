# syntax=docker/dockerfile:1.7
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ARG BASE_IMAGE=mirror.ccs.tencentyun.com/lmsysorg/sglang:v0.5.21-cu130
ARG TRAIN_IMAGE=train
ARG HYBRID_EP_COMMIT=eb9cee7de5a24193bf09500668d3a619d3d3f3fb
# One amd64 image for Hopper and Blackwell; GPU kernels select their backend at runtime.
FROM ${BASE_IMAGE} as base

ENV TARGETARCH=amd64 \
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

FROM base as sglang

WORKDIR /sgl-workspace/sglang

FROM base as train

# FA2 uses generic SM100 cubins on both B200 (SM100) and B300 (SM103).
# This version's build script accepts "100", not "103".
RUN FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS="80;90;100" MAX_JOBS=64 \
    pip -v install flash-attn==2.7.4.post1 --force-reinstall --no-deps --no-build-isolation --no-cache-dir && \
    pip install --no-cache-dir flash-linear-attention==0.5.2

# Keep TileLang 0.1.12 with the Blackwell FP16 vector-store fix.
COPY docker/patch/tilelang/0.1.12-blackwell-fp16-store.patch /tmp/tilelang-blackwell-fp16-store.patch
RUN cd "$(python -c 'import importlib.util; from pathlib import Path; print(Path(importlib.util.find_spec("tilelang").origin).parent)')" && \
    patch --batch --forward --fuzz=0 -p1 < /tmp/tilelang-blackwell-fp16-store.patch

# DeepSeek-V4 DSA deps, both listed under mcore's `no_pypi_wheels` so built from pinned source:
#   - fast-hadamard-transform: required by the Lightning Indexer (dsa.rotate_activation asserts on it)
#   - FlashMLA: fused DSA sparse attention; without it the CSA backward is quadratic in seq len
#     (mcore only allows the fusion on SM90 when dsa_indexer_loss_coeff == 0)
ARG FLASH_MLA_COMMIT=b7643bd54521f563b839b98289b5cd048c062ba2
RUN FAST_HADAMARD_TRANSFORM_FORCE_BUILD=TRUE MAX_JOBS=64 \
    pip install --force-reinstall --no-cache-dir --no-build-isolation --no-deps \
        git+https://github.com/Dao-AILab/fast-hadamard-transform.git@f134af63deb2df17e1171a9ec1ea4a7d8604d5ca && \
    git clone --recurse-submodules https://github.com/deepseek-ai/FlashMLA.git /opt/FlashMLA && \
    cd /opt/FlashMLA && git checkout ${FLASH_MLA_COMMIT} && \
    git submodule update --init --recursive && \
    CPLUS_INCLUDE_PATH="${CUDA_HOME:-/usr/local/cuda}/include/cccl${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}" \
        MAX_JOBS=64 pip install --no-cache-dir --no-build-isolation --no-deps . && \
    cd / && python -c "import flash_mla; assert hasattr(flash_mla, 'flash_mla_sparse_fwd'), \
        'flash_mla built but flash_mla_sparse_fwd missing'" && \
    rm -rf /opt/FlashMLA

# FA3 stays Hopper-only; Blackwell uses FA2/cuDNN with TE's FA4 disabled below.
# TE 2.18 imports flash_attn_interface; existing callers use flash_attn_3.
# Alias both names to one module so its Torch custom ops are registered once.
# Keep flash_attn_config.py, which the interface imports during forward.
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
    printf '%s\n' 'import sys' 'from flash_attn_3 import flash_attn_interface' \
        'sys.modules[__name__] = flash_attn_interface' > "$python_path/flash_attn_interface.py" && \
    cd / && python -c "import flash_attn_interface as top; from flash_attn_3 import flash_attn_interface as fa3; assert top is fa3; import flash_attn_config" && \
    rm -rf /opt/flash-attention/

# TE 2.18 defers optional cuBLAS symbol resolution for CUDA 13.0 compatibility.
RUN NVTE_PYTORCH_FORCE_BUILD=TRUE MAX_JOBS=32 \
    pip -v install --no-cache-dir --no-build-isolation "transformer_engine[core_cu13,pytorch]==2.18.0" && \
    TMS_CUDA_MAJOR=$(python -c 'import torch; print(torch.version.cuda.split(".")[0])') pip install git+https://github.com/redai-studio/torch_memory_saver.git@afc13785c50119048e2dd8ac497cc9e29ec75bd4 --no-cache-dir --force-reinstall --no-deps && \
    pip install "nvidia-modelopt==0.44.0" --no-build-isolation --no-cache-dir && \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0;10.3" NVCC_APPEND_FLAGS="--threads 32" \
    pip -v install --force-reinstall --no-deps --disable-pip-version-check --no-cache-dir \
    --no-build-isolation \
    --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" \
    git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4

# TE 2.18 still omits B300 from the FA2 head_dim > 192 whitelist.
COPY docker/patch/transformer_engine/2.18.0-fa2-sm103.patch /tmp/te-fa2-sm103.patch
RUN cd "$(python -c 'import importlib.util; from pathlib import Path; print(Path(importlib.util.find_spec("transformer_engine").origin).parent.parent)')" && \
    patch --batch --forward --fuzz=0 -p1 < /tmp/te-fa2-sm103.patch

# FA4 b19 fails Qwen3.5 vision backward; b35 still fails TE's causal head256 backward.
ENV NVTE_FLASH_ATTN_V4=0

WORKDIR /root

# Build the INT4 fake-quant CUDA extension required by INT4 QAT rollout weight sync.
COPY relax/backends/megatron/kernels/int4_qat /tmp/int4_qat
RUN cd /tmp/int4_qat && \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0;10.3" pip install . --no-deps --no-build-isolation --no-cache-dir && \
    rm -rf /tmp/int4_qat

ARG TRAIN_IMAGE=train
# Build HybridEP for Hopper and Blackwell against the final image's Torch/CUDA ABI.
FROM ${TRAIN_IMAGE} as hybridep_builder
COPY docker/patch/deepep/eb9cee7-hybridep-namespace.patch /tmp/hybridep-namespace.patch
ARG HYBRID_EP_COMMIT
ARG HYBRID_EP_CUDA_ARCH_LIST="9.0;10.0a;10.3a"
RUN pip install --no-deps "nvidia-nvshmem-cu13==3.4.5" "pynvml==13.0.1" && \
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
    pip install --no-cache-dir "compressed_tensors>=0.13.0" tensordict==0.10.0 pyvers==0.1.0 --no-deps && \
    pip install "transferqueue @ git+https://github.com/redai-studio/TransferQueue.git@9784ad0d8a5f0db93ff8326379ab2a6d06e640a0" --no-deps && \
    TQ_SITE="$(python -c 'import importlib.util; from pathlib import Path; print(Path(importlib.util.find_spec("transfer_queue").origin).parent.parent)')" && \
    patch --batch --forward --fuzz=0 -d "$TQ_SITE" -p1 < /tmp/tq-assignment.patch

# MCore's pinned revision requires the NVRx 0.6 async checkpoint API.
# Build the fixed v0.6.0 source for bases older than the wheel's glibc 2.39.
# The final stage also covers externally supplied TRAIN_IMAGE builds.
RUN cupti_root="$(python -c 'import sysconfig; print(sysconfig.get_paths()["platlib"])')/nvidia/cu13" && \
    ln -sf libcupti.so.13 "$cupti_root/lib/libcupti.so" && \
    CUDA_PATH="$cupti_root" CPLUS_INCLUDE_PATH="${CUDA_HOME}/include:${CPLUS_INCLUDE_PATH}" \
    pip install --no-cache-dir \
    "nvidia-resiliency-ext @ git+https://github.com/NVIDIA/nvidia-resiliency-ext.git@6c5f2a13c7688d92a7ac7ee6e464721eb8b7345d"

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
COPY docker/scripts/setup_memory_saver.py /tmp/setup_memory_saver.py
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
    apt update && apt install -y jq && \
    python /tmp/setup_memory_saver.py

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

# Resolve training dependency constraints after all other package installs.
# grpcio-tools must stay on protobuf 6 alongside CUTLASS and gRPC health/reflection.
# NIXL 1.5.0 requires both CUDA backend wheels.
RUN pip install --no-cache-dir packaging==25.0 protobuf==6.33.6 grpcio-tools==1.81.1 \
        hydra-core==1.3.2 typeguard==4.6.0 pillow==11.3.0 imageio==2.38.1 nixl-cu12==1.5.0

# Keep one OpenCV provider after resolving the base image metadata.
RUN pip uninstall -y opencv-python-headless && \
    pip install --no-cache-dir opencv-python==4.10.0.84

# Restore the official DeepEP runtime after Torch's NCCL dependency resolution.
RUN pip install --no-cache-dir --force-reinstall --no-deps nvidia-nccl-cu13==2.30.7

# Keep Torch's pip cuDNN; mixing it with the base image's apt cuDNN breaks TE imports.
RUN apt-get remove -y --purge --allow-change-held-packages \
        libcudnn9-cuda-13 libcudnn9-dev-cuda-13 libcudnn9-headers-cuda-13 && \
    rm -rf /var/lib/apt/lists/*

# TE can load NCCL before Torch; prefer the restored wheel over the system library.
ENV LD_LIBRARY_PATH=/opt/sglang/lib/python${PYVER}/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH}

# Keep legacy FLA isolated for recipes that explicitly select it.
COPY docker/scripts/install_legacy_fla.py \
    docker/scripts/validate_hybridep_install.py \
    docker/scripts/record_qwen35_runtime.py \
    docker/scripts/validate_final_runtime.py \
    docker/scripts/check_gpu_imports.py /opt/relax/
COPY docker/patch/fla/0.4.2-legacy-runtime.patch /tmp/fla042.patch
RUN python /opt/relax/install_legacy_fla.py --target /opt/relax/compat/fla-0.4.2 \
        --patch /tmp/fla042.patch --manifest /etc/relax-fla042-runtime.json && \
    rm /tmp/fla042.patch

COPY --from=hybridep_builder /opt/ep-wheels /tmp/ep-wheels
RUN pip install --no-deps "nvidia-nvshmem-cu13==3.4.5" "pynvml==13.0.1" && \
    pip install --force-reinstall --no-deps /tmp/ep-wheels/*.whl && \
    rm -r /tmp/ep-wheels && \
    python /opt/relax/validate_hybridep_install.py && \
    python /opt/relax/validate_final_runtime.py --profile unified
