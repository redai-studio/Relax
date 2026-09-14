ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ARG BASE_IMAGE=mirror.ccs.tencentyun.com/lmsysorg/sglang:v0.5.15.post1-cu129
ARG TRAIN_IMAGE=train
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

FROM base as sglang

WORKDIR /sgl-workspace/sglang

RUN pip install nvidia-cudnn-cu12==9.16.0.29

FROM base as train

# flash-linear-attention 0.4.2 (pulls fla-core 0.4.2) is the first release carrying
# `fla.ops.cp`, which GDN chunkwise context parallel needs. NOTE: `fla-core` depends
# on an unpinned `torch`, so pip re-resolves torch's own looser `nvidia-cudnn-cu12` pin here;
# the explicit `nvidia-cudnn-cu12==9.16.0.29` install further down runs after this layer
# and is what the final image keeps.
RUN MAX_JOBS=64 pip -v install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir && \
    pip install --no-cache-dir flash-linear-attention==0.4.2 && \
    pip install --no-cache-dir tilelang -f https://tile-ai.github.io/whl/nightly/cu128/

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

RUN pip -v install --no-cache-dir --no-build-isolation "transformer_engine[pytorch]==2.14.1" && \
    TMS_CUDA_MAJOR=$(python -c 'import torch; print(torch.version.cuda.split(".")[0])') pip install git+https://github.com/redai-studio/torch_memory_saver.git@afc13785c50119048e2dd8ac497cc9e29ec75bd4 --no-cache-dir --force-reinstall && \
    pip install nvidia-modelopt[torch]>=0.37.0 --no-build-isolation --no-cache-dir && \
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
FROM ${TRAIN_IMAGE} as relax

WORKDIR /root

ARG PATCH_VERSION=latest
ARG ENABLE_SGLANG_PATCH=1

ARG MEGATRON_BRIDGE_COMMIT=2faedbf6fe3c422835a44b2b360cadcb2a116a54
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

RUN pip install --ignore-installed PyJWT && \
    pip install -r /tmp/requirements.txt --no-cache-dir && \
    pip install --no-cache-dir "compressed_tensors>=0.13.0" tensordict==0.10.0 pyvers==0.1.0 'nvidia-modelopt[hf]==0.44.0' --no-deps && \
    pip install "transferqueue @ git+https://github.com/redai-studio/TransferQueue.git@58054a33834aadbcf76aacd6b1e32e25c030f2c9" --no-deps

# sgl-router: override the official wheel (pulled by requirements.txt above) with
# slime's r3-capable fork. The official sglang-router drops the routed_experts
# field on the /generate round-trip, breaking R3 (rollout routing replay,
# --use-rollout-routing-replay) with a step-0 deadlock; this fork carries the
# `support r3` passthrough patch. Must run AFTER the requirements install so it
# wins.
RUN pip install --no-cache-dir --force-reinstall --no-deps \
        https://github.com/zhuzilin/sgl-router/releases/download/v0.3.2-9daabcd/sglang_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl && \
    python -c "import sglang_router; assert 'slime' in sglang_router.__version__, sglang_router.__version__"


COPY docker/patch/${PATCH_VERSION}/megatron.patch /root/Megatron-LM/
RUN cd Megatron-LM && \
    patch -p1 < /root/Megatron-LM/megatron.patch || (cat megatron/bridge/models/qwen_vl/modelling_qwen3_vl/text_model.py.rej && exit 1 )&& \
    if grep -R -n '^<<<<<<< ' .; then \
    echo "Patch failed to apply cleanly. Please resolve conflicts." && \
    exit 1; \
    fi && \
    rm megatron.patch && \
    apt update && apt install -y jq && ln -s /usr/local/lib/python3.12/dist-packages/torch_memory_saver_hook_mode_preload_cu12.abi3.so \
    /usr/local/lib/python3.12/dist-packages/torch_memory_saver_hook_mode_preload.abi3.so

COPY docker/patch/${PATCH_VERSION}/sglang.patch /sgl-workspace/sglang/
RUN if [ "$ENABLE_SGLANG_PATCH" = "1" ]; then \
    cd /sgl-workspace/sglang && git add . && \
    git update-index --refresh && \
    git apply sglang.patch --3way && \
    if grep -R -n '^<<<<<<< ' .; then \
    echo "Patch failed to apply cleanly. Please resolve conflicts." && \
    exit 1; \
    fi && \
    rm sglang.patch; \
fi
