# syntax=docker/dockerfile:1.7
# Build context: this directory, not the workspace containing datasets/models.
# BASE_IMAGE must be the complete CUDA 13 Blackwell image with Megatron f78809f6.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

USER root
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
ARG MEGATRON_ROOT=/root/Megatron-LM
ARG MOK_ROOT=/opt/relax/mixture-of-kittens
ARG MOK_ARCH=SM103
ARG PYTHON_BIN=/opt/sglang/bin
ENV PATH=${PYTHON_BIN}:${PATH}

COPY . /opt/relax/mok-build/
RUN CUDA_VISIBLE_DEVICES="" python /opt/relax/mok-build/build.py \
        --megatron-root "${MEGATRON_ROOT}" --mok-root "${MOK_ROOT}" --arch "${MOK_ARCH}"

# MoK is installed as a wheel; do not prepend its source tree or an old experiment overlay.
ENV PYTHONPATH=${MEGATRON_ROOT}:${PYTHONPATH} \
    RELAX_MEGATRON_ROOT=${MEGATRON_ROOT} \
    RELAX_MOK_SOURCE_ROOT=${MOK_ROOT}

# Source/installed-wheel checks need no GPU or NVIDIA driver mount.
RUN CUDA_VISIBLE_DEVICES="" python /opt/relax/mok-build/verify.py
