"""Relax's HybridEP namespace, re-exported from the installed DeepEP build.

Based on:
  PPU source   the DeepEP wheel in the PPU base image (``deep_ep
               2.2.0+v0.1.0.ppu2.2.0``); it has no public source repository
  Relax patch  docker/patch/deepep/eb9cee7-hybridep-namespace.patch, which builds
               this same ``_relax_hybrid_ep`` name on the CUDA line

Megatron-Core imports ``_relax_hybrid_ep`` directly (``fused_a2a.py``) and does
not fall back to ``deep_ep``, while the upstream image builds HybridEP from
source and installs it under that name.  HybridEP links ``nvidia-nvshmem-cu12``
and CUDA 12 SM targets, so it cannot be built on this base; the PPU DeepEP that
ships here exposes the same ``Buffer`` API, so the namespace forwards to it
instead.
"""

import deep_ep as _deep_ep
from deep_ep import Buffer  # noqa: F401


def __getattr__(name):
    return getattr(_deep_ep, name)


def __dir__():
    return dir(_deep_ep)
