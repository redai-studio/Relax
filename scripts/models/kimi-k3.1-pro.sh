#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# Kimi-K3.1-Pro keeps K3's backbone dimensions. Bridge reads the DSA placement
# and indexer dimensions from the checkpoint's text_config.dsa_config.
# Use --megatron-to-hf-mode bridge and --trust-remote-code with the K3.1 HF directory.
# Indexer weights are frozen during SFT/RL (hard top-k has no LM-loss gradient).
# Rollout requires SGLang with K3.1 support (upstream PR #48415) and Relax's
# weight-reload patches; the repository's SGLang v0.5.21 base lacks this model.
source "$(dirname -- "${BASH_SOURCE[0]}")/kimi-k3.sh"
