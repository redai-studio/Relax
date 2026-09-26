#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Starts the cluster-shared WebShop env server before training. Holds one shared
# SimServer (several-GB catalog + Lucene index) and serves /reset /step /close /
# health over HTTP.
#
# Configure WEBSHOP_CONDA_ENV / WEBSHOP_HOME / WEBSHOP_HOST / WEBSHOP_PORT in
# the server process environment.

set -eu

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "${SCRIPT_DIR}"

WEBSHOP_CONDA_ENV="${WEBSHOP_CONDA_ENV:-/root/miniconda3/envs/relax-opd-webshop}"

# WebShop simulator source (dir containing web_agent_site); server.py adds it to
# sys.path and loads the catalog it points at (README installs the 1000 subset).
export WEBSHOP_HOME="${WEBSHOP_HOME:-/root/WebShop}"
export WEBSHOP_HOST="${WEBSHOP_HOST:-0.0.0.0}"
export WEBSHOP_PORT="${WEBSHOP_PORT:-36001}"

exec "${WEBSHOP_CONDA_ENV}/bin/python" -m app.server
