# syntax=docker/dockerfile:1

ARG RELAX_IMAGE=ghcr.io/redai-studio/relaxrl:latest
FROM ${RELAX_IMAGE}

ARG NEMO_GYM_COMMIT=a85670eb167ba9b48cc53a36a070eed815e6c40d
ARG PIP_INDEX_URL
ARG PIP_TRUSTED_HOST
ARG UV_INDEX_URL
ARG UV_INSECURE_HOST
ARG UV_HTTP_TIMEOUT=300
ARG UV_CONCURRENT_DOWNLOADS=4

# HTTP(S) proxy build args.
#
# These are Docker "predefined" ARGs: if you pass --build-arg http_proxy=... the
# value is auto-exported as an env var inside every RUN and is excluded from
# `docker history`, so it will NOT leak into the final image ENV. We declare
# them explicitly here so the proxy contract is visible in the Dockerfile.
#
# Usage:
#   docker build \
#     --build-arg http_proxy=$http_proxy \
#     --build-arg https_proxy=$https_proxy \
#     --build-arg no_proxy=$no_proxy \
#     ...
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ARG http_proxy
ARG https_proxy
ARG no_proxy

RUN apt-get update \
    && apt-get install -y --no-install-recommends bubblewrap socat \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir uv==0.9.30 \
    && git clone https://github.com/NVIDIA-NeMo/Gym.git /opt/nemo-gym \
    && cd /opt/nemo-gym \
    && git checkout "${NEMO_GYM_COMMIT}" \
    && uv sync --frozen --extra sandbox

COPY examples/nemo_gym_agentic/service/patches/workplace_assistant_cleanup.patch /tmp/workplace_assistant_cleanup.patch
COPY examples/nemo_gym_agentic/service/patches/nemo_gym_http_error_traceback.patch \
    /tmp/nemo_gym_http_error_traceback.patch
COPY examples/nemo_gym_agentic/service/patches/swe_agents_r2e.patch /tmp/swe_agents_r2e.patch
COPY examples/nemo_gym_agentic/service/patches/openhands_setup_portability.patch \
    /tmp/openhands_setup_portability.patch
COPY examples/nemo_gym_agentic/service/patches/openhands_platform_release.patch \
    /tmp/openhands_platform_release.patch
COPY examples/nemo_gym_agentic/service/patches/openhands_rollout_prefix.patch \
    /opt/nemo-gym/responses_api_agents/swe_agents/setup_scripts/openhands_rollout_prefix.patch
COPY examples/nemo_gym_agentic/service/patches/openhands_r2e_runtime.patch \
    /opt/nemo-gym/responses_api_agents/swe_agents/setup_scripts/openhands_r2e_runtime.patch
COPY examples/nemo_gym_agentic/service/patches/openhands_r2e_runtime_setup.patch \
    /tmp/openhands_r2e_runtime_setup.patch
COPY examples/nemo_gym_agentic/service/patches/claude_code_agent_cleanup.patch \
    /tmp/claude_code_agent_cleanup.patch
COPY examples/nemo_gym_agentic/service/patches/claude_code_agent_pinned_install.patch \
    /tmp/claude_code_agent_pinned_install.patch
COPY examples/nemo_gym_agentic/service/nemo_gym_gateway_model \
    /opt/nemo-gym/responses_api_models/relax_gateway_model

RUN git -C /opt/nemo-gym apply /tmp/nemo_gym_http_error_traceback.patch \
    && git -C /opt/nemo-gym apply /tmp/workplace_assistant_cleanup.patch \
    && git -C /opt/nemo-gym apply /tmp/swe_agents_r2e.patch \
    && git -C /opt/nemo-gym apply /tmp/openhands_r2e_runtime_setup.patch \
    && git -C /opt/nemo-gym apply /tmp/openhands_setup_portability.patch \
    && git -C /opt/nemo-gym apply /tmp/openhands_platform_release.patch \
    && git -C /opt/nemo-gym apply /tmp/claude_code_agent_cleanup.patch \
    && git -C /opt/nemo-gym apply /tmp/claude_code_agent_pinned_install.patch \
    # Use whatever ray version nemo-gym's own lockfile shipped in the Gym venv.
    # The base image already carries a (possibly private-fork) Ray; we do NOT
    # force-align Gym to it, because that either requires the internal mirror
    # or an ad-hoc X.Y.Z stripping. Downstream gateway/sub-server venvs pin to
    # the Gym-venv ray version so all NeMo-side processes stay consistent.
    && GYM_RAY_VERSION="$(/opt/nemo-gym/.venv/bin/python -c 'import ray; print(ray.__version__)')" \
    && GYM_OPENAI_VERSION="$(/opt/nemo-gym/.venv/bin/python -c 'import openai; print(openai.__version__)')" \
    && GATEWAY_VENV=/opt/nemo-gym/responses_api_models/relax_gateway_model/.venv \
    && uv venv --seed --python /opt/nemo-gym/.venv/bin/python "${GATEWAY_VENV}" \
    && uv pip install \
      --python "${GATEWAY_VENV}/bin/python" \
      -e /opt/nemo-gym \
      -e /opt/nemo-gym/responses_api_models/relax_gateway_model \
      "ray[default]==${GYM_RAY_VERSION}" \
      "openai==${GYM_OPENAI_VERSION}" \
    && for server_dir in \
      responses_api_agents/simple_agent \
      responses_api_agents/claude_code_agent \
      resources_servers/calendar \
      resources_servers/example_multi_step \
      resources_servers/math_with_judge \
      resources_servers/reasoning_gym \
      resources_servers/workplace_assistant \
      responses_api_agents/swe_agents; do \
        uv venv --seed --python /opt/nemo-gym/.venv/bin/python "/opt/nemo-gym/${server_dir}/.venv"; \
        (cd "/opt/nemo-gym/${server_dir}" \
          && uv pip install \
            --python ".venv/bin/python" \
            -r requirements.txt \
            "ray[default]==${GYM_RAY_VERSION}" \
            "openai==${GYM_OPENAI_VERSION}") \
          || exit 1; \
      done \
    && cd /opt/nemo-gym/responses_api_agents/claude_code_agent \
    # Restrict PATH to system shell/coreutils only — this ensures
    # setup_claude_code truly bootstraps its own node/npm/claude and doesn't
    # silently reuse anything from the base image, while still letting `npm`
    # spawn `sh` for package install scripts.
    && PATH=/usr/bin:/bin .venv/bin/python -c \
      "from responses_api_agents.claude_code_agent.setup_claude_code import ensure_claude_code; ensure_claude_code('2.1.237')" \
    && test "$(PATH="${PWD}/.claude_node/bin:${PATH}" .claude_node/bin/claude --version | awk '{print $1}')" = "2.1.237"

COPY examples/nemo_gym_agentic /opt/relax-integration/examples/nemo_gym_agentic

ENV NEMO_GYM_COMMIT=${NEMO_GYM_COMMIT}

# Keep the Relax base image's Python and Ray CLI as the container defaults.
# NeMo Gym launchers use absolute paths into /opt/nemo-gym/.venv and inject
# their integration-only PYTHONPATH explicitly.

WORKDIR /opt/nemo-gym
