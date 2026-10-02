#!/bin/bash
set -eu
exec "${SEARCH_RUNTIME_TEST_PYTHON}" "${SEARCH_RUNTIME_TEST_FIXTURES}/ray_probe.py" "$@"
