test -z "${RUNTIME_ENV_JSON:-}"
printf '%s\n' judge >> "${SEARCH_RUNTIME_TEST_EFFECTS_FILE}"
export DEEPEYES_JUDGE_BASE_URL="https://judge.example.test/v1"
export DEEPEYES_JUDGE_MODELS="fixture-judge"
export DEEPEYES_JUDGE_API_KEY="${SEARCH_RUNTIME_TEST_JUDGE_TOKEN}"
set -x
