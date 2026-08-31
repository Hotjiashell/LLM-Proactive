#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${LLM_PROACTIVE_CLARQ_CONFIG:-${SCRIPT_DIR}/.env}"

if [[ -f "${CONFIG_FILE}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${CONFIG_FILE}"
    set +a
fi

exec python3 "${SCRIPT_DIR}/run_evaluation.py" "$@"
