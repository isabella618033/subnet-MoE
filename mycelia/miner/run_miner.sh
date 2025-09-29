#!/usr/bin/env bash
# Run the miner service with sane defaults.
# Works whether you've installed the package (entrypoint `weightnet-miner`)
# or you’re running from source (`python -m llm_weightnet.miner.cli`).

set -euo pipefail

# --- repo root (best-effort) ---
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$REPO_ROOT"

# --- load env (optional) ---
if [ -f ".env" ]; then
  # shellcheck source=/dev/null
  source .env
fi

# --- defaults (override via .env or inline env) ---
export ROLE="${ROLE:-miner}"
export LOG_FORMAT="${LOG_FORMAT:-console}"          # console | json
export LOG_LEVEL="${LOG_LEVEL:-INFO}"
export METRICS_PORT_Miner="${METRICS_PORT_Miner:-8002}"
export BROKER_URL="${BROKER_URL:-redis://localhost:6379/0}"
export ARTIFACT_STORE="${ARTIFACT_STORE:-/tmp/mycelia-artifacts}"
export DEVICE="${DEVICE:-cuda}"                     # cuda | cpu
export MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3-8B}"
# If you’re using the validator API discovery stub:
# export VALIDATOR_API_BASE="http://localhost:8000"

# Optional config file (Hydra/Pydantic-style layered configs if you added them)
CONFIG_PATH="${1:-}"  # pass a path or leave blank

echo "[miner] starting…"
echo "  ROLE=$ROLE LOG_FORMAT=$LOG_FORMAT LOG_LEVEL=$LOG_LEVEL DEVICE=$DEVICE"
echo "  BROKER_URL=$BROKER_URL ARTIFACT_STORE=$ARTIFACT_STORE"
if [ -n "${VALIDATOR_API_BASE:-}" ]; then
  echo "  VALIDATOR_API_BASE=$VALIDATOR_API_BASE"
fi
if [ -n "$CONFIG_PATH" ]; then
  echo "  CONFIG=$CONFIG_PATH"
fi

# --- pick a runner: uv (if available) -> package entrypoint -> module run ---
run_with_uv() {
  if command -v uv >/dev/null 2>&1; then
    if uv tool list | grep -q "uvicorn" >/dev/null 2>&1; then :; fi
    if uv pip show mycelia >/dev/null 2>&1; then
      uv run weightnet-miner ${CONFIG_PATH:+--config "$CONFIG_PATH"}
    else
      # Run from source; uv resolves deps on the fly if needed.
      uv run python -m llm_weightnet.miner.cli ${CONFIG_PATH:+--config "$CONFIG_PATH"}
    fi
    return 0
  fi
  return 1
}

run_with_entrypoint() {
  if command -v weightnet-miner >/dev/null 2>&1; then
    weightnet-miner ${CONFIG_PATH:+--config "$CONFIG_PATH"}
    return 0
  fi
  return 1
}

run_with_python_module() {
  # Ensure repository root is importable
  export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
  python -m llm_weightnet.miner.cli ${CONFIG_PATH:+--config "$CONFIG_PATH"}
}

if ! run_with_uv; then
  if ! run_with_entrypoint; then
    run_with_python_module
  fi
fi
