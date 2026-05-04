# Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
# SPDX-License-Identifier: Epsilon-Hollow

#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
VENV_DIR="$REPO_ROOT/.venv"
REQ_FILE="$REPO_ROOT/requirements/ubuntu-runtime.txt"
BACKEND_DIR="$REPO_ROOT/kernel/epsilon/epsilon-ide/pentesting/backend"
BACKEND_MAIN="$BACKEND_DIR/main.py"
MODEL_DOWNLOADER="$REPO_ROOT/scripts/download_models.py"
LOG_DIR="$REPO_ROOT/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/ubuntu_one_click_${TIMESTAMP}.log"

NO_RUN=0
SKIP_MODELS=0
TIERS="all"
INSTALL_TORCH=1
WORKSPACE_DIR="$REPO_ROOT"
DEV_MODE=1
RECREATE_VENV=1
FORCE_MODEL_DOWNLOADS=1
FORCE_PIP_REINSTALL=1
MODEL_WORKERS=3
MODEL_RETRIES=2
HF_WORKERS=8
ENABLE_OPENCLAW_FALLBACK=1
OPENCLAW_REPO_URL="${OPENCLAW_REPO_URL:-}"
AUTO_OPEN_FRONTEND=1
FRONTEND_URL="http://127.0.0.1:8742/"
BACKEND_HEALTH_URL="http://127.0.0.1:8742/api/v1/status"
BACKEND_READY_TIMEOUT=90

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

provision_openclaw_debug() {
  if [[ "$ENABLE_OPENCLAW_FALLBACK" -ne 1 ]]; then
    return
  fi

  echo "[debug] attempting OpenClaw fallback provisioning..."
  local openclaw_venv="$REPO_ROOT/.openclaw-venv"
  local openclaw_bin=""

  if command -v openclaw >/dev/null 2>&1; then
    openclaw_bin="$(command -v openclaw)"
  fi

  if [[ -z "$openclaw_bin" ]]; then
    python3 -m venv "$openclaw_venv" || true
    if [[ -x "$openclaw_venv/bin/pip" ]]; then
      "$openclaw_venv/bin/python" -m pip install --upgrade pip setuptools wheel || true
      "$openclaw_venv/bin/pip" install --upgrade openclaw || true
      if [[ -x "$openclaw_venv/bin/openclaw" ]]; then
        openclaw_bin="$openclaw_venv/bin/openclaw"
      fi
    fi
  fi

  if [[ -z "$openclaw_bin" && -n "$OPENCLAW_REPO_URL" ]]; then
    local openclaw_repo_dir="$REPO_ROOT/tools/openclaw"
    rm -rf "$openclaw_repo_dir"
    git clone "$OPENCLAW_REPO_URL" "$openclaw_repo_dir" || true
    if [[ -d "$openclaw_repo_dir" && -x "$openclaw_venv/bin/pip" ]]; then
      "$openclaw_venv/bin/pip" install -e "$openclaw_repo_dir" || true
      if [[ -x "$openclaw_venv/bin/openclaw" ]]; then
        openclaw_bin="$openclaw_venv/bin/openclaw"
      fi
    fi
  fi

  local launcher="$REPO_ROOT/launch_openclaw_debug.sh"
  cat > "$launcher" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/.openclaw-venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/.openclaw-venv/bin/activate"
fi
export EPSILON_WORKSPACE_ROOT="${EPSILON_WORKSPACE_ROOT:-$SCRIPT_DIR}"
export EPSILON_DEV_MODE=1
if command -v openclaw >/dev/null 2>&1; then
  exec openclaw "$@"
fi
echo "[openclaw] command not found in PATH." >&2
exit 127
EOF
  chmod +x "$launcher"

  if [[ -n "$openclaw_bin" ]]; then
    echo "[debug] OpenClaw available: $openclaw_bin"
    echo "[debug] launch with: $launcher"
  else
    echo "[warn] OpenClaw auto-install was not successful."
    echo "[warn] set OPENCLAW_REPO_URL and rerun, then launch: $launcher"
  fi
}

open_frontend_browser() {
  local target_url="$1"
  echo "[run] opening frontend: ${target_url}"

  if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$target_url" >/dev/null 2>&1 && return 0
  fi

  if command -v gio >/dev/null 2>&1; then
    gio open "$target_url" >/dev/null 2>&1 && return 0
  fi

  if command -v sensible-browser >/dev/null 2>&1; then
    sensible-browser "$target_url" >/dev/null 2>&1 && return 0
  fi

  if command -v wslview >/dev/null 2>&1; then
    wslview "$target_url" >/dev/null 2>&1 && return 0
  fi

  if command -v powershell.exe >/dev/null 2>&1; then
    powershell.exe -NoProfile -Command "Start-Process '$target_url'" >/dev/null 2>&1 && return 0
  fi

  echo "[warn] could not auto-open browser. Open manually: ${target_url}"
  return 1
}

wait_for_backend() {
  local health_url="$1"
  local timeout_s="$2"
  local elapsed=0

  while (( elapsed < timeout_s )); do
    if curl -fsS --max-time 4 "$health_url" >/dev/null 2>&1; then
      echo "[run] backend is healthy"
      return 0
    fi
    sleep 1
    elapsed=$((elapsed + 1))
  done

  echo "[warn] backend did not become healthy within ${timeout_s}s"
  return 1
}

on_error() {
  local exit_code=$?
  local line_number=$1
  local cmd="${BASH_COMMAND:-unknown}"
  trap - ERR
  set +e
  provision_openclaw_debug
  echo "[error] bootstrap failed at line ${line_number} with exit code ${exit_code}"
  echo "[error] failed command: ${cmd}"
  echo "[error] full log: ${LOG_FILE}"
  exit "$exit_code"
}
trap 'on_error $LINENO' ERR

print_help() {
  cat <<'EOF'
Usage: ./ubuntu_one_click.sh [options]

Options:
  --no-run              Setup only; do not start backend server
  --skip-models         Skip LLM downloads
  --tiers <list>        Comma-separated tiers: foreman,logicgate,architect
  --workspace <path>    Workspace path exposed by backend (default: repo root)
  --no-open-frontend    Do not auto-open frontend in browser
  --frontend-url <url>  Frontend URL to open (default: http://127.0.0.1:8742/)
  --ready-timeout <sec> Backend health wait timeout before opening UI (default: 90)
  --reuse-venv          Reuse existing .venv instead of recreating it
  --no-force-models     Do not force re-download model snapshots
  --no-force-pip        Do not force reinstall Python dependencies
  --workers <n>         Parallel tier download workers (default: 3)
  --retries <n>         Retries per model tier (default: 2)
  --hf-workers <n>      HuggingFace workers per tier (default: 8)
  --no-openclaw-fallback Disable OpenClaw auto-provisioning on failure
  --openclaw-repo <url> Optional OpenClaw git repo URL for fallback install
  --no-torch            Skip torch install
  --prod                Run backend with EPSILON_DEV_MODE=0
  -h, --help            Show this help

Examples:
  ./ubuntu_one_click.sh
  ./ubuntu_one_click.sh --tiers foreman,logicgate --no-run
  ./ubuntu_one_click.sh --skip-models --no-run
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-run)
      NO_RUN=1
      ;;
    --skip-models)
      SKIP_MODELS=1
      ;;
    --tiers)
      TIERS="${2:-}"
      shift
      ;;
    --workspace)
      WORKSPACE_DIR="${2:-}"
      shift
      ;;
    --no-open-frontend)
      AUTO_OPEN_FRONTEND=0
      ;;
    --frontend-url)
      FRONTEND_URL="${2:-}"
      shift
      ;;
    --ready-timeout)
      BACKEND_READY_TIMEOUT="${2:-}"
      shift
      ;;
    --reuse-venv)
      RECREATE_VENV=0
      ;;
    --no-force-models)
      FORCE_MODEL_DOWNLOADS=0
      ;;
    --no-force-pip)
      FORCE_PIP_REINSTALL=0
      ;;
    --workers)
      MODEL_WORKERS="${2:-}"
      shift
      ;;
    --retries)
      MODEL_RETRIES="${2:-}"
      shift
      ;;
    --hf-workers)
      HF_WORKERS="${2:-}"
      shift
      ;;
    --no-openclaw-fallback)
      ENABLE_OPENCLAW_FALLBACK=0
      ;;
    --openclaw-repo)
      OPENCLAW_REPO_URL="${2:-}"
      shift
      ;;
    --no-torch)
      INSTALL_TORCH=0
      ;;
    --prod)
      DEV_MODE=0
      ;;
    -h|--help)
      print_help
      exit 0
      ;;
    *)
      echo "[error] Unknown argument: $1" >&2
      print_help
      exit 2
      ;;
  esac
  shift
done

echo "[setup] bootstrap log: $LOG_FILE"

if [[ ! -f "$BACKEND_MAIN" ]]; then
  echo "[error] Backend entrypoint not found: $BACKEND_MAIN" >&2
  exit 2
fi

if [[ ! -f "$REQ_FILE" ]]; then
  echo "[error] Requirements file not found: $REQ_FILE" >&2
  exit 2
fi

if [[ ! -f "$MODEL_DOWNLOADER" ]]; then
  echo "[error] Model downloader not found: $MODEL_DOWNLOADER" >&2
  exit 2
fi

if [[ ! -d "$WORKSPACE_DIR" ]]; then
  echo "[error] Workspace path does not exist: $WORKSPACE_DIR" >&2
  exit 2
fi

if ! [[ "$MODEL_WORKERS" =~ ^[0-9]+$ && "$MODEL_WORKERS" -ge 1 ]]; then
  echo "[error] --workers must be a positive integer" >&2
  exit 2
fi

if ! [[ "$MODEL_RETRIES" =~ ^[0-9]+$ ]]; then
  echo "[error] --retries must be a non-negative integer" >&2
  exit 2
fi

if ! [[ "$HF_WORKERS" =~ ^[0-9]+$ && "$HF_WORKERS" -ge 1 ]]; then
  echo "[error] --hf-workers must be a positive integer" >&2
  exit 2
fi

if ! [[ "$BACKEND_READY_TIMEOUT" =~ ^[0-9]+$ && "$BACKEND_READY_TIMEOUT" -ge 1 ]]; then
  echo "[error] --ready-timeout must be a positive integer" >&2
  exit 2
fi

if [[ -z "$FRONTEND_URL" ]]; then
  echo "[error] --frontend-url must not be empty" >&2
  exit 2
fi

# Keep health probing aligned with the frontend host/port when overridden.
FRONTEND_BASE="${FRONTEND_URL%/}"
BACKEND_HEALTH_URL="${FRONTEND_BASE}/api/v1/status"

WORKSPACE_DIR="$(cd "$WORKSPACE_DIR" && pwd)"

if [[ -f /etc/os-release ]]; then
  # shellcheck disable=SC1091
  source /etc/os-release
  case "${ID:-}" in
    ubuntu|debian)
      ;;
    *)
      if [[ "${ID_LIKE:-}" != *"debian"* ]]; then
        echo "[warn] Non-Debian distro detected (${PRETTY_NAME:-unknown}). Continuing anyway."
      fi
      ;;
  esac
else
  echo "[warn] /etc/os-release not found. Continuing without distro check."
fi

SUDO=""
if command -v sudo >/dev/null 2>&1; then
  SUDO="sudo"
fi

if [[ "$(id -u)" -ne 0 && -z "$SUDO" ]]; then
  echo "[error] This script needs apt package installation but sudo is unavailable." >&2
  echo "        Run as root or install dependencies manually." >&2
  exit 1
fi

echo "[setup] Installing Ubuntu packages..."
export DEBIAN_FRONTEND=noninteractive
$SUDO apt-get update -y
$SUDO apt-get install -y \
  python3 \
  python3-venv \
  python3-pip \
  python3-dev \
  build-essential \
  cmake \
  pkg-config \
  git \
  curl \
  wget \
  ca-certificates \
  libssl-dev \
  libffi-dev

echo "[setup] Checking network endpoints needed for bootstrap..."
if ! curl -fsS --max-time 20 https://pypi.org/ >/dev/null; then
  echo "[warn] Could not reach https://pypi.org; package installation may fail."
fi
if ! curl -fsS --max-time 20 https://huggingface.co/ >/dev/null; then
  echo "[warn] Could not reach https://huggingface.co; model downloads may fail."
fi

export HF_HUB_ENABLE_HF_TRANSFER=1

if [[ "$RECREATE_VENV" -eq 1 && -d "$VENV_DIR" ]]; then
  echo "[setup] Removing existing virtual environment at $VENV_DIR"
  rm -rf "$VENV_DIR"
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[setup] Creating virtual environment at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
else
  echo "[setup] Reusing virtual environment at $VENV_DIR"
fi

PYTHON_BIN="$VENV_DIR/bin/python"
PIP_BIN="$VENV_DIR/bin/pip"
PIP_ARGS=(--upgrade --no-cache-dir)

if [[ "$FORCE_PIP_REINSTALL" -eq 1 ]]; then
  PIP_ARGS+=(--force-reinstall)
fi

echo "[setup] Upgrading pip tooling..."
"$PYTHON_BIN" -m pip install "${PIP_ARGS[@]}" pip setuptools wheel

echo "[setup] Installing Python dependencies from $REQ_FILE"
"$PIP_BIN" install "${PIP_ARGS[@]}" -r "$REQ_FILE"

if [[ "$INSTALL_TORCH" -eq 1 ]]; then
  echo "[setup] Installing torch..."
  if command -v nvidia-smi >/dev/null 2>&1; then
    "$PIP_BIN" install "${PIP_ARGS[@]}" torch --index-url https://download.pytorch.org/whl/cu124 || \
      "$PIP_BIN" install "${PIP_ARGS[@]}" torch
  else
    "$PIP_BIN" install "${PIP_ARGS[@]}" torch
  fi
else
  echo "[setup] Skipping torch install (--no-torch)"
fi

if [[ "$SKIP_MODELS" -eq 0 ]]; then
  echo "[setup] Downloading model tiers: $TIERS"
  DOWNLOADER_ARGS=(
    --backend-dir "$BACKEND_DIR"
    --tiers "$TIERS"
    --workers "$MODEL_WORKERS"
    --retries "$MODEL_RETRIES"
    --hf-workers "$HF_WORKERS"
  )
  if [[ "$FORCE_MODEL_DOWNLOADS" -eq 1 ]]; then
    DOWNLOADER_ARGS+=(--force)
  fi
  "$PYTHON_BIN" "$MODEL_DOWNLOADER" "${DOWNLOADER_ARGS[@]}"
else
  echo "[setup] Skipping model download (--skip-models)"
fi

echo "[setup] Verifying core Python imports..."
"$PYTHON_BIN" - <<'PY'
import importlib
modules = [
    "fastapi",
    "uvicorn",
    "pydantic",
    "huggingface_hub",
    "numpy",
    "transformers",
    "torch",
]
for mod in modules:
    importlib.import_module(mod)
print("[setup] Python import verification passed")
PY

echo "[setup] Verifying backend and PicoClaw route health..."
"$PYTHON_BIN" - <<PY
import os
import sys

backend_dir = r"""$BACKEND_DIR"""
dev_mode = "$DEV_MODE" == "1"
sys.path.insert(0, backend_dir)
os.environ["EPSILON_DEV_MODE"] = "1" if dev_mode else "0"

import main  # noqa: E402

routes = {getattr(route, "path", "") for route in main.app.routes}
required = {"/api/v1/status", "/api/v1/models/download", "/api/v1/workspace"}
missing = sorted(required - routes)
if missing:
  raise SystemExit(f"missing backend routes: {missing}")

if dev_mode:
  dev_required = {"/api/v1/claw/execute", "/ws/terminal"}
  missing_dev = sorted(dev_required - routes)
  if missing_dev:
    raise SystemExit(f"missing dev routes: {missing_dev}")

print("[setup] backend route verification passed")
PY

echo "[setup] Writing runtime environment file .env.ubuntu.local"
cat > "$REPO_ROOT/.env.ubuntu.local" <<EOF
EPSILON_WORKSPACE_ROOT=$WORKSPACE_DIR
EPSILON_DEV_MODE=$DEV_MODE
EOF

if [[ "$NO_RUN" -eq 1 ]]; then
  cat <<EOF
[done] Setup complete.

To run backend manually:
  source "$VENV_DIR/bin/activate"
  export EPSILON_WORKSPACE_ROOT="$WORKSPACE_DIR"
  export EPSILON_DEV_MODE=$DEV_MODE
  python "$BACKEND_MAIN"
  Log file: "$LOG_FILE"
EOF
  exit 0
fi

echo "[run] Starting backend server on http://127.0.0.1:8742"
export EPSILON_WORKSPACE_ROOT="$WORKSPACE_DIR"
export EPSILON_DEV_MODE="$DEV_MODE"

if [[ "$AUTO_OPEN_FRONTEND" -eq 0 ]]; then
  exec "$PYTHON_BIN" "$BACKEND_MAIN"
fi

"$PYTHON_BIN" "$BACKEND_MAIN" &
BACKEND_PID=$!

cleanup_backend() {
  if [[ -n "${BACKEND_PID:-}" ]] && kill -0 "$BACKEND_PID" >/dev/null 2>&1; then
    kill "$BACKEND_PID" >/dev/null 2>&1 || true
    wait "$BACKEND_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup_backend EXIT

wait_for_backend "$BACKEND_HEALTH_URL" "$BACKEND_READY_TIMEOUT" || true
open_frontend_browser "$FRONTEND_URL" || true

wait "$BACKEND_PID"
