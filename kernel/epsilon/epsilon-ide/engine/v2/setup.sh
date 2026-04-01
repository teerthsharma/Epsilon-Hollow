#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Epsilon IDE Engine v2 — Automated Setup Script
#
# What this script does:
#   1. Checks your system (OS, GPU, CUDA, disk space)
#   2. Installs system dependencies (cmake, clang, python3.11)
#   3. Creates Python virtual environment and installs packages
#   4. Clones BitNet and builds it via setup_env.py (correct build path)
#   5. Downloads the fast tier model (Qwen2.5-Coder 1.5B)
#   6. Creates config.yaml from template
#   7. Runs verification tests
#
# Usage:
#   bash setup.sh              # full setup
#   bash setup.sh --skip-build # skip bitnet.cpp compile (already built)
#   bash setup.sh --skip-model # skip model download (already downloaded)
#
# WHY setup_env.py INSTEAD OF RAW CMAKE:
#   BitNet requires a kernel code-generation step (codegen_tl1.py / codegen_tl2.py)
#   that produces include/bitnet-lut-kernels.h BEFORE cmake runs.
#   Calling cmake directly skips this step and always fails with:
#     "No SOURCES given to target: ggml"
#   setup_env.py runs both phases in the correct order.
#
# If any step fails:
#   The script prints the exact error, the cause, and the fix.
#   It never silently continues after a failure.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail  # exit on error, undefined var, or pipe failure

# ── Colours for output ────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # no colour

# ── Argument parsing ──────────────────────────────────────────────────────────
SKIP_BUILD=false
SKIP_MODEL=false
for arg in "$@"; do
    case $arg in
        --skip-build) SKIP_BUILD=true ;;
        --skip-model) SKIP_MODEL=true ;;
    esac
done

# ── Helper functions ──────────────────────────────────────────────────────────

log_step() {
    echo -e "\n${BLUE}${BOLD}[STEP]${NC} $1"
}

log_ok() {
    echo -e "${GREEN}✓${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

log_fail() {
    echo -e "${RED}✗ FAILED:${NC} $1"
}

# Called when any command fails
handle_error() {
    local exit_code=$?
    local line_number=$1
    log_fail "Error on line $line_number (exit code $exit_code)"
    echo ""
    echo "To debug: run the failing command manually with full output"
    echo "To skip this step: re-run with --skip-build or --skip-model"
    exit $exit_code
}
trap 'handle_error $LINENO' ERR

# ── Configuration ─────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V2_DIR="$SCRIPT_DIR"
BITNET_DIR="$V2_DIR/BitNet"
VENV_DIR="$V2_DIR/venv"
MODELS_DIR="$V2_DIR/models"          # NOTE: models live outside BitNet dir now
CONFIG_FILE="$V2_DIR/config.yaml"

echo ""
echo -e "${BOLD}════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  Epsilon IDE Engine v2 — Automated Setup${NC}"
echo -e "${BOLD}════════════════════════════════════════════════════${NC}"
echo "  Working directory: $V2_DIR"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — System checks
# ─────────────────────────────────────────────────────────────────────────────
log_step "System checks"

# Check OS
if ! grep -q "ubuntu\|debian" /etc/os-release 2>/dev/null; then
    log_warn "This script is designed for Ubuntu/Debian (WSL2). Other distros may need manual adjustments."
fi
log_ok "OS: $(grep PRETTY_NAME /etc/os-release | cut -d'"' -f2)"

# Check Python 3.11
if ! command -v python3.11 &>/dev/null; then
    log_fail "Python 3.11 not found"
    echo ""
    echo "Fix: run these commands then re-run this script:"
    echo "  sudo apt update"
    echo "  sudo apt install -y python3.11 python3.11-venv"
    exit 1
fi
log_ok "Python: $(python3.11 --version)"

# Check disk space (need at least 5 GB free)
AVAILABLE_GB=$(df "$V2_DIR" | awk 'NR==2 {printf "%.0f", $4/1024/1024}')
if [ "$AVAILABLE_GB" -lt 5 ]; then
    log_fail "Only ${AVAILABLE_GB}GB free disk space. Need at least 5GB."
    echo ""
    echo "Fix: free up disk space and re-run"
    echo "  Large files: ls -lh ~/  and  du -sh /mnt/d/*"
    exit 1
fi
log_ok "Disk space: ${AVAILABLE_GB}GB available"

# Check NVIDIA GPU
GPU_AVAILABLE=false
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -1)
    VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    log_ok "GPU: $GPU_NAME (${VRAM_MB}MB VRAM)"
    GPU_AVAILABLE=true
else
    log_warn "nvidia-smi not found — GPU inference will be disabled"
    log_warn "The engine will run in CPU-only mode (slower)"
fi

# Check CUDA — export PATH so nvcc is always found if installed
export PATH="/usr/local/cuda/bin:$PATH"
CUDA_AVAILABLE=false
if command -v nvcc &>/dev/null; then
    CUDA_VER=$(nvcc --version | grep release | awk '{print $6}' | tr -d ',')
    log_ok "CUDA: $CUDA_VER"
    CUDA_AVAILABLE=true
else
    log_warn "CUDA toolkit not found — BitNet will be built without GPU support"
    echo ""
    echo "To enable GPU support (recommended), install CUDA toolkit:"
    echo "  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb"
    echo "  sudo dpkg -i cuda-keyring_1.1-1_all.deb"
    echo "  sudo apt-get update && sudo apt-get install -y cuda-toolkit-12-4"
    echo "  echo 'export PATH=/usr/local/cuda/bin:\$PATH' >> ~/.bashrc"
    echo "  source ~/.bashrc"
    echo ""
    read -p "Continue without CUDA? GPU inference will not work. [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Setup cancelled. Install CUDA and re-run."
        exit 0
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — System dependencies
# ─────────────────────────────────────────────────────────────────────────────
log_step "Installing system dependencies"

sudo apt-get update -qq

PACKAGES="git cmake build-essential clang lld python3.11 python3.11-venv python3-pip pybind11-dev"
MISSING_PACKAGES=""

for pkg in $PACKAGES; do
    if ! dpkg -l "$pkg" &>/dev/null 2>&1; then
        MISSING_PACKAGES="$MISSING_PACKAGES $pkg"
    fi
done

if [ -n "$MISSING_PACKAGES" ]; then
    echo "Installing:$MISSING_PACKAGES"
    sudo apt-get install -y $MISSING_PACKAGES
else
    log_ok "All system packages already installed"
fi

# Verify critical tools
for tool in git cmake clang python3.11; do
    if command -v $tool &>/dev/null; then
        log_ok "$tool: $(command -v $tool)"
    else
        log_fail "$tool not found after installation"
        echo "Fix: sudo apt install -y $tool"
        exit 1
    fi
done

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Python virtual environment
# ─────────────────────────────────────────────────────────────────────────────
log_step "Setting up Python virtual environment"

if [ ! -d "$VENV_DIR" ]; then
    python3.11 -m venv "$VENV_DIR"
    log_ok "Created venv at $VENV_DIR"
else
    log_ok "Venv already exists at $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
log_ok "Venv activated: $(python --version)"

pip install --upgrade pip --quiet

if [ -f "$V2_DIR/requirements.txt" ]; then
    echo "Installing Python packages from requirements.txt..."
    pip install -r "$V2_DIR/requirements.txt" --quiet
    log_ok "Python packages installed"
else
    log_warn "requirements.txt not found — installing core packages manually"
    pip install --quiet \
        tinygrad sqlite-vec \
        tree-sitter tree-sitter-python \
        httpx requests \
        pyyaml psutil \
        huggingface-hub \
        python-telegram-bot \
        aiohttp \
        python-docx \
        numpy
    log_ok "Core Python packages installed"
fi

# Verify critical imports
python3 -c "import tinygrad; import sqlite_vec; import httpx; import yaml; print('OK')" && \
    log_ok "Python package imports verified" || {
    log_fail "Some packages failed to import"
    echo "Fix: source venv/bin/activate && pip install -r requirements.txt"
    exit 1
}

# Auto-activate venv on new terminal
BASHRC_LINE="cd $V2_DIR && source venv/bin/activate"
if ! grep -qF "$BASHRC_LINE" ~/.bashrc 2>/dev/null; then
    echo "" >> ~/.bashrc
    echo "# Epsilon IDE Engine v2 — auto-activate" >> ~/.bashrc
    echo "$BASHRC_LINE" >> ~/.bashrc
    log_ok "Added auto-activate to ~/.bashrc"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Clone and build BitNet via setup_env.py
#
# ROOT CAUSE OF THE PREVIOUS FAILURE:
#   BitNet's CMakeLists.txt expects include/bitnet-lut-kernels.h to exist.
#   This file is NOT in the repo — it is GENERATED by codegen_tl1.py or
#   codegen_tl2.py (called by setup_env.py) during the configuration phase.
#   Running cmake directly skips codegen, so the header never exists and
#   cmake fails: "No SOURCES given to target: ggml".
#
# THE FIX:
#   Always go through setup_env.py, which runs codegen first, then cmake.
#   setup_env.py also builds llama-server at build/bin/llama-server.
# ─────────────────────────────────────────────────────────────────────────────
log_step "Building BitNet inference engine via setup_env.py"

if [ "$SKIP_BUILD" = true ]; then
    log_warn "Skipping build (--skip-build flag set)"
else
    # ── 4a. Clone ──────────────────────────────────────────────────────────────
    if [ ! -d "$BITNET_DIR" ]; then
        echo "Cloning BitNet repository..."
        git clone --recursive https://github.com/microsoft/BitNet.git "$BITNET_DIR"
        log_ok "Cloned BitNet"
    else
        log_ok "BitNet already cloned at $BITNET_DIR"
        # Make sure submodules are fully initialised (handles partial clones)
        cd "$BITNET_DIR"
        git submodule update --init --recursive
        cd "$V2_DIR"
    fi

    cd "$BITNET_DIR"

    # ── 4b. Install BitNet's own Python requirements ───────────────────────────
    # setup_env.py needs numpy, etc. Install into our venv.
    if [ -f "$BITNET_DIR/requirements.txt" ]; then
        pip install -r "$BITNET_DIR/requirements.txt" --quiet
        log_ok "BitNet Python requirements installed"
    fi

    # ── 4c. Apply the known ggml-bitnet-mad.cpp compile-error fix ─────────────
    MAD_CPP="$BITNET_DIR/src/ggml-bitnet-mad.cpp"
    if [ -f "$MAD_CPP" ] && grep -q "int8_t \* y_col" "$MAD_CPP" 2>/dev/null; then
        sed -i 's/int8_t \* y_col/const int8_t * y_col/g' "$MAD_CPP"
        log_ok "Applied ggml-bitnet-mad.cpp compile fix"
    else
        log_ok "ggml-bitnet-mad.cpp fix not needed (already applied or file absent)"
    fi

    # ── 4d. Set compiler env vars so setup_env.py / cmake use clang ───────────
    export CC=clang
    export CXX=clang++
    if [ "$CUDA_AVAILABLE" = true ]; then
        export CUDACXX=$(which nvcc)
    fi

    # ── 4e. Run setup_env.py with a supported BitNet model ────────────────────
    #
    # setup_env.py must be run with a BitNet 1-bit model so it can:
    #   1. Generate include/bitnet-lut-kernels.h  (codegen phase)
    #   2. Configure cmake with the correct flags  (cmake phase)
    #   3. Compile all binaries incl. llama-server (build phase)
    #
    # We use the smallest supported model (BitNet-b1.58-2B-4T) to minimise
    # download time during build.  Our Qwen GGUF models are downloaded
    # separately in Step 5 and referenced in config.yaml.
    #
    BITNET_MODEL_DIR="$BITNET_DIR/models/BitNet-b1.58-2B-4T"
    BITNET_MODEL_GGUF="$BITNET_MODEL_DIR/ggml-model-i2_s.gguf"

    if [ ! -f "$BITNET_MODEL_GGUF" ]; then
        echo "Downloading BitNet-b1.58-2B-4T (bootstrap model for build, ~1.3 GB)..."
        python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='microsoft/BitNet-b1.58-2B-4T-gguf',
    local_dir='$BITNET_MODEL_DIR',
    ignore_patterns=['*.bin'],
)
print('Bootstrap model downloaded.')
" || {
            log_fail "Bootstrap model download failed"
            echo ""
            echo "Common causes:"
            echo "  1. No internet:  ping huggingface.co"
            echo "  2. Login needed: huggingface-cli login"
            echo "  3. Disk full:    df -h"
            exit 1
        }
        log_ok "Bootstrap model downloaded to $BITNET_MODEL_DIR"
    else
        log_ok "Bootstrap model already present"
    fi

    # Run setup_env.py — this generates kernels, configures cmake, and compiles
    echo "Running BitNet setup_env.py (generates kernels + compiles, 5-15 min)..."
    python3 setup_env.py \
        --model-dir "$BITNET_MODEL_DIR" \
        --quant-type i2_s 2>&1 | tee /tmp/bitnet_build.log | tail -20

    # ── 4f. Verify the binary was produced ────────────────────────────────────
    BINARY="$BITNET_DIR/build/bin/llama-server"
    if [ ! -f "$BINARY" ]; then
        log_fail "Build failed — llama-server binary not found at $BINARY"
        echo ""
        echo "Full build log: /tmp/bitnet_build.log"
        echo ""
        echo "Common causes and fixes:"
        echo ""
        echo "1. Missing clang:"
        echo "   sudo apt install -y clang lld"
        echo ""
        echo "2. CUDA/nvcc not on PATH:"
        echo "   export PATH=/usr/local/cuda/bin:\$PATH"
        echo "   then re-run this script"
        echo ""
        echo "3. Out of disk space during compile:"
        echo "   df -h — need at least 3GB free for build artefacts"
        echo "   rm -rf $BITNET_DIR/build && bash setup.sh"
        echo ""
        echo "4. See full build log:"
        echo "   less /tmp/bitnet_build.log"
        exit 1
    fi

    log_ok "Build successful: $BINARY"
    "$BINARY" --version 2>&1 | head -1 && log_ok "Binary runs correctly"

    cd "$V2_DIR"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Download Qwen2.5-Coder GGUF models
# ─────────────────────────────────────────────────────────────────────────────
log_step "Downloading Qwen2.5-Coder model files"

if [ "$SKIP_MODEL" = true ]; then
    log_warn "Skipping model download (--skip-model flag set)"
else
    source "$VENV_DIR/bin/activate"

    mkdir -p "$MODELS_DIR/Qwen2.5-Coder-1.5B"

    FAST_MODEL="$MODELS_DIR/Qwen2.5-Coder-1.5B/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"

    if [ -f "$FAST_MODEL" ]; then
        SIZE=$(du -sh "$FAST_MODEL" | cut -f1)
        log_ok "Fast tier model already downloaded ($SIZE)"
    else
        echo "Downloading Qwen2.5-Coder 1.5B (~1 GB)..."
        python3 -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id='Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF',
    filename='qwen2.5-coder-1.5b-instruct-q4_k_m.gguf',
    local_dir='$MODELS_DIR/Qwen2.5-Coder-1.5B',
)
print(f'Downloaded: {path}')
" || {
            log_fail "Fast tier model download failed"
            echo ""
            echo "Common causes and fixes:"
            echo ""
            echo "1. No internet: ping huggingface.co"
            echo "2. HuggingFace login required:"
            echo "   huggingface-cli login"
            echo "   (token from huggingface.co/settings/tokens)"
            echo "3. Disk full: df -h"
            echo "4. Manual download:"
            echo "   Visit: https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF"
            echo "   Download: qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
            echo "   Place at: $FAST_MODEL"
            exit 1
        }
        log_ok "Fast tier model downloaded"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Create config.yaml
# ─────────────────────────────────────────────────────────────────────────────
log_step "Creating configuration file"

if [ -f "$CONFIG_FILE" ]; then
    log_ok "config.yaml already exists — not overwriting"
    log_warn "To reset config: rm config.yaml && bash setup.sh --skip-build --skip-model"
else
    cat > "$CONFIG_FILE" << CONFIGEOF
# ─────────────────────────────────────────────────────────────────────────────
# Epsilon IDE Engine v2.2 — Configuration
# Generated by setup.sh on $(date)
# ─────────────────────────────────────────────────────────────────────────────

model_tier: "auto"

llama_server_bin: "$BITNET_DIR/build/bin/llama-server"

models:
  fast:
    path:        "$MODELS_DIR/Qwen2.5-Coder-1.5B/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
    gpu_layers:  28
    context_len: 1024
    max_tokens:  256
    temperature: 0.0
    description: "Qwen2.5-Coder 1.5B — tab completions and simple functions"

  balanced:
    path:        "$MODELS_DIR/Qwen2.5-Coder-7B/qwen2.5-coder-7b-instruct-q4_k_m.gguf"
    gpu_layers:  28
    context_len: 2048
    max_tokens:  1024
    temperature: 0.05
    description: "Qwen2.5-Coder 7B — full files and complex generation"

  deep:
    path:        "$MODELS_DIR/DeepSeek-Coder-33B/deepseek-coder-33b-instruct.Q4_K_M.gguf"
    gpu_layers:  0
    context_len: 4096
    max_tokens:  2048
    temperature: 0.1
    description: "DeepSeek-Coder 33B — system design (CPU only)"

routing:
  fast_max:     2
  balanced_max: 6
  deep_min:     7

server_port:    8088
server_host:    localhost
cpu_threads:    4

db_path:        $V2_DIR/clara.db
project_dir:    $V2_DIR
crawl_on_start: false
memory_path:    $V2_DIR/conversation.json
memory_turns:   10

telegram_token:         "YOUR_BOT_TOKEN_HERE"
telegram_allowed_users: []

idle_timeout: 300
CONFIGEOF
    log_ok "Created config.yaml"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Verification tests
# ─────────────────────────────────────────────────────────────────────────────
log_step "Running verification tests"

source "$VENV_DIR/bin/activate"

# Test 1: Python imports
python3 -c "
import sys
sys.path.insert(0, '$V2_DIR')
failures = []
packages = [
    ('tinygrad',         'tinygrad'),
    ('sqlite_vec',       'sqlite-vec'),
    ('httpx',            'httpx'),
    ('yaml',             'pyyaml'),
    ('psutil',           'psutil'),
    ('huggingface_hub',  'huggingface-hub'),
    ('docx',             'python-docx'),
]
for module, pkg in packages:
    try:
        __import__(module)
        print(f'  OK: {module}')
    except ImportError:
        failures.append(pkg)
        print(f'  MISSING: {module}')
if failures:
    print(f'Install missing: pip install {\" \".join(failures)}')
    sys.exit(1)
" || {
    log_fail "Some Python packages are missing"
    echo "Fix: source venv/bin/activate && pip install -r requirements.txt"
    exit 1
}
log_ok "All Python packages import correctly"

# Test 2: Binary exists and runs
BINARY="$BITNET_DIR/build/bin/llama-server"
if [ -f "$BINARY" ] && [ -x "$BINARY" ]; then
    log_ok "llama-server binary: $BINARY"
else
    log_fail "llama-server binary missing or not executable"
    echo "Fix: bash setup.sh (without --skip-build)"
    exit 1
fi

# Test 3: Fast model exists
FAST_MODEL="$MODELS_DIR/Qwen2.5-Coder-1.5B/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
if [ -f "$FAST_MODEL" ]; then
    SIZE=$(du -sh "$FAST_MODEL" | cut -f1)
    log_ok "Fast tier model: $FAST_MODEL ($SIZE)"
else
    log_fail "Fast tier model not found at $FAST_MODEL"
    echo "Fix: bash setup.sh --skip-build (downloads model only)"
    exit 1
fi

# Test 4: Config file valid YAML
if [ -f "$CONFIG_FILE" ]; then
    python3 -c "import yaml; yaml.safe_load(open('$CONFIG_FILE'))" && \
        log_ok "config.yaml is valid YAML" || {
        log_fail "config.yaml has syntax errors"
        echo "Fix: python3 -c \"import yaml; yaml.safe_load(open('config.yaml'))\" to see the error"
        exit 1
    }
else
    log_fail "config.yaml not found"
    exit 1
fi

# Test 5: GPU availability
if command -v nvidia-smi &>/dev/null; then
    VRAM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
    log_ok "GPU free VRAM: ${VRAM}MB"
    if [ "$VRAM" -lt 1500 ]; then
        log_warn "Less than 1.5GB VRAM free — close other GPU applications before running the engine"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}${BOLD}  Setup complete!${NC}"
echo -e "${BOLD}════════════════════════════════════════════════════${NC}"
echo ""
echo "To start the engine:"
echo "  source venv/bin/activate"
echo "  python3 backend/main.py"
echo ""
echo "To test with a single prompt:"
echo "  echo '{\"prompt\": \"write a hello world function\"}' | python3 backend/main.py 2>/dev/null"
echo ""
echo "To start with Telegram bot:"
echo "  1. Edit config.yaml — set telegram_token"
echo "  2. python3 backend/main.py --telegram"
echo ""
echo "To download the balanced tier (7B, better quality):"
echo "  python3 -c \""
echo "  from huggingface_hub import hf_hub_download; import os"
echo "  os.makedirs('$MODELS_DIR/Qwen2.5-Coder-7B', exist_ok=True)"
echo "  hf_hub_download("
echo "      repo_id='Qwen/Qwen2.5-Coder-7B-Instruct-GGUF',"
echo "      filename='qwen2.5-coder-7b-instruct-q4_k_m.gguf',"
echo "      local_dir='$MODELS_DIR/Qwen2.5-Coder-7B',"
echo "  )\""
echo ""