#!/usr/bin/env bash
# LAAMBA GOVERNOR — Windows Launcher (Git Bash / MSYS2)
# WUBBA LUBBA DUB DUB!

set -e

echo "=================================================="
echo "  LAAMBA GOVERNOR — Topology Engine Launcher"
echo "=================================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check prerequisites
check_cmd() {
    if ! command -v "$1" &> /dev/null; then
        echo -e "${RED}ERROR: $1 not found in PATH${NC}"
        echo "Install $2 and try again."
        exit 1
    fi
}

echo -e "${CYAN}Checking prerequisites...${NC}"
check_cmd node "Node.js 18+ (https://nodejs.org)"
check_cmd cargo "Rust via rustup (https://rustup.rs)"
check_cmd python "Python 3.11+ (https://python.org)"

echo -e "${GREEN}✓ node$(node --version | head -c 10)${NC}"
echo -e "${GREEN}✓ cargo $(cargo --version | cut -d' ' -f2)${NC}"
echo -e "${GREEN}✓ python $(python --version 2>&1 | cut -d' ' -f2)${NC}"

# Project root (where this script lives)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Install frontend deps if needed
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}Installing npm dependencies...${NC}"
    npm install
else
    echo -e "${GREEN}✓ node_modules exists${NC}"
fi

# Generate seed data if missing
if [ ! -f "data/index.json" ]; then
    echo -e "${YELLOW}Generating seed datasets...${NC}"
    python cli/generate_seed_data.py
else
    echo -e "${GREEN}✓ Seed datasets present${NC}"
fi

# Check Python deps
echo -e "${CYAN}Checking Python dependencies...${NC}"
python -c "import numpy, sklearn" 2>/dev/null || {
    echo -e "${YELLOW}Installing Python deps (numpy, scikit-learn)...${NC}"
    pip install numpy scikit-learn
}
echo -e "${GREEN}✓ Python deps OK${NC}"

# Launch
echo ""
echo -e "${CYAN}Launching LAAMBA GOVERNOR...${NC}"
echo -e "${CYAN}This may take 30-120s on first compile.${NC}"
echo ""

npm run tauri dev
