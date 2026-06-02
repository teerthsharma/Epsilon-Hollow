#!/usr/bin/env bash
# LAAMBA GOVERNOR - native Rust + Aether launcher

set -e

echo "=================================================="
echo "  LAAMBA GOVERNOR - Native Topology Workstation"
echo "=================================================="

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

check_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo -e "${RED}ERROR: $1 not found in PATH${NC}"
        echo "Install $2 and try again."
        exit 1
    fi
}

echo -e "${CYAN}Checking prerequisites...${NC}"
check_cmd node "Node.js 18+"
check_cmd cargo "Rust via rustup"

echo -e "${GREEN}node $(node --version)${NC}"
echo -e "${GREEN}cargo $(cargo --version | cut -d' ' -f2)${NC}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}Installing npm dependencies...${NC}"
    npm install
else
    echo -e "${GREEN}node_modules exists${NC}"
fi

if [ ! -f "data/index.json" ]; then
    echo -e "${RED}data/index.json missing. Restore the checked-in seed data bundle.${NC}"
    exit 1
else
    echo -e "${GREEN}seed datasets present${NC}"
fi

echo ""
echo -e "${CYAN}Launching LAAMBA GOVERNOR...${NC}"
echo -e "${CYAN}This may take 30-120s on first compile.${NC}"
echo ""

npm run tauri dev
