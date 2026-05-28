#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────
# Local CI — runs all pre-push checks in order.
# Exits 1 on first failure.
# ──────────────────────────────────────────────────────────────

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PASS=0
FAIL=0

run_step() {
    local name="$1"
    shift
    local start=$SECONDS

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "▶ $name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if "$@"; then
        local elapsed=$((SECONDS - start))
        echo -e "${GREEN}✅ PASS${NC} — $name (${elapsed}s)"
        ((PASS++)) || true
    else
        local elapsed=$((SECONDS - start))
        echo -e "${RED}❌ FAIL${NC} — $name (${elapsed}s)"
        ((FAIL++)) || true
        echo ""
        echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${RED}Local CI blocked — fix failures and retry.${NC}"
        echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        exit 1
    fi
}

echo "🚀 Local CI started at $(date)"
OVERALL_START=$SECONDS

run_step "cargo fmt" cargo fmt --all -- --check
run_step "cargo clippy" cargo clippy --workspace --all-targets -- -D warnings
run_step "cargo test" cargo test --workspace

run_step "seal-os nightly build" bash -c 'cd kernel/seal-os && cargo +nightly build --release'

# Python checks (skip if python/pytest not installed)
PYTHON=""
if command -v python &>/dev/null; then
    PYTHON=python
elif command -v python3 &>/dev/null; then
    PYTHON=python3
fi

if [ -n "$PYTHON" ]; then
    if $PYTHON -m pytest --version &>/dev/null; then
        run_step "pytest" $PYTHON -m pytest tests/ -v
    else
        echo -e "${YELLOW}⚠️  pytest not available — skipping${NC}"
    fi

    run_step "python compileall" $PYTHON -m compileall -q infrastructure scripts tests kernel/epsilon/epsilon_core
else
    echo -e "${YELLOW}⚠️  python not available — skipping Python checks${NC}"
fi

run_step "BOM check" ./scripts/check_no_bom.sh
run_step "cargo doc" cargo doc --workspace --no-deps

TOTAL=$((SECONDS - OVERALL_START))
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}🎉 All $PASS checks passed in ${TOTAL}s${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
