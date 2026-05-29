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
run_step "aether-core no_std check" cargo +stable check --manifest-path kernel/epsilon/epsilon/crates/aether-core/Cargo.toml --no-default-features --features no_std
run_step "aether-core migration gate" cargo +stable test --manifest-path kernel/epsilon/epsilon/crates/aether-core/Cargo.toml --test legacy_migration_gate

run_step "seal-os test-mode build" bash -c 'cd kernel/seal-os && cargo +nightly build --release --features test-mode'
run_step "seal-os nightly build" bash -c 'cd kernel/seal-os && cargo +nightly build --release'
run_step "seal-os image build" bash -c 'cd kernel/seal-mkimage && cargo +stable run --release'
run_step "seal-os image verify" cargo +stable run --manifest-path kernel/seal-mkimage/Cargo.toml --release -- --verify kernel/seal-os/target/x86_64-unknown-uefi/release/seal-os.img kernel/seal-os/target/x86_64-unknown-uefi/release/seal-os.efi

run_step "seal abi audit" cargo +stable run --manifest-path kernel/seal-mkimage/Cargo.toml --release -- --check-seal-abi .
run_step "language hygiene audit" cargo +stable run --manifest-path kernel/seal-mkimage/Cargo.toml --release -- --check-language-hygiene .
run_step "doc claim contract audit" cargo +stable run --manifest-path kernel/seal-mkimage/Cargo.toml --release -- --check-doc-claim-contract .
run_step "aether migration audit" cargo +stable run --manifest-path kernel/seal-mkimage/Cargo.toml --release -- --check-aether-migration .
run_step "o1 allocator audit" cargo +stable run --manifest-path kernel/seal-mkimage/Cargo.toml --release -- --check-o1-allocator .
run_step "runtime theorem coverage audit" cargo +stable run --manifest-path kernel/seal-mkimage/Cargo.toml --release -- --check-runtime-theorems .
run_step "unsafe inventory" cargo +stable run --manifest-path kernel/seal-mkimage/Cargo.toml --release -- --unsafe-inventory .
run_step "ubuntu allocator harness build" cargo +stable build --manifest-path tools/ubuntu-alloc-bench/Cargo.toml --release

if [ -f kernel/seal-os/target/x86_64-unknown-uefi/release/qemu-proof/serial.log ]; then
    run_step "vm proof audit" cargo +stable run --manifest-path kernel/seal-mkimage/Cargo.toml --release -- --check-vm-proof kernel/seal-os/target/x86_64-unknown-uefi/release/qemu-proof/serial.log
    run_step "aether runtime audit" cargo +stable run --manifest-path kernel/seal-mkimage/Cargo.toml --release -- --check-aether-runtime kernel/seal-os/target/x86_64-unknown-uefi/release/qemu-proof/serial.log
    run_step "desktop soak audit" cargo +stable run --manifest-path kernel/seal-mkimage/Cargo.toml --release -- --check-desktop-soak kernel/seal-os/target/x86_64-unknown-uefi/release/qemu-proof/serial.log
    run_step "benchmark log audit" cargo +stable run --manifest-path kernel/seal-mkimage/Cargo.toml --release -- --check-benchmark-log kernel/seal-os/target/x86_64-unknown-uefi/release/qemu-proof/serial.log
else
    echo "SKIP vm proof audit: run kernel/seal-os/run-qemu.ps1 -HeadlessProof to create qemu-proof/serial.log"
fi

if [ -f tools/ubuntu-alloc-bench/ubuntu-alloc.log ]; then
    run_step "ubuntu benchmark audit" cargo +stable run --manifest-path kernel/seal-mkimage/Cargo.toml --release -- --check-ubuntu-benchmark-log tools/ubuntu-alloc-bench/ubuntu-alloc.log
    if [ -f kernel/seal-os/target/x86_64-unknown-uefi/release/qemu-proof/serial.log ]; then
        run_step "seal vs ubuntu allocator comparison" cargo +stable run --manifest-path kernel/seal-mkimage/Cargo.toml --release -- --compare-benchmark-logs kernel/seal-os/target/x86_64-unknown-uefi/release/qemu-proof/serial.log tools/ubuntu-alloc-bench/ubuntu-alloc.log
    else
        echo "SKIP seal vs ubuntu allocator comparison: missing qemu-proof/serial.log"
    fi
else
    echo "SKIP ubuntu benchmark audit: capture tools/ubuntu-alloc-bench/ubuntu-alloc.log on Ubuntu 26.04 first"
fi

run_step "BOM check" ./scripts/check_no_bom.sh
run_step "cargo doc" cargo doc --workspace --no-deps

TOTAL=$((SECONDS - OVERALL_START))
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}🎉 All $PASS checks passed in ${TOTAL}s${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
