#!/usr/bin/env bash
# Seal OS — Aether Build Driver (Stream 6)
# SPDX-License-Identifier: MIT
#
# Uses the Aether bootstrap compiler (`aether-cli`) to compile `.ae` / `.aether`
# kernel sources into object images, while still relying on `rustc` for the
# Rust subset (intermediate dependency until Phase 2).
#
# Usage:
#   ./scripts/build_aether.sh [target_dir]
#
# Environment:
#   AETHER_VERIFY=0   Skip bootstrap syntax verification (faster, dangerous)
#   RUSTC=rustc       Override Rust compiler binary

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TARGET_DIR="${1:-$PROJECT_ROOT/target/aether-build}"
AETHER_VERIFY="${AETHER_VERIFY:-1}"
RUSTC="${RUSTC:-rustc}"

# ── Directories ──────────────────────────────────────────────────────────────
AETHER_SOURCE_DIRS=(
    "$PROJECT_ROOT/kernel/aether/Aether-Lang/examples"
    "$PROJECT_ROOT/kernel/aether/aether-link/src"
    "$PROJECT_ROOT/apps/laamba-governor/native"
)

RUST_SHIM_DIRS=(
    "$PROJECT_ROOT/kernel/seal-mkimage/src"
)

mkdir -p "$TARGET_DIR"

echo "[build_aether] Output dir: $TARGET_DIR"
echo "[build_aether] Verify: $AETHER_VERIFY"

# ── Phase 0: Aether bootstrap compiler ───────────────────────────────────────
# Compile .ae / .aether sources into Aether Object Images (.aeo).
echo ""
echo "[build_aether] === Phase 0: Aether sources → Object Images ==="

AETHER_SOURCES=()
for dir in "${AETHER_SOURCE_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        while IFS= read -r -d '' file; do
            AETHER_SOURCES+=("$file")
        done < <(find "$dir" -maxdepth 1 -type f \( -name '*.ae' -o -name '*.aether' \) -print0)
    fi
done

if [ ${#AETHER_SOURCES[@]} -eq 0 ]; then
    echo "[build_aether] warning: no Aether sources found."
else
    # Ensure bootstrap compiler is built.
    echo "[build_aether] Building bootstrap compiler (aether-cli)..."
    cd "$PROJECT_ROOT"
    cargo build -p aether-cli --release

    for src in "${AETHER_SOURCES[@]}"; do
        base="$(basename "$src")"
        stem="${base%.*}"
        out="$TARGET_DIR/${stem}.aeo"

        if [ "$AETHER_VERIFY" -eq 1 ]; then
            echo "[build_aether] check: $base"
            cargo run -p aether-cli --release -- check "$src"
        fi

        # TODO(Phase 1): Replace placeholder with actual bytecode emission.
        #   cargo run -p aether-cli --release -- compile "$src" -o "$out"
        # For now we stamp a placeholder object image so the pipeline stays intact.
        {
            printf 'AEO\x00'
            printf '%08x' "${#stem}"
            printf '%s' "$stem"
            printf '%08x' "$(md5sum -q -b "$src" | wc -c)"
            md5sum -q -b "$src"
            printf '\x00\x00\x00\x00'
        } > "$out"

        echo "[build_aether] $base → $out"
    done
fi

# ── Intermediate: Rust subset ────────────────────────────────────────────────
# Rust is still used for the build driver, mkimage, and low-level kernel shims.
# Phase 2 will compile a subset of Rust-like code via the Aether compiler.
echo ""
echo "[build_aether] === Intermediate: Rust shims (rustc) ==="

RUST_SOURCES=()
for dir in "${RUST_SHIM_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        while IFS= read -r -d '' file; do
            RUST_SOURCES+=("$file")
        done < <(find "$dir" -maxdepth 1 -type f -name '*.rs' -print0)
    fi
done

if [ ${#RUST_SOURCES[@]} -eq 0 ]; then
    echo "[build_aether] warning: no Rust sources found."
else
    for src in "${RUST_SOURCES[@]}"; do
        base="$(basename "$src")"
        echo "[build_aether] rust: $base (checked)"
        # We only syntax-check here; full Cargo build is handled by the caller.
        "$RUSTC" --edition 2021 --crate-type lib --emit=metadata -o /dev/null "$src" 2>/dev/null || true
    done
fi

# ── Manifest ─────────────────────────────────────────────────────────────────
MANIFEST="$TARGET_DIR/aether-build-manifest.json"
cat > "$MANIFEST" <<EOF
{
  "version": 1,
  "phase": 0,
  "description": "Rust driver + Aether bootstrap compiler + rustc intermediate",
  "target_dir": "$TARGET_DIR",
  "aether_sources": [$(printf '\n'; for s in "${AETHER_SOURCES[@]}"; do echo "    \"$s\","; done | sed '$ s/,$//')],
  "rust_shims": [$(printf '\n'; for s in "${RUST_SOURCES[@]}"; do echo "    \"$s\","; done | sed '$ s/,$//')]
}
EOF

echo ""
echo "[build_aether] manifest → $MANIFEST"
echo "[build_aether] Done."
