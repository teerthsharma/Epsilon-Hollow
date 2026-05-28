#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────
# Push with CI — runs local_ci.sh, then git push.
# Blocks push if any check fails.
# ──────────────────────────────────────────────────────────────

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if bash scripts/local_ci.sh; then
    git push "$@"
else
    echo "Push blocked — fix failures first"
    exit 1
fi
