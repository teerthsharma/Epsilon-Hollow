#!/usr/bin/env bash
# Epsilon-Hollow Topological OS launcher
# Usage: ./run.sh [--api-key <MINIMAX_KEY>]
#    or: MINIMAX_API_KEY=<key> ./run.sh
set -e
cd "$(dirname "$0")"
cargo run -p epsilon-os --release -- "$@"
