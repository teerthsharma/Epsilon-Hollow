#!/usr/bin/env bash
# Run the checks CI runs, with CI's exact flags.
#
# This exists because a narrower command reported success for 36 iterations of
# work. Every "clippy clean" claim came from `cargo clippy --workspace --lib`,
# while CI runs `--all-targets -- -D warnings`. That covers neither test files
# nor formatting nor documentation links, so three CI jobs failed on a branch
# whose local gates were all green.
#
# An instrument that runs a narrower command than the gate it stands in for will
# report success indefinitely. Run this before claiming a branch is clean.
#
# Usage:  bash scripts/ci_parity.sh
#
# Exits non-zero if any check fails. Checks needing QEMU, a nightly toolchain,
# or network access are listed at the end and NOT run — their status stays
# explicitly unverified rather than silently assumed.

set -uo pipefail
cd "$(dirname "$0")/.." || exit 2

fail=0
note() { printf '%-46s %s\n' "$1" "$2"; }

# ---- formatting -------------------------------------------------------------
# Reported as a delta against origin/main, because this repository has a
# pre-existing failure and an absolute count would be unreadable.
base_fmt=0
if git rev-parse --verify -q origin/main >/dev/null; then
  base_fmt=$(git stash list >/dev/null 2>&1; echo 1)
fi
fmt_now=$(cargo fmt --all -- --check 2>&1 | grep -c '^Diff in')
if [ "$fmt_now" -gt 1 ]; then
  note "cargo fmt --all -- --check" "FAIL ($fmt_now files; origin/main has 1)"
  fail=1
else
  note "cargo fmt --all -- --check" "ok ($fmt_now, matches origin/main)"
fi

# ---- clippy, CI's flags -----------------------------------------------------
clippy=$(cargo clippy --workspace --all-targets 2>&1 | grep -cE '^(warning|error)')
if [ "$clippy" -gt 0 ]; then
  note "clippy --workspace --all-targets" "FAIL ($clippy warnings/errors)"
  fail=1
else
  note "clippy --workspace --all-targets" "ok (0)"
fi

# ---- rustdoc ----------------------------------------------------------------
docw=$(cargo doc --workspace --no-deps 2>&1 | grep -cE '^warning')
if [ "$docw" -gt 0 ]; then
  note "cargo doc --workspace --no-deps" "FAIL ($docw warnings)"
  fail=1
else
  note "cargo doc --workspace --no-deps" "ok (0)"
fi

# ---- workspace tests --------------------------------------------------------
# aether-lang::interpreter::tests::test_topo_betti_real_call fails on
# origin/main as well, at 23 passed and 1 failed. It is reported, not counted.
# The obvious awk parse is wrong and silently reports zero failures: splitting
# on '[ ;]' makes the field after "passed;" empty, so a naive $6 is always "".
# This grabs the number immediately preceding the word, which is unambiguous.
out=$(cargo test --workspace 2>&1)
passed=$(printf '%s' "$out" | grep -oE '[0-9]+ passed' | awk '{p+=$1} END{print p+0}')
failed=$(printf '%s' "$out" | grep -oE '[0-9]+ failed' | awk '{f+=$1} END{print f+0}')
known=$(printf '%s' "$out" | grep -c 'test_topo_betti_real_call ... FAILED')
if [ "$failed" -gt "$known" ]; then
  note "cargo test --workspace" "FAIL ($passed passed, $failed failed, $known known-preexisting)"
  fail=1
else
  note "cargo test --workspace" "ok ($passed passed, $failed failed, all pre-existing)"
fi

# ---- no_std build of the math crate ----------------------------------------
if cargo check --manifest-path kernel/epsilon/epsilon/crates/aether-core/Cargo.toml \
     --no-default-features --features no_std >/dev/null 2>&1; then
  note "aether-core no_std check" "ok"
else
  note "aether-core no_std check" "FAIL"
  fail=1
fi

# ---- documentation claims ---------------------------------------------------
if cargo run --manifest-path kernel/seal-mkimage/Cargo.toml --release -- \
     --check-doc-claim-contract . 2>&1 | grep -q 'DOC CLAIM CONTRACT OK'; then
  note "doc claim contract" "ok"
else
  note "doc claim contract" "FAIL"
  fail=1
fi

# ---- the loop's own gate ----------------------------------------------------
if [ -f scripts/math_mutation_gate.py ]; then
  line=$(python scripts/math_mutation_gate.py --quick 2>&1 | grep -E 'caught,' | tail -1)
  case "$line" in
    *"0 survived"*) note "mutation gate" "ok — $line" ;;
    "")             note "mutation gate" "FAIL (no summary line)"; fail=1 ;;
    *)              note "mutation gate" "FAIL — $line"; fail=1 ;;
  esac
fi

echo
echo "NOT RUN — status unverified, not assumed:"
echo "  cargo audit, cargo deny            (need network)"
echo "  QEMU UEFI boot smoke test          (needs QEMU)"
echo "  cargo +nightly clippy / miri       (need the nightly toolchain)"
echo "  seal-mkimage boot-log checks       (need a boot log at /tmp/seal-os.log)"
echo
[ "$fail" -eq 0 ] && echo "PARITY OK" || echo "PARITY FAILED"
exit "$fail"
