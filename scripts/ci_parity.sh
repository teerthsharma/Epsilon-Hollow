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

# ---- SELF-TEST -------------------------------------------------------------
# Every parser below is fed input it MUST flag. A counter that reports zero on
# known-bad input reports zero on everything, which is how this script's own
# failure counter shipped broken: `awk -F'[ ;]'` on a "test result" line yields
# an EMPTY field after "passed;", so the naive $6 was "" for every input and the
# failed count was always 0.
#
# Three checks written during this work could not fail. Two were caught by a
# required-misfire control and one by hand. This runs the control on the
# controls.
if [ "${1:-}" = "--self-test" ]; then
  st_fail=0
  chk() { # name, actual, expected-nonzero
    if [ "$2" -gt 0 ]; then printf '%-46s %s
' "$1" "ok (flags $2)"
    else printf '%-46s %s
' "$1" "BROKEN — reported 0 on known-bad input"; st_fail=1; fi
  }

  sample_fail='test result: FAILED. 23 passed; 1 failed; 0 ignored; 0 measured'
  chk "failure counter sees a failure"       "$(printf '%s' "$sample_fail" | grep -oE '[0-9]+ failed' | awk '{f+=$1} END{print f+0}')"

  sample_pass='test result: ok. 351 passed; 0 failed; 0 ignored; 0 measured'
  chk "pass counter sees passes"       "$(printf '%s' "$sample_pass" | grep -oE '[0-9]+ passed' | awk '{p+=$1} END{print p+0}')"

  chk "fmt counter sees a diff"       "$(printf 'Diff in /x/y.rs:1:
Diff in /x/z.rs:9:
' | grep -c '^Diff in')"

  chk "clippy counter sees a warning"       "$(printf 'warning: unused import
error: bad
' | grep -cE '^(warning|error)')"

  chk "doc counter sees a warning"       "$(printf 'warning: public documentation links to private item
' | grep -cE '^warning')"

  # The mutation-gate branch must NOT accept a summary reporting survivors.
  bad_gate='17 caught, 2 survived, 0 did not compile, 0 skipped'
  case "$bad_gate" in
    *"0 survived"*) printf '%-46s %s
' "gate parser rejects survivors" "BROKEN — accepted survivors"; st_fail=1 ;;
    *)              printf '%-46s %s
' "gate parser rejects survivors" "ok" ;;
  esac
  # ...and must NOT accept one the compiler rejected, which proves nothing.
  nc_gate='19 caught, 0 survived, 1 did not compile, 0 skipped'
  case "$nc_gate" in
    *"0 did not compile"*) printf '%-46s %s
' "gate parser rejects non-compiling" "BROKEN - accepted a rejected mutation"; st_fail=1 ;;
    *)                     printf '%-46s %s
' "gate parser rejects non-compiling" "ok" ;;
  esac

  # ...and must NOT accept one reporting skips, which silently stop guarding a file.
  skip_gate='19 caught, 0 survived, 0 did not compile, 3 skipped'
  case "$skip_gate" in
    *"0 skipped"*) printf '%-46s %s
' "gate parser rejects skips" "BROKEN — accepted skips"; st_fail=1 ;;
    *)             printf '%-46s %s
' "gate parser rejects skips" "ok" ;;
  esac

  echo
  [ "$st_fail" -eq 0 ] && echo "SELF-TEST OK — every counter flags known-bad input"                        || echo "SELF-TEST FAILED — a counter cannot detect its own failure mode"
  exit "$st_fail"
fi

fail=0
note() { printf '%-46s %s\n' "$1" "$2"; }

# ---- formatting -------------------------------------------------------------
# The single pre-existing violation this allowed for was in
# epsilon-os/src/main.rs and is repaired, so the threshold is now zero. It was
# written as "at most 1" against origin/main; an allowance kept after the thing
# it excused is gone hides the next violation behind it, which is the same
# defect as the tolerated test failure removed below.
fmt_now=$(cargo fmt --all -- --check 2>&1 | grep -c '^Diff in')
if [ "$fmt_now" -gt 0 ]; then
  note "cargo fmt --all -- --check" "FAIL ($fmt_now files need formatting)"
  fail=1
else
  note "cargo fmt --all -- --check" "ok (0)"
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
# There is no longer a tolerated failure here. test_topo_betti_real_call was
# excused as pre-existing for six iterations; it was in fact a stale assertion
# pinning the beta_0 defect this effort had already repaired - it asserted
# beta_0([0,50,100]) == 1 where three values separated by gaps of 50 against a
# threshold of 15 give 3. Repaired in iteration 42, so the tolerance is gone.
# An allowance for a failure that no longer exists hides the next one.
# The obvious awk parse is wrong and silently reports zero failures: splitting
# on '[ ;]' makes the field after "passed;" empty, so a naive $6 is always "".
# This grabs the number immediately preceding the word, which is unambiguous.
out=$(cargo test --workspace 2>&1)
passed=$(printf '%s' "$out" | grep -oE '[0-9]+ passed' | awk '{p+=$1} END{print p+0}')
failed=$(printf '%s' "$out" | grep -oE '[0-9]+ failed' | awk '{f+=$1} END{print f+0}')
if [ "$failed" -gt 0 ]; then
  note "cargo test --workspace" "FAIL ($passed passed, $failed failed)"
  fail=1
else
  note "cargo test --workspace" "ok ($passed passed, 0 failed)"
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
    *"0 survived"*"0 did not compile"*"0 skipped"*) note "mutation gate" "ok - $line" ;;
    *"did not compile"*) note "mutation gate" "FAIL - a mutation the compiler rejected proves nothing: $line"; fail=1 ;;
    *"skipped"*)    note "mutation gate" "FAIL — skipped mutations stop guarding their file: $line"; fail=1 ;;
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
