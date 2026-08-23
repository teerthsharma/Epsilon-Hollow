#!/usr/bin/env python3
"""Mutation gate for the mathematics in aether-core, epsilon and aether-verified.

A test that cannot fail is decoration. This applies a named one-line mutation to
a source file, runs the owning crate's tests, and records whether anything went
red. A mutation that NO test catches is a gap in the suite, not a bug in the
mutation.

Every mutation is reverted from an in-memory copy of the original file inside a
`finally` block, so the tree is left exactly as found even if a run is
interrupted mid-way. `persistence.rs` and `diagram.rs` are mutated transiently
for coverage measurement only; they are never left modified.

A mutation that does not COMPILE is reported as NOCOMPILE rather than counted as
caught. The compiler is not a test, and crediting it would overstate the suite.

Usage:  python scripts/math_mutation_gate.py [--quick]

`--quick` runs only the crate owning each mutated file rather than the whole
workspace.
"""

import io
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AC = "kernel/epsilon/epsilon/crates/aether-core/src"
EP = "kernel/epsilon/epsilon/crates/epsilon/src"
AV = "kernel/aether/aether-verified/src"
EO = "kernel/epsilon/epsilon/crates/epsilon-os/src"

# (id, crate, file, original fragment, mutated fragment, what it breaks)
MUTATIONS = [
    # ---- repairs made by this loop -----------------------------------------
    ("beta0-drop-plus-one", "aether-core", f"{AC}/topology.rs",
     "None => components = 1,",
     "None => components = 0,",
     "beta_0 counts gaps instead of gaps+1"),

    ("beta0-gap-comparison", "aether-core", f"{AC}/topology.rs",
     "Some(p) if value - p > threshold => components += 1,",
     "Some(p) if value - p >= threshold => components += 1,",
     "off-by-one at the threshold boundary"),

    ("voronoi-betti0-returns-k", "aether-core", f"{AC}/tss.rs",
     "        u32::from(K > 0)",
     "        K as u32",
     "beta_0 of the tiling counts cells instead of components"),

    ("telemetry-lipschitz-optimistic", "aether-core", f"{AC}/scm.rs",
     "        1.0 - self.alpha_min.min(self.alpha_max)",
     "        1.0 - self.alpha_min",
     "Lipschitz constant optimistic when the gains are supplied swapped"),

    ("scm-converged-reads-the-bound", "aether-core", f"{AC}/scm.rs",
     "            converged: contracts && final_error <= tolerance,",
     "            converged: final_error <= theoretical_error_bound + 1e-9,",
     "convergence check becomes an identity again"),

    ("nettree-packing-boundary", "aether-core", f"{AC}/nettree.rs",
     "points[c].distance(&points[k]) <= radius",
     "points[c].distance(&points[k]) < radius",
     "net points may sit exactly at the separation radius"),

    ("gc-core-precision-floor", "aether-core", f"{AC}/geodesic_consolidation.rs",
     "sqrt(h.clamp(0.0, 1.0))",
     "sqrt(h.clamp(1e-16, 1.0))",
     "floors tiny separations: distinct points within 1e-8 read as coincident"),

    ("gc-core-latitude", "aether-core", f"{AC}/geodesic_consolidation.rs",
     "    let h = half_dt * half_dt + sin(theta_a) * sin(theta_b) * half_dp * half_dp;",
     "    let h = half_dp * half_dp + sin(theta_a) * sin(theta_b) * half_dt * half_dt;",
     "swaps the haversine terms: the latitude convention again"),

    ("gc-verified-acos-form", "aether_verified", f"{AV}/aether_tss.rs",
     "    2.0 * libm::asin(libm::sqrt(h.clamp(0.0, 1.0)))",
     "    libm::acos((1.0 - 2.0 * h).clamp(-1.0, 1.0))",
     "reverts to acos-of-dot-product in the verified crate"),

    ("gc-verified-latitude", "aether_verified", f"{AV}/aether_tss.rs",
     "    let h = half_dt * half_dt + libm::sin(t1) * libm::sin(t2) * half_dp * half_dp;",
     "    let h = half_dp * half_dp + libm::sin(t1) * libm::sin(t2) * half_dt * half_dt;",
     "swaps the haversine terms in the verified crate"),

    ("deletion-times-drop-eps-scaling", "aether-core", f"{AC}/nettree.rs",
     "        let scale = 1.0 / (eps * (1.0 - 2.0 * eps));",
     "        let scale = 1.0;",
     "deletion times lose their eps dependence: Sheehy section 6 discarded"),

    ("deletion-times-own-radius", "aether-core", f"{AC}/nettree.rs",
     "                let parent = (l + 1).min(top);",
     "                let parent = l;",
     "uses the node's own radius instead of its parent's"),

    ("sparse-liveness-dropped", "aether-core", f"{AC}/nettree.rs",
     "    d.is_finite() && d >= 0.0",
     "    d.is_finite() && d >= -1.0",
     "entry time accepts negative distances: sparsification predicate corrupted"),

    # EQUIVALENT MUTANT - retained, not counted. Changing the early-exit probe
    # from admits(d) to admits(d*0.5) alters only WHEN the short circuit fires,
    # never what it returns: the branch still returns d, and admits is monotone
    # by Lemma 4.1 so the mutant fires strictly less often. When it does not
    # fire, the bisection produces the same answer. No test can kill it.
    # ("sparse-prefilter-unsound", ...) - disabled, see above.

    # ---- oracle coverage ----------------------------------------------------
    # persistence.rs and diagram.rs are the reference every claim in this loop
    # is measured against. Section 5 of the loop prompt forbids CHANGING them;
    # these mutations are transient and restored in the `finally` block. The
    # question they answer is whether the oracle guards itself.
    # --- Sheehy section 4, which had no mutant at all until iteration 45 ----
    # The coverage map showed every test in house_relaxed_distance.rs and
    # house_relaxed_entry_time.rs going green for all 31 mutations. Not because
    # they cannot fail, but because `weight` - the function the whole sparse
    # construction rests on - was never mutated.
    ("weight-middle-slope", "aether-core", f"{AC}/nettree.rs",
     "        0.5 * (alpha - knee)",
     "        1.0 * (alpha - knee)",
     "the middle weight piece doubles its slope, breaking the half-Lipschitz "
     "bound and continuity at t"),

    ("weight-eps-branch-unscaled", "aether-core", f"{AC}/nettree.rs",
     "        eps * alpha",
     "        alpha",
     "the deleted-regime weight loses its eps factor, so d_alpha exceeds alpha "
     "for every pair and nothing ever enters"),

    ("verify-shape-length-cap", "aether-core", f"{AC}/topology.rs",
     "    let max_assessable = (256.0 / DENSITY_MIN) as usize;",
     "    let max_assessable = (256.0 * DENSITY_MIN) as usize;",
     "the assessable-length cap collapses from 2560 to 25, so ordinary inputs "
     "are declined instead of judged"),

    # --- corrections found by adversarial verification (iteration 44) ------
    # The interval probe formed `lo + hi` before halving, which overflows to
    # infinity for large deletion times and drops both weights into the eps
    # piece. Found by a verifier, not by the sweep, which caps t at 50.
    ("entry-exact-probe-overflow", "aether-core", f"{AC}/nettree.rs",
     "            lo + 0.5 * (hi - lo)",
     "            0.5 * (lo + hi)",
     "interval probe overflows for large deletion times: returns a root 8.7% "
     "late at d=9.2e307, t_p=1e308, t_q=1.11111e308, eps=0.05"),

    ("hyperbolic-restore-ball-clamp", "aether-core", f"{AC}/hyperbolic_geometry.rs",
     "        sqrt(sum)" + chr(10) + "    }",
     "        sqrt(sum).min(self.max_norm)" + chr(10) + "    }",
     "hyperbolic distance saturates at 2 atanh(1 - BALL_MARGIN) = "
     "12.206067645522225 for every point past the knee"),

    ("tss-fixture-latitude-points", "epsilon-os", f"{EO}/world.rs",
     "        let centroids = [(0.0, 0.0), (1.2, 0.0), (2.4, 0.0)];",
     "        let centroids = [(0.0, 0.0), (1.2, 0.0), (0.0, 1.2)];",
     "T1_TSS fixture reverts to latitude-convention points, two of which are "
     "the same pole under colatitude"),

    # --- beta_2 is not an Euler solve (iteration 43) -----------------------
    ("betti-2-euler-solve-restored", "epsilon", f"{EP}/manifold.rs",
     "    pub fn betti_2(&self) -> u32 {" + chr(10) + "        0" + chr(10) + "    }",
     "    pub fn betti_2(&self) -> u32 {" + chr(10) + "        self.euler_defect()"
     + chr(10) + "    }",
     "beta_2 solves the Euler identity again: reports 512 spherical voids in a "
     "flat disc"),

    ("estimate-betti1-drops-components", "epsilon", f"{EP}/manifold.rs",
     "        let b1 = e - v + b0;",
     "        let b1 = e - v;",
     "graph beta_1 loses its component term, so E - V + beta_0 no longer holds"),

    # --- beta_1 is not a rank (iteration 42) -------------------------------
    ("betti-1-returns-the-statistic", "aether-core", f"{AC}/topology.rs",
     "pub fn betti_1(_data: &[u8]) -> u32 {" + chr(10) + "    0" + chr(10) + "}",
     "pub fn betti_1(_data: &[u8]) -> u32 {" + chr(10)
     + "    oscillation_count(_data)" + chr(10) + "}",
     "beta_1 reports the window statistic again: grows with sample count and "
     "depends on arrival order"),

    ("shape-uses-true-betti1", "aether-core", f"{AC}/topology.rs",
     "    let oscillation = oscillation_count(data);",
     "    let oscillation = betti_1(data);",
     "TopologicalShape carries the identically-zero beta_1, which makes the "
     "MAX_OSCILLATION branch of verify_shape unreachable"),

    ("oscillation-window-width", "aether-core", f"{AC}/topology.rs",
     "    for window in data.windows(4) {",
     "    for window in data.windows(5) {",
     "the oscillation statistic scans 5-windows, so its exact closed form on a "
     "period-3 pattern moves from len-3 to len-4"),

    # --- the closed-form entry time (iteration 41) ------------------------
    # The bisection is retained as the reference implementation; these check
    # that the closed form is genuinely doing the work rather than agreeing by
    # luck. Each is a one-line edit to an affine coefficient or a root.
    ("entry-exact-root-sign", "aether-core", f"{AC}/nettree.rs",
     "            let root = -c / s;",
     "            let root = c / s;",
     "closed-form root takes the wrong sign of the affine solve"),

    ("entry-exact-probe-at-endpoint", "aether-core", f"{AC}/nettree.rs",
     "            lo + 0.5 * (hi - lo)",
     "            lo",
     "affine piece sampled at the left endpoint, which belongs to the piece "
     "before it: every interval reads the wrong slope"),

    ("entry-exact-drop-knee-intercept", "aether-core", f"{AC}/nettree.rs",
     "        (0.5, -0.5 * knee)",
     "        (0.5, 0.0)",
     "middle weight piece loses its intercept, so it no longer meets the "
     "flat piece at the knee"),

    ("oracle-face-before-coface", "aether-core", f"{AC}/persistence.rs",
     "        .then(a.dimension.cmp(&b.dimension))",
     "        .then(b.dimension.cmp(&a.dimension))",
     "equal-filtration simplices ordered coface-first: not a filtration"),

    # EQUIVALENT MUTANT - deliberately retained, and deliberately not counted.
    # The lookup key is (face, face_len) with face_len = simplex.len - 1, while
    # the simplex's own key is (vertices, simplex.len). Different lengths give
    # different keys, so idx == simplex_idx is unreachable and `<` versus `<=`
    # cannot differ. No test can kill it because it changes no behaviour. Kept
    # as a worked example: a survivor is not automatically a gap.
    # ("oracle-boundary-guard", ... ) - disabled, see above.

    ("oracle-wrong-pivot", "aether-core", f"{AC}/persistence.rs",
     "        while let Some(&low) = column.last() {",
     "        while let Some(&low) = column.first() {",
     "reduction pivots on the lowest index instead of the highest"),

    ("oracle-rips-max3-to-min", "aether-core", f"{AC}/persistence.rs",
     "    a.max(b).max(c)" + chr(10) + "}",
     "    a.min(b).min(c)" + chr(10) + "}",
     "triangle enters at its shortest edge, not its longest"),

    # max3's fragment is a strict PREFIX of max6's, and replace(old, new, 1)
    # takes the first match, so an unanchored "a.max(b).max(c)" only ever
    # mutated max3. max6 computes the tetrahedron filtration value and drives
    # every H2 bar; it was unguarded until this entry existed.
    ("oracle-rips-max6-to-min", "aether-core", f"{AC}/persistence.rs",
     "    a.max(b).max(c).max(d).max(e).max(f)",
     "    a.min(b).min(c).min(d).min(e).min(f)",
     "tetrahedron enters at its shortest edge: every H2 birth is wrong"),

    ("oracle-betti-halfopen", "aether-core", f"{AC}/persistence.rs",
     "pair.death.map(|death| radius < death).unwrap_or(true)",
     "pair.death.map(|death| radius <= death).unwrap_or(true)",
     "betti_at counts bars at their exact death radius"),

    ("oracle-diagonal-cost", "aether-core", f"{AC}/diagram.rs",
     "                (true, false) => (a[i].1 - a[i].0) / 2.0,",
     "                (true, false) => a[i].1 - a[i].0,",
     "distance to the diagonal doubled: bottleneck inflated"),

    ("oracle-linf-to-min", "aether-core", f"{AC}/diagram.rs",
     "                    fmax((b0 - b1).abs(), (d0 - d1).abs())",
     "                    (b0 - b1).abs().min((d0 - d1).abs())",
     "L-infinity cost takes the smaller coordinate difference"),
]


NL = chr(10)  # written as chr(10) so no escape survives a shell round trip
CRLF = chr(13) + chr(10)


LOCK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".gate.lock")


def acquire_lock():
    """Refuse to start while another gate run, or any other cargo build, is live.

    This gate edits shared source files in place. Anything else compiling the
    workspace at the same time sees a mutated tree, and the symptom is
    *unrelated tests failing* - which reads as a regression in whatever the
    other command was checking. That has happened twice: once from two gate
    runs launched concurrently, once from a `cargo test --workspace` run
    started while `ci_parity.sh` was in its gate step, which reported two
    great-circle failures that had nothing to do with great circles.

    The lock only stops a second *gate*. It cannot stop an unrelated cargo
    command, so the message says what the hazard is rather than implying the
    tree is protected.
    """
    try:
        fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            with open(LOCK_PATH) as fh:
                held_by = fh.read().strip()
        except OSError:
            held_by = "unknown"
        print("ANOTHER GATE RUN HOLDS THE LOCK (pid " + held_by + ").")
        print()
        print("This gate rewrites source files in place, so two runs corrupt")
        print("each other and any concurrent cargo build sees a mutated tree.")
        print("If no gate is actually running, the previous one was killed:")
        print("  rm " + LOCK_PATH)
        print("and rerun - startup recovery will restore any mutated file.")
        return False
    os.write(fd, str(os.getpid()).encode("ascii"))
    os.close(fd)
    return True


def release_lock():
    try:
        os.remove(LOCK_PATH)
    except OSError:
        pass


def as_bytes(fragment, body):
    """Encode a fragment to match the line endings `body` actually uses.

    A fragment anchored on a newline is the only way to disambiguate a target
    that is a prefix of another. Encoded naively it cannot match a CRLF file,
    which turns a real guard into a silent SKIP. Returns whichever encoding
    occurs in `body`, preferring LF.
    """
    lf = fragment.encode("utf-8")
    if lf in body:
        return lf
    crlf = fragment.replace(NL, CRLF).encode("utf-8")
    return crlf if crlf in body else lf


def audit_fragments():
    """Check every mutation targets exactly what it claims to target.

    Two failure modes, both silent, both of which this gate shipped with:

    * **Prefix collision.** max6 contains max3 as a prefix, and
      replace(old, new, 1) takes the first match. Every run mutated max3
      twice over and left the entire H2 filtration path unguarded, while the
      summary line reported 19 caught, 0 survived.
    * **Ambiguous target.** A fragment occurring more than once mutates only
      the first site. The others are unguarded and nothing says so.

    Returns a list of problems. Empty means every fragment resolves to exactly
    one site and no fragment shadows another.
    """
    problems = []
    seen = {}
    for mid, _crate, relpath, old, _new, _breaks in MUTATIONS:
        path = os.path.join(ROOT, relpath)
        if not os.path.exists(path):
            continue
        with open(path, "rb") as fh:
            body = fh.read()
        n = body.count(as_bytes(old, body))
        if n == 0:
            problems.append(mid + ": fragment absent from " + relpath)
        elif n > 1:
            problems.append(
                mid + ": fragment occurs " + str(n) + " times in " + relpath
                + "; only the first is mutated, the rest are unguarded")
        seen.setdefault(relpath, []).append((mid, old))

    for relpath, entries in seen.items():
        for i, (mid_a, old_a) in enumerate(entries):
            for mid_b, old_b in entries[i + 1:]:
                if old_a != old_b and (old_a in old_b or old_b in old_a):
                    short, long_ = ((mid_a, mid_b) if len(old_a) < len(old_b)
                                    else (mid_b, mid_a))
                    problems.append(
                        short + "'s fragment is contained in " + long_
                        + "'s in " + relpath + "; the shorter one shadows it")
    return problems


BACKUP_SUFFIX = ".gate-backup"


def recover_pending():
    """Restore any file a previous run left mutated, before doing anything.

    A gate killed mid-mutation leaves the source mutated and reports nothing;
    the next reader sees a poisoned tree with no indication why. The obvious
    fix - handle SIGTERM and let the existing try/finally unwind - does not
    work here: on Windows a terminate() is TerminateProcess, which delivers no
    signal and runs no handler. Verified by killing a live run, which still
    left a mutation in scm.rs with the handler installed.

    So the guarantee is moved off the process and onto the disk. A byte copy
    of the original is written before the mutation and removed only after the
    restore, which makes recovery independent of how the process died.
    """
    recovered = []
    for _mid, _crate, relpath, _old, _new, _breaks in MUTATIONS:
        path = os.path.join(ROOT, relpath)
        backup = path + BACKUP_SUFFIX
        if not os.path.exists(backup):
            continue
        with open(backup, "rb") as fh:
            body = fh.read()
        with open(path, "wb") as fh:
            fh.write(body)
        os.remove(backup)
        if relpath not in recovered:
            recovered.append(relpath)
    for relpath in recovered:
        print("RECOVERED  a previous run was killed mid-mutation; restored "
              + relpath, flush=True)
    if recovered:
        print()


def run(cmd):
    return subprocess.run(cmd, cwd=ROOT, shell=True, capture_output=True, text=True)


def main():
    quick = "--quick" in sys.argv
    if not acquire_lock():
        return 3
    try:
        return _run(quick)
    finally:
        release_lock()


def _run(quick):
    recover_pending()

    # A mutation that does not resolve to exactly one site proves nothing,
    # and says nothing about it. Audit before running anything.
    problems = audit_fragments()
    if problems:
        print("FRAGMENT AUDIT FAILED - mutations that miss their target:")
        for pr in problems:
            print("  " + pr)
        return 2
    print("fragment audit: " + str(len(MUTATIONS))
          + " mutations, each resolving to exactly one site")
    print()

    results = []
    killers = set()

    for mid, crate, relpath, old, new, breaks in MUTATIONS:
        path = os.path.join(ROOT, relpath)
        if not os.path.exists(path):
            results.append((mid, "SKIP", "file missing", breaks))
            print(f"{'SKIP':9} {mid:34} file missing")
            continue
        # Binary I/O: text mode rewrites LF as CRLF on Windows, so the
        # "restored exactly as found" guarantee held only up to line
        # endings and left files showing as modified in git status.
        with open(path, "rb") as fh:
            original = fh.read()
        old_b = as_bytes(old, original)
        new_b = (new.replace(NL, CRLF).encode("utf-8")
                 if CRLF.encode("utf-8") in old_b else new.encode("utf-8"))
        if old_b not in original:
            results.append((mid, "SKIP", "fragment not found - source moved", breaks))
            print(f"{'SKIP':9} {mid:34} fragment not found - source moved")
            continue

        backup = path + BACKUP_SUFFIX
        with open(backup, "wb") as fh:
            fh.write(original)
        with open(path, "wb") as fh:
            fh.write(original.replace(old_b, new_b, 1))
        try:
            target = f"-p {crate}" if quick else "--workspace"
            # --no-fail-fast is load-bearing for the coverage map, not a nicety.
            # cargo stops running later test BINARIES once one target
            # fails, so without it the gate only ever sees failures from
            # the first failing binary and every later one is never run.
            # That made 12 tests in house_relaxed_distance.rs and
            # house_relaxed_entry_time.rs look permanently green: they
            # were not passing, they were not being executed.
            proc = run(f"cargo test {target} --no-fail-fast 2>&1")
            names = [ln.split(" ... ")[0].replace("test ", "").strip()
                     for ln in proc.stdout.splitlines()
                     if " ... FAILED" in ln]
            compile_failed = ("error[E" in proc.stdout
                              or "could not compile" in proc.stdout)
            if names:
                verdict, detail = "CAUGHT", f"{len(names)} test(s), first: {names[0]}"
                killers.update(names)
            elif compile_failed:
                verdict = "NOCOMPILE"
                detail = "does not compile - not a valid mutation, rewrite it"
            elif proc.returncode != 0:
                verdict, detail = "CAUGHT", "non-zero exit with no named failure"
            else:
                verdict, detail = "SURVIVED", "NO TEST DETECTED THIS"
        finally:
            with open(path, "wb") as fh:
                fh.write(original)
            os.remove(backup)
        results.append((mid, verdict, detail, breaks))
        print(f"{verdict:9} {mid:34} {detail}")

    # Which tests are load-bearing? A mutation gate proves each MUTANT is
    # caught. It says nothing about a test that no mutant can make fail, and
    # this effort has already shipped five checks that could not fail. This is
    # the inverse map: every test in the effort's own files that never went red
    # for any of the 31 mutations.
    if "--coverage" in sys.argv:
        import glob
        import re

        never = []
        for path in sorted(glob.glob(os.path.join(ROOT, "kernel", "**", "tests",
                                                  "house_*.rs"), recursive=True)):
            body = open(path, encoding="utf-8", errors="replace").read()
            names_in_file = re.findall(r"fn\s+([a-z0-9_]+)\s*\(", body)
            tests = [n for n in names_in_file
                     if re.search(r"#\[test\][^#]*?fn\s+" + n + r"\s*\(", body,
                                  re.S)]
            for t in tests:
                if not any(t == k or k.endswith("::" + t) for k in killers):
                    never.append(os.path.basename(path) + "::" + t)
        print()
        print(f"COVERAGE - {len(killers)} distinct tests went red for at least "
              f"one mutation")
        if quick:
            print("  NOTE: --quick runs only the mutated crate's tests, so a")
            print("  mutation in one crate can never redden a test in another.")
            print("  Part of the list below is an artefact of that, not a")
            print("  property of the tests. Run without --quick for a true map.")
        if never:
            print(f"{len(never)} test(s) in this effort never went red for any "
                  f"mutation. Each is either guarding something no mutant "
                  f"expresses, or is unfalsifiable:")
            for n in never:
                print("  " + n)
        else:
            print("every test in the effort's own files is killed by "
                  "some mutation")

    survived = [r for r in results if r[1] == "SURVIVED"]
    skipped = [r for r in results if r[1] == "SKIP"]
    nocomp = [r for r in results if r[1] == "NOCOMPILE"]
    caught = len(results) - len(survived) - len(skipped) - len(nocomp)
    print()
    print(f"{caught} caught, {len(survived)} survived, "
          f"{len(nocomp)} did not compile, {len(skipped)} skipped")

    if nocomp:
        print()
        print("NOT VALID MUTATIONS - the compiler rejected these, so they prove")
        print("nothing about the tests. Rewrite them to compile:")
        for mid, _, _, breaks in nocomp:
            print(f"  {mid}: {breaks}")
    if skipped:
        print()
        print("SKIPPED - the source moved. Retarget these or the gate silently")
        print("stops checking that file:")
        for mid, _, _, breaks in skipped:
            print(f"  {mid}: {breaks}")
    if survived:
        print()
        print("GAPS - these mutations were not detected by any test:")
        for mid, _, _, breaks in survived:
            print(f"  {mid}: {breaks}")
    # A skipped or non-compiling mutation proves nothing about the tests,
    # so it must not exit clean. This returned 0 for both and the
    # ci_parity gate then read the run as a pass.
    return 1 if (survived or skipped or nocomp) else 0


if __name__ == "__main__":
    sys.exit(main())
