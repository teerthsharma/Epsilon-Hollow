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
     "    if !d.is_finite() || d < 0.0 {",
     "    if !d.is_finite() || d < -1.0 {",
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

    ("oracle-rips-max-to-min", "aether-core", f"{AC}/persistence.rs",
     "    a.max(b).max(c)",
     "    a.min(b).min(c)",
     "triangle enters at its shortest edge, not its longest"),

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


def run(cmd):
    return subprocess.run(cmd, cwd=ROOT, shell=True, capture_output=True, text=True)


def main():
    quick = "--quick" in sys.argv
    results = []

    for mid, crate, relpath, old, new, breaks in MUTATIONS:
        path = os.path.join(ROOT, relpath)
        if not os.path.exists(path):
            results.append((mid, "SKIP", "file missing", breaks))
            print(f"{'SKIP':9} {mid:34} file missing")
            continue
        original = io.open(path, encoding="utf-8").read()
        if old not in original:
            results.append((mid, "SKIP", "fragment not found - source moved", breaks))
            print(f"{'SKIP':9} {mid:34} fragment not found - source moved")
            continue

        io.open(path, "w", encoding="utf-8").write(original.replace(old, new, 1))
        try:
            target = f"-p {crate}" if quick else "--workspace"
            proc = run(f"cargo test {target} 2>&1")
            names = [ln.split(" ... ")[0].replace("test ", "").strip()
                     for ln in proc.stdout.splitlines()
                     if " ... FAILED" in ln]
            compile_failed = ("error[E" in proc.stdout
                              or "could not compile" in proc.stdout)
            if names:
                verdict, detail = "CAUGHT", f"{len(names)} test(s), first: {names[0]}"
            elif compile_failed:
                verdict = "NOCOMPILE"
                detail = "does not compile - not a valid mutation, rewrite it"
            elif proc.returncode != 0:
                verdict, detail = "CAUGHT", "non-zero exit with no named failure"
            else:
                verdict, detail = "SURVIVED", "NO TEST DETECTED THIS"
        finally:
            io.open(path, "w", encoding="utf-8").write(original)
        results.append((mid, verdict, detail, breaks))
        print(f"{verdict:9} {mid:34} {detail}")

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
    return 1 if survived else 0


if __name__ == "__main__":
    sys.exit(main())
