//! Mutation gate for the mathematics in aether-core, epsilon and aether-verified.
//!
//! A test that cannot fail is decoration. This applies a named one-line mutation to
//! a source file, runs the owning crate's tests, and records whether anything went
//! red. A mutation that NO test catches is a gap in the suite, not a bug in the
//! mutation.
//!
//! Every mutation is reverted from an in-memory copy of the original file, with a
//! drop guard covering early exits and panics, so the tree is left exactly as found
//! even if a run fails mid-way. A byte copy on disk covers a killed process.
//! `persistence.rs` and `diagram.rs` are mutated transiently for coverage
//! measurement only; they are never left modified.
//!
//! A mutation that does not COMPILE is reported as NOCOMPILE rather than counted as
//! caught. The compiler is not a test, and crediting it would overstate the suite.
//!
//! Usage:  `cargo run -p math-mutation-gate -- [--quick] [--coverage]`
//!
//! `--quick` runs only the crate owning each mutated file rather than the whole
//! workspace. `--coverage` also lists every test in `kernel/**/tests/house_*.rs`
//! that no mutation turned red.

use std::collections::HashSet;
use std::fs;
use std::io::{self, Write as _};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

const AC: &str = "kernel/epsilon/epsilon/crates/aether-core/src";
const EP: &str = "kernel/epsilon/epsilon/crates/epsilon/src";
const AV: &str = "kernel/aether/aether-verified/src";
const EO: &str = "kernel/epsilon/epsilon/crates/epsilon-os/src";

const BACKUP_SUFFIX: &str = ".gate-backup";

/// A named one-line mutation: `old` is replaced by `new` in `dir/file`, and the
/// tests of `krate` are expected to go red because of what `breaks` describes.
struct Mutation {
    id: &'static str,
    krate: &'static str,
    dir: &'static str,
    file: &'static str,
    old: &'static str,
    new: &'static str,
    breaks: &'static str,
}

impl Mutation {
    fn relpath(&self) -> String {
        format!("{}/{}", self.dir, self.file)
    }
}

const MUTATIONS: &[Mutation] = &[
    // ---- repairs made by this loop -----------------------------------------
    Mutation {
        id: "beta0-drop-plus-one",
        krate: "aether-core",
        dir: AC,
        file: "topology.rs",
        old: "None => components = 1,",
        new: "None => components = 0,",
        breaks: "beta_0 counts gaps instead of gaps+1",
    },

    Mutation {
        id: "beta0-gap-comparison",
        krate: "aether-core",
        dir: AC,
        file: "topology.rs",
        old: "Some(p) if value - p > threshold => components += 1,",
        new: "Some(p) if value - p >= threshold => components += 1,",
        breaks: "off-by-one at the threshold boundary",
    },

    Mutation {
        id: "voronoi-betti0-returns-k",
        krate: "aether-core",
        dir: AC,
        file: "tss.rs",
        old: "        u32::from(K > 0)",
        new: "        K as u32",
        breaks: "beta_0 of the tiling counts cells instead of components",
    },

    Mutation {
        id: "telemetry-lipschitz-optimistic",
        krate: "aether-core",
        dir: AC,
        file: "scm.rs",
        old: "        1.0 - self.alpha_min.min(self.alpha_max)",
        new: "        1.0 - self.alpha_min",
        breaks: "Lipschitz constant optimistic when the gains are supplied swapped",
    },

    Mutation {
        id: "scm-converged-reads-the-bound",
        krate: "aether-core",
        dir: AC,
        file: "scm.rs",
        old: "            converged: contracts && final_error <= tolerance,",
        new: "            converged: final_error <= theoretical_error_bound + 1e-9,",
        breaks: "convergence check becomes an identity again",
    },

    Mutation {
        id: "nettree-packing-boundary",
        krate: "aether-core",
        dir: AC,
        file: "nettree.rs",
        old: "points[c].distance(&points[k]) <= radius",
        new: "points[c].distance(&points[k]) < radius",
        breaks: "net points may sit exactly at the separation radius",
    },

    Mutation {
        id: "gc-core-precision-floor",
        krate: "aether-core",
        dir: AC,
        file: "geodesic_consolidation.rs",
        old: "sqrt(h.clamp(0.0, 1.0))",
        new: "sqrt(h.clamp(1e-16, 1.0))",
        breaks: "floors tiny separations: distinct points within 1e-8 read as coincident",
    },

    Mutation {
        id: "gc-core-latitude",
        krate: "aether-core",
        dir: AC,
        file: "geodesic_consolidation.rs",
        old: "    let h = half_dt * half_dt + sin(theta_a) * sin(theta_b) * half_dp * half_dp;",
        new: "    let h = half_dp * half_dp + sin(theta_a) * sin(theta_b) * half_dt * half_dt;",
        breaks: "swaps the haversine terms: the latitude convention again",
    },

    Mutation {
        id: "gc-verified-acos-form",
        krate: "aether_verified",
        dir: AV,
        file: "aether_tss.rs",
        old: "    2.0 * libm::asin(libm::sqrt(h.clamp(0.0, 1.0)))",
        new: "    libm::acos((1.0 - 2.0 * h).clamp(-1.0, 1.0))",
        breaks: "reverts to acos-of-dot-product in the verified crate",
    },

    Mutation {
        id: "gc-verified-latitude",
        krate: "aether_verified",
        dir: AV,
        file: "aether_tss.rs",
        old: "    let h = half_dt * half_dt + libm::sin(t1) * libm::sin(t2) * half_dp * half_dp;",
        new: "    let h = half_dp * half_dp + libm::sin(t1) * libm::sin(t2) * half_dt * half_dt;",
        breaks: "swaps the haversine terms in the verified crate",
    },

    Mutation {
        id: "deletion-times-drop-eps-scaling",
        krate: "aether-core",
        dir: AC,
        file: "nettree.rs",
        old: "        let scale = 1.0 / (eps * (1.0 - 2.0 * eps));",
        new: "        let scale = 1.0;",
        breaks: "deletion times lose their eps dependence: Sheehy section 6 discarded",
    },

    Mutation {
        id: "deletion-times-own-radius",
        krate: "aether-core",
        dir: AC,
        file: "nettree.rs",
        old: "                let parent = (l + 1).min(top);",
        new: "                let parent = l;",
        breaks: "uses the node's own radius instead of its parent's",
    },

    Mutation {
        id: "sparse-liveness-dropped",
        krate: "aether-core",
        dir: AC,
        file: "nettree.rs",
        old: "    d.is_finite() && d >= 0.0",
        new: "    d.is_finite() && d >= -1.0",
        breaks: "entry time accepts negative distances: sparsification predicate corrupted",
    },

    // EQUIVALENT MUTANT - retained, not counted. Changing the early-exit probe
    // from admits(d) to admits(d*0.5) alters only WHEN the short circuit fires,
    // never what it returns: the branch still returns d, and admits is monotone
    // by Lemma 4.1 so the mutant fires strictly less often. When it does not
    // fire, the bisection produces the same answer. No test can kill it.
    // ("sparse-prefilter-unsound", ...) - disabled, see above.

    // ---- oracle coverage ----------------------------------------------------
    // persistence.rs and diagram.rs are the reference every claim in this loop
    // is measured against. Section 5 of the loop prompt forbids CHANGING them;
    // these mutations are transient and restored by the drop guard. The
    // question they answer is whether the oracle guards itself.
    // --- Sheehy section 4, which had no mutant at all until iteration 45 ----
    // The coverage map showed every test in house_relaxed_distance.rs and
    // house_relaxed_entry_time.rs going green for all 31 mutations. Not because
    // they cannot fail, but because `weight` - the function the whole sparse
    // construction rests on - was never mutated.
    Mutation {
        id: "weight-middle-slope",
        krate: "aether-core",
        dir: AC,
        file: "nettree.rs",
        old: "        0.5 * (alpha - knee)",
        new: "        1.0 * (alpha - knee)",
        breaks: "the middle weight piece doubles its slope, breaking the half-Lipschitz bound and continuity at t",
    },

    Mutation {
        id: "weight-eps-branch-unscaled",
        krate: "aether-core",
        dir: AC,
        file: "nettree.rs",
        old: "        eps * alpha",
        new: "        alpha",
        breaks: "the deleted-regime weight loses its eps factor, so d_alpha exceeds alpha for every pair and nothing ever enters",
    },

    Mutation {
        id: "verify-shape-length-cap",
        krate: "aether-core",
        dir: AC,
        file: "topology.rs",
        old: "    let max_assessable = (256.0 / DENSITY_MIN) as usize;",
        new: "    let max_assessable = (256.0 * DENSITY_MIN) as usize;",
        breaks: "the assessable-length cap collapses from 2560 to 25, so ordinary inputs are declined instead of judged",
    },

    // --- corrections found by adversarial verification (iteration 44) ------
    // The interval probe formed `lo + hi` before halving, which overflows to
    // infinity for large deletion times and drops both weights into the eps
    // piece. Found by a verifier, not by the sweep, which caps t at 50.
    Mutation {
        id: "entry-exact-probe-overflow",
        krate: "aether-core",
        dir: AC,
        file: "nettree.rs",
        old: "            lo + 0.5 * (hi - lo)",
        new: "            0.5 * (lo + hi)",
        breaks: "interval probe overflows for large deletion times: returns a root 8.7% late at d=9.2e307, t_p=1e308, t_q=1.11111e308, eps=0.05",
    },

    Mutation {
        id: "hyperbolic-restore-ball-clamp",
        krate: "aether-core",
        dir: AC,
        file: "hyperbolic_geometry.rs",
        old: "        sqrt(sum)\n    }",
        new: "        sqrt(sum).min(self.max_norm)\n    }",
        breaks: "hyperbolic distance saturates at 2 atanh(1 - BALL_MARGIN) = 12.206067645522225 for every point past the knee",
    },

    Mutation {
        id: "tss-fixture-latitude-points",
        krate: "epsilon-os",
        dir: EO,
        file: "world.rs",
        old: "        let centroids = [(0.0, 0.0), (1.2, 0.0), (2.4, 0.0)];",
        new: "        let centroids = [(0.0, 0.0), (1.2, 0.0), (0.0, 1.2)];",
        breaks: "T1_TSS fixture reverts to latitude-convention points, two of which are the same pole under colatitude",
    },

    // --- beta_2 is not an Euler solve (iteration 43) -----------------------
    Mutation {
        id: "betti-2-euler-solve-restored",
        krate: "epsilon",
        dir: EP,
        file: "manifold.rs",
        old: "    pub fn betti_2(&self) -> u32 {\n        0\n    }",
        new: "    pub fn betti_2(&self) -> u32 {\n        self.euler_defect()\n    }",
        breaks: "beta_2 solves the Euler identity again: reports 512 spherical voids in a flat disc",
    },

    Mutation {
        id: "estimate-betti1-drops-components",
        krate: "epsilon",
        dir: EP,
        file: "manifold.rs",
        old: "        let b1 = e - v + b0;",
        new: "        let b1 = e - v;",
        breaks: "graph beta_1 loses its component term, so E - V + beta_0 no longer holds",
    },

    // --- beta_1 is not a rank (iteration 42) -------------------------------
    Mutation {
        id: "betti-1-returns-the-statistic",
        krate: "aether-core",
        dir: AC,
        file: "topology.rs",
        old: "pub fn betti_1(_data: &[u8]) -> u32 {\n    0\n}",
        new: "pub fn betti_1(_data: &[u8]) -> u32 {\n    oscillation_count(_data)\n}",
        breaks: "beta_1 reports the window statistic again: grows with sample count and depends on arrival order",
    },

    Mutation {
        id: "shape-uses-true-betti1",
        krate: "aether-core",
        dir: AC,
        file: "topology.rs",
        old: "    let oscillation = oscillation_count(data);",
        new: "    let oscillation = betti_1(data);",
        breaks: "TopologicalShape carries the identically-zero beta_1, which makes the MAX_OSCILLATION branch of verify_shape unreachable",
    },

    Mutation {
        id: "oscillation-window-width",
        krate: "aether-core",
        dir: AC,
        file: "topology.rs",
        old: "    for window in data.windows(4) {",
        new: "    for window in data.windows(5) {",
        breaks: "the oscillation statistic scans 5-windows, so its exact closed form on a period-3 pattern moves from len-3 to len-4",
    },

    // --- the closed-form entry time (iteration 41) ------------------------
    // The bisection is retained as the reference implementation; these check
    // that the closed form is genuinely doing the work rather than agreeing by
    // luck. Each is a one-line edit to an affine coefficient or a root.
    Mutation {
        id: "entry-exact-root-sign",
        krate: "aether-core",
        dir: AC,
        file: "nettree.rs",
        old: "            let root = -c / s;",
        new: "            let root = c / s;",
        breaks: "closed-form root takes the wrong sign of the affine solve",
    },

    Mutation {
        id: "entry-exact-probe-at-endpoint",
        krate: "aether-core",
        dir: AC,
        file: "nettree.rs",
        old: "            lo + 0.5 * (hi - lo)",
        new: "            lo",
        breaks: "affine piece sampled at the left endpoint, which belongs to the piece before it: every interval reads the wrong slope",
    },

    Mutation {
        id: "entry-exact-drop-knee-intercept",
        krate: "aether-core",
        dir: AC,
        file: "nettree.rs",
        old: "        (0.5, -0.5 * knee)",
        new: "        (0.5, 0.0)",
        breaks: "middle weight piece loses its intercept, so it no longer meets the flat piece at the knee",
    },

    Mutation {
        id: "oracle-face-before-coface",
        krate: "aether-core",
        dir: AC,
        file: "persistence.rs",
        old: "        .then(a.dimension.cmp(&b.dimension))",
        new: "        .then(b.dimension.cmp(&a.dimension))",
        breaks: "equal-filtration simplices ordered coface-first: not a filtration",
    },

    // EQUIVALENT MUTANT - deliberately retained, and deliberately not counted.
    // The lookup key is (face, face_len) with face_len = simplex.len - 1, while
    // the simplex's own key is (vertices, simplex.len). Different lengths give
    // different keys, so idx == simplex_idx is unreachable and `<` versus `<=`
    // cannot differ. No test can kill it because it changes no behaviour. Kept
    // as a worked example: a survivor is not automatically a gap.
    // ("oracle-boundary-guard", ... ) - disabled, see above.

    Mutation {
        id: "oracle-wrong-pivot",
        krate: "aether-core",
        dir: AC,
        file: "persistence.rs",
        old: "        while let Some(&low) = column.last() {",
        new: "        while let Some(&low) = column.first() {",
        breaks: "reduction pivots on the lowest index instead of the highest",
    },

    Mutation {
        id: "oracle-rips-max3-to-min",
        krate: "aether-core",
        dir: AC,
        file: "persistence.rs",
        old: "    a.max(b).max(c)\n}",
        new: "    a.min(b).min(c)\n}",
        breaks: "triangle enters at its shortest edge, not its longest",
    },

    // max3's fragment is a strict PREFIX of max6's, and replace(old, new, 1)
    // takes the first match, so an unanchored "a.max(b).max(c)" only ever
    // mutated max3. max6 computes the tetrahedron filtration value and drives
    // every H2 bar; it was unguarded until this entry existed.
    Mutation {
        id: "oracle-rips-max6-to-min",
        krate: "aether-core",
        dir: AC,
        file: "persistence.rs",
        old: "    a.max(b).max(c).max(d).max(e).max(f)",
        new: "    a.min(b).min(c).min(d).min(e).min(f)",
        breaks: "tetrahedron enters at its shortest edge: every H2 birth is wrong",
    },

    Mutation {
        id: "oracle-betti-halfopen",
        krate: "aether-core",
        dir: AC,
        file: "persistence.rs",
        old: "pair.death.map(|death| radius < death).unwrap_or(true)",
        new: "pair.death.map(|death| radius <= death).unwrap_or(true)",
        breaks: "betti_at counts bars at their exact death radius",
    },

    Mutation {
        id: "oracle-diagonal-cost",
        krate: "aether-core",
        dir: AC,
        file: "diagram.rs",
        old: "                (true, false) => (a[i].1 - a[i].0) / 2.0,",
        new: "                (true, false) => a[i].1 - a[i].0,",
        breaks: "distance to the diagonal doubled: bottleneck inflated",
    },

    Mutation {
        id: "oracle-linf-to-min",
        krate: "aether-core",
        dir: AC,
        file: "diagram.rs",
        old: "                    fmax((b0 - b1).abs(), (d0 - d1).abs())",
        new: "                    (b0 - b1).abs().min((d0 - d1).abs())",
        breaks: "L-infinity cost takes the smaller coordinate difference",
    },
];

/// Holds the gate's lock file for the life of a run.
///
/// This gate edits shared source files in place. Anything else compiling the
/// workspace at the same time sees a mutated tree, and the symptom is
/// *unrelated tests failing* - which reads as a regression in whatever the
/// other command was checking. That has happened twice: once from two gate
/// runs launched concurrently, once from a `cargo test --workspace` run
/// started while `ci_parity.sh` was in its gate step, which reported two
/// great-circle failures that had nothing to do with great circles.
///
/// The lock only stops a second *gate*. It cannot stop an unrelated cargo
/// command, so the message says what the hazard is rather than implying the
/// tree is protected.
struct Lock(PathBuf);

impl Lock {
    fn acquire(path: PathBuf) -> io::Result<Option<Lock>> {
        match fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(mut fh) => {
                let lock = Lock(path);
                write!(fh, "{}", std::process::id())?;
                Ok(Some(lock))
            }
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {
                let held_by = fs::read_to_string(&path)
                    .map_or_else(|_| "unknown".to_string(), |s| s.trim().to_string());
                println!("ANOTHER GATE RUN HOLDS THE LOCK (pid {held_by}).");
                println!();
                println!("This gate rewrites source files in place, so two runs corrupt");
                println!("each other and any concurrent cargo build sees a mutated tree.");
                println!("If no gate is actually running, the previous one was killed:");
                println!("  rm {}", path.display());
                println!("and rerun - startup recovery will restore any mutated file.");
                Ok(None)
            }
            Err(e) => Err(e),
        }
    }
}

impl Drop for Lock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

/// Non-overlapping occurrences of `needle`.
fn count(hay: &[u8], needle: &[u8]) -> usize {
    let (mut n, mut at) = (0, 0);
    while let Some(i) = find(&hay[at..], needle) {
        n += 1;
        at += i + needle.len();
    }
    n
}

/// Encodes a fragment to match the line endings `body` actually uses.
///
/// A fragment anchored on a newline is the only way to disambiguate a target
/// that is a prefix of another. Encoded naively it cannot match a CRLF file,
/// which turns a real guard into a silent SKIP. Returns whichever encoding
/// occurs in `body`, preferring LF.
fn encode_like(fragment: &str, body: &[u8]) -> Vec<u8> {
    let crlf = fragment.replace('\n', "\r\n");
    if find(body, fragment.as_bytes()).is_none() && find(body, crlf.as_bytes()).is_some() {
        crlf.into_bytes()
    } else {
        fragment.as_bytes().to_vec()
    }
}

/// Checks every mutation targets exactly what it claims to target.
///
/// Two failure modes, both silent, both of which this gate shipped with:
///
/// * **Prefix collision.** max6 contains max3 as a prefix, and a first-match
///   replace takes the first match. Every run mutated max3 twice over and left
///   the entire H2 filtration path unguarded, while the summary line reported
///   19 caught, 0 survived.
/// * **Ambiguous target.** A fragment occurring more than once mutates only
///   the first site. The others are unguarded and nothing says so.
///
/// Returns a list of problems. Empty means every fragment resolves to exactly
/// one site and no fragment shadows another.
fn audit_fragments(root: &Path) -> io::Result<Vec<String>> {
    let mut problems = Vec::new();
    let mut seen: Vec<(String, Vec<&Mutation>)> = Vec::new();
    for m in MUTATIONS {
        let relpath = m.relpath();
        let path = root.join(&relpath);
        if !path.exists() {
            continue;
        }
        let body = fs::read(&path)?;
        match count(&body, &encode_like(m.old, &body)) {
            0 => problems.push(format!("{}: fragment absent from {relpath}", m.id)),
            1 => {}
            n => problems.push(format!(
                "{}: fragment occurs {n} times in {relpath}; only the first is mutated, \
                 the rest are unguarded",
                m.id
            )),
        }
        match seen.iter_mut().find(|(p, _)| *p == relpath) {
            Some((_, entries)) => entries.push(m),
            None => seen.push((relpath, vec![m])),
        }
    }

    for (relpath, entries) in &seen {
        for (i, a) in entries.iter().enumerate() {
            for b in &entries[i + 1..] {
                if a.old != b.old && (b.old.contains(a.old) || a.old.contains(b.old)) {
                    let (short, long) = if a.old.len() < b.old.len() {
                        (a, b)
                    } else {
                        (b, a)
                    };
                    problems.push(format!(
                        "{}'s fragment is contained in {}'s in {relpath}; \
                         the shorter one shadows it",
                        short.id, long.id
                    ));
                }
            }
        }
    }
    Ok(problems)
}

fn backup_path(path: &Path) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(BACKUP_SUFFIX);
    PathBuf::from(s)
}

/// Restores any file a previous run left mutated, before doing anything.
///
/// A gate killed mid-mutation leaves the source mutated and reports nothing;
/// the next reader sees a poisoned tree with no indication why. An in-process
/// guard cannot prevent that: on Windows a kill is `TerminateProcess`, which
/// delivers no signal and runs no handler or destructor.
///
/// So the guarantee is moved off the process and onto the disk. A byte copy
/// of the original is written before the mutation and removed only after the
/// restore, which makes recovery independent of how the process died.
fn recover_pending(root: &Path) -> io::Result<()> {
    let mut recovered: Vec<String> = Vec::new();
    for m in MUTATIONS {
        let relpath = m.relpath();
        let path = root.join(&relpath);
        let backup = backup_path(&path);
        if !backup.exists() {
            continue;
        }
        fs::write(&path, fs::read(&backup)?)?;
        fs::remove_file(&backup)?;
        if !recovered.contains(&relpath) {
            recovered.push(relpath);
        }
    }
    // stdout is line-buffered, so each notice is flushed as it is written: a
    // recovery notice lost to buffering is the silent failure this prevents.
    for relpath in &recovered {
        println!("RECOVERED  a previous run was killed mid-mutation; restored {relpath}");
    }
    if !recovered.is_empty() {
        println!();
    }
    Ok(())
}

/// Puts a mutated file's original bytes back, then removes its backup.
///
/// The normal path calls `restore` and propagates its error, so a failed
/// restore stops the run instead of testing later mutations on a poisoned
/// tree. `Drop` covers every other exit, including a panic.
struct Restore<'a> {
    path: &'a Path,
    backup: PathBuf,
    original: &'a [u8],
    done: bool,
}

impl Restore<'_> {
    fn restore(&mut self) -> io::Result<()> {
        fs::write(self.path, self.original)?;
        fs::remove_file(&self.backup)?;
        self.done = true;
        Ok(())
    }
}

impl Drop for Restore<'_> {
    fn drop(&mut self) {
        if !self.done {
            let _ = self.restore();
        }
    }
}

/// Runs the tests that should catch a mutation. Returns whether they passed and
/// their output.
fn run_tests(root: &Path, krate: &str, quick: bool) -> io::Result<(bool, String)> {
    let mut cmd = Command::new("cargo");
    cmd.arg("test");
    if quick {
        cmd.args(["-p", krate]);
    } else {
        cmd.arg("--workspace");
    }
    // --no-fail-fast is load-bearing for the coverage map, not a nicety.
    // cargo stops running later test BINARIES once one target
    // fails, so without it the gate only ever sees failures from
    // the first failing binary and every later one is never run.
    // That made 12 tests in house_relaxed_distance.rs and
    // house_relaxed_entry_time.rs look permanently green: they
    // were not passing, they were not being executed.
    let out = cmd.arg("--no-fail-fast").current_dir(root).output()?;
    // stderr is appended rather than interleaved. Every verdict reads either
    // the `... FAILED` lines, which libtest writes to stdout in order, or a
    // compiler marker whose position does not matter.
    let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&out.stderr));
    Ok((out.status.success(), text))
}

fn is_space(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c)
}

/// Matches `fn\s+([a-z0-9_]+)\s*\(` at byte `i`: the name and where the match
/// ends.
fn fn_name_at(b: &[u8], i: usize) -> Option<(&str, usize)> {
    if !b.get(i..)?.starts_with(b"fn") {
        return None;
    }
    let mut j = i + 2;
    let ws = j;
    while j < b.len() && is_space(b[j]) {
        j += 1;
    }
    let name_start = j;
    while j < b.len() && (b[j].is_ascii_lowercase() || b[j].is_ascii_digit() || b[j] == b'_') {
        j += 1;
    }
    let name_end = j;
    while j < b.len() && is_space(b[j]) {
        j += 1;
    }
    if ws == name_start || name_start == name_end || b.get(j) != Some(&b'(') {
        return None;
    }
    let name = std::str::from_utf8(&b[name_start..name_end]).ok()?;
    Some((name, j + 1))
}

/// Every function name in `body`, left to right, without overlap.
fn fn_names(body: &str) -> Vec<&str> {
    let b = body.as_bytes();
    let mut names = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if let Some((name, end)) = fn_name_at(b, i) {
            names.push(name);
            i = end;
        } else {
            i += 1;
        }
    }
    names
}

/// Whether `name` is a test: some `#[test]` precedes `fn name(` with no `#`
/// between them.
fn is_test(body: &str, name: &str) -> bool {
    let b = body.as_bytes();
    let mut from = 0;
    while let Some(at) = find(&b[from..], b"#[test]") {
        let start = from + at + b"#[test]".len();
        let end = b[start..]
            .iter()
            .position(|&c| c == b'#')
            .map_or(b.len(), |k| start + k);
        if (start..end).any(|s| fn_name_at(&b[..end], s).is_some_and(|(n, _)| n == name)) {
            return true;
        }
        from = start;
    }
    false
}

/// Every `house_*.rs` in a `tests` directory under `dir`, at any depth. Hidden
/// and unreadable directories are skipped, as a `**` glob skips them.
fn collect_house_tests(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') {
            continue;
        }
        if path.is_dir() {
            collect_house_tests(&path, out);
        } else if name.starts_with("house_")
            && name.ends_with(".rs")
            && dir.file_name().is_some_and(|d| d == "tests")
        {
            out.push(path);
        }
    }
}

fn gate(root: &Path, quick: bool, coverage: bool) -> io::Result<u8> {
    recover_pending(root)?;

    // A mutation that does not resolve to exactly one site proves nothing,
    // and says nothing about it. Audit before running anything.
    let problems = audit_fragments(root)?;
    if !problems.is_empty() {
        println!("FRAGMENT AUDIT FAILED - mutations that miss their target:");
        for pr in &problems {
            println!("  {pr}");
        }
        return Ok(2);
    }
    println!(
        "fragment audit: {} mutations, each resolving to exactly one site",
        MUTATIONS.len()
    );
    println!();

    let mut results: Vec<(&Mutation, &str)> = Vec::new();
    let mut killers: HashSet<String> = HashSet::new();

    for m in MUTATIONS {
        let path = root.join(m.relpath());
        if !path.exists() {
            results.push((m, "SKIP"));
            println!("{:9} {:34} file missing", "SKIP", m.id);
            continue;
        }
        // Binary I/O, so the restore puts back the exact bytes, line endings
        // included, and leaves nothing showing as modified in git status.
        let original = fs::read(&path)?;
        let old = encode_like(m.old, &original);
        let new = if find(&old, b"\r\n").is_some() {
            m.new.replace('\n', "\r\n")
        } else {
            m.new.to_string()
        };
        let Some(at) = find(&original, &old) else {
            results.push((m, "SKIP"));
            println!("{:9} {:34} fragment not found - source moved", "SKIP", m.id);
            continue;
        };

        let mut restore = Restore {
            path: &path,
            backup: backup_path(&path),
            original: &original,
            done: false,
        };
        fs::write(&restore.backup, &original)?;
        fs::write(
            &path,
            [&original[..at], new.as_bytes(), &original[at + old.len()..]].concat(),
        )?;
        let outcome = run_tests(root, m.krate, quick);
        restore.restore()?;
        let (passed, text) = outcome?;

        let names: Vec<String> = text
            .lines()
            .filter(|ln| ln.contains(" ... FAILED"))
            .map(|ln| {
                let head = ln.split_once(" ... ").map_or(ln, |(head, _)| head);
                head.replace("test ", "").trim().to_string()
            })
            .collect();
        let compile_failed = text.contains("error[E") || text.contains("could not compile");
        let (verdict, detail) = if let Some(first) = names.first() {
            ("CAUGHT", format!("{} test(s), first: {first}", names.len()))
        } else if compile_failed {
            (
                "NOCOMPILE",
                "does not compile - not a valid mutation, rewrite it".to_string(),
            )
        } else if !passed {
            ("CAUGHT", "non-zero exit with no named failure".to_string())
        } else {
            ("SURVIVED", "NO TEST DETECTED THIS".to_string())
        };
        killers.extend(names);
        results.push((m, verdict));
        println!("{verdict:9} {:34} {detail}", m.id);
    }

    // Which tests are load-bearing? A mutation gate proves each MUTANT is
    // caught. It says nothing about a test that no mutant can make fail, and
    // this effort has already shipped five checks that could not fail. This is
    // the inverse map: every test in the effort's own files that never went red
    // for any mutation.
    if coverage {
        let mut files = Vec::new();
        collect_house_tests(&root.join("kernel"), &mut files);
        files.sort_by_key(|p| p.to_string_lossy().into_owned());
        let mut never = Vec::new();
        for path in &files {
            let body = String::from_utf8_lossy(&fs::read(path)?).into_owned();
            let base = path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            for name in fn_names(&body) {
                let killed = killers
                    .iter()
                    .any(|k| k == name || k.ends_with(&format!("::{name}")));
                if is_test(&body, name) && !killed {
                    never.push(format!("{base}::{name}"));
                }
            }
        }
        println!();
        println!(
            "COVERAGE - {} distinct tests went red for at least one mutation",
            killers.len()
        );
        if quick {
            println!("  NOTE: --quick runs only the mutated crate's tests, so a");
            println!("  mutation in one crate can never redden a test in another.");
            println!("  Part of the list below is an artefact of that, not a");
            println!("  property of the tests. Run without --quick for a true map.");
        }
        if never.is_empty() {
            println!("every test in the effort's own files is killed by some mutation");
        } else {
            println!(
                "{} test(s) in this effort never went red for any mutation. Each is \
                 either guarding something no mutant expresses, or is unfalsifiable:",
                never.len()
            );
            for n in &never {
                println!("  {n}");
            }
        }
    }

    let tally = |v: &str| results.iter().filter(|(_, r)| *r == v).count();
    let (survived, skipped, nocomp) = (tally("SURVIVED"), tally("SKIP"), tally("NOCOMPILE"));
    let caught = results.len() - survived - skipped - nocomp;
    println!();
    println!("{caught} caught, {survived} survived, {nocomp} did not compile, {skipped} skipped");

    list(
        &results,
        "NOCOMPILE",
        &[
            "NOT VALID MUTATIONS - the compiler rejected these, so they prove",
            "nothing about the tests. Rewrite them to compile:",
        ],
    );
    list(
        &results,
        "SKIP",
        &[
            "SKIPPED - the source moved. Retarget these or the gate silently",
            "stops checking that file:",
        ],
    );
    list(
        &results,
        "SURVIVED",
        &["GAPS - these mutations were not detected by any test:"],
    );
    // A skipped or non-compiling mutation proves nothing about the tests,
    // so it must not exit clean. This returned 0 for both and the
    // ci_parity gate then read the run as a pass.
    Ok(u8::from(survived + skipped + nocomp > 0))
}

fn list(results: &[(&Mutation, &str)], verdict: &str, header: &[&str]) {
    let hits: Vec<&Mutation> = results
        .iter()
        .filter(|(_, r)| *r == verdict)
        .map(|(m, _)| *m)
        .collect();
    if hits.is_empty() {
        return;
    }
    println!();
    for line in header {
        println!("{line}");
    }
    for m in hits {
        println!("  {}: {}", m.id, m.breaks);
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let flag = |f: &str| args.iter().any(|a| a == f);
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let root = manifest
        .ancestors()
        .nth(2)
        .expect("tools/math-mutation-gate sits two levels below the repository root");
    let code = match Lock::acquire(manifest.join(".gate.lock")) {
        Ok(Some(_lock)) => gate(root, flag("--quick"), flag("--coverage")),
        Ok(None) => Ok(3),
        Err(e) => Err(e),
    };
    match code {
        Ok(c) => ExitCode::from(c),
        Err(e) => {
            eprintln!("math-mutation-gate: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fragments_and_test_names_resolve_as_the_gate_needs() {
        // Non-overlapping, so a fragment counted twice really is two sites.
        assert_eq!(count(b"aaaa", b"aa"), 2);
        // An LF-anchored fragment matches a CRLF file in its CRLF form.
        assert_eq!(encode_like("x\n}", b"a\r\nx\r\n}\r\n"), b"x\r\n}");
        assert_eq!(encode_like("x\n}", b"x\n}"), b"x\n}");
        // A test is any fn after `#[test]` with no `#` in between.
        let body = "#[test]\nfn a() {}\nfn b() {}\n#[inline]\nfn c() {}\nfn helper_d (x)";
        assert_eq!(fn_names(body), ["a", "b", "c", "helper_d"]);
        assert!(is_test(body, "a") && is_test(body, "b"));
        assert!(!is_test(body, "c") && !is_test(body, "helper_d"));
    }
}
