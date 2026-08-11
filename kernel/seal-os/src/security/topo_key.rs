// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! `topo_key` — a fuzzy extractor whose sketch is an H₀ persistence diagram.
//!
//! # What this is
//!
//! A password is mapped deterministically to a point cloud, the H₀ persistence
//! diagram of the Vietoris–Rips filtration of that cloud is computed exactly,
//! the diagram is quantised to a fingerprint, and the fingerprint feeds a
//! salted iterated-SHA-256 KDF whose output is committed to with an Ed25519
//! public key and a signature over the parameter block.
//!
//! **This is not a cipher.** Every bit of the security rests on SHA-256 and
//! Ed25519. The topology buys exactly one property: bounded typo tolerance,
//! because the Cohen-Steiner–Edelsbrunner–Harer stability theorem bounds the
//! bottleneck movement of the diagram by `2ε` when every point of the cloud
//! moves by at most `ε`, so a quantisation grain coarser than `2ε` maps a small
//! edit onto the same fingerprint. [`GRAIN`] is `16 × 2ε` here, with `ε`
//! [`CASE_STEP`].
//!
//! # The verdict: do not ship this in front of a password
//!
//! Fuzzy extraction trades entropy for tolerance, and the trade this
//! construction makes is a bad one. [`measure_entropy`] runs the real pipeline
//! over the exhaustive 16⁴ corpus in [`CORPUS_ALPHABET`] and measures it, and
//! [`CLAIMED_SHIPPABLE`] is asserted against [`EntropyReport::worth_shipping`]
//! by a registered test, so the verdict cannot drift away from the measurement.
//! [`topo_key_proof_line`] formats the whole measurement and the registered
//! `topo_key::proof_line_passes` test runs it; [`emit_boot_proof`] prints it,
//! and nothing calls that yet — wiring it into the boot sequence needs an edit
//! to `main.rs`, which is outside this module.
//!
//! Measured over the exhaustive 16⁴ corpus — every password, no sampling, so
//! the fingerprint distribution and its entropies are exact:
//!
//! | construction | distinct keys | H₂ | H∞ | case typos tolerated |
//! |---|---|---|---|---|
//! | `KDF(password)` | 65536 | 16.000 | 16.000 | 0 / 65536 |
//! | `KDF(ascii_lowercase(password))` | 4096 | 12.000 | 12.000 | 65536 / 65536 |
//! | this module, `GRAIN` = 32 | 200 | **6.293** | 4.871 | 63364 / 65536 |
//! | this module, raw diagram (`PROBE_GRAIN`) | 3001 | 10.182 | — | 20 / 65536 |
//!
//! Collision rate at the shipped grain is 1.0000: every password in the corpus
//! shares its key with another.
//!
//! The comparison that decides it is the second row. `KDF(ascii_lowercase(pw))`
//! is one line, buys the *same* typo class with no boundary failures at all,
//! and costs exactly the case bits — 4 on this corpus. This module costs 9.707
//! bits for 96.69% of the same tolerance.
//!
//! The obvious objection is that 32 is simply the wrong grain, so the fourth
//! row answers it: fingerprinting at a grain finer than any tolerated edit can
//! move a death time *is* fingerprinting the raw diagram, and that is the
//! ceiling no choice of grain can beat. It is 10.182 bits — still 1.8 bits
//! below the one-line baseline, and by then the tolerance is gone (20 / 65536).
//! **The lossy step is the sketch, not the quantiser.** An H₀ diagram is an
//! unlabelled multiset of merge heights: it keeps the geometry of the cloud and
//! discards which point contributed what, and for this cloud that discarded
//! part is worth more than the case bits the tolerance costs. `perm_collisions`
//! is 0/128, so at 12 characters the ordering is *not* what is being lost —
//! the loss is the collapse of a 4-point cloud onto 3 merge heights.
//!
//! So: correct, tested, measured, and **not wired into
//! [`crate::security::shadow`] or any authentication path**. It is kept as the
//! measurement that settles the question, re-runnable at every boot.
//!
//! # Pipeline
//!
//! 1. [`cloud`] — byte `i` becomes the point
//!    `(i·POS_STEP, fold(b)·FOLD_STEP + upper(b)·CASE_STEP)` in ℝ², where
//!    `fold` is ASCII case folding. A case typo therefore moves exactly one
//!    point by exactly [`CASE_STEP`]; that is the `ε` the whole scheme is built
//!    around. Substitutions that change the folded letter move a point by at
//!    least [`FOLD_STEP`] and are *not* tolerated, by design.
//! 2. [`h0_persistence`] — exact H₀ of the Rips filtration by Kruskal over all
//!    pairwise distances with union-find and the elder rule. All vertices enter
//!    at `t = 0`, so the diagram is `{(0, dₖ)}` for the `n−1` merge heights,
//!    plus the one essential class that never dies and is not returned.
//! 3. [`fingerprint`] — `floor(death / GRAIN)` per bar, in the non-decreasing
//!    order the bars are produced.
//! 4. [`fingerprint_digest`] — SHA-256 over a domain tag, the grain, the true
//!    password length, and the bins.
//! 5. [`enroll_with_salt`] / [`open`] — salted iterated SHA-256 to a root key,
//!    split into an Ed25519 signing seed and a file key, committed to by the
//!    Ed25519 public key plus a signature over `(version, grain, salt, commit)`.
//!
//! H₁ is **not** computed. H₀ alone is exact, testable and O(n²) in bounded
//! memory; a Rips H₁ needs the 2-skeleton, which is O(n³) simplices.
//!
//! # Bounds
//!
//! At most [`MAX_POINTS`] = 32 points, so at most 496 pairwise distances and a
//! 496-entry edge list (~12 KiB) per fingerprint. Password bytes past the cap
//! contribute through a plain SHA-256 of the tail — they still reach the key,
//! but they get no typo tolerance. [`measure_entropy`] holds one
//! `Vec<u64>` of [`CORPUS_SIZE`] entries (512 KiB) and is the only allocation
//! in this module that is not per-call and tiny.
//!
//! # Failure model
//!
//! Fail closed everywhere. A non-finite coordinate, an overflowed distance, a
//! cloud over the cap, or a bin past `u32` makes [`h0_persistence`] and
//! [`fingerprint`] return `None` rather than a plausible number they did not
//! measure — the defect `ml_engine::stratum::mst_edge_stats` has, where a cloud
//! whose every distance overflows returns `(0.0, 0.0)` and reads as a single
//! coincident point. A record with an unknown version, an unknown grain, a
//! malformed public key or a bad signature is a failed authentication.
//!
//! The salt comes from [`crate::drivers::entropy::getrandom`], which is the
//! RDSEED-then-RDRAND arm (`entropy.rs:80-104`) and returns `false` when CPUID
//! advertises neither. [`fresh_salt`] then returns `None` and [`enroll`] fails:
//! the `fallback_random_u64` xorshift path is deliberately not reached, because
//! a salt drawn from an unreseeded linear generator is a salt an attacker can
//! reproduce.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::cmp::Ordering;
use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};
use sha2::{Digest, Sha256};

// ── Constants ───────────────────────────────────────────────────────────────

/// Largest cloud built from a password. Caps the pairwise work at
/// `32·31/2 = 496` distances and the edge list at 496 entries.
pub const MAX_POINTS: usize = 32;

/// Spacing along the position axis.
pub const POS_STEP: f64 = 256.0;

/// Spacing between adjacent case-folded byte values.
pub const FOLD_STEP: f64 = 256.0;

/// How far a case flip moves one point. This is the `ε` of the stability
/// theorem, and every other scale here is expressed as a multiple of it.
pub const CASE_STEP: f64 = 1.0;

/// Quantisation grain. `16 × 2·CASE_STEP`, so the `2ε` bottleneck movement a
/// case typo can induce is 1/16 of a bin — and `FOLD_STEP / GRAIN = 8`, so a
/// genuine one-letter substitution still lands in a different bin.
pub const GRAIN: f64 = 32.0;

/// Probe grain: finer than the smallest death-time change any tolerated edit
/// can cause (`1/(2·3·POS_STEP)` = 6.5·10⁻⁴), so fingerprinting at this grain
/// *is* fingerprinting the raw diagram.
///
/// Only [`measure_entropy_at_grain`] uses it, and only to answer the obvious
/// objection to the verdict: that the numbers are an artifact of one chosen
/// [`GRAIN`]. Measuring the unquantised diagram gives the ceiling *no* choice
/// of grain can beat, and the proof line reports it next to the shipped grain.
pub const PROBE_GRAIN: f64 = 0.0001;

/// `GRAIN` in thousandths, so the grain can be bound into the signed parameter
/// block as an exact integer.
pub const GRAIN_MILLI: u32 = 32_000;

/// KDF iterations. Follows the `security::shadow` idiom (iterated SHA-256).
pub const KDF_ROUNDS: u32 = 4096;

/// Record format version.
pub const VERSION: u8 = 1;

const FP_DOMAIN: &[u8] = b"seal-os/topo-key/fingerprint/v1";
const TAIL_DOMAIN: &[u8] = b"seal-os/topo-key/tail/v1";
const KDF_DOMAIN: &[u8] = b"seal-os/topo-key/kdf/v1";
const COMMIT_DOMAIN: &[u8] = b"seal-os/topo-key/commit/v1";
const SIGN_SUBKEY: &[u8] = b"seal-os/topo-key/subkey/sign";
const FILE_SUBKEY: &[u8] = b"seal-os/topo-key/subkey/file";

/// What this module claims about its own worth, checked against
/// [`EntropyReport::worth_shipping`] by a registered test so the module
/// documentation cannot drift away from the measurement.
pub const CLAIMED_SHIPPABLE: bool = false;

fn sha256(parts: &[&[u8]]) -> [u8; 32] {
    let mut h = Sha256::new();
    for p in parts {
        h.update(p);
    }
    h.finalize().into()
}

// ── Point cloud ─────────────────────────────────────────────────────────────

/// Map a password to its point cloud. Deterministic and local: byte `i` decides
/// point `i` and nothing else, which is what makes a single-character edit a
/// bounded perturbation of the cloud rather than a fresh cloud.
///
/// Only the first [`MAX_POINTS`] bytes take part; [`fingerprint_digest`] folds
/// the tail in separately.
pub fn cloud(password: &[u8]) -> Vec<[f64; 2]> {
    let n = password.len().min(MAX_POINTS);
    let mut pts = Vec::with_capacity(n);
    for (i, &b) in password[..n].iter().enumerate() {
        let upper = b.is_ascii_uppercase();
        let folded = if upper { b + 32 } else { b };
        pts.push([
            i as f64 * POS_STEP,
            folded as f64 * FOLD_STEP + if upper { CASE_STEP } else { 0.0 },
        ]);
    }
    pts
}

fn dist(a: [f64; 2], b: [f64; 2]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    libm::sqrt(dx * dx + dy * dy)
}

// ── H₀ persistence ──────────────────────────────────────────────────────────

/// One finite H₀ bar. Birth is `0` for every H₀ class of a Rips filtration on a
/// point cloud, so only the death is carried.
///
/// `elder` and `dying` are the elder-rule bookkeeping: the two merging
/// components are represented by their oldest vertex (smallest index, the order
/// vertices enter the filtration at `t = 0`), the older of the two survives,
/// and the younger is the class that dies here. `elder < dying` always.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct H0Bar {
    /// Scale at which the class dies — the merge height.
    pub death: f64,
    /// Oldest vertex of the surviving component.
    pub elder: usize,
    /// Oldest vertex of the component that dies.
    pub dying: usize,
}

fn uf_find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

/// Exact H₀ persistence of the Vietoris–Rips filtration of `points`, by Kruskal
/// over every pairwise distance with union-find and the elder rule.
///
/// Returns the `n−1` finite bars in non-decreasing death order. The one
/// essential class (the component that never dies) is not returned; its bar is
/// `(0, ∞)` and carries no information beyond `n`.
///
/// `None` — never a fabricated diagram — when the cloud is over [`MAX_POINTS`],
/// when a coordinate is non-finite, or when a pairwise distance overflows to
/// infinity. That last arm is the one `stratum::mst_edge_stats` gets wrong: it
/// reports `(0.0, 0.0)` for a cloud whose every distance overflowed, which the
/// caller cannot tell from a single coincident point.
pub fn h0_persistence(points: &[[f64; 2]]) -> Option<Vec<H0Bar>> {
    let n = points.len();
    if n > MAX_POINTS {
        return None;
    }
    if points.iter().any(|p| !p[0].is_finite() || !p[1].is_finite()) {
        return None;
    }
    if n < 2 {
        return Some(Vec::new());
    }

    let mut edges: Vec<(f64, usize, usize)> = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let d = dist(points[i], points[j]);
            if !d.is_finite() {
                return None;
            }
            edges.push((d, i, j));
        }
    }
    // Ties broken by vertex index so the traversal is deterministic. The death
    // *multiset* does not depend on the tie-break — every spanning tree that
    // Kruskal can produce has the same sorted edge lengths — which is why the
    // fingerprint is permutation invariant while `elder`/`dying` are not.
    edges.sort_unstable_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(Ordering::Equal)
            .then(a.1.cmp(&b.1))
            .then(a.2.cmp(&b.2))
    });

    let mut parent: Vec<usize> = (0..n).collect();
    let mut oldest: Vec<usize> = (0..n).collect();
    let mut bars = Vec::with_capacity(n - 1);
    for (d, i, j) in edges {
        let a = uf_find(&mut parent, i);
        let b = uf_find(&mut parent, j);
        if a == b {
            continue;
        }
        // Elder rule: the component whose oldest vertex is older survives.
        let (survivor, absorbed) = if oldest[a] <= oldest[b] { (a, b) } else { (b, a) };
        let elder = oldest[survivor];
        let dying = oldest[absorbed];
        parent[absorbed] = survivor;
        oldest[survivor] = elder;
        bars.push(H0Bar { death: d, elder, dying });
        if bars.len() == n - 1 {
            break;
        }
    }
    Some(bars)
}

/// Upper bound on the bottleneck distance between two H₀ diagrams.
///
/// Every H₀ class of a Rips filtration is born at `0`, so both diagrams are
/// point sets on one vertical line and the order-preserving matching of the
/// sorted deaths is optimal among off-diagonal matchings; matching a bar to the
/// diagonal instead can only lower the cost. The returned value is therefore
/// `≥` the true bottleneck distance, which is the direction an assertion of the
/// form `bottleneck ≤ 2ε` needs.
///
/// Diagrams of different cardinality return `∞` rather than a number derived
/// from a matching this function does not compute.
pub fn bottleneck_h0(a: &[H0Bar], b: &[H0Bar]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    let mut worst = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = libm::fabs(x.death - y.death);
        if d > worst {
            worst = d;
        }
    }
    worst
}

// ── Fingerprint ─────────────────────────────────────────────────────────────

/// Quantised H₀ death times of a password's cloud, in non-decreasing order, at
/// an explicit grain.
///
/// `None` when the diagram is unmeasurable, when the grain is not a positive
/// finite number, or when a bin does not fit `u32`.
pub fn fingerprint_at_grain(password: &[u8], grain: f64) -> Option<Vec<u32>> {
    if !(grain.is_finite() && grain > 0.0) {
        return None;
    }
    let bars = h0_persistence(&cloud(password))?;
    let mut bins = Vec::with_capacity(bars.len());
    for bar in &bars {
        let q = libm::floor(bar.death / grain);
        if !(0.0..=(u32::MAX as f64)).contains(&q) {
            return None;
        }
        bins.push(q as u32);
    }
    Some(bins)
}

/// [`fingerprint_at_grain`] at the shipped [`GRAIN`].
pub fn fingerprint(password: &[u8]) -> Option<Vec<u32>> {
    fingerprint_at_grain(password, GRAIN)
}

/// SHA-256 over the grain, the fingerprint, the true password length, and — for
/// passwords longer than [`MAX_POINTS`] — a digest of the bytes past the cap.
///
/// The grain is inside the digest, so two grains can never produce the same
/// key. The tail is hashed rather than dropped so a long passphrase is not
/// silently truncated to its first 32 bytes; it simply gets no typo tolerance.
pub fn fingerprint_digest_at_grain(password: &[u8], grain: f64) -> Option<[u8; 32]> {
    let bins = fingerprint_at_grain(password, grain)?;
    let milli = libm::round(grain * 1000.0);
    if !(0.0..=(u32::MAX as f64)).contains(&milli) {
        return None;
    }
    let mut h = Sha256::new();
    h.update(FP_DOMAIN);
    h.update((milli as u32).to_le_bytes());
    h.update((password.len() as u64).to_le_bytes());
    h.update((bins.len() as u32).to_le_bytes());
    for b in &bins {
        h.update(b.to_le_bytes());
    }
    if password.len() > MAX_POINTS {
        h.update(sha256(&[TAIL_DOMAIN, &password[MAX_POINTS..]]));
    }
    Some(h.finalize().into())
}

/// [`fingerprint_digest_at_grain`] at the shipped [`GRAIN`].
pub fn fingerprint_digest(password: &[u8]) -> Option<[u8; 32]> {
    fingerprint_digest_at_grain(password, GRAIN)
}

// ── Key derivation and commitment ───────────────────────────────────────────

fn kdf(fp: &[u8; 32], salt: &[u8; 16]) -> [u8; 32] {
    let mut h = sha256(&[KDF_DOMAIN, salt, fp]);
    for _ in 0..KDF_ROUNDS {
        h = sha256(&[&h, salt]);
    }
    h
}

/// Split the root key so the Ed25519 seed and the file key are never the same
/// bytes used under two primitives.
fn subkeys(root: &[u8; 32]) -> ([u8; 32], [u8; 32]) {
    (sha256(&[root, SIGN_SUBKEY]), sha256(&[root, FILE_SUBKEY]))
}

fn commit_context(version: u8, grain_milli: u32, salt: &[u8; 16], commit: &[u8; 32]) -> Vec<u8> {
    let mut v = Vec::with_capacity(COMMIT_DOMAIN.len() + 53);
    v.extend_from_slice(COMMIT_DOMAIN);
    v.push(version);
    v.extend_from_slice(&grain_milli.to_le_bytes());
    v.extend_from_slice(salt);
    v.extend_from_slice(commit);
    v
}

fn ct_eq(a: &[u8; 32], b: &[u8; 32]) -> bool {
    let mut diff = 0u8;
    for i in 0..32 {
        diff |= a[i] ^ b[i];
    }
    diff == 0
}

/// What is stored. Neither the key nor the password appears here: `commit` is
/// an Ed25519 public key whose secret seed is the password-derived signing
/// subkey, and `sig` is that key's signature over the parameter block, so the
/// version, grain and salt a verifier will use are bound to the commitment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TopoRecord {
    pub version: u8,
    pub grain_milli: u32,
    pub salt: [u8; 16],
    pub commit: [u8; 32],
    pub sig: [u8; 64],
}

/// Draw a 16-byte salt from hardware entropy.
///
/// `None` when [`crate::drivers::entropy::getrandom`] reports no hardware
/// source, and `None` when every byte it returned is identical — the shape a
/// stuck RDRAND takes (all-zero, or the all-ones an AMD part can latch), not a
/// 2⁻¹²⁰ draw worth honouring. There is deliberately no software fallback.
///
/// ponytail: the stuck-source arm is not covered by any registered test,
/// because nothing in the kernel can decide what RDRAND returns; the host
/// harness covers it by stubbing `getrandom`. To cover it in-kernel, `getrandom`
/// would need a seam that feeds it bytes — which belongs in `drivers/entropy.rs`,
/// not here.
pub fn fresh_salt() -> Option<[u8; 16]> {
    let mut salt = [0u8; 16];
    if !crate::drivers::entropy::getrandom(&mut salt) {
        return None;
    }
    if salt.iter().all(|&b| b == salt[0]) {
        return None;
    }
    Some(salt)
}

/// Enrol a password under a caller-supplied salt.
pub fn enroll_with_salt(password: &[u8], salt: &[u8; 16]) -> Option<TopoRecord> {
    let fp = fingerprint_digest(password)?;
    let (k_sign, _) = subkeys(&kdf(&fp, salt));
    let sk = SigningKey::from_bytes(&k_sign);
    let commit = sk.verifying_key().to_bytes();
    let ctx = commit_context(VERSION, GRAIN_MILLI, salt, &commit);
    Some(TopoRecord {
        version: VERSION,
        grain_milli: GRAIN_MILLI,
        salt: *salt,
        commit,
        sig: sk.sign(&ctx).to_bytes(),
    })
}

/// Enrol a password under a fresh hardware salt. `None` when no hardware
/// entropy is available: enrolment fails rather than salting from a counter.
pub fn enroll(password: &[u8]) -> Option<TopoRecord> {
    enroll_with_salt(password, &fresh_salt()?)
}

/// Recover the file key for `password`, or `None`.
///
/// Fails closed on an unknown version, a grain this build cannot reproduce, a
/// malformed public key, a signature that does not verify under
/// `verify_strict`, an unmeasurable fingerprint, or a commitment mismatch. The
/// commitment comparison is constant time.
pub fn open(password: &[u8], record: &TopoRecord) -> Option<[u8; 32]> {
    if record.version != VERSION || record.grain_milli != GRAIN_MILLI {
        return None;
    }
    let stored = VerifyingKey::from_bytes(&record.commit).ok()?;
    let ctx = commit_context(
        record.version,
        record.grain_milli,
        &record.salt,
        &record.commit,
    );
    stored
        .verify_strict(&ctx, &Signature::from_bytes(&record.sig))
        .ok()?;
    let fp = fingerprint_digest(password)?;
    let (k_sign, k_file) = subkeys(&kdf(&fp, &record.salt));
    let derived = SigningKey::from_bytes(&k_sign).verifying_key().to_bytes();
    if ct_eq(&derived, &record.commit) {
        Some(k_file)
    } else {
        None
    }
}

/// Whether `password` opens `record`.
pub fn verify(password: &[u8], record: &TopoRecord) -> bool {
    open(password, record).is_some()
}

// ── Entropy measurement ─────────────────────────────────────────────────────

/// Corpus alphabet: 8 letters in both cases, so the case bits — the dimension
/// the tolerance is bought in — are inside the corpus rather than assumed.
pub const CORPUS_ALPHABET: &[u8; 16] = b"abcdefghABCDEFGH";

/// Corpus password length.
pub const CORPUS_LEN: usize = 4;

/// `16⁴`. The corpus is exhaustive, so the fingerprint distribution is exact
/// and the entropies below are computed, not estimated from a sample.
pub const CORPUS_SIZE: usize = 65536;

/// Password whose permutations measure the ordering information an unlabelled
/// diagram cannot carry.
pub const PERM_PASSWORD: &[u8; 12] = b"aBcDeFgHiJkL";

/// Permutations tried.
pub const PERM_TRIALS: u32 = 128;

/// Floor on the measured collision entropy. A grain coarse enough to collapse
/// the corpus onto one fingerprint drives H₂ to 0; this is the assertion that
/// notices. Set half a bit below the measured 6.293.
pub const MIN_H2_BITS: f64 = 5.8;

/// Floor on the measured case-typo tolerance. A grain fine enough to lose the
/// tolerance the entropy was spent on drives this down: measured 0.9669 at
/// [`GRAIN`] and 0.0003 at [`PROBE_GRAIN`], the two ends of the same trade.
pub const MIN_TYPO_RATE: f64 = 0.95;

fn corpus_password(index: usize) -> [u8; CORPUS_LEN] {
    let mut out = [0u8; CORPUS_LEN];
    let mut n = index;
    for slot in out.iter_mut() {
        *slot = CORPUS_ALPHABET[n % CORPUS_ALPHABET.len()];
        n /= CORPUS_ALPHABET.len();
    }
    out
}

/// What the exhaustive corpus measured.
#[derive(Clone, Copy, Debug)]
pub struct EntropyReport {
    /// Grain the corpus was fingerprinted at.
    pub grain: f64,
    /// Passwords enumerated.
    pub corpus: u32,
    /// Distinct fingerprints they produced.
    pub distinct: u32,
    /// Fraction of the corpus sharing a fingerprint with another password.
    pub collision_rate: f64,
    /// Rényi collision entropy of the fingerprint distribution, exact.
    pub h2_bits: f64,
    /// Min-entropy of the fingerprint distribution, exact. This is the number a
    /// guessing attacker faces.
    pub hmin_bits: f64,
    /// Entropy of the corpus itself — what `KDF(password)` would deliver.
    pub plain_bits: f64,
    /// What `KDF(ascii_lowercase(password))` would deliver: the boring
    /// construction that buys the same typo class for the case bits alone.
    pub folded_bits: f64,
    /// Case-typo trials attempted (one per corpus member).
    pub typo_trials: u32,
    /// Trials where the typo produced an identical fingerprint.
    pub typo_tolerated: u32,
    /// Permutations of [`PERM_PASSWORD`] tried.
    pub perm_trials: u32,
    /// Permutations that reproduced the original fingerprint.
    pub perm_collisions: u32,
    /// Corpus members whose diagram could not be measured. Non-zero is a bug.
    pub unmeasurable: u32,
}

impl EntropyReport {
    /// Fraction of case typos that produced the same key.
    pub fn typo_rate(&self) -> f64 {
        if self.typo_trials == 0 {
            0.0
        } else {
            self.typo_tolerated as f64 / self.typo_trials as f64
        }
    }

    /// Bits given up against hashing the password directly.
    pub fn bits_lost_vs_plain(&self) -> f64 {
        self.plain_bits - self.h2_bits
    }

    /// Bits given up against the boring construction that buys the same
    /// tolerance: case-fold, then hash.
    pub fn bits_lost_vs_folded(&self) -> f64 {
        self.folded_bits - self.h2_bits
    }

    /// The shipping test. Worth shipping only if every diagram was measurable,
    /// every case typo was tolerated, and the fingerprint retains at least as
    /// much entropy as case-folding then hashing — because that alternative is
    /// one line and buys the same tolerance.
    pub fn worth_shipping(&self) -> bool {
        self.unmeasurable == 0
            && self.typo_tolerated == self.typo_trials
            && self.h2_bits >= self.folded_bits
    }
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// How many of [`PERM_TRIALS`] deterministic permutations of [`PERM_PASSWORD`]
/// reproduce its fingerprint. Seeded explicitly, so every boot measures the
/// same permutations.
fn permutation_collisions(grain: f64) -> u32 {
    let Some(base) = fingerprint_digest_at_grain(PERM_PASSWORD, grain) else {
        return 0;
    };
    let mut state = 0x0517_0BEE_F00D_1234u64;
    let mut hits = 0u32;
    for _ in 0..PERM_TRIALS {
        let mut p = *PERM_PASSWORD;
        for i in (1..p.len()).rev() {
            let j = (splitmix64(&mut state) % (i as u64 + 1)) as usize;
            p.swap(i, j);
        }
        if p == *PERM_PASSWORD {
            continue;
        }
        if fingerprint_digest_at_grain(&p, grain) == Some(base) {
            hits += 1;
        }
    }
    hits
}

/// Run the real pipeline over the exhaustive corpus at an explicit grain and
/// measure what it costs.
///
/// Fingerprints are keyed by the first 8 bytes of their digest; over 65536
/// items an accidental 64-bit collision has probability ≈ 1.2·10⁻¹⁰, four
/// orders below the effect being measured.
pub fn measure_entropy_at_grain(grain: f64) -> EntropyReport {
    let mut keys: Vec<u64> = Vec::with_capacity(CORPUS_SIZE);
    let mut unmeasurable = 0u32;
    let mut typo_tolerated = 0u32;
    let mut typo_trials = 0u32;

    for idx in 0..CORPUS_SIZE {
        let pw = corpus_password(idx);
        let Some(d) = fingerprint_digest_at_grain(&pw, grain) else {
            unmeasurable += 1;
            continue;
        };
        keys.push(u64::from_le_bytes(d[..8].try_into().unwrap_or([0u8; 8])));
        // The typo class this scheme exists to tolerate: one flipped case bit.
        let mut typo = pw;
        typo[0] ^= 0x20;
        typo_trials += 1;
        if fingerprint_digest_at_grain(&typo, grain) == Some(d) {
            typo_tolerated += 1;
        }
    }

    keys.sort_unstable();
    let mut distinct = 0u32;
    let mut colliding = 0u32;
    let mut max_run = 1u32;
    let mut sum_sq = 0.0f64;
    let mut i = 0usize;
    while i < keys.len() {
        let mut j = i + 1;
        while j < keys.len() && keys[j] == keys[i] {
            j += 1;
        }
        let run = (j - i) as u32;
        distinct += 1;
        sum_sq += run as f64 * run as f64;
        if run > max_run {
            max_run = run;
        }
        if run > 1 {
            colliding += run;
        }
        i = j;
    }

    let n = keys.len() as f64;
    let alphabet = CORPUS_ALPHABET.len() as f64;
    EntropyReport {
        grain,
        corpus: keys.len() as u32,
        distinct,
        collision_rate: if n > 0.0 { colliding as f64 / n } else { 0.0 },
        h2_bits: if sum_sq > 0.0 {
            libm::log2(n * n / sum_sq)
        } else {
            0.0
        },
        hmin_bits: if n > 0.0 {
            libm::log2(n / max_run as f64)
        } else {
            0.0
        },
        plain_bits: CORPUS_LEN as f64 * libm::log2(alphabet),
        folded_bits: CORPUS_LEN as f64 * libm::log2(alphabet / 2.0),
        typo_trials,
        typo_tolerated,
        perm_trials: PERM_TRIALS,
        perm_collisions: permutation_collisions(grain),
        unmeasurable,
    }
}

/// [`measure_entropy_at_grain`] at the shipped [`GRAIN`].
pub fn measure_entropy() -> EntropyReport {
    measure_entropy_at_grain(GRAIN)
}

// ── Proof emitter ───────────────────────────────────────────────────────────

/// Single-line boot proof. Every number is measured at this boot by running the
/// real pipeline.
///
/// `result` is the self-check: the crypto round-trips, a wrong password is
/// refused, a tampered record is refused, and the measured verdict still equals
/// [`CLAIMED_SHIPPABLE`]. `verdict` is the separate question of whether this
/// construction should be put in front of a password, and it is `reject`.
pub fn topo_key_proof_line() -> String {
    let report = measure_entropy();
    let raw = measure_entropy_at_grain(PROBE_GRAIN);

    let salt = [0x11u8; 16];
    let record = enroll_with_salt(b"correct horse", &salt);
    let roundtrip = record
        .as_ref()
        .map(|r| verify(b"correct horse", r))
        .unwrap_or(false);
    let wrong_refused = record
        .as_ref()
        .map(|r| !verify(b"correct zebra", r))
        .unwrap_or(false);
    let tamper_refused = record
        .as_ref()
        .map(|r| {
            let mut bad = *r;
            bad.sig[0] ^= 1;
            let mut bad_salt = *r;
            bad_salt.salt[0] ^= 1;
            !verify(b"correct horse", &bad) && !verify(b"correct horse", &bad_salt)
        })
        .unwrap_or(false);
    let case_typo = record
        .as_ref()
        .map(|r| verify(b"Correct horse", r))
        .unwrap_or(false);
    let salt_arm = if fresh_salt().is_some() {
        "hardware"
    } else {
        "unavailable"
    };
    let verdict_locked = report.worth_shipping() == CLAIMED_SHIPPABLE;

    let pass = roundtrip
        && wrong_refused
        && tamper_refused
        && case_typo
        && verdict_locked
        && report.unmeasurable == 0
        && report.h2_bits >= MIN_H2_BITS
        && !raw.worth_shipping()
        && raw.h2_bits < report.folded_bits;

    format!(
        "[TOPOKEY] proof version=1 subsystem=topo_key homology=H0 max_points={} grain={:.3} \
         eps={:.3} grain_over_2eps={:.1} kdf=sha256x{} commitment=ed25519 salt_source={} \
         corpus={} distinct={} collision_rate={:.4} h2_bits={:.3} hmin_bits={:.3} \
         plain_kdf_bits={:.3} folded_kdf_bits={:.3} bits_lost_vs_plain={:.3} \
         bits_lost_vs_folded={:.3} typo_tolerated={}/{} perm_collisions={}/{} unmeasurable={} \
         raw_grain={:.5} raw_distinct={} raw_h2_bits={:.3} raw_typo_tolerated={}/{} \
         raw_beats_folded={} roundtrip={} wrong_refused={} tamper_refused={} \
         case_typo_opens={} verdict={} result={}",
        MAX_POINTS,
        GRAIN,
        CASE_STEP,
        GRAIN / (2.0 * CASE_STEP),
        KDF_ROUNDS,
        salt_arm,
        report.corpus,
        report.distinct,
        report.collision_rate,
        report.h2_bits,
        report.hmin_bits,
        report.plain_bits,
        report.folded_bits,
        report.bits_lost_vs_plain(),
        report.bits_lost_vs_folded(),
        report.typo_tolerated,
        report.typo_trials,
        report.perm_collisions,
        report.perm_trials,
        report.unmeasurable,
        raw.grain,
        raw.distinct,
        raw.h2_bits,
        raw.typo_tolerated,
        raw.typo_trials,
        if raw.h2_bits >= report.folded_bits { 1 } else { 0 },
        if roundtrip { 1 } else { 0 },
        if wrong_refused { 1 } else { 0 },
        if tamper_refused { 1 } else { 0 },
        if case_typo { 1 } else { 0 },
        if report.worth_shipping() {
            "ship"
        } else {
            "reject"
        },
        if pass { "pass" } else { "fail" }
    )
}

/// Print the boot proof to serial.
pub fn emit_boot_proof() {
    crate::serial_println!("{}", topo_key_proof_line());
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use super::*;
    use crate::testing::TestResult;
    use crate::{test_assert, test_assert_eq};

    /// `test_assert!` expands to `if !$cond`, and on a float comparison that
    /// negation is what clippy's `neg_cmp_op_on_partial_ord` flags. Binding the
    /// comparison to a `bool` first makes the negation a plain boolean one,
    /// which is what it always meant, and keeps the assertion text unchanged.
    macro_rules! assert_num {
        ($cond:expr, $msg:expr) => {{
            let ok: bool = $cond;
            test_assert!(ok, $msg);
        }};
    }

    /// Deterministic cloud with no password behind it, for the pure topology
    /// invariants. Seeded explicitly.
    fn synthetic_cloud(n: usize, seed: u64) -> Vec<[f64; 2]> {
        let mut state = seed;
        let mut pts = Vec::with_capacity(n);
        for _ in 0..n {
            let x = (splitmix64(&mut state) % 10_000) as f64 / 10.0;
            let y = (splitmix64(&mut state) % 10_000) as f64 / 10.0;
            pts.push([x, y]);
        }
        pts
    }

    fn deaths(bars: &[H0Bar]) -> Vec<f64> {
        bars.iter().map(|b| b.death).collect()
    }

    /// Independently written minimum spanning tree, by Prim rather than by
    /// Kruskal and with no union-find at all. The H₀ death multiset must equal
    /// its edge multiset.
    fn prim_edges(points: &[[f64; 2]]) -> Vec<f64> {
        let n = points.len();
        let mut out = Vec::new();
        if n < 2 {
            return out;
        }
        let mut inside = alloc::vec![false; n];
        let mut key = alloc::vec![f64::INFINITY; n];
        key[0] = 0.0;
        for _ in 0..n {
            let mut best = usize::MAX;
            let mut best_key = f64::INFINITY;
            for i in 0..n {
                if !inside[i] && key[i] < best_key {
                    best_key = key[i];
                    best = i;
                }
            }
            if best == usize::MAX {
                break;
            }
            inside[best] = true;
            if best_key > 0.0 && best_key.is_finite() {
                out.push(best_key);
            }
            for i in 0..n {
                if !inside[i] {
                    let d = dist(points[best], points[i]);
                    if d < key[i] {
                        key[i] = d;
                    }
                }
            }
        }
        out.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        out
    }

    /// INVARIANT 1. Shuffling the input points must not move the diagram.
    /// Catches order-dependent tie-breaking.
    fn test_permutation_invariance() -> TestResult {
        let pts = synthetic_cloud(24, 0xA11CE);
        let Some(base) = h0_persistence(&pts) else {
            return TestResult::Fail("base diagram must be measurable");
        };
        let mut state = 0xC0FFEEu64;
        for _ in 0..16 {
            let mut shuffled = pts.clone();
            for i in (1..shuffled.len()).rev() {
                let j = (splitmix64(&mut state) % (i as u64 + 1)) as usize;
                shuffled.swap(i, j);
            }
            let Some(got) = h0_persistence(&shuffled) else {
                return TestResult::Fail("shuffled diagram must be measurable");
            };
            test_assert!(
                deaths(&got) == deaths(&base),
                "H0 death multiset must not depend on input order"
            );
        }
        // Ties are where an order-dependent tie-break would show: a regular
        // grid has many equal pairwise distances.
        let grid: Vec<[f64; 2]> = (0..16).map(|k| [(k % 4) as f64, (k / 4) as f64]).collect();
        let mut reversed = grid.clone();
        reversed.reverse();
        let (Some(a), Some(b)) = (h0_persistence(&grid), h0_persistence(&reversed)) else {
            return TestResult::Fail("grid diagrams must be measurable");
        };
        test_assert!(
            deaths(&a) == deaths(&b),
            "a tied grid must give the same deaths in either order"
        );
        TestResult::Pass
    }

    /// INVARIANT 2. Scaling the cloud by `c > 0` scales every death by exactly
    /// `c`. Catches an absolute threshold pretending to be a relative one.
    fn test_scale_equivariance() -> TestResult {
        let pts = synthetic_cloud(20, 0xBEEF);
        let Some(base) = h0_persistence(&pts) else {
            return TestResult::Fail("base diagram must be measurable");
        };
        // The sweep has to reach scales on both sides of any constant that
        // could be hiding in the filtration: `1e-6` puts every death far below
        // 1.0 and `1e6` far above it. A version of this test that only tried
        // factors near 1 let a `dist(..).max(1.0)` mutation through.
        for c in [1e-6f64, 0.25, 3.0, 1024.0, 1e6] {
            let scaled: Vec<[f64; 2]> = pts.iter().map(|p| [p[0] * c, p[1] * c]).collect();
            let Some(got) = h0_persistence(&scaled) else {
                return TestResult::Fail("scaled diagram must be measurable");
            };
            test_assert_eq!(got.len(), base.len());
            for (g, b) in got.iter().zip(base.iter()) {
                assert_num!(b.death > 0.0, "the control cloud must have positive deaths");
                let want = b.death * c;
                // Purely relative: an absolute tolerance is vacuous once the
                // deaths are smaller than it.
                assert_num!(
                    libm::fabs(g.death - want) <= 1e-9 * want,
                    "death must scale by exactly c"
                );
            }
        }
        TestResult::Pass
    }

    /// INVARIANT 3, the load-bearing one. Move every point by at most `eps`;
    /// the bottleneck distance between the diagrams must not exceed `2·eps`.
    /// This is the Rips constant, and it is the theorem the typo tolerance
    /// rests on — without it the tolerance is a coincidence.
    fn test_stability_bottleneck() -> TestResult {
        let pts = synthetic_cloud(24, 0xD15EA5E);
        let Some(base) = h0_persistence(&pts) else {
            return TestResult::Fail("base diagram must be measurable");
        };

        // Calibrate the yardstick before using it. An upper bound that always
        // returns 0 satisfies every stability assertion below, so the bound is
        // first checked against a diagram whose distance from `base` is known:
        // doubling the cloud doubles every death, so the bottleneck is exactly
        // the largest death.
        let doubled: Vec<[f64; 2]> = pts.iter().map(|p| [p[0] * 2.0, p[1] * 2.0]).collect();
        let Some(twice) = h0_persistence(&doubled) else {
            return TestResult::Fail("doubled diagram must be measurable");
        };
        let longest = base[base.len() - 1].death;
        assert_num!(longest > 0.0, "the control cloud must have positive deaths");
        assert_num!(
            libm::fabs(bottleneck_h0(&base, &twice) - longest) <= 1e-9 * longest,
            "bottleneck_h0 must measure a known separation, not return zero"
        );
        assert_num!(
            bottleneck_h0(&base, &base) == 0.0,
            "a diagram must be at distance zero from itself"
        );
        assert_num!(
            bottleneck_h0(&base, &base[1..]) == f64::INFINITY,
            "diagrams of different cardinality must not report a finite bound"
        );

        let mut state = 0x5EED_1234u64;
        for eps in [0.01f64, 0.5, 5.0] {
            for _ in 0..8 {
                let moved: Vec<[f64; 2]> = pts
                    .iter()
                    .map(|p| {
                        // Displacement drawn inside the closed eps-ball, so the
                        // hypothesis of the theorem holds exactly.
                        let theta = (splitmix64(&mut state) % 100_000) as f64
                            * core::f64::consts::TAU
                            / 100_000.0;
                        let r = (splitmix64(&mut state) % 100_001) as f64 * eps / 100_000.0;
                        [p[0] + r * libm::cos(theta), p[1] + r * libm::sin(theta)]
                    })
                    .collect();
                for (m, p) in moved.iter().zip(pts.iter()) {
                    assert_num!(
                        dist(*m, *p) <= eps * (1.0 + 1e-12),
                        "the perturbation must respect its own eps, or the test proves nothing"
                    );
                }
                let Some(got) = h0_persistence(&moved) else {
                    return TestResult::Fail("perturbed diagram must be measurable");
                };
                assert_num!(
                    bottleneck_h0(&base, &got) <= 2.0 * eps * (1.0 + 1e-12),
                    "CSEH stability: bottleneck must not exceed 2*eps for a Rips filtration"
                );
            }
        }
        TestResult::Pass
    }

    /// INVARIANT 4. The elder rule: on a merge the younger class dies and the
    /// older survives. Cross-checked against an independently written Prim MST,
    /// which uses no union-find.
    fn test_elder_rule_and_independent_mst() -> TestResult {
        for seed in [1u64, 2, 99, 0xFACE] {
            let pts = synthetic_cloud(18, seed);
            let Some(bars) = h0_persistence(&pts) else {
                return TestResult::Fail("diagram must be measurable");
            };
            test_assert_eq!(bars.len(), pts.len() - 1);
            for bar in &bars {
                test_assert!(
                    bar.elder < bar.dying,
                    "elder rule: the surviving class must be the older one"
                );
            }
            for w in bars.windows(2) {
                assert_num!(
                    w[0].death <= w[1].death,
                    "bars must come out in non-decreasing death order"
                );
            }
            let independent = prim_edges(&pts);
            test_assert_eq!(independent.len(), bars.len());
            for (a, b) in independent.iter().zip(bars.iter()) {
                assert_num!(
                    libm::fabs(a - b.death) < 1e-12,
                    "H0 deaths must equal an independently computed MST edge multiset"
                );
            }
        }
        TestResult::Pass
    }

    /// INVARIANT 5. Negative control: an unstructured cloud must produce no
    /// long bar. Paired with a positive control, because a test that only
    /// checks the negative side passes for a pipeline that always returns
    /// nothing.
    fn test_negative_control_no_long_bars() -> TestResult {
        let noise = synthetic_cloud(28, 0x9911);
        let Some(bars) = h0_persistence(&noise) else {
            return TestResult::Fail("noise diagram must be measurable");
        };
        let d = deaths(&bars);
        let median = d[d.len() / 2];
        let longest = d[d.len() - 1];
        assert_num!(median > 0.0, "noise must have a non-degenerate scale");
        assert_num!(
            longest / median < 4.0,
            "an unstructured cloud must not produce a long bar"
        );

        // Positive control: two tight clusters 1000 apart must show exactly one
        // bar far above the rest.
        let mut clustered = Vec::new();
        for p in synthetic_cloud(12, 0x2211) {
            clustered.push([p[0] * 0.01, p[1] * 0.01]);
        }
        for p in synthetic_cloud(12, 0x3311) {
            clustered.push([p[0] * 0.01 + 1000.0, p[1] * 0.01]);
        }
        let Some(cbars) = h0_persistence(&clustered) else {
            return TestResult::Fail("clustered diagram must be measurable");
        };
        let cd = deaths(&cbars);
        assert_num!(
            cd[cd.len() - 1] / cd[cd.len() / 2] > 20.0,
            "two separated clusters must produce one long bar"
        );
        TestResult::Pass
    }

    /// A cloud whose distances overflow must report that it could not be
    /// measured, not a diagram it did not compute. `stratum::mst_edge_stats`
    /// returns `(0.0, 0.0)` in exactly this case.
    fn test_unmeasurable_is_none() -> TestResult {
        test_assert!(
            h0_persistence(&[[0.0, 0.0], [f64::NAN, 1.0]]).is_none(),
            "a NaN coordinate must not produce a diagram"
        );
        test_assert!(
            h0_persistence(&[[0.0, 0.0], [f64::INFINITY, 0.0]]).is_none(),
            "an infinite coordinate must not produce a diagram"
        );
        test_assert!(
            h0_persistence(&[[-1e308, 0.0], [1e308, 0.0], [0.0, 1e308]]).is_none(),
            "an overflowed distance must not read as a coincident point"
        );
        let over: Vec<[f64; 2]> = (0..(MAX_POINTS + 1)).map(|i| [i as f64, 0.0]).collect();
        test_assert!(
            h0_persistence(&over).is_none(),
            "a cloud over the cap must be refused, not truncated"
        );
        test_assert!(
            h0_persistence(&[[1.0, 2.0]]) == Some(Vec::new()),
            "a single point has no finite bar"
        );
        TestResult::Pass
    }

    /// The grain has to be coarser than the bottleneck movement a tolerated
    /// edit can cause, and finer than the movement a real substitution causes.
    /// Both halves, because only checking one lets the other be tuned away.
    fn test_grain_brackets_the_stability_bound() -> TestResult {
        assert_num!(
            GRAIN > 2.0 * CASE_STEP,
            "grain must exceed the 2*eps bottleneck bound or tolerance is luck"
        );
        assert_num!(
            FOLD_STEP > 4.0 * GRAIN,
            "grain must stay well under a real substitution or entropy is gone"
        );
        // The perturbation a case flip actually causes, measured rather than
        // assumed.
        let a = cloud(b"topology");
        let b = cloud(b"Topology");
        test_assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_num!(
                dist(*x, *y) <= CASE_STEP,
                "a case flip must move a point by at most CASE_STEP"
            );
        }
        let (Some(da), Some(db)) = (h0_persistence(&a), h0_persistence(&b)) else {
            return TestResult::Fail("both diagrams must be measurable");
        };
        assert_num!(
            bottleneck_h0(&da, &db) <= 2.0 * CASE_STEP,
            "a case typo must move the diagram by at most 2*CASE_STEP"
        );
        TestResult::Pass
    }

    /// The property the whole construction exists for: a case typo opens the
    /// record and yields the identical key.
    fn test_case_typo_opens_the_record() -> TestResult {
        let salt = [0x5Au8; 16];
        let Some(record) = enroll_with_salt(b"manifold", &salt) else {
            return TestResult::Fail("enrolment must succeed");
        };
        let Some(key) = open(b"manifold", &record) else {
            return TestResult::Fail("the enrolled password must open the record");
        };
        for typo in [
            b"Manifold".as_slice(),
            b"manifolD".as_slice(),
            b"MANIFOLD".as_slice(),
        ] {
            let Some(k) = open(typo, &record) else {
                return TestResult::Fail("a case typo must still open the record");
            };
            test_assert!(k == key, "a tolerated typo must yield the identical key");
        }
        TestResult::Pass
    }

    /// The other half: a real substitution must not open it. Without this the
    /// tolerance test is satisfied by a scheme that accepts everything.
    fn test_wrong_password_refused() -> TestResult {
        let salt = [0x5Au8; 16];
        let Some(record) = enroll_with_salt(b"manifold", &salt) else {
            return TestResult::Fail("enrolment must succeed");
        };
        for wrong in [
            b"manifolds".as_slice(),
            b"manifoldd".as_slice(),
            b"manifole".as_slice(),
            b"nanifold".as_slice(),
            b"".as_slice(),
        ] {
            test_assert!(
                open(wrong, &record).is_none(),
                "a different password must not open the record"
            );
        }
        TestResult::Pass
    }

    /// A malformed or tampered record is a failed authentication, never a
    /// successful one.
    fn test_tampered_record_refused() -> TestResult {
        let salt = [0x77u8; 16];
        let Some(record) = enroll_with_salt(b"correct horse", &salt) else {
            return TestResult::Fail("enrolment must succeed");
        };
        test_assert!(verify(b"correct horse", &record), "control must open");

        let mut bad_sig = record;
        bad_sig.sig[0] ^= 1;
        test_assert!(
            !verify(b"correct horse", &bad_sig),
            "a flipped signature bit must fail authentication"
        );

        let mut bad_commit = record;
        bad_commit.commit[0] ^= 1;
        test_assert!(
            !verify(b"correct horse", &bad_commit),
            "a flipped commitment bit must fail authentication"
        );

        let mut bad_salt = record;
        bad_salt.salt[0] ^= 1;
        test_assert!(
            !verify(b"correct horse", &bad_salt),
            "a swapped salt must fail authentication"
        );

        let mut bad_version = record;
        bad_version.version = 2;
        test_assert!(
            !verify(b"correct horse", &bad_version),
            "an unknown record version must fail authentication"
        );

        let mut bad_grain = record;
        bad_grain.grain_milli = GRAIN_MILLI * 4;
        test_assert!(
            !verify(b"correct horse", &bad_grain),
            "a grain this build cannot reproduce must fail authentication"
        );

        let zeroed = TopoRecord {
            version: VERSION,
            grain_milli: GRAIN_MILLI,
            salt: [0u8; 16],
            commit: [0u8; 32],
            sig: [0u8; 64],
        };
        test_assert!(
            !verify(b"correct horse", &zeroed),
            "an all-zero record must fail authentication"
        );

        // A record carrying parameters this build does not implement, signed
        // *correctly* for those parameters. Flipping a byte is not enough to
        // exercise the parameter gate, because the signature covers the
        // parameter block and catches the flip first.
        let fp = fingerprint_digest(b"correct horse").unwrap_or([0u8; 32]);
        let (k_sign, _) = subkeys(&kdf(&fp, &salt));
        let sk = SigningKey::from_bytes(&k_sign);
        let commit = sk.verifying_key().to_bytes();
        for (version, grain_milli) in [(VERSION + 1, GRAIN_MILLI), (VERSION, GRAIN_MILLI * 4)] {
            let ctx = commit_context(version, grain_milli, &salt, &commit);
            let future = TopoRecord {
                version,
                grain_milli,
                salt,
                commit,
                sig: sk.sign(&ctx).to_bytes(),
            };
            test_assert!(
                VerifyingKey::from_bytes(&future.commit)
                    .map(|vk| vk
                        .verify_strict(&ctx, &Signature::from_bytes(&future.sig))
                        .is_ok())
                    .unwrap_or(false),
                "the forward-version record must really be signed, or it proves nothing"
            );
            test_assert!(
                !verify(b"correct horse", &future),
                "a validly signed record with parameters this build cannot reproduce \
                 must fail authentication, not be honoured on its signature alone"
            );
        }
        TestResult::Pass
    }

    /// The salt has to reach the key, and it has to come from hardware. A KDF
    /// that ignores its salt gives two enrolments of one password the same
    /// commitment, which is the whole reason salts exist.
    fn test_salt_reaches_the_key() -> TestResult {
        let a = enroll_with_salt(b"manifold", &[0x01u8; 16]);
        let b = enroll_with_salt(b"manifold", &[0x02u8; 16]);
        let (Some(a), Some(b)) = (a, b) else {
            return TestResult::Fail("both enrolments must succeed");
        };
        test_assert!(
            a.commit != b.commit,
            "two salts must give one password two commitments"
        );
        test_assert!(
            a.sig != b.sig,
            "the signature must cover the salt it was made under"
        );
        test_assert!(
            open(b"manifold", &a) != open(b"manifold", &b),
            "the derived key must depend on the salt"
        );
        // No software fallback: enrolment succeeds exactly when the hardware
        // path answers.
        test_assert_eq!(
            enroll(b"manifold").is_some(),
            crate::drivers::entropy::getrandom(&mut [0u8; 16])
        );
        TestResult::Pass
    }

    /// The stored record must not be the key, and the key must not be the
    /// Ed25519 seed.
    fn test_record_stores_no_key() -> TestResult {
        let salt = [0x33u8; 16];
        let Some(record) = enroll_with_salt(b"manifold", &salt) else {
            return TestResult::Fail("enrolment must succeed");
        };
        let Some(key) = open(b"manifold", &record) else {
            return TestResult::Fail("the record must open");
        };
        test_assert!(
            key != record.commit,
            "the file key must not be the stored commitment"
        );
        let fp = fingerprint_digest(b"manifold").unwrap_or([0u8; 32]);
        let (k_sign, k_file) = subkeys(&kdf(&fp, &salt));
        test_assert!(k_sign != k_file, "the two subkeys must differ");
        test_assert!(key == k_file, "open must return the file subkey");
        test_assert!(
            SigningKey::from_bytes(&k_sign).verifying_key().to_bytes() == record.commit,
            "the commitment must be the signing subkey's public key"
        );
        TestResult::Pass
    }

    /// The measurement, and the verdict locked to it. This is the deliverable:
    /// the numbers say the construction costs more entropy than the boring
    /// alternative that buys the same tolerance, so `worth_shipping` is false
    /// and `CLAIMED_SHIPPABLE` says so in the source.
    fn test_entropy_measured_and_verdict_locked() -> TestResult {
        let r = measure_entropy();
        test_assert_eq!(r.corpus, CORPUS_SIZE as u32);
        test_assert_eq!(r.unmeasurable, 0);
        assert_num!(
            r.h2_bits >= MIN_H2_BITS,
            "collision entropy floor: a grain that collapses the corpus fails here"
        );
        assert_num!(
            r.h2_bits <= r.plain_bits,
            "a fuzzy extractor cannot add entropy to its input"
        );
        assert_num!(
            r.hmin_bits <= r.h2_bits,
            "min-entropy cannot exceed collision entropy"
        );
        assert_num!(
            r.typo_rate() >= MIN_TYPO_RATE,
            "the tolerance the entropy was spent on must actually be there"
        );
        test_assert_eq!(r.worth_shipping(), CLAIMED_SHIPPABLE);
        assert_num!(
            r.bits_lost_vs_folded() > 0.0,
            "the recorded verdict is that case-folding then hashing wins; if this \
             no longer holds, re-derive the verdict instead of deleting the check"
        );

        // The verdict must not be an artifact of one chosen grain. Fingerprint
        // the corpus at a grain finer than any tolerated edit can move a death
        // time, i.e. measure the raw diagram: that is the ceiling no quantiser
        // of this diagram can beat, and it is still below the folded baseline.
        let raw = measure_entropy_at_grain(PROBE_GRAIN);
        assert_num!(
            raw.h2_bits > r.h2_bits,
            "a finer grain must keep more entropy, or the sweep measures nothing"
        );
        assert_num!(
            raw.typo_rate() < r.typo_rate(),
            "a finer grain must tolerate fewer typos, or the sweep measures nothing"
        );
        assert_num!(
            raw.h2_bits < r.folded_bits,
            "the unquantised diagram is the ceiling for any grain; if it ever \
             clears the folded baseline, the verdict has to be re-derived"
        );
        TestResult::Pass
    }

    fn test_proof_line_passes() -> TestResult {
        let line = topo_key_proof_line();
        test_assert!(
            line.starts_with("[TOPOKEY] proof version=1"),
            "proof prefix"
        );
        test_assert!(line.contains("result=pass"), "self-check must pass");
        test_assert!(
            line.contains("verdict=reject"),
            "the measured verdict must be reported, not hidden"
        );
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test(
            "topo_key::permutation_invariance",
            test_permutation_invariance,
        );
        crate::testing::register_test("topo_key::scale_equivariance", test_scale_equivariance);
        crate::testing::register_test("topo_key::stability_bottleneck", test_stability_bottleneck);
        crate::testing::register_test(
            "topo_key::elder_rule_and_independent_mst",
            test_elder_rule_and_independent_mst,
        );
        crate::testing::register_test(
            "topo_key::negative_control_no_long_bars",
            test_negative_control_no_long_bars,
        );
        crate::testing::register_test("topo_key::unmeasurable_is_none", test_unmeasurable_is_none);
        crate::testing::register_test(
            "topo_key::grain_brackets_the_stability_bound",
            test_grain_brackets_the_stability_bound,
        );
        crate::testing::register_test(
            "topo_key::case_typo_opens_the_record",
            test_case_typo_opens_the_record,
        );
        crate::testing::register_test(
            "topo_key::wrong_password_refused",
            test_wrong_password_refused,
        );
        crate::testing::register_test(
            "topo_key::tampered_record_refused",
            test_tampered_record_refused,
        );
        crate::testing::register_test("topo_key::salt_reaches_the_key", test_salt_reaches_the_key);
        crate::testing::register_test("topo_key::record_stores_no_key", test_record_stores_no_key);
        crate::testing::register_test(
            "topo_key::entropy_measured_and_verdict_locked",
            test_entropy_measured_and_verdict_locked,
        );
        crate::testing::register_test("topo_key::proof_line_passes", test_proof_line_passes);
    }
}
