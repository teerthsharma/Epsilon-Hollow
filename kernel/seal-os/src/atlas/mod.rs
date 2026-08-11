// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Atlas — the kernel's loadable coordinate patches.
//!
//! Seal OS treats the running kernel as a manifold. Built-in code is the
//! ambient space; a **chart** is a coordinate patch that can be *grafted* onto
//! it at runtime and *pruned* off again. A chart imports kernel functionality
//! through **germs** — named values the kernel publishes at a point of the
//! manifold — and charts that share structure record that sharing as an
//! **overlap**. The overlap relation is the atlas's *nerve*, and the nerve is
//! required to stay acyclic: a cover whose nerve contains a cycle is not a
//! cover you can walk, so [`Atlas::bind_overlap`] refuses to close one.
//!
//! Nothing here is POSIX and nothing here is Linux: the object format is
//! borrowed (ELF64 `ET_REL` is just how x86_64 toolchains emit relocatable
//! code), the interface is not.

pub mod fixture;
pub mod relobj;

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};
use sha2::{Digest, Sha256};
use spin::Mutex;

use self::relobj::{ChartImage, LoadReport, ObjError};

/// Domain separator mixed into every chart signature.
pub const CHART_SIG_DOMAIN: &[u8] = b"seal-chart-v1";

/// Release key charts grafted through the Seal ABI are checked against.
pub const SEAL_CHART_PUBLIC_KEY: [u8; 32] = [
    0x1a, 0x7c, 0x93, 0x40, 0x5b, 0xe2, 0x86, 0x11, 0xd4, 0x0f, 0x77, 0xa9, 0x35, 0xc8, 0x62, 0x0e,
    0x53, 0xbb, 0x2d, 0x94, 0x68, 0xf1, 0x07, 0xac, 0x31, 0x5e, 0xd0, 0x89, 0x46, 0x22, 0xfb, 0x70,
];

/// Fixture key used to sign the embedded proof chart at boot. Same pattern as
/// `pkg::PROOF_PKG_SIGNING_KEY`: the kernel signs the fixture it is about to
/// verify, so the verification path is exercised end to end without shipping a
/// production private key.
const PROOF_CHART_SIGNING_KEY: [u8; 32] = [
    0x2e, 0x84, 0x1b, 0xc7, 0x59, 0xa0, 0x36, 0xd2, 0x0b, 0x6f, 0xe4, 0x18, 0x7d, 0x95, 0x43, 0xca,
    0x60, 0x2f, 0xb8, 0x11, 0x87, 0x3d, 0xe6, 0x5a, 0x09, 0xcf, 0x74, 0x22, 0x9b, 0x18, 0x50, 0xd3,
];

// ---------------------------------------------------------------------------
// Germs — the kernel's published symbols
// ---------------------------------------------------------------------------

struct Germ {
    name: &'static str,
    addr: u64,
}

static GERMS: Mutex<Vec<Germ>> = Mutex::new(Vec::new());

/// Publish a kernel symbol so charts can bind to it. Re-publishing a name
/// replaces the previous address.
pub fn publish_germ(name: &'static str, addr: u64) {
    let mut germs = GERMS.lock();
    if let Some(existing) = germs.iter_mut().find(|g| g.name == name) {
        existing.addr = addr;
        return;
    }
    germs.push(Germ { name, addr });
}

/// Resolve a germ by name.
pub fn germ(name: &str) -> Option<u64> {
    GERMS.lock().iter().find(|g| g.name == name).map(|g| g.addr)
}

pub fn germ_count() -> usize {
    GERMS.lock().len()
}

/// Publish `$path` under its own identifier.
#[macro_export]
macro_rules! publish_germ {
    ($path:path) => {
        $crate::atlas::publish_germ(stringify!($path), $path as *const () as usize as u64)
    };
    ($name:expr, $path:path) => {
        $crate::atlas::publish_germ($name, $path as *const () as usize as u64)
    };
}

/// Germ the proof chart imports; returns a constant the chart folds into its
/// init return code, so a botched `R_X86_64_PLT32` shows up as a wrong code.
extern "C" fn atlas_germ_probe() -> i64 {
    fixture::GERM_PROBE_VALUE
}

/// Germ giving charts a way to write to the kernel serial log.
extern "C" fn atlas_germ_serial_write(ptr: *const u8, len: usize) -> i64 {
    if ptr.is_null() || len > 4096 {
        return -22; // EINVAL
    }
    let bytes = unsafe { core::slice::from_raw_parts(ptr, len) };
    crate::serial_print!("{}", String::from_utf8_lossy(bytes));
    len as i64
}

/// Publish the germs every chart may assume. Idempotent.
pub fn publish_default_germs() {
    publish_germ(
        "atlas_germ_probe",
        atlas_germ_probe as *const () as usize as u64,
    );
    publish_germ(
        "atlas_germ_serial_write",
        atlas_germ_serial_write as *const () as usize as u64,
    );
}

// ---------------------------------------------------------------------------
// Signatures
// ---------------------------------------------------------------------------

fn chart_digest(object: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(CHART_SIG_DOMAIN);
    hasher.update(object);
    hasher.finalize().into()
}

/// Verify a detached ed25519 signature over `sha256(domain || object)`.
pub fn verify_chart_signature(object: &[u8], signature: &[u8; 64], public_key: &[u8; 32]) -> bool {
    let Ok(vk) = VerifyingKey::from_bytes(public_key) else {
        return false;
    };
    vk.verify_strict(&chart_digest(object), &Signature::from_bytes(signature))
        .is_ok()
}

fn sign_proof_chart(object: &[u8]) -> [u8; 64] {
    SigningKey::from_bytes(&PROOF_CHART_SIGNING_KEY)
        .sign(&chart_digest(object))
        .to_bytes()
}

fn proof_chart_public_key() -> [u8; 32] {
    SigningKey::from_bytes(&PROOF_CHART_SIGNING_KEY)
        .verifying_key()
        .to_bytes()
}

// ---------------------------------------------------------------------------
// Atlas registry
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AtlasError {
    Object(ObjError),
    BadSignature,
    DuplicateChart,
    UnknownChart,
    MissingOverlap,
    /// Refused: something still references this chart.
    Busy(u32),
    /// Refused: the overlap would close a cycle in the nerve.
    NerveCycle,
    /// `chart_init` returned non-zero; the chart was rolled back.
    InitRefused(i64),
}

impl AtlasError {
    pub fn tag(&self) -> &'static str {
        match self {
            AtlasError::Object(_) => "object",
            AtlasError::BadSignature => "bad_signature",
            AtlasError::DuplicateChart => "duplicate",
            AtlasError::UnknownChart => "unknown",
            AtlasError::MissingOverlap => "missing_overlap",
            AtlasError::Busy(_) => "busy",
            AtlasError::NerveCycle => "nerve_cycle",
            AtlasError::InitRefused(_) => "init_refused",
        }
    }
}

struct Chart {
    name: String,
    // REASON: held only to keep the mapping alive; dropping it unmaps the image.
    image: ChartImage,
    exit_addr: u64,
    /// Charts this one sits over, i.e. outgoing nerve edges.
    overlaps: Vec<String>,
    /// Incoming references: dependents plus explicit holds.
    refs: u32,
    report: LoadReport,
}

pub struct Atlas {
    charts: Vec<Chart>,
}

/// The kernel atlas.
pub static ATLAS: Mutex<Atlas> = Mutex::new(Atlas::new());

fn call_entry(addr: u64) -> i64 {
    let f: extern "C" fn() -> i64 = unsafe { core::mem::transmute(addr as usize) };
    f()
}

impl Atlas {
    pub const fn new() -> Self {
        Self { charts: Vec::new() }
    }

    pub fn len(&self) -> usize {
        self.charts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.charts.is_empty()
    }

    fn index_of(&self, name: &str) -> Option<usize> {
        self.charts.iter().position(|c| c.name == name)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.index_of(name).is_some()
    }

    pub fn refs(&self, name: &str) -> Option<u32> {
        self.index_of(name).map(|i| self.charts[i].refs)
    }

    /// `name -> refs` for every grafted chart.
    pub fn list(&self) -> Vec<(String, u32)> {
        self.charts
            .iter()
            .map(|c| (c.name.clone(), c.refs))
            .collect()
    }

    /// Can `from` reach `to` by following overlap edges?
    fn reaches(&self, from: &str, to: &str) -> bool {
        let mut stack: Vec<&str> = alloc::vec![from];
        let mut seen: Vec<&str> = Vec::new();
        while let Some(node) = stack.pop() {
            if node == to {
                return true;
            }
            if seen.contains(&node) {
                continue;
            }
            seen.push(node);
            if let Some(idx) = self.index_of(node) {
                for edge in &self.charts[idx].overlaps {
                    stack.push(edge.as_str());
                }
            }
        }
        false
    }

    /// Record that `from` sits over `to`, refusing to close a cycle in the
    /// nerve. Succeeds idempotently if the edge already exists.
    pub fn bind_overlap(&mut self, from: &str, to: &str) -> Result<(), AtlasError> {
        let from_idx = self.index_of(from).ok_or(AtlasError::UnknownChart)?;
        if self.index_of(to).is_none() {
            return Err(AtlasError::MissingOverlap);
        }
        if self.charts[from_idx].overlaps.iter().any(|o| o == to) {
            return Ok(());
        }
        // A cycle appears exactly when `to` can already reach `from`.
        if from == to || self.reaches(to, from) {
            return Err(AtlasError::NerveCycle);
        }
        self.charts[from_idx].overlaps.push(to.to_string());
        let to_idx = self.index_of(to).ok_or(AtlasError::MissingOverlap)?;
        self.charts[to_idx].refs = self.charts[to_idx].refs.saturating_add(1);
        Ok(())
    }

    /// Take an explicit reference, pinning the chart against pruning.
    pub fn hold(&mut self, name: &str) -> Result<u32, AtlasError> {
        let idx = self.index_of(name).ok_or(AtlasError::UnknownChart)?;
        self.charts[idx].refs = self.charts[idx].refs.saturating_add(1);
        Ok(self.charts[idx].refs)
    }

    /// Drop an explicit reference.
    pub fn release(&mut self, name: &str) -> Result<u32, AtlasError> {
        let idx = self.index_of(name).ok_or(AtlasError::UnknownChart)?;
        self.charts[idx].refs = self.charts[idx].refs.saturating_sub(1);
        Ok(self.charts[idx].refs)
    }

    /// Verify, relocate, map and initialise a chart.
    ///
    /// Returns `(init_code, load_report)`. On any failure nothing is registered
    /// and the image is unmapped before returning.
    pub fn graft(
        &mut self,
        name: &str,
        object: &[u8],
        signature: &[u8; 64],
        public_key: &[u8; 32],
        overlaps: &[&str],
    ) -> Result<(i64, LoadReport), AtlasError> {
        if self.contains(name) {
            return Err(AtlasError::DuplicateChart);
        }
        if !verify_chart_signature(object, signature, public_key) {
            return Err(AtlasError::BadSignature);
        }
        for dep in overlaps {
            if !self.contains(dep) {
                return Err(AtlasError::MissingOverlap);
            }
        }

        let (mut image, report) =
            relobj::load(object, &|sym: &str| germ(sym)).map_err(AtlasError::Object)?;

        // Contract: `chart_init` returns a negative value to decline the graft;
        // any non-negative value is accepted and reported verbatim.

        if !image.seal() {
            drop(image);
            return Err(AtlasError::Object(ObjError::Truncated));
        }

        let init_code = call_entry(report.init_addr);
        if init_code < 0 {
            drop(image); // unmap without ever calling exit
            return Err(AtlasError::InitRefused(init_code));
        }

        self.charts.push(Chart {
            name: name.to_string(),
            image,
            exit_addr: report.exit_addr,
            overlaps: Vec::new(),
            refs: 0,
            report,
        });
        for dep in overlaps {
            // A freshly pushed chart has no incoming edges, so this cannot
            // cycle; the check runs anyway rather than being assumed.
            if let Err(e) = self.bind_overlap(name, dep) {
                let _ = self.force_prune(name);
                return Err(e);
            }
        }
        Ok((init_code, report))
    }

    /// Call `chart_exit`, unmap the image and remove the chart.
    ///
    /// Refused with [`AtlasError::Busy`] while anything still references it.
    pub fn prune(&mut self, name: &str) -> Result<i64, AtlasError> {
        let idx = self.index_of(name).ok_or(AtlasError::UnknownChart)?;
        let refs = self.charts[idx].refs;
        if refs > 0 {
            return Err(AtlasError::Busy(refs));
        }
        self.force_prune(name)
    }

    fn force_prune(&mut self, name: &str) -> Result<i64, AtlasError> {
        let idx = self.index_of(name).ok_or(AtlasError::UnknownChart)?;
        let exit_code = call_entry(self.charts[idx].exit_addr);
        let chart = self.charts.remove(idx);
        for dep in &chart.overlaps {
            if let Some(d) = self.index_of(dep) {
                self.charts[d].refs = self.charts[d].refs.saturating_sub(1);
            }
        }
        drop(chart); // ChartImage::drop unmaps and returns the frames
        Ok(exit_code)
    }

    /// Load report of a grafted chart.
    pub fn report(&self, name: &str) -> Option<LoadReport> {
        self.index_of(name).map(|i| self.charts[i].report)
    }

    /// Whether a chart's text half is mapped read+execute (W^X sealed).
    pub fn sealed(&self, name: &str) -> Option<bool> {
        self.index_of(name)
            .map(|i| self.charts[i].image.is_sealed())
    }
}

impl Default for Atlas {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Boot proof
// ---------------------------------------------------------------------------

const PROOF_ALPHA: &str = "seal-proof-chart-alpha";
const PROOF_BETA: &str = "seal-proof-chart-beta";

fn ok_fail(v: bool) -> &'static str {
    if v {
        "ok"
    } else {
        "fail"
    }
}

/// Render `Atlas::sealed`'s tri-state outcome for the boot proof line.
///
/// `None` means the chart never made it into the atlas at all — grafting was
/// never attempted, or it ran and failed before the chart was registered.
/// `Some(false)` means the chart *is* registered and its text half is still
/// `RW+NX`, i.e. W^X enforcement itself is broken. Collapsing these two with
/// `unwrap_or(false)` prints the same `wx=fail` for both, and a reader sees
/// only the W^X story: that conflation sent a real investigation into page
/// permissions and TLB shootdown for hours when the actual defect was
/// `relobj::parse` failing to load a chart at all (fixed in `56dd589`), so
/// `graft` never reached the mapping code this field is supposed to describe.
fn wx_field(sealed: Option<bool>) -> &'static str {
    match sealed {
        Some(true) => "text_rx_data_rw_nx",
        Some(false) => "unsealed",
        None => "absent",
    }
}

/// Graft the embedded proof chart, exercise every guard, and report what
/// measurably happened as one line.
pub fn module_proof_line() -> String {
    publish_default_germs();

    let object = fixture::proof_chart();
    let signature = sign_proof_chart(&object);
    let public_key = proof_chart_public_key();

    let mut atlas = ATLAS.lock();
    // Re-entrant safety: clear anything a previous run left behind.
    for stale in [PROOF_BETA, PROOF_ALPHA, fixture::PROOF_CHART_NAME] {
        let _ = atlas.force_prune(stale);
    }
    let charts_before = atlas.len();

    // --- negative paths: none of these may register a chart ---------------
    let truncated = fixture::truncated_chart();
    let truncated_refused = matches!(
        atlas.graft(
            "seal-proof-truncated",
            &truncated,
            &sign_proof_chart(&truncated),
            &public_key,
            &[],
        ),
        Err(AtlasError::Object(_))
    );

    let unresolved = fixture::unresolved_chart();
    let unresolved_refused = matches!(
        atlas.graft(
            "seal-proof-unresolved",
            &unresolved,
            &sign_proof_chart(&unresolved),
            &public_key,
            &[],
        ),
        Err(AtlasError::Object(ObjError::UnresolvedGerm(_)))
    );

    let mut tampered = signature;
    tampered[0] ^= 0xFF;
    let bad_signature_refused = matches!(
        atlas.graft("seal-proof-badsig", &object, &tampered, &public_key, &[]),
        Err(AtlasError::BadSignature)
    );

    // --- the real graft ----------------------------------------------------
    let grafted = atlas.graft(
        fixture::PROOF_CHART_NAME,
        &object,
        &signature,
        &public_key,
        &[],
    );
    let (init_code, report) = match grafted {
        Ok(v) => (v.0, v.1),
        Err(_) => (i64::MIN, LoadReport::default()),
    };
    let graft_ok = init_code == fixture::PROOF_INIT_EXPECT;
    let wx_state = atlas.sealed(fixture::PROOF_CHART_NAME);
    let wx_sealed = wx_state == Some(true);

    // --- refcount guard: explicit hold -------------------------------------
    let held = atlas.hold(fixture::PROOF_CHART_NAME).unwrap_or(0);
    let hold_guard_ok = matches!(
        atlas.prune(fixture::PROOF_CHART_NAME),
        Err(AtlasError::Busy(_))
    ) && held == 1;
    let _ = atlas.release(fixture::PROOF_CHART_NAME);

    // --- nerve: beta sits over alpha, closing the loop must be refused -----
    let alpha_ok = atlas
        .graft(PROOF_ALPHA, &object, &signature, &public_key, &[])
        .is_ok();
    let beta_ok = atlas
        .graft(PROOF_BETA, &object, &signature, &public_key, &[PROOF_ALPHA])
        .is_ok();
    let cycle_refused = matches!(
        atlas.bind_overlap(PROOF_ALPHA, PROOF_BETA),
        Err(AtlasError::NerveCycle)
    );
    let dep_guard_ok = matches!(atlas.prune(PROOF_ALPHA), Err(AtlasError::Busy(_)));
    let charts_peak = atlas.len();

    let beta_exit = atlas.prune(PROOF_BETA);
    let alpha_exit = atlas.prune(PROOF_ALPHA);
    let nerve_ok = alpha_ok
        && beta_ok
        && cycle_refused
        && dep_guard_ok
        && beta_exit == Ok(fixture::PROOF_EXIT_EXPECT)
        && alpha_exit == Ok(fixture::PROOF_EXIT_EXPECT);

    // --- prune the proof chart --------------------------------------------
    let exit_result = atlas.prune(fixture::PROOF_CHART_NAME);
    let exit_code = exit_result.unwrap_or(i64::MIN);
    let exit_ok = exit_code == fixture::PROOF_EXIT_EXPECT;
    let charts_after = atlas.len();

    let counts_ok = charts_peak == charts_before + 3 && charts_after == charts_before;
    let reloc_ok = report.reloc_64 == 1
        && report.reloc_pc32 == 3
        && report.reloc_plt32 == 1
        && report.reloc_32s == 1
        && report.relocations_applied == 6;

    let result = graft_ok
        && wx_sealed
        && exit_ok
        && reloc_ok
        && counts_ok
        && hold_guard_ok
        && nerve_ok
        && truncated_refused
        && unresolved_refused
        && bad_signature_refused
        && report.germs_bound == 1;

    format!(
        "[Atlas] proof version={} source=embedded_chart format=elf64_rel machine=x86_64 object_bytes={} sections_placed={} symbols_resolved={} germs_published={} germs_bound={} plt_veneers={} relocations_applied={} r64={} rpc32={} rplt32={} r32s={} image_bytes={} wx={} signature=ed25519_fixture truncated_object={} unresolved_germ={} bad_signature={} init_code={:#x} init_expect={:#x} exit_code={:#x} exit_expect={:#x} refcount_hold_guard={} refcount_dependency_guard={} nerve_cycle={} charts_before={} charts_peak={} charts_after={} result={}",
        relobj::CHART_FORMAT_VERSION,
        object.len(),
        report.sections_placed,
        report.symbols_resolved,
        germ_count(),
        report.germs_bound,
        report.plt_slots,
        report.relocations_applied,
        report.reloc_64,
        report.reloc_pc32,
        report.reloc_plt32,
        report.reloc_32s,
        report.image_bytes,
        wx_field(wx_state),
        ok_fail(truncated_refused),
        ok_fail(unresolved_refused),
        ok_fail(bad_signature_refused),
        init_code,
        fixture::PROOF_INIT_EXPECT,
        exit_code,
        fixture::PROOF_EXIT_EXPECT,
        if hold_guard_ok { "refused_busy" } else { "fail" },
        if dep_guard_ok { "refused_busy" } else { "fail" },
        if cycle_refused { "refused" } else { "fail" },
        charts_before,
        charts_peak,
        charts_after,
        if result { "pass" } else { "fail" }
    )
}

/// Emit the Atlas boot proof to serial.
pub fn emit_boot_proof() {
    crate::serial_println!("{}", module_proof_line());
}

// ---------------------------------------------------------------------------
// Seal ABI helpers (used by syscall::table)
// ---------------------------------------------------------------------------

/// Graft a chart supplied as raw object + detached signature bytes.
pub fn abi_graft(name: &str, object: &[u8], signature: &[u8]) -> Result<i64, AtlasError> {
    if signature.len() != 64 {
        return Err(AtlasError::BadSignature);
    }
    let mut sig = [0u8; 64];
    sig.copy_from_slice(signature);
    publish_default_germs();
    ATLAS
        .lock()
        .graft(name, object, &sig, &SEAL_CHART_PUBLIC_KEY, &[])
        .map(|(code, _)| code)
}

pub fn abi_prune(name: &str) -> Result<i64, AtlasError> {
    ATLAS.lock().prune(name)
}

pub fn abi_list() -> String {
    let atlas = ATLAS.lock();
    if atlas.is_empty() {
        return String::from("no charts grafted");
    }
    atlas
        .list()
        .iter()
        .map(|(name, refs)| format!("{} refs={}\n", name, refs))
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(feature = "test-mode")]
pub mod tests {
    use super::*;
    use crate::testing::TestResult;
    use crate::{test_assert, test_assert_eq};

    fn fixture_key() -> ([u8; 32], alloc::vec::Vec<u8>, [u8; 64]) {
        publish_default_germs();
        let object = fixture::proof_chart();
        let sig = sign_proof_chart(&object);
        (proof_chart_public_key(), object, sig)
    }

    fn test_graft_valid_chart() -> TestResult {
        let (key, object, sig) = fixture_key();
        let mut atlas = ATLAS.lock();
        let _ = atlas.force_prune("t-valid");
        let grafted = atlas.graft("t-valid", &object, &sig, &key, &[]);
        test_assert!(grafted.is_ok(), "valid chart should graft");
        let (code, report) = grafted.unwrap_or((0, LoadReport::default()));
        test_assert_eq!(code, fixture::PROOF_INIT_EXPECT);
        test_assert_eq!(report.relocations_applied, 6);
        test_assert_eq!(report.germs_bound, 1);
        test_assert_eq!(atlas.prune("t-valid"), Ok(fixture::PROOF_EXIT_EXPECT));
        TestResult::Pass
    }

    fn test_truncated_object_refused() -> TestResult {
        let (key, _, _) = fixture_key();
        let object = fixture::truncated_chart();
        let sig = sign_proof_chart(&object);
        let mut atlas = ATLAS.lock();
        let r = atlas.graft("t-trunc", &object, &sig, &key, &[]);
        test_assert!(
            matches!(r, Err(AtlasError::Object(_))),
            "truncated object must be refused, not panic"
        );
        test_assert!(!atlas.contains("t-trunc"), "nothing may be registered");
        TestResult::Pass
    }

    fn test_unresolved_germ_refused() -> TestResult {
        let (key, _, _) = fixture_key();
        let object = fixture::unresolved_chart();
        let sig = sign_proof_chart(&object);
        let mut atlas = ATLAS.lock();
        let r = atlas.graft("t-unres", &object, &sig, &key, &[]);
        test_assert!(
            matches!(r, Err(AtlasError::Object(ObjError::UnresolvedGerm(_)))),
            "unresolved germ must fail the graft"
        );
        TestResult::Pass
    }

    fn test_bad_signature_refused() -> TestResult {
        let (key, object, mut sig) = fixture_key();
        sig[63] ^= 0x01;
        let mut atlas = ATLAS.lock();
        let r = atlas.graft("t-badsig", &object, &sig, &key, &[]);
        test_assert_eq!(r, Err(AtlasError::BadSignature));
        TestResult::Pass
    }

    fn test_prune_while_referenced_refused() -> TestResult {
        let (key, object, sig) = fixture_key();
        let mut atlas = ATLAS.lock();
        let _ = atlas.force_prune("t-busy");
        test_assert!(atlas.graft("t-busy", &object, &sig, &key, &[]).is_ok());
        test_assert_eq!(atlas.hold("t-busy"), Ok(1));
        test_assert_eq!(atlas.prune("t-busy"), Err(AtlasError::Busy(1)));
        test_assert_eq!(atlas.release("t-busy"), Ok(0));
        test_assert_eq!(atlas.prune("t-busy"), Ok(fixture::PROOF_EXIT_EXPECT));
        TestResult::Pass
    }

    fn test_sealed_true_after_successful_graft() -> TestResult {
        let (key, object, sig) = fixture_key();
        let mut atlas = ATLAS.lock();
        let _ = atlas.force_prune("t-sealed");
        test_assert!(atlas.graft("t-sealed", &object, &sig, &key, &[]).is_ok());
        test_assert_eq!(atlas.sealed("t-sealed"), Some(true));
        test_assert!(atlas.prune("t-sealed").is_ok());
        TestResult::Pass
    }

    fn test_nerve_cycle_refused() -> TestResult {
        let (key, object, sig) = fixture_key();
        let mut atlas = ATLAS.lock();
        for name in ["t-cyc-c", "t-cyc-b", "t-cyc-a"] {
            let _ = atlas.force_prune(name);
        }
        test_assert!(atlas.graft("t-cyc-a", &object, &sig, &key, &[]).is_ok());
        test_assert!(atlas
            .graft("t-cyc-b", &object, &sig, &key, &["t-cyc-a"])
            .is_ok());
        test_assert!(atlas
            .graft("t-cyc-c", &object, &sig, &key, &["t-cyc-b"])
            .is_ok());
        // c -> b -> a already; a -> c would close the loop.
        test_assert_eq!(
            atlas.bind_overlap("t-cyc-a", "t-cyc-c"),
            Err(AtlasError::NerveCycle)
        );
        test_assert_eq!(
            atlas.bind_overlap("t-cyc-a", "t-cyc-a"),
            Err(AtlasError::NerveCycle)
        );
        test_assert_eq!(atlas.prune("t-cyc-a"), Err(AtlasError::Busy(1)));
        test_assert!(atlas.prune("t-cyc-c").is_ok());
        test_assert!(atlas.prune("t-cyc-b").is_ok());
        test_assert!(atlas.prune("t-cyc-a").is_ok());
        TestResult::Pass
    }

    fn test_sealed_none_for_unknown_chart() -> TestResult {
        let atlas = ATLAS.lock();
        test_assert_eq!(atlas.sealed("t-does-not-exist"), None);
        TestResult::Pass
    }

    /// `wx_field` is what turns `Atlas::sealed`'s `Option<bool>` into the
    /// boot proof's `wx=` value. A chart absent from the atlas and a chart
    /// present but still `RW+NX` must not print the same string — that
    /// collapse is exactly what sent the original investigation into page
    /// permissions instead of `relobj::parse`.
    fn test_wx_field_distinguishes_absent_from_unsealed() -> TestResult {
        test_assert_eq!(wx_field(None), "absent");
        test_assert_eq!(wx_field(Some(false)), "unsealed");
        test_assert_eq!(wx_field(Some(true)), "text_rx_data_rw_nx");
        test_assert!(
            wx_field(None) != wx_field(Some(false)),
            "absent chart and present-but-unsealed chart must not share a wx= value"
        );
        TestResult::Pass
    }

    fn test_relocation_arithmetic() -> TestResult {
        use relobj::{compute_reloc, RelocWrite};
        // S + A
        test_assert_eq!(
            compute_reloc(relobj::R_X86_64_64, 0xFFFF_9000_0000_1000, 0x20, 0),
            Ok(RelocWrite::W8(0xFFFF_9000_0000_1020))
        );
        // S + A - P, the encoding a `call rel32` at P wants.
        test_assert_eq!(
            compute_reloc(
                relobj::R_X86_64_PC32,
                0xFFFF_9000_0000_1100,
                -4,
                0xFFFF_9000_0000_1005
            ),
            Ok(RelocWrite::W4(0xF7))
        );
        // Negative displacement sign-extends.
        test_assert_eq!(
            compute_reloc(
                relobj::R_X86_64_PLT32,
                0xFFFF_9000_0000_1000,
                -4,
                0xFFFF_9000_0000_1100
            ),
            Ok(RelocWrite::W4(0xFFFF_FF00))
        );
        // 32S truncates to a sign-extendable 32-bit value.
        test_assert_eq!(
            compute_reloc(relobj::R_X86_64_32S, 0x42, 0, 0),
            Ok(RelocWrite::W4(0x42))
        );
        // Out of +/-2 GiB must be refused, not silently wrapped.
        test_assert_eq!(
            compute_reloc(
                relobj::R_X86_64_PC32,
                0xFFFF_FFFF_8000_0000,
                0,
                0xFFFF_9000_0000_0000
            ),
            Err(ObjError::RelocationOverflow)
        );
        test_assert_eq!(
            compute_reloc(relobj::R_X86_64_32S, 0xFFFF_9000_0000_0000, 0, 0),
            Err(ObjError::RelocationOverflow)
        );
        test_assert_eq!(
            compute_reloc(9, 0, 0, 0),
            Err(ObjError::UnsupportedRelocation(9))
        );
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test("atlas::graft_valid_chart", test_graft_valid_chart);
        crate::testing::register_test(
            "atlas::truncated_object_refused",
            test_truncated_object_refused,
        );
        crate::testing::register_test(
            "atlas::unresolved_germ_refused",
            test_unresolved_germ_refused,
        );
        crate::testing::register_test("atlas::bad_signature_refused", test_bad_signature_refused);
        crate::testing::register_test(
            "atlas::prune_while_referenced_refused",
            test_prune_while_referenced_refused,
        );
        crate::testing::register_test(
            "atlas::sealed_true_after_successful_graft",
            test_sealed_true_after_successful_graft,
        );
        crate::testing::register_test("atlas::nerve_cycle_refused", test_nerve_cycle_refused);
        crate::testing::register_test("atlas::relocation_arithmetic", test_relocation_arithmetic);
        crate::testing::register_test(
            "atlas::sealed_none_for_unknown_chart",
            test_sealed_none_for_unknown_chart,
        );
        crate::testing::register_test(
            "atlas::wx_field_distinguishes_absent_from_unsealed",
            test_wx_field_distinguishes_absent_from_unsealed,
        );
    }
}
