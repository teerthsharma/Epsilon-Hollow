// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Permission as a **total field** rather than a lookup table.
//!
//! # The problem this replaces
//!
//! [`crate::security::mac::check_file_permission`] is a lookup: it walks a rule
//! list and, when nothing matches, returns `true` (mac.rs:141), and returns
//! `true` outright for uid 0 (mac.rs:110) and when no policy is loaded
//! (mac.rs:117). A lookup table has holes. Every hole is a question — either a
//! silent default-allow, or a prompt someone has to answer. An agent driving
//! this OS hits those holes constantly.
//!
//! This module makes permission a function that is **defined at every point by
//! construction**:
//!
//! ```text
//! evaluate : (subject, resource, action) -> Verdict ∈ {Allow, Deny, Unknown}
//! ```
//!
//! Totality is what removes the prompt. It is *not* totality-by-permitting: the
//! value at an uncovered point is [`Verdict::Unknown`], a third value the caller
//! must handle, never a grant.
//!
//! Nothing here is wired into the live `mac.rs` path. This lands the field with
//! its proofs; replacing the live check is a separate change.
//!
//! # The coordinate embedding, and why it is defensible
//!
//! A field is only as meaningful as its metric. A metric where unrelated things
//! are close is *worse* than a lookup table, because it generalises wrongly and
//! with confidence. So each axis gets the metric it actually has, and no more:
//!
//! | axis | metric | why |
//! |------|--------|-----|
//! | subject (uid) | discrete: equal, or a wildcard | uid 1000 and 1001 are two unrelated people. There is no sense in which a grant to one should leak to the other, so there is no distance between them — only equality, or an explicit "any subject" source. |
//! | action | discrete: equal, or a wildcard | `Read` is not "near" `Write`. Interpolating between them has no meaning; the only defensible relation is equality. |
//! | resource (path) | **descendant depth in the path tree** | This is the one axis with a real metric, and it is not invented here: `mac.rs` already expresses every rule as a path prefix matched at component boundaries. `/data/logs/a` and `/data/logs/b` are genuinely security-similar precisely because the existing policy language cannot distinguish them without a new rule. |
//!
//! The path metric is **asymmetric on purpose**. [`steps_below`] measures how
//! many components `path` lies *below* `anchor`, and is undefined otherwise. A
//! symmetric tree metric (up-steps + down-steps) would put `/etc` two steps from
//! `/data`, so a grant on `/data` with radius 2 would reach `/etc` — siblings
//! are not security-similar, and a symmetric metric is exactly how a permission
//! field becomes an escalation engine. Ancestors are excluded for the same
//! reason: granting `/data/x` must not grant `/data`.
//!
//! What the embedding will *not* do is put two applications near each other.
//! There is no observable in this kernel that makes "similar application" mean
//! anything defensible, so the subject axis stays discrete. See the ceiling note
//! at the bottom of this doc.
//!
//! # Influence kernel
//!
//! A source influences a query point iff its subject and action match (or are
//! wildcards) and `steps_below(anchor, path) <= radius`. That is a step kernel,
//! and it is the whole of the "distance weighting" on purpose: the verdict is a
//! three-valued lattice under deny-dominance, so *any* monotone decreasing
//! kernel with the same support produces the same verdict at every point. A
//! smooth falloff would be decoration that cannot change an answer.
//!
//! # The three safety properties
//!
//! 1. **Deny dominates.** [`evaluate`] returns [`Verdict::Deny`] the moment any
//!    influencing source is a denial, before it has looked at how near or how
//!    numerous the grants are. There is no comparison of nearness anywhere in
//!    the function, so the verdict is independent of source order and no
//!    quantity of grants can outvote one denial.
//! 2. **Totality without invention.** Every point evaluates. A point with no
//!    source in range is [`Verdict::Unknown`], which is neither a grant nor a
//!    silent deny. See "What `Unknown` means at the call site" below.
//! 3. **Inference may only narrow.** [`infer`] returns a [`Narrowing`], whose
//!    entries have **no polarity field**. A widening is not merely something
//!    `infer` declines to emit — there is nowhere in the returned type to put
//!    one. [`PermField::narrow`] can only append denials.
//!
//! # What `Unknown` means at the call site
//!
//! `Unknown` is **refuse now, and report the point as uncovered**. Concretely, a
//! future `shell::deny` reads [`Verdict::permits`], which is true for `Allow`
//! alone, so the operation is refused; and because `Unknown` is a distinct value
//! from `Deny`, the audit record says "no source covers this point" rather than
//! "policy forbids this point".
//!
//! That distinction is the entire reason for the third value:
//!
//! * mapping `Unknown` to a grant is totality-by-permitting, which is the
//!   default-allow hole in `mac.rs` rebuilt with more machinery;
//! * mapping it to a plain `Deny` is a silent deny that hides a
//!   misconfiguration — indistinguishable from a policy that meant it.
//!
//! Refusing *without prompting* is what makes the field total in the sense that
//! matters. The kernel never has to ask a human; the caller gets a
//! machine-readable "uncovered" that an agent resolves by adding an explicit
//! source, once, rather than by answering the same question forever.
//!
//! [`Verdict::permits`] exists so this cannot be got wrong by writing
//! `verdict != Verdict::Deny`.
//!
//! # Userspace membership is the same field
//!
//! "Is this in userspace" stops being a bit. It is [`Action::Member`] evaluated
//! against the same sources by the same [`evaluate`] — same deny-dominance, same
//! `Unknown`. See [`in_userspace`], which is one line because it is not a second
//! mechanism.
//!
//! # Bounds
//!
//! Fixed heap. [`MAX_SOURCES`] sources per field, [`MAX_RADIUS`] components of
//! reach per source, [`MAX_ANCHOR_BYTES`] per anchor, [`MAX_OBSERVATIONS`]
//! observations per inference. The radius bound is enforced in two independent
//! places — [`PermField::push`] refuses an over-wide source, and [`evaluate`]
//! clamps at the point of use — so a source that reaches [`evaluate`] without
//! passing through `push` still cannot exceed it.

use alloc::string::String;
use alloc::vec::Vec;

/// Sources one field may hold. A field is a policy, not a database.
pub const MAX_SOURCES: usize = 64;

/// Path components a single source may reach below its anchor.
///
/// This is the generalisation budget of one explicit decision. `mac.rs` prefix
/// rules reach infinitely deep; here a point deeper than this is `Unknown` and
/// therefore refused, which is a hole that fails closed rather than a rule that
/// silently covers a subtree nobody looked at.
pub const MAX_RADIUS: u8 = 8;

/// Bytes in a source anchor.
pub const MAX_ANCHOR_BYTES: usize = 128;

/// Observations one [`infer`] call will read. Inference is O(n² · MAX_RADIUS).
///
/// ponytail: quadratic scan, because each denial re-scans the observation set
/// for a blocking grant. 128² · 8 ≈ 131k component comparisons, once, off any
/// hot path. Upgrade path is sorting observations by path and walking ancestors
/// with a stack, which is O(n log n) and about three times the code.
pub const MAX_OBSERVATIONS: usize = 128;

/// The value of the field at a point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// An explicit grant covers this point and no denial does.
    Allow,
    /// A denial covers this point. Dominant.
    Deny,
    /// No source covers this point. Neither approval nor refusal — see the
    /// module docs for what a call site does with it.
    Unknown,
}

impl Verdict {
    /// True for [`Verdict::Allow`] alone.
    ///
    /// The only sanctioned way to turn a verdict into a boolean. `Unknown` is
    /// not permission, and a call site written as `v != Verdict::Deny` would say
    /// it was.
    pub fn permits(self) -> bool {
        matches!(self, Verdict::Allow)
    }
}

/// What is being attempted. Discrete: no action is "near" another.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Read,
    Write,
    Execute,
    /// Membership of userspace, evaluated by the same field as everything else.
    Member,
}

/// Which way a source pushes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Polarity {
    Grant,
    Deny,
}

/// One explicit decision, placed as a point with a radius of influence.
///
/// `subject: None` and `action: None` are wildcards — "every subject", "every
/// action". Wildcards exist because the alternative is one source per uid,
/// which does not fit in [`MAX_SOURCES`].
#[derive(Debug, Clone)]
pub struct Source {
    /// uid this source speaks for, or `None` for every subject.
    pub subject: Option<u32>,
    /// Action this source speaks for, or `None` for every action.
    pub action: Option<Action>,
    /// Canonical absolute path the source sits at.
    pub anchor: String,
    /// Components of reach below `anchor`. 0 is the anchor alone.
    pub radius: u8,
    pub polarity: Polarity,
}

impl Source {
    /// True when this source's support contains `q`.
    ///
    /// The radius is clamped to [`MAX_RADIUS`] here as well as refused in
    /// [`PermField::push`], so the bound holds on a `&[Source]` handed straight
    /// to [`evaluate`].
    fn influences(&self, q: &Query<'_>) -> bool {
        if self.subject.is_some_and(|s| s != q.subject) {
            return false;
        }
        if self.action.is_some_and(|a| a != q.action) {
            return false;
        }
        let reach = self.radius.min(MAX_RADIUS) as usize;
        steps_below(&self.anchor, q.path).is_some_and(|d| d <= reach)
    }
}

/// A point at which the field is evaluated.
///
/// `path` must be canonical — absolute, single `/` separators, no trailing
/// slash, no `.`/`..` components. Unlike `mac::check_file_permission`, which
/// documents the contract and trusts it, [`steps_below`] *checks* it: a
/// non-canonical path is inside nothing, so it evaluates to `Unknown` and is
/// refused rather than walking out of an anchor via `..`.
#[derive(Debug, Clone, Copy)]
pub struct Query<'a> {
    pub subject: u32,
    pub path: &'a str,
    pub action: Action,
}

/// Evaluate the field generated by `sources` at `q`. Total: every input has a
/// value.
///
/// Deny-dominance is structural. The loop returns on the first influencing
/// denial, having compared no distances and counted no grants, so the result
/// does not depend on source order and no number of nearer grants can beat one
/// denial in range.
pub fn evaluate(sources: &[Source], q: &Query<'_>) -> Verdict {
    let mut granted = false;
    for s in sources {
        if !s.influences(q) {
            continue;
        }
        match s.polarity {
            Polarity::Deny => return Verdict::Deny,
            Polarity::Grant => granted = true,
        }
    }
    if granted {
        Verdict::Allow
    } else {
        Verdict::Unknown
    }
}

/// A bounded set of sources. The field is the function they generate.
#[derive(Debug, Clone, Default)]
pub struct PermField {
    sources: Vec<Source>,
}

impl PermField {
    pub const fn new() -> Self {
        Self {
            sources: Vec::new(),
        }
    }

    /// Add an explicit source. Returns false, and changes nothing, when the
    /// source is outside a bound or its anchor is not canonical.
    ///
    /// Refused, never clamped: clamping a radius down would silently shrink a
    /// *denial* the author meant to reach further, and clamping one up would
    /// widen a grant nobody approved. A refusal leaves a hole, and a hole is
    /// `Unknown`, which the call site refuses.
    pub fn push(&mut self, s: Source) -> bool {
        if self.sources.len() >= MAX_SOURCES
            || s.radius > MAX_RADIUS
            || s.anchor.len() > MAX_ANCHOR_BYTES
            || !is_canonical(&s.anchor)
        {
            return false;
        }
        self.sources.push(s);
        true
    }

    /// Value of the field at `q`.
    pub fn evaluate(&self, q: &Query<'_>) -> Verdict {
        evaluate(&self.sources, q)
    }

    pub fn sources(&self) -> &[Source] {
        &self.sources
    }

    /// Apply an inferred narrowing. Returns the number of cuts installed, which
    /// is below `n.len()` only when [`MAX_SOURCES`] is reached.
    ///
    /// This is the only consumer of a [`Narrowing`], and it writes
    /// [`Polarity::Deny`] as a literal. A [`Cut`] carries no polarity, so there
    /// is no value of `n` for which this function could widen the field.
    pub fn narrow(&mut self, n: &Narrowing) -> usize {
        let mut applied = 0;
        for c in &n.cuts {
            let ok = self.push(Source {
                subject: c.subject,
                action: c.action,
                anchor: c.anchor.clone(),
                radius: c.radius,
                polarity: Polarity::Deny,
            });
            if ok {
                applied += 1;
            }
        }
        applied
    }
}

/// Membership of userspace at a point — the same field, a different support.
pub fn in_userspace(field: &PermField, subject: u32, path: &str) -> Verdict {
    field.evaluate(&Query {
        subject,
        path,
        action: Action::Member,
    })
}

// ── Inference ───────────────────────────────────────────────────────────────

/// One decision that was actually observed, e.g. replayed from the audit log.
#[derive(Debug, Clone, Copy)]
pub struct Observation<'a> {
    pub subject: u32,
    pub action: Action,
    pub path: &'a str,
    /// What the decision was. `true` observations are read by [`infer`] **only
    /// in a negative position** — they block a denial from being lifted — so a
    /// grant can never be the thing inference emits.
    pub allowed: bool,
}

/// One restriction. Deliberately has no polarity field.
#[derive(Debug, Clone)]
pub struct Cut {
    pub subject: Option<u32>,
    pub action: Option<Action>,
    pub anchor: String,
    pub radius: u8,
}

/// The output of [`infer`]: restrictions and nothing else.
///
/// This type is the structural half of "inference may only narrow". A widening
/// is unrepresentable here — [`Cut`] has no polarity to set — so the property
/// does not rest on `infer` being careful, and cannot be broken by a later edit
/// to `infer` that forgets.
#[derive(Debug, Clone, Default)]
pub struct Narrowing {
    cuts: Vec<Cut>,
}

impl Narrowing {
    pub fn len(&self) -> usize {
        self.cuts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cuts.is_empty()
    }

    pub fn cuts(&self) -> &[Cut] {
        &self.cuts
    }
}

/// Fit restrictions to observed decisions.
///
/// Every denied observation becomes a cut. The cut is then *sharpened* by
/// lifting its anchor toward the root for as long as no observation under the
/// candidate ancestor, for the same subject and action, was allowed — so the
/// generalisation stops at the first point where the evidence disagrees with
/// it. The radius is the number of lifts, i.e. exactly enough to reach the
/// denial that justified the cut and no further.
///
/// Observations beyond [`MAX_OBSERVATIONS`] are ignored.
pub fn infer(observations: &[Observation<'_>]) -> Narrowing {
    let obs = &observations[..observations.len().min(MAX_OBSERVATIONS)];
    let mut out = Narrowing { cuts: Vec::new() };
    for d in obs.iter().filter(|o| !o.allowed) {
        if !is_canonical(d.path) || out.cuts.len() >= MAX_SOURCES {
            continue;
        }
        let mut anchor = d.path;
        let mut lift = 0u8;
        while lift < MAX_RADIUS {
            let Some(parent) = parent_of(anchor) else {
                break;
            };
            let contradicted = obs.iter().any(|o| {
                o.allowed
                    && o.subject == d.subject
                    && o.action == d.action
                    && steps_below(parent, o.path).is_some()
            });
            if contradicted {
                break;
            }
            anchor = parent;
            lift += 1;
        }
        out.cuts.push(Cut {
            subject: Some(d.subject),
            action: Some(d.action),
            anchor: String::from(anchor),
            radius: lift,
        });
    }
    out
}

// ── Path metric ─────────────────────────────────────────────────────────────

/// Components `path` lies below `anchor`, or `None` when `path` is not inside
/// `anchor`.
///
/// Asymmetric by design (see the module docs): siblings and ancestors are not
/// at any distance, they are outside the support entirely. Matching is at
/// component boundaries, so `/database` is not inside `/data` — the
/// over-matching half of the bug class `mac::path_matches_rule` documents.
///
/// Non-canonical input on either side is inside nothing. That is what stops
/// `/data/../root` from being two steps below `/data`.
fn steps_below(anchor: &str, path: &str) -> Option<usize> {
    if !is_canonical(anchor) || !is_canonical(path) {
        return None;
    }
    let rest = if anchor == "/" {
        path.strip_prefix('/')?
    } else {
        match path.strip_prefix(anchor) {
            Some("") => return Some(0),
            Some(r) => r.strip_prefix('/')?,
            None => return None,
        }
    };
    if rest.is_empty() {
        return Some(0);
    }
    Some(rest.split('/').count())
}

/// Absolute, single separators, no trailing slash, no `.`/`..`.
fn is_canonical(path: &str) -> bool {
    if path == "/" {
        return true;
    }
    if !path.starts_with('/') || path.ends_with('/') {
        return false;
    }
    path.split('/')
        .skip(1)
        .all(|c| !c.is_empty() && c != "." && c != "..")
}

/// Parent component path, or `None` at the root.
fn parent_of(path: &str) -> Option<&str> {
    if path == "/" || !path.starts_with('/') {
        return None;
    }
    match path.rfind('/') {
        Some(0) => Some("/"),
        Some(i) => Some(&path[..i]),
        None => None,
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use super::*;
    use crate::testing::TestResult;
    use crate::{test_assert, test_assert_eq};
    use alloc::vec;
    use alloc::vec::Vec;

    fn src(
        subject: Option<u32>,
        action: Option<Action>,
        anchor: &str,
        radius: u8,
        polarity: Polarity,
    ) -> Source {
        Source {
            subject,
            action,
            anchor: String::from(anchor),
            radius,
            polarity,
        }
    }

    fn q<'a>(subject: u32, path: &'a str, action: Action) -> Query<'a> {
        Query {
            subject,
            path,
            action,
        }
    }

    /// Probe grid used by the totality and narrowing sweeps.
    fn probes() -> Vec<&'static str> {
        vec![
            "/",
            "/data",
            "/data/a",
            "/data/a/b",
            "/data/a/b/c",
            "/data/secret",
            "/database",
            "/etc",
            "/etc/passwd",
            "/root",
            "/root/deep/deep/deep/deep/deep/deep/deep/deep/deep/key",
            "/home",
            "/home/agent",
            "/home/agent/work",
            "/tmp",
        ]
    }

    /// **Property 1.** A denial in range wins over any number of nearer,
    /// tighter grants, at any position in the source list.
    fn test_deny_dominates_every_grant() -> TestResult {
        // Deny sits 3 components up; grants sit exactly on the point.
        let deny = src(None, None, "/data", 8, Polarity::Deny);
        let near1 = src(Some(7), Some(Action::Read), "/data/a/b/c", 0, Polarity::Grant);
        let near2 = src(Some(7), Some(Action::Read), "/data/a/b", 1, Polarity::Grant);
        let near3 = src(None, None, "/data/a", 2, Polarity::Grant);
        let point = q(7, "/data/a/b/c", Action::Read);

        // Grants first, denial last.
        let a = vec![near1.clone(), near2.clone(), near3.clone(), deny.clone()];
        test_assert_eq!(evaluate(&a, &point), Verdict::Deny);
        // Denial first, grants after.
        let b = vec![deny.clone(), near1.clone(), near2.clone(), near3.clone()];
        test_assert_eq!(evaluate(&b, &point), Verdict::Deny);
        // Denial buried in the middle.
        let c = vec![near1.clone(), deny.clone(), near2, near3];
        test_assert_eq!(evaluate(&c, &point), Verdict::Deny);

        // Interpolating between a grant and a deny never produces a grant: walk
        // every point on the segment from the grant's anchor down to the deny's.
        let seg = [
            "/data",
            "/data/a",
            "/data/a/b",
            "/data/a/b/c",
            "/data/a/b/c/d",
        ];
        let field = vec![
            src(None, None, "/data/a/b/c/d", 0, Polarity::Grant),
            src(None, None, "/data", 8, Polarity::Deny),
        ];
        for p in seg {
            test_assert!(
                evaluate(&field, &q(7, p, Action::Read)) == Verdict::Deny,
                "a point between a grant and a deny resolved to something other than Deny"
            );
        }

        // A denial the point is *outside* of must not deny it — dominance is
        // not "any denial anywhere".
        let outside = vec![near1, src(None, None, "/etc", 8, Polarity::Deny)];
        test_assert_eq!(evaluate(&outside, &point), Verdict::Allow);
        TestResult::Pass
    }

    /// **Property 2.** The field is defined at every point, and an uncovered
    /// point is `Unknown` — never `Allow`, and distinguishable from `Deny`.
    fn test_totality_without_invention() -> TestResult {
        let mut f = PermField::new();
        test_assert!(f.push(src(
            Some(7),
            Some(Action::Read),
            "/data",
            1,
            Polarity::Grant
        )));
        test_assert!(f.push(src(None, None, "/root", 2, Polarity::Deny)));

        // Empty field: total, and every point is Unknown.
        let empty = PermField::new();
        for p in probes() {
            let v = empty.evaluate(&q(7, p, Action::Read));
            test_assert!(v == Verdict::Unknown, "empty field invented a verdict");
            test_assert!(!v.permits(), "Unknown permitted an operation");
        }

        // Populated field: every probe still evaluates, and Allow appears only
        // inside a grant's support.
        for p in probes() {
            let v = f.evaluate(&q(7, p, Action::Read));
            let covered = steps_below("/data", p).is_some_and(|d| d <= 1);
            test_assert!(
                (v == Verdict::Allow) == (covered && steps_below("/root", p).is_none()),
                "Allow appeared at a point no grant covers, or failed to appear where one does"
            );
        }

        // uid 0 is not special here: mac.rs:110 short-circuits root, this does
        // not. An uncovered point is uncovered for root too.
        test_assert_eq!(f.evaluate(&q(0, "/etc/passwd", Action::Read)), Verdict::Unknown);
        // And a deny still binds root.
        test_assert_eq!(f.evaluate(&q(0, "/root/secret", Action::Read)), Verdict::Deny);

        // The three values are distinct, and only one of them permits.
        test_assert!(Verdict::Allow.permits());
        test_assert!(!Verdict::Deny.permits());
        test_assert!(!Verdict::Unknown.permits());
        test_assert!(Verdict::Unknown != Verdict::Deny);
        TestResult::Pass
    }

    /// **Property 3, load-bearing.** A trace whose natural generalisation is a
    /// grant must not produce one.
    ///
    /// uid 7 was allowed `/data/a`, `/data/b` and `/data/c`, and denied
    /// `/data/secret`. The obvious fit is "grant `/data` radius 1 to uid 7",
    /// which would hand uid 7 `/data/d` — a point nobody approved. The inferred
    /// field must not grant it, and must not grant *anything* the base field
    /// did not already grant.
    fn test_inference_only_narrows() -> TestResult {
        let mut base = PermField::new();
        for leaf in ["/data/a", "/data/b", "/data/c"] {
            test_assert!(base.push(src(
                Some(7),
                Some(Action::Read),
                leaf,
                0,
                Polarity::Grant
            )));
        }

        let trace = [
            Observation {
                subject: 7,
                action: Action::Read,
                path: "/data/a",
                allowed: true,
            },
            Observation {
                subject: 7,
                action: Action::Read,
                path: "/data/b",
                allowed: true,
            },
            Observation {
                subject: 7,
                action: Action::Read,
                path: "/data/c",
                allowed: true,
            },
            Observation {
                subject: 7,
                action: Action::Read,
                path: "/data/secret",
                allowed: false,
            },
        ];

        let before: Vec<Verdict> = probes()
            .iter()
            .map(|p| base.evaluate(&q(7, p, Action::Read)))
            .collect();

        let n = infer(&trace);
        test_assert_eq!(n.len(), 1);
        // The cut could not be lifted to /data: three allowed observations sit
        // under it. Sharpening stops where the evidence contradicts it.
        test_assert_eq!(n.cuts()[0].anchor.as_str(), "/data/secret");
        test_assert_eq!(n.cuts()[0].radius, 0u8);

        let mut after = base.clone();
        test_assert_eq!(after.narrow(&n), 1usize);

        // The natural generalisation, refused.
        test_assert_eq!(
            base.evaluate(&q(7, "/data/d", Action::Read)),
            Verdict::Unknown
        );
        test_assert!(
            after.evaluate(&q(7, "/data/d", Action::Read)) != Verdict::Allow,
            "inference granted a point no explicit source granted"
        );
        // ...and not by widening some other axis either.
        test_assert!(
            after.evaluate(&q(8, "/data/a", Action::Read)) != Verdict::Allow,
            "inference widened across the subject axis"
        );
        test_assert!(
            after.evaluate(&q(7, "/data/a", Action::Write)) != Verdict::Allow,
            "inference widened across the action axis"
        );

        // Universally: no probe gained an Allow, and the observed denial lost
        // one it never had.
        for (i, p) in probes().iter().enumerate() {
            let now = after.evaluate(&q(7, p, Action::Read));
            test_assert!(
                now != Verdict::Allow || before[i] == Verdict::Allow,
                "inference widened a grant to a point the base field did not grant"
            );
        }
        test_assert_eq!(
            after.evaluate(&q(7, "/data/secret", Action::Read)),
            Verdict::Deny
        );
        // Every cut is a denial, whatever the trace said.
        test_assert!(
            after
                .sources()
                .iter()
                .filter(|s| s.polarity == Polarity::Grant)
                .count()
                == 3,
            "narrowing added a grant"
        );

        // Sharpening does happen when nothing contradicts it: a denial with no
        // allowed observation above it lifts to the root of the evidence.
        let lone = [Observation {
            subject: 9,
            action: Action::Write,
            path: "/etc/ssl/private/key",
            allowed: false,
        }];
        let m = infer(&lone);
        test_assert_eq!(m.len(), 1);
        test_assert_eq!(m.cuts()[0].anchor.as_str(), "/");
        test_assert_eq!(m.cuts()[0].radius, 4u8);
        let mut wide = PermField::new();
        test_assert_eq!(wide.narrow(&m), 1usize);
        test_assert_eq!(
            wide.evaluate(&q(9, "/etc/ssl/private/key", Action::Write)),
            Verdict::Deny
        );
        // The lift is bounded by the evidence: radius 4 reaches the denial and
        // nothing deeper.
        test_assert_eq!(
            wide.evaluate(&q(9, "/etc/ssl/private/key/sub", Action::Write)),
            Verdict::Unknown
        );
        // One allowed observation anywhere under a candidate ancestor stops the
        // lift there.
        let mixed = [
            Observation {
                subject: 9,
                action: Action::Write,
                path: "/etc/ssl/private/key",
                allowed: false,
            },
            Observation {
                subject: 9,
                action: Action::Write,
                path: "/etc/ssl/certs/ca",
                allowed: true,
            },
        ];
        let k = infer(&mixed);
        test_assert_eq!(k.len(), 1);
        test_assert_eq!(k.cuts()[0].anchor.as_str(), "/etc/ssl/private");
        test_assert_eq!(k.cuts()[0].radius, 1u8);
        TestResult::Pass
    }

    /// The radius bound, tested in both directions and on both enforcement
    /// layers.
    fn test_radius_bounds_influence() -> TestResult {
        let f = vec![src(Some(7), Some(Action::Read), "/data", 2, Polarity::Grant)];
        // At the radius: in. One past it: out. Both directions of the threshold.
        test_assert_eq!(evaluate(&f, &q(7, "/data", Action::Read)), Verdict::Allow);
        test_assert_eq!(evaluate(&f, &q(7, "/data/a", Action::Read)), Verdict::Allow);
        test_assert_eq!(
            evaluate(&f, &q(7, "/data/a/b", Action::Read)),
            Verdict::Allow
        );
        test_assert_eq!(
            evaluate(&f, &q(7, "/data/a/b/c", Action::Read)),
            Verdict::Unknown
        );
        test_assert_eq!(steps_below("/data", "/data/a/b"), Some(2usize));
        test_assert_eq!(steps_below("/data", "/data/a/b/c"), Some(3usize));

        // Layer 1: push refuses an over-wide source outright.
        let mut pf = PermField::new();
        test_assert!(
            !pf.push(src(None, None, "/", MAX_RADIUS + 1, Polarity::Grant)),
            "push accepted a radius above MAX_RADIUS"
        );
        test_assert!(pf.push(src(None, None, "/", MAX_RADIUS, Polarity::Grant)));
        test_assert_eq!(pf.sources().len(), 1usize);

        // Layer 2: a source that never went through push is still clamped at
        // evaluation, so MAX_RADIUS holds on the free-function path.
        let raw = vec![src(Some(7), Some(Action::Read), "/", 255, Polarity::Grant)];
        let deep = "/a/b/c/d/e/f/g/h"; // 8 components: at the clamp.
        let deeper = "/a/b/c/d/e/f/g/h/i"; // 9: past it.
        test_assert_eq!(steps_below("/", deep), Some(MAX_RADIUS as usize));
        test_assert_eq!(evaluate(&raw, &q(7, deep, Action::Read)), Verdict::Allow);
        test_assert!(
            evaluate(&raw, &q(7, deeper, Action::Read)) == Verdict::Unknown,
            "a radius of 255 reached past MAX_RADIUS on the free-function path"
        );

        // Source count is bounded, and the bound refuses rather than evicts.
        let mut full = PermField::new();
        for i in 0..MAX_SOURCES {
            test_assert!(
                full.push(src(Some(i as u32), None, "/data", 0, Polarity::Grant)),
                "push refused inside MAX_SOURCES"
            );
        }
        test_assert_eq!(full.sources().len(), MAX_SOURCES);
        test_assert!(
            !full.push(src(None, None, "/data", 0, Polarity::Deny)),
            "push accepted a source past MAX_SOURCES"
        );
        test_assert_eq!(full.sources().len(), MAX_SOURCES);

        // Anchor length is bounded, and a non-canonical anchor is refused.
        let long = {
            let mut s = String::from("/");
            for _ in 0..MAX_ANCHOR_BYTES {
                s.push('x');
            }
            s
        };
        let mut pf2 = PermField::new();
        test_assert!(
            !pf2.push(src(None, None, &long, 0, Polarity::Grant)),
            "push accepted an anchor past MAX_ANCHOR_BYTES"
        );
        test_assert!(
            !pf2.push(src(None, None, "/data/../root", 0, Polarity::Grant)),
            "push accepted a non-canonical anchor"
        );
        test_assert!(
            !pf2.push(src(None, None, "/data/", 0, Polarity::Grant)),
            "push accepted a trailing slash"
        );
        test_assert_eq!(pf2.sources().len(), 0usize);
        TestResult::Pass
    }

    /// Shaping the support: include one application, omit another, carve one
    /// out. Same path, same action, three different values of the field.
    fn test_support_shaping_by_application() -> TestResult {
        let mut f = PermField::new();
        // Included: uid 7 only.
        test_assert!(f.push(src(
            Some(7),
            Some(Action::Read),
            "/data",
            2,
            Polarity::Grant
        )));
        // Carved out: uid 9, explicitly.
        test_assert!(f.push(src(Some(9), None, "/data", 2, Polarity::Deny)));

        let p = "/data/logs";
        // Included application.
        test_assert_eq!(f.evaluate(&q(7, p, Action::Read)), Verdict::Allow);
        // Omitted application — no source reaches it, so it is uncovered, not
        // permitted.
        test_assert_eq!(f.evaluate(&q(8, p, Action::Read)), Verdict::Unknown);
        test_assert!(!f.evaluate(&q(8, p, Action::Read)).permits());
        // Excluded application.
        test_assert_eq!(f.evaluate(&q(9, p, Action::Read)), Verdict::Deny);
        // The action axis shapes the support the same way.
        test_assert_eq!(f.evaluate(&q(7, p, Action::Write)), Verdict::Unknown);

        // A wildcard subject covers everyone, and a per-subject denial still
        // dominates it.
        let mut g = PermField::new();
        test_assert!(g.push(src(None, Some(Action::Read), "/data", 2, Polarity::Grant)));
        test_assert!(g.push(src(Some(9), None, "/data/logs", 0, Polarity::Deny)));
        test_assert_eq!(g.evaluate(&q(8, p, Action::Read)), Verdict::Allow);
        test_assert_eq!(g.evaluate(&q(9, p, Action::Read)), Verdict::Deny);
        TestResult::Pass
    }

    /// Userspace membership is a value of the same field, with the same
    /// dominance and the same `Unknown`.
    fn test_membership_is_the_same_field() -> TestResult {
        let mut f = PermField::new();
        test_assert!(f.push(src(None, Some(Action::Member), "/home", 4, Polarity::Grant)));
        test_assert!(f.push(src(None, Some(Action::Member), "/tmp", 4, Polarity::Grant)));
        // A kernel-owned subtree inside userspace is carved back out.
        test_assert!(f.push(src(
            None,
            Some(Action::Member),
            "/home/.kernel",
            4,
            Polarity::Deny
        )));

        test_assert_eq!(in_userspace(&f, 7, "/home/agent/work"), Verdict::Allow);
        test_assert_eq!(in_userspace(&f, 7, "/tmp"), Verdict::Allow);
        // Deny dominates inside the grant, exactly as for a file permission.
        test_assert_eq!(in_userspace(&f, 7, "/home/.kernel/state"), Verdict::Deny);
        // Outside every support: uncovered, and not a member.
        test_assert_eq!(in_userspace(&f, 7, "/dev/sda"), Verdict::Unknown);
        test_assert!(!in_userspace(&f, 7, "/dev/sda").permits());
        // Membership does not leak into file permission, or the reverse.
        test_assert_eq!(f.evaluate(&q(7, "/home/agent", Action::Read)), Verdict::Unknown);
        TestResult::Pass
    }

    /// The metric refuses to be walked out of. Non-canonical paths are inside
    /// nothing, siblings share no support, and ancestors are not covered by
    /// their descendants' grants.
    fn test_metric_has_no_escapes() -> TestResult {
        let f = vec![src(None, None, "/data", 8, Polarity::Grant)];

        // `..` traversal out of the anchor.
        test_assert_eq!(steps_below("/data", "/data/../root"), None);
        test_assert_eq!(
            evaluate(&f, &q(7, "/data/../root", Action::Read)),
            Verdict::Unknown
        );
        // Doubled separators.
        test_assert_eq!(steps_below("/data", "//data/x"), None);
        test_assert_eq!(steps_below("/", "//root/secret"), None);
        // Trailing slash and `.`.
        test_assert_eq!(steps_below("/data", "/data/"), None);
        test_assert_eq!(steps_below("/data", "/data/./x"), None);
        // Relative paths are absolute-only.
        test_assert_eq!(steps_below("/data", "data/x"), None);
        test_assert_eq!(steps_below("/data", ""), None);

        // Component boundary: the mac.rs over-matching class.
        test_assert_eq!(steps_below("/data", "/database"), None);
        test_assert_eq!(
            evaluate(&f, &q(7, "/database", Action::Read)),
            Verdict::Unknown
        );
        test_assert_eq!(steps_below("/root", "/rootkit"), None);

        // Ancestors are outside a descendant's support; siblings likewise. A
        // symmetric tree metric would put both in range.
        let g = vec![src(None, None, "/data/x", 8, Polarity::Grant)];
        test_assert_eq!(steps_below("/data/x", "/data"), None);
        test_assert_eq!(evaluate(&g, &q(7, "/data", Action::Read)), Verdict::Unknown);
        test_assert_eq!(steps_below("/data/x", "/data/y"), None);
        test_assert_eq!(
            evaluate(&g, &q(7, "/data/y", Action::Read)),
            Verdict::Unknown
        );

        // The root anchor reaches everything within its radius, and only that.
        test_assert_eq!(steps_below("/", "/"), Some(0usize));
        test_assert_eq!(steps_below("/", "/etc/passwd"), Some(2usize));

        // Inference will not emit a cut for a path it cannot embed.
        let bad = [Observation {
            subject: 7,
            action: Action::Read,
            path: "/data/../root",
            allowed: false,
        }];
        test_assert_eq!(infer(&bad).len(), 0usize);
        TestResult::Pass
    }

    /// Inference reads a bounded prefix of its input and emits a bounded
    /// number of cuts.
    fn test_inference_input_is_bounded() -> TestResult {
        let mut paths: Vec<String> = Vec::new();
        for i in 0..(MAX_OBSERVATIONS + 40) {
            let mut s = String::from("/deny");
            s.push_str(match i % 4 {
                0 => "/a",
                1 => "/b",
                2 => "/c",
                _ => "/d",
            });
            let mut t = s.clone();
            t.push('/');
            t.push((b'a' + (i % 26) as u8) as char);
            paths.push(t);
        }
        let obs: Vec<Observation<'_>> = paths
            .iter()
            .map(|p| Observation {
                subject: 7,
                action: Action::Read,
                path: p.as_str(),
                allowed: false,
            })
            .collect();
        let n = infer(&obs);
        test_assert!(
            n.len() <= MAX_SOURCES,
            "inference emitted more cuts than a field can hold"
        );
        test_assert!(
            n.len() <= MAX_OBSERVATIONS,
            "inference read past MAX_OBSERVATIONS"
        );
        for c in n.cuts() {
            test_assert!(c.radius <= MAX_RADIUS, "an inferred cut exceeded MAX_RADIUS");
        }

        // The input bound is observable, not merely a comment: a trace of
        // exactly MAX_OBSERVATIONS allowed decisions followed by one denial
        // yields no cut, because the denial sits past the prefix `infer` reads.
        // Moving it one place earlier yields the cut.
        let mut tail: Vec<Observation<'_>> = (0..MAX_OBSERVATIONS)
            .map(|_| Observation {
                subject: 7,
                action: Action::Read,
                path: "/data/seen",
                allowed: true,
            })
            .collect();
        tail.push(Observation {
            subject: 7,
            action: Action::Read,
            path: "/data/unseen",
            allowed: false,
        });
        test_assert_eq!(tail.len(), MAX_OBSERVATIONS + 1);
        test_assert!(
            infer(&tail).is_empty(),
            "inference read an observation past MAX_OBSERVATIONS"
        );
        tail.swap(MAX_OBSERVATIONS - 1, MAX_OBSERVATIONS);
        test_assert_eq!(infer(&tail).len(), 1usize);
        // An empty trace is a no-op, not a default.
        let none: [Observation<'_>; 0] = [];
        test_assert!(infer(&none).is_empty());
        let mut f = PermField::new();
        test_assert_eq!(f.narrow(&infer(&none)), 0usize);
        test_assert_eq!(f.evaluate(&q(7, "/data", Action::Read)), Verdict::Unknown);
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test(
            "security::perm_field_deny_dominates",
            test_deny_dominates_every_grant,
        );
        crate::testing::register_test(
            "security::perm_field_totality_without_invention",
            test_totality_without_invention,
        );
        crate::testing::register_test(
            "security::perm_field_inference_only_narrows",
            test_inference_only_narrows,
        );
        crate::testing::register_test(
            "security::perm_field_radius_bounds_influence",
            test_radius_bounds_influence,
        );
        crate::testing::register_test(
            "security::perm_field_support_shaping",
            test_support_shaping_by_application,
        );
        crate::testing::register_test(
            "security::perm_field_membership_same_field",
            test_membership_is_the_same_field,
        );
        crate::testing::register_test(
            "security::perm_field_metric_has_no_escapes",
            test_metric_has_no_escapes,
        );
        crate::testing::register_test(
            "security::perm_field_inference_input_bounded",
            test_inference_input_is_bounded,
        );
    }
}
