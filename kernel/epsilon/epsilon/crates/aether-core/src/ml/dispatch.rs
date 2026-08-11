// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow

//! HFT-signal-priority dispatcher + LASSO kernel selector.
//!
//! Inference op comes in with a feature vector (shape, dtype-code, batch, ...).
//! LASSO-fit sparse weights per candidate kernel score each; argmax picks.
//! Chosen op enters an alpha-priority queue; the dispatcher pops highest
//! (alpha - staleness_decay * age_ticks) next. HFT playbook, OS-side.

extern crate alloc;
use alloc::collections::BinaryHeap;
use alloc::vec::Vec;
use core::cmp::Ordering;

/// LASSO-fit sparse weights for one kernel variant. Zeros already dropped.
#[derive(Clone, Debug)]
pub struct KernelScore {
    pub kernel_id: u16,
    /// (feature_index, weight) — sparse, LASSO output already thresholded.
    pub nz: Vec<(u16, f32)>,
    pub bias: f32,
}

impl KernelScore {
    /// Sparse dot product. features indexed by position.
    pub fn score(&self, features: &[f32]) -> f32 {
        let mut s = self.bias;
        for &(i, w) in &self.nz {
            if let Some(&x) = features.get(i as usize) {
                s += w * x;
            }
        }
        s
    }
}

/// Pick the highest-scoring kernel for these op features. None if empty.
pub fn select_kernel(candidates: &[KernelScore], features: &[f32]) -> Option<u16> {
    candidates
        .iter()
        .map(|k| (k.kernel_id, k.score(features)))
        .fold(None, |acc, (id, s)| match acc {
            None => Some((id, s)),
            Some((_, bs)) if s > bs => Some((id, s)),
            other => other,
        })
        .map(|(id, _)| id)
}

/// One queued inference request. Higher alpha runs first; alpha decays with age.
#[derive(Clone, Debug)]
pub struct Signal<P> {
    pub alpha: f32,
    pub enqueued_tick: u64,
    pub kernel_id: u16,
    pub payload: P,
}

impl<P> Signal<P> {
    /// ponytail: linear decay, swap for exp if staleness curve matters.
    pub fn effective(&self, now_tick: u64, decay_per_tick: f32) -> f32 {
        let age = now_tick.saturating_sub(self.enqueued_tick) as f32;
        self.alpha - decay_per_tick * age
    }
}

/// Heap entry: (effective_alpha at enqueue, seq for tiebreak, signal).
/// Effective alpha is recomputed on pop-check, but enqueue value seeds ordering.
struct Entry<P> {
    key: OrdF32,
    seq: u64,
    sig: Signal<P>,
}
impl<P> PartialEq for Entry<P> {
    fn eq(&self, o: &Self) -> bool {
        self.key == o.key && self.seq == o.seq
    }
}
impl<P> Eq for Entry<P> {}
impl<P> PartialOrd for Entry<P> {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl<P> Ord for Entry<P> {
    fn cmp(&self, o: &Self) -> Ordering {
        // higher key first; older seq wins ties (FIFO within same alpha)
        self.key.cmp(&o.key).then_with(|| o.seq.cmp(&self.seq))
    }
}

/// Total-order f32 wrapper (NaN sinks to bottom). Kernel-safe, no std.
#[derive(Copy, Clone, PartialEq)]
struct OrdF32(f32);
impl Eq for OrdF32 {}
impl PartialOrd for OrdF32 {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for OrdF32 {
    fn cmp(&self, o: &Self) -> Ordering {
        self.0
            .partial_cmp(&o.0)
            .unwrap_or_else(|| match (self.0.is_nan(), o.0.is_nan()) {
                (true, true) => Ordering::Equal,
                (true, false) => Ordering::Less,
                (false, true) => Ordering::Greater,
                _ => Ordering::Equal,
            })
    }
}

/// Alpha-priority dispatcher. ponytail: global heap, shard per-core when contention shows.
pub struct Dispatcher<P> {
    heap: BinaryHeap<Entry<P>>,
    seq: u64,
    pub decay_per_tick: f32,
}

impl<P> Dispatcher<P> {
    pub fn new(decay_per_tick: f32) -> Self {
        Self {
            heap: BinaryHeap::new(),
            seq: 0,
            decay_per_tick,
        }
    }

    pub fn enqueue(&mut self, sig: Signal<P>) {
        self.seq += 1;
        let key = OrdF32(sig.effective(sig.enqueued_tick, self.decay_per_tick));
        self.heap.push(Entry {
            key,
            seq: self.seq,
            sig,
        });
    }

    /// Pop the currently-highest effective-alpha signal at `now_tick`.
    /// Rechecks staleness so heavily-aged entries don't beat fresh ones.
    pub fn pop(&mut self, now_tick: u64) -> Option<Signal<P>> {
        // Re-key every entry against `now`. Small heaps → cheap.
        // ponytail: O(n) rebuild per pop; use a decay-aware indexed heap when queue grows.
        let drained: Vec<_> = self.heap.drain().collect();
        let mut best: Option<Entry<P>> = None;
        for mut e in drained {
            e.key = OrdF32(e.sig.effective(now_tick, self.decay_per_tick));
            match &best {
                Some(b) if b.key >= e.key => self.heap.push(e),
                _ => {
                    if let Some(prev) = best.take() {
                        self.heap.push(prev);
                    }
                    best = Some(e);
                }
            }
        }
        best.map(|e| e.sig)
    }

    pub fn len(&self) -> usize {
        self.heap.len()
    }
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }
}

#[cfg(test)]
mod check {
    use super::*;

    #[test]
    fn lasso_pick_and_dispatch() {
        // Two kernel variants: 0 favors small batch, 1 favors large batch.
        // features = [batch_size_normalized, dtype_is_f16]
        let cands = [
            KernelScore {
                kernel_id: 0,
                nz: alloc::vec![(0, -1.0), (1, 0.2)],
                bias: 0.5,
            },
            KernelScore {
                kernel_id: 1,
                nz: alloc::vec![(0, 1.0)],
                bias: 0.0,
            },
        ];
        assert_eq!(select_kernel(&cands, &[0.1, 1.0]), Some(0));
        assert_eq!(select_kernel(&cands, &[0.9, 0.0]), Some(1));
        assert_eq!(select_kernel(&[], &[0.0]), None);

        // Dispatch: high alpha wins fresh; stale high loses to fresh medium.
        let mut d = Dispatcher::new(0.1);
        d.enqueue(Signal {
            alpha: 10.0,
            enqueued_tick: 0,
            kernel_id: 0,
            payload: "stale-high",
        });
        d.enqueue(Signal {
            alpha: 5.0,
            enqueued_tick: 90,
            kernel_id: 1,
            payload: "fresh-mid",
        });
        // at t=100: stale-high eff = 10 - 0.1*100 = 0.0; fresh-mid eff = 5 - 0.1*10 = 4.0
        assert_eq!(d.pop(100).unwrap().payload, "fresh-mid");
        assert_eq!(d.pop(100).unwrap().payload, "stale-high");
        assert!(d.is_empty());
    }
}
