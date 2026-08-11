// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow

//! ═══════════════════════════════════════════════════════════════════════════════
//! AEGIS Data Loaders
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Efficient data loading with batching and shuffling.
//!
//! ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use super::rng::Lcg;
use super::tensor::Tensor;

/// Odd mixing constant (golden-ratio derived) that decorrelates per-epoch seeds.
const EPOCH_MIX: u64 = 0x9E37_79B9_7F4A_7C15;

/// Data Loader
#[derive(Debug, Clone)]
pub struct DataLoader {
    pub features: Vec<Tensor>,
    pub targets: Vec<Tensor>,
    pub batch_size: usize,
    pub shuffle: bool,
    /// Base seed for the shuffle. Combined with the epoch index in
    /// [`DataLoader::epoch_order`], so each epoch draws a different permutation.
    pub seed: u64,
    /// Discard the trailing partial batch. Required when the consuming kernel
    /// is compiled for a fixed batch width.
    pub drop_last: bool,
}

impl DataLoader {
    /// Create new DataLoader.
    ///
    /// Panics if the feature and target counts differ, or if `batch_size` is 0 —
    /// a zero-width batch cannot advance the iterator.
    pub fn new(
        features: Vec<Tensor>,
        targets: Vec<Tensor>,
        batch_size: usize,
        shuffle: bool,
    ) -> Self {
        assert_eq!(
            features.len(),
            targets.len(),
            "Features and targets must have same length"
        );
        assert!(batch_size > 0, "batch_size must be non-zero");
        Self {
            features,
            targets,
            batch_size,
            shuffle,
            seed: 42,
            drop_last: false,
        }
    }

    /// Convert raw slices to DataLoader. Enforces the same invariants as [`Self::new`].
    pub fn from_slice(x: &[Tensor], y: &[Tensor], batch_size: usize, shuffle: bool) -> Self {
        Self::new(x.to_vec(), y.to_vec(), batch_size, shuffle)
    }

    /// Override the shuffle seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set whether the trailing partial batch is discarded.
    pub fn with_drop_last(mut self, drop_last: bool) -> Self {
        self.drop_last = drop_last;
        self
    }

    /// Number of samples.
    pub fn len(&self) -> usize {
        self.features.len()
    }

    /// Whether the loader holds no samples.
    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }

    /// Sample order for `epoch`. Identity when `shuffle` is off, otherwise a
    /// fresh permutation per epoch, reproducible for a given `(seed, epoch)`.
    pub fn epoch_order(&self, epoch: u64) -> Vec<usize> {
        if !self.shuffle {
            return (0..self.len()).collect();
        }
        Lcg::new(self.seed ^ epoch.wrapping_mul(EPOCH_MIX)).permutation(self.len())
    }

    /// Iterate over the first epoch's batches.
    pub fn iter(&self) -> BatchIterator<'_> {
        self.iter_epoch(0)
    }

    /// Iterate over the batches of a specific epoch.
    pub fn iter_epoch(&self, epoch: u64) -> BatchIterator<'_> {
        BatchIterator {
            loader: self,
            indices: self.epoch_order(epoch),
            current_idx: 0,
        }
    }

    /// Split into `(first, second)` by sample count, `first` receiving
    /// `round(len * fraction)` samples. The train/held-out primitive.
    ///
    /// Panics unless `fraction` lies in `[0, 1]`.
    pub fn split(&self, fraction: f64) -> (Self, Self) {
        assert!(
            (0.0..=1.0).contains(&fraction),
            "split fraction must lie in [0, 1], got {fraction}"
        );
        let cut = libm::round(self.len() as f64 * fraction) as usize;
        let build = |range: core::ops::Range<usize>| Self {
            features: self.features[range.clone()].to_vec(),
            targets: self.targets[range].to_vec(),
            ..self.clone()
        };
        (build(0..cut), build(cut..self.len()))
    }
}

/// Iterator over batches
pub struct BatchIterator<'a> {
    loader: &'a DataLoader,
    indices: Vec<usize>,
    current_idx: usize,
}

impl Iterator for BatchIterator<'_> {
    type Item = (Vec<Tensor>, Vec<Tensor>);

    fn next(&mut self) -> Option<Self::Item> {
        let start = self.current_idx;
        let end = (start + self.loader.batch_size).min(self.loader.features.len());
        // Empty range ends the epoch; a short tail ends it too under drop_last.
        if start >= end || (self.loader.drop_last && end - start < self.loader.batch_size) {
            return None;
        }
        self.current_idx = end;

        let mut batch_x = Vec::with_capacity(end - start);
        let mut batch_y = Vec::with_capacity(end - start);

        for i in start..end {
            let idx = self.indices[i];
            batch_x.push(self.loader.features[idx].clone());
            batch_y.push(self.loader.targets[idx].clone());
        }

        Some((batch_x, batch_y))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataloader() {
        let x = ones(5);
        let y = x.clone();

        let loader = DataLoader::new(x, y, 2, false);
        let mut iter = loader.iter();

        let batch1 = iter.next();
        assert!(batch1.is_some());
        assert_eq!(batch1.unwrap().0.len(), 2);

        let batch2 = iter.next();
        assert!(batch2.is_some());
        assert_eq!(batch2.unwrap().0.len(), 2);

        let batch3 = iter.next(); // Last batch of 1
        assert!(batch3.is_some());
        assert_eq!(batch3.unwrap().0.len(), 1);

        assert!(iter.next().is_none());
    }

    fn ones(n: usize) -> Vec<Tensor> {
        (0..n).map(|_| Tensor::zeros(&[1])).collect()
    }

    /// `new` asserts the lengths match; `from_slice` did not, so a short target
    /// list reached `BatchIterator::next` and indexed out of bounds.
    #[test]
    #[should_panic(expected = "same length")]
    fn from_slice_rejects_mismatched_lengths() {
        let _ = DataLoader::from_slice(&ones(4), &ones(3), 2, false);
    }

    /// `batch_size == 0` left `current_idx` unchanged, so the iterator emitted
    /// empty batches forever instead of terminating.
    #[test]
    #[should_panic(expected = "batch_size")]
    fn zero_batch_size_is_rejected_at_construction() {
        let _ = DataLoader::new(ones(4), ones(4), 0, false);
    }

    /// The seed was the literal 42, so every epoch replayed one permutation and
    /// shuffling bought nothing across a training run.
    #[test]
    fn distinct_epochs_give_distinct_orders() {
        let loader = DataLoader::new(ones(32), ones(32), 4, true);
        let e0 = loader.epoch_order(0);
        let e1 = loader.epoch_order(1);
        assert_ne!(e0, e1, "epoch 0 and 1 must not share a permutation");

        // Still a permutation, and still reproducible for a given epoch.
        let mut sorted = e0.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..32).collect::<Vec<_>>());
        assert_eq!(
            e0,
            loader.epoch_order(0),
            "epoch order must be reproducible"
        );

        // A different seed must move the whole schedule.
        let other = DataLoader::new(ones(32), ones(32), 4, true).with_seed(7);
        assert_ne!(other.epoch_order(0), e0, "seed must change the schedule");
    }

    /// Fixed-shape kernels need every batch the same width; `drop_last` discards
    /// the short tail rather than emitting a ragged final batch.
    #[test]
    fn drop_last_discards_the_short_tail() {
        let ragged = DataLoader::new(ones(5), ones(5), 2, false);
        assert_eq!(
            ragged.iter().map(|(x, _)| x.len()).collect::<Vec<_>>(),
            [2, 2, 1]
        );

        let even = ragged.clone().with_drop_last(true);
        assert_eq!(
            even.iter().map(|(x, _)| x.len()).collect::<Vec<_>>(),
            [2, 2]
        );

        // A dataset shorter than one batch yields nothing rather than a partial.
        let tiny = DataLoader::new(ones(1), ones(1), 4, false).with_drop_last(true);
        assert_eq!(tiny.iter().count(), 0);
    }

    /// Overfitting detection needs a held-out split; the loader had no way to
    /// produce one, so callers had to slice the tensors by hand.
    #[test]
    fn split_partitions_without_overlap() {
        let loader = DataLoader::new(ones(10), ones(10), 2, false);
        let (train, val) = loader.split(0.7);
        assert_eq!(train.len(), 7);
        assert_eq!(val.len(), 3);
        assert_eq!(train.batch_size, loader.batch_size);

        // Boundary fractions keep every sample somewhere.
        let (all, none) = loader.split(1.0);
        assert_eq!((all.len(), none.len()), (10, 0));
        let (none2, all2) = loader.split(0.0);
        assert_eq!((none2.len(), all2.len()), (0, 10));
    }

    #[test]
    #[should_panic(expected = "fraction")]
    fn split_rejects_a_fraction_outside_the_unit_interval() {
        let _ = DataLoader::new(ones(4), ones(4), 2, false).split(1.5);
    }
}
