// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow

//! Persistent Homology KV-cache Partitioning (PHKP) in `no_std` Rust.
//!
//! This replaces the legacy `epsilon_core/persistent_kv_partition.py` theorem
//! slice with fixed-capacity Rust. It keeps Betti-0 clusters intact while
//! greedily packing larger clusters into faster tiers.

/// H100 HBM3 access latency in nanoseconds.
pub const HBM3_LATENCY_NS: f64 = 100.0;
/// Host DDR5 access latency in nanoseconds.
pub const DDR5_LATENCY_NS: f64 = 300.0;
/// NVMe access latency in nanoseconds.
pub const NVME_LATENCY_NS: f64 = 50_000.0;
/// H100 HBM3 bandwidth in GB/s.
pub const HBM3_BANDWIDTH_GBS: f64 = 3350.0;
/// Host DDR5 bandwidth in GB/s.
pub const DDR5_BANDWIDTH_GBS: f64 = 204.8;
/// NVMe bandwidth in GB/s.
pub const NVME_BANDWIDTH_GBS: f64 = 7.0;
/// Default H100 HBM3 capacity in GB.
pub const HBM3_CAPACITY_GB: f64 = 80.0;
/// Default host DDR5 capacity in GB.
pub const DDR5_CAPACITY_GB: f64 = 512.0;
/// Default NVMe capacity in GB.
pub const NVME_CAPACITY_GB: f64 = 8000.0;
const BYTES_PER_GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// Memory tier used for KV-cache placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryTier {
    /// H100 HBM3.
    Hbm3,
    /// Host DDR5.
    Ddr5,
    /// NVMe storage.
    Nvme,
}

impl MemoryTier {
    /// Tier access latency in nanoseconds.
    pub const fn latency_ns(self) -> f64 {
        match self {
            Self::Hbm3 => HBM3_LATENCY_NS,
            Self::Ddr5 => DDR5_LATENCY_NS,
            Self::Nvme => NVME_LATENCY_NS,
        }
    }

    /// Tier bandwidth in GB/s.
    pub const fn bandwidth_gbs(self) -> f64 {
        match self {
            Self::Hbm3 => HBM3_BANDWIDTH_GBS,
            Self::Ddr5 => DDR5_BANDWIDTH_GBS,
            Self::Nvme => NVME_BANDWIDTH_GBS,
        }
    }
}

/// Compute KV-cache size in GiB.
pub fn kv_cache_size_gb(
    seq_len: u64,
    n_layers: u32,
    head_dim: u32,
    n_kv_heads: u32,
    dtype_bytes: u32,
    batch_size: u32,
) -> f64 {
    let total_bytes = seq_len as f64
        * 2.0
        * f64::from(head_dim)
        * f64::from(n_kv_heads)
        * f64::from(n_layers)
        * f64::from(dtype_bytes)
        * f64::from(batch_size);
    total_bytes / BYTES_PER_GIB
}

/// Compute topological locality `L(pi)` for a partition and cluster-id stream.
///
/// `0` means each cluster lives entirely in one tier. A positive value means
/// one or more Betti-0 clusters were split across tiers.
pub fn topological_locality(partition: &[MemoryTier], cluster_ids: &[usize]) -> usize {
    let len = partition.len().min(cluster_ids.len());
    let mut locality = 0_usize;

    for idx in 0..len {
        let cid = cluster_ids[idx];
        let mut first_seen = true;
        if cluster_ids[..idx].iter().any(|&c| c == cid) {
            first_seen = false;
        }
        if !first_seen {
            continue;
        }

        let mut tiers = [false; 3];
        for scan in 0..len {
            if cluster_ids[scan] == cid {
                tiers[tier_index(partition[scan])] = true;
            }
        }
        let tier_count = tiers.iter().filter(|used| **used).count();
        locality += tier_count.saturating_sub(1);
    }

    locality
}

/// Greedy Betti-guided KV-cache partitioner.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BettiGuidedPartitioner {
    hbm_capacity_gb: f64,
    ddr_capacity_gb: f64,
}

impl Default for BettiGuidedPartitioner {
    fn default() -> Self {
        Self::new(HBM3_CAPACITY_GB, DDR5_CAPACITY_GB)
    }
}

impl BettiGuidedPartitioner {
    /// Create a partitioner with explicit HBM and DDR capacities in GiB.
    pub const fn new(hbm_capacity_gb: f64, ddr_capacity_gb: f64) -> Self {
        Self {
            hbm_capacity_gb,
            ddr_capacity_gb,
        }
    }

    /// HBM capacity in GiB.
    pub const fn hbm_capacity_gb(&self) -> f64 {
        self.hbm_capacity_gb
    }

    /// DDR capacity in GiB.
    pub const fn ddr_capacity_gb(&self) -> f64 {
        self.ddr_capacity_gb
    }

    /// Assign each Betti-0 cluster to one memory tier without splitting it.
    pub fn partition<const N: usize>(
        &self,
        cluster_sizes: &[u64; N],
        bytes_per_key: u64,
    ) -> PartitionReport<N> {
        let mut order = [0_usize; N];
        for (idx, slot) in order.iter_mut().enumerate() {
            *slot = idx;
        }
        sort_indices_by_size_desc(&mut order, cluster_sizes);

        let mut assignments = [MemoryTier::Nvme; N];
        let mut hbm_used_gb = 0.0;
        let mut ddr_used_gb = 0.0;
        let mut n_hbm = 0_u64;
        let mut n_ddr = 0_u64;
        let mut n_nvme = 0_u64;

        for &cluster_idx in &order {
            let size = cluster_sizes[cluster_idx];
            let size_gb = size as f64 * bytes_per_key as f64 / BYTES_PER_GIB;
            if hbm_used_gb + size_gb <= self.hbm_capacity_gb {
                assignments[cluster_idx] = MemoryTier::Hbm3;
                hbm_used_gb += size_gb;
                n_hbm += size;
            } else if ddr_used_gb + size_gb <= self.ddr_capacity_gb {
                assignments[cluster_idx] = MemoryTier::Ddr5;
                ddr_used_gb += size_gb;
                n_ddr += size;
            } else {
                assignments[cluster_idx] = MemoryTier::Nvme;
                n_nvme += size;
            }
        }

        let total_keys = cluster_sizes.iter().copied().sum::<u64>();
        let denom = total_keys.max(1) as f64;
        let latency_betti_ns = HBM3_LATENCY_NS * n_hbm as f64 / denom
            + DDR5_LATENCY_NS * n_ddr as f64 / denom
            + NVME_LATENCY_NS * n_nvme as f64 / denom;

        let total_capacity = self.hbm_capacity_gb + self.ddr_capacity_gb + NVME_CAPACITY_GB;
        let hbm_frac = self.hbm_capacity_gb / total_capacity;
        let ddr_frac = self.ddr_capacity_gb / total_capacity;
        let nvme_frac = 1.0 - hbm_frac - ddr_frac;
        let latency_random_ns =
            HBM3_LATENCY_NS * hbm_frac + DDR5_LATENCY_NS * ddr_frac + NVME_LATENCY_NS * nvme_frac;

        let mut cluster_ids = [0_usize; N];
        for (idx, slot) in cluster_ids.iter_mut().enumerate() {
            *slot = idx;
        }
        let locality = topological_locality(&assignments, &cluster_ids);
        let p_hbm = assignments
            .iter()
            .filter(|tier| matches!(tier, MemoryTier::Hbm3))
            .count();

        PartitionReport {
            assignments,
            n_hbm,
            n_ddr,
            n_nvme,
            hbm_used_gb,
            ddr_used_gb,
            p_total: N,
            p_hbm,
            latency_betti_ns,
            latency_random_ns,
            speedup: latency_random_ns / latency_betti_ns.max(0.001),
            topological_locality: locality,
            perfect_locality: locality == 0,
        }
    }

    /// Sparse-attention latency bound after pruning fraction `sparsity`.
    pub fn sparse_latency(&self, base_latency_ns: f64, sparsity: f64) -> f64 {
        (1.0 - sparsity.clamp(0.0, 1.0)) * base_latency_ns
    }
}

/// PHKP partition report for fixed-size cluster arrays.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PartitionReport<const N: usize> {
    /// Tier assignment for each cluster.
    pub assignments: [MemoryTier; N],
    /// Keys placed in HBM3.
    pub n_hbm: u64,
    /// Keys placed in DDR5.
    pub n_ddr: u64,
    /// Keys placed in NVMe.
    pub n_nvme: u64,
    /// HBM3 capacity consumed in GiB.
    pub hbm_used_gb: f64,
    /// DDR5 capacity consumed in GiB.
    pub ddr_used_gb: f64,
    /// Total cluster count.
    pub p_total: usize,
    /// Number of clusters placed in HBM3.
    pub p_hbm: usize,
    /// Expected Betti-guided latency in nanoseconds.
    pub latency_betti_ns: f64,
    /// Expected random-placement latency in nanoseconds.
    pub latency_random_ns: f64,
    /// Random latency divided by Betti-guided latency.
    pub speedup: f64,
    /// Locality penalty. Zero means no cluster is split across tiers.
    pub topological_locality: usize,
    /// True when `topological_locality == 0`.
    pub perfect_locality: bool,
}

fn tier_index(tier: MemoryTier) -> usize {
    match tier {
        MemoryTier::Hbm3 => 0,
        MemoryTier::Ddr5 => 1,
        MemoryTier::Nvme => 2,
    }
}

fn sort_indices_by_size_desc<const N: usize>(order: &mut [usize; N], sizes: &[u64; N]) {
    for i in 0..N {
        let mut best = i;
        for j in (i + 1)..N {
            if sizes[order[j]] > sizes[order[best]] {
                best = j;
            }
        }
        if best != i {
            order.swap(i, best);
        }
    }
}
