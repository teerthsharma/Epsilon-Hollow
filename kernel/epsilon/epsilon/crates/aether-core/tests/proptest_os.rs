// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow
//
// Property-based tests for the O(1) OS / state primitives in `aether-core`.

use aether_core::os::{HardwareTopology, MemoryRegion, MemoryType, PageTableEntry, PhysAddr};
use aether_core::state::SystemState;
use proptest::prelude::*;

proptest! {
    /// Property 1: `set_addr(a)` round-trips through `addr()` after applying ADDR_MASK,
    /// for any u64 input `a`.
    #[test]
    fn prop_pte_set_addr_round_trip(a in any::<u64>(), flags in any::<u64>()) {
        let mut pte = PageTableEntry::new(flags);
        pte.set_addr(a);
        prop_assert_eq!(pte.addr(), a & PageTableEntry::ADDR_MASK);
    }

    /// Property 2: `set_addr` preserves all bits outside ADDR_MASK (i.e. the flags).
    #[test]
    fn prop_pte_set_addr_preserves_flags(a in any::<u64>(), flags in any::<u64>()) {
        let mut pte = PageTableEntry::new(flags);
        let original_flags = flags & !PageTableEntry::ADDR_MASK;
        pte.set_addr(a);
        let surviving_flags = pte.entry & !PageTableEntry::ADDR_MASK;
        prop_assert_eq!(surviving_flags, original_flags);
    }

    /// Property 3: After N `add_region` calls, `memory_map_len == min(N, 32)`.
    #[test]
    fn prop_topology_capacity_bounded(n in 0usize..128) {
        let mut t = HardwareTopology::new();
        let region = MemoryRegion {
            start: PhysAddr(0),
            length: 1,
            region_type: MemoryType::Reserved, // doesn't bump total_memory
        };
        for _ in 0..n {
            t.add_region(region);
        }
        prop_assert_eq!(t.memory_map_len, n.min(32));
    }

    /// Property 4: `SystemState::deviation` is symmetric and non-negative for any
    /// pair of [f64; 4] vectors.
    #[test]
    fn prop_deviation_symmetric_non_negative(
        a in prop::array::uniform4(-1.0e6f64..1.0e6),
        b in prop::array::uniform4(-1.0e6f64..1.0e6),
    ) {
        let s1 = SystemState::<4>::new(a, 0);
        let s2 = SystemState::<4>::new(b, 0);
        let d_ab = s1.deviation(&s2);
        let d_ba = s2.deviation(&s1);
        prop_assert!(d_ab >= 0.0, "deviation must be non-negative, got {}", d_ab);
        prop_assert!(d_ba >= 0.0, "deviation must be non-negative, got {}", d_ba);
        prop_assert!(
            (d_ab - d_ba).abs() < 1e-9,
            "deviation asymmetric: {} vs {}",
            d_ab,
            d_ba
        );
    }
}
