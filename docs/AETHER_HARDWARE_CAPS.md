# Aether Hardware Capability Model

Aether-Lang is the native language layer of Seal OS. That does not mean every
Aether script gets arbitrary hardware power.

Kernel Rust owns hardware discovery, MMIO mapping, PCI BAR ownership, ACPI
tables, interrupt routing, and driver lifetimes. Aether-Lang can ask for
hardware actions only through Seal capability objects minted by kernel Rust.

## Rules

1. Raw integer addresses from script space are not hardware authority.
2. MMIO reads and writes require a capability tied to a kernel-approved range.
3. Capabilities are width-checked, range-checked, and driver-owned.
4. Untrusted `.aether` scripts cannot access arbitrary physical memory.
5. A driver may expose a narrow Aether API for a device, but the driver remains
   responsible for synchronization and safety.

## Current Implementation

The kernel bridge keeps low-level callback names for the current Aether-Lang API,
but `kernel/seal-os/src/lang/mod.rs` denies MMIO by default. A raw address gains
authority only when kernel Rust registers a driver-owned range with
`register_aether_mmio_range(base, len)`.

Closure tests still needed:

- raw `0xffff_ffff` MMIO read/write is rejected
- a fake whitelisted BAR is accepted in test mode
- out-of-range offset is rejected
- wrong width is rejected

Until those tests land, Aether hardware access is improved but still tracked as
a safety boundary under construction.
