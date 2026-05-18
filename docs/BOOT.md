# Boot Sequence Technical Reference

> Sources: `kernel/seal-os/src/boot/boot.S`, `kernel/seal-os/src/boot/multiboot2.rs`, `kernel/seal-os/linker.ld`, `kernel/seal-os/build.rs`

---

## Complete Boot Flow

```
BIOS/UEFI
  -> GRUB (Multiboot2 compliant)
    -> _boot (boot.S, 32-bit protected mode)
      -> Set up page tables (identity map 4GB)
      -> Enable PAE (CR4.PAE)
      -> Enable Long Mode (IA32_EFER.LME)
      -> Enable Paging (CR0.PG + CR0.PE)
      -> Load 64-bit GDT
      -> Far jump to long_mode_entry (CS=0x08)
        -> Set segment registers (DS=ES=SS=FS=GS = 0x10)
        -> Set 64-bit stack
        -> call _start(multiboot_info_ptr)
          -> Rust kernel initialization
```

---

## Multiboot2 Header Layout

Source: `kernel/seal-os/src/boot/multiboot2.rs` lines 6-38

The header is a 48-byte `[u32; 12]` array placed in `.multiboot_header` section:

| Offset | Field | Value | Notes |
|--------|-------|-------|-------|
| 0 | Magic | `0xE85250D6` | Multiboot2 magic number (line 6) |
| 4 | Architecture | `0x00000000` | i386 protected mode (line 7) |
| 8 | Header Length | `48` | Total header size in bytes (line 10) |
| 12 | Checksum | `0u32 - (magic + arch + len)` | Two's complement (line 11-12) |
| 16 | FB Tag Type | `5` | Framebuffer request tag type (line 24) |
| 20 | FB Tag Size | `20` | Tag size in bytes (line 26) |
| 24 | Width | `1024` | Requested framebuffer width (line 28) |
| 28 | Height | `768` | Requested framebuffer height (line 30) |
| 32 | Depth | `32` | Bits per pixel (line 32) |
| 36 | Padding | `0` | 8-byte alignment for next tag (line 34) |
| 40 | End Tag Type | `0` | End tag (line 36) |
| 44 | End Tag Size | `8` | Minimum tag size (line 37) |

The header is linked into `.multiboot_header` via `#[link_section = ".multiboot_header"]` (line 15). GRUB scans the first 32KB of the kernel binary for this magic.

---

## Page Table Setup

Source: `kernel/seal-os/src/boot/boot.S` lines 19-55

### Structure: PML4 -> PDPT -> PD0-PD3

Six 4KB-aligned pages are allocated in `.bss` (lines 109-115):

| Page | Purpose | Size |
|------|---------|------|
| `pml4` | Page Map Level 4 | 4096 bytes |
| `pdpt` | Page Directory Pointer Table | 4096 bytes |
| `pd0` | Page Directory 0 (0-1GB) | 4096 bytes |
| `pd1` | Page Directory 1 (1-2GB) | 4096 bytes |
| `pd2` | Page Directory 2 (2-3GB) | 4096 bytes |
| `pd3` | Page Directory 3 (3-4GB) | 4096 bytes |

### Zeroing (lines 21-23)

All 6 pages (6 * 4096 / 4 = 6144 dwords) are zeroed via `rep stosl`.

### Wiring (lines 26-55)

1. **PML4[0] -> PDPT**: `pml4[0] = &pdpt | 0x3` (present + writable) (lines 27-30)
2. **PDPT[0] -> PD0**: `pdpt[0] = &pd0 | 0x3` (line 35-36)
3. **PDPT[1] -> PD1**: `pdpt[8] = &pd1 | 0x3` (line 37-39)
4. **PDPT[2] -> PD2**: `pdpt[16] = &pd2 | 0x3` (line 40-42)
5. **PDPT[3] -> PD3**: `pdpt[24] = &pd3 | 0x3` (line 43-45)
6. **PD entries**: 4 * 512 = 2048 entries, each maps a 2MB huge page (lines 48-55)
   - Flags: `0x83` = present (bit 0) + writable (bit 1) + huge/PS (bit 7)
   - Each entry increments physical address by `0x200000` (2MB)

### Result

4GB identity map using 2MB huge pages. Physical address == virtual address for the entire first 4GB. This covers both kernel memory and MMIO regions (framebuffer at ~0xFD000000).

---

## GDT Entries

Source: `kernel/seal-os/src/boot/boot.S` lines 124-130

The 64-bit GDT contains three entries in `.rodata`:

| Index | Quad Value | Description |
|-------|-----------|-------------|
| 0 | `0x0000000000000000` | Null descriptor (required) (line 125) |
| 1 | `0x00AF9A000000FFFF` | 64-bit code segment: base=0, limit=max, execute/read, long mode (L=1) (line 126) |
| 2 | `0x00CF92000000FFFF` | 64-bit data segment: base=0, limit=max, read/write (line 127) |

The GDT pointer (lines 128-129):
- Limit: `gdt64_ptr - gdt64 - 1` (23 bytes = 3 entries * 8 - 1)
- Base: address of `gdt64`

The far jump `ljmp $0x08, $long_mode_entry` (line 79) uses selector 0x08 (entry 1 = code segment). Segment registers are loaded with 0x10 (entry 2 = data segment) at lines 84-89.

---

## Mode Transitions

### Real -> Protected (handled by GRUB)

GRUB enters the kernel at `_boot` already in 32-bit protected mode. EAX contains the Multiboot2 magic `0x36d76289`, EBX contains the physical address of the multiboot2 info structure (lines 4-5).

### Protected -> Long Mode (boot.S lines 57-79)

1. **Enable PAE**: `CR4.PAE` (bit 5) set (lines 58-59)
2. **Load PML4 into CR3**: `CR3 = &pml4` (lines 63-64)
3. **Enable Long Mode**: `IA32_EFER` MSR (`0xC0000080`), set `LME` bit 8 (lines 67-69)
4. **Enable Paging**: `CR0.PG` (bit 31) + `CR0.PE` (bit 0) set (lines 72-73)
5. **Load GDT**: `lgdt gdt64_ptr` (line 78)
6. **Far Jump**: `ljmp $0x08, $long_mode_entry` -- switches CS to 64-bit code segment (line 79)

After the far jump, the processor is in 64-bit long mode. The `long_mode_entry` label (line 82) runs `.code64` instructions.

---

## Linker Script Sections

Source: `kernel/seal-os/linker.ld`

Entry point: `_boot` (line 2). Kernel loaded at virtual/physical address 1MB (line 5).

| Section | Alignment | Contents |
|---------|-----------|----------|
| `.multiboot_header` | 8 bytes | Multiboot2 header (KEEP) -- must be in first 32KB (line 7) |
| `.multiboot_boot` | 4KB | 32-bit boot trampoline (boot.S `.code32` code) (line 11) |
| `.text` | 4KB | All code sections (line 15) |
| `.rodata` | 4KB | Read-only data (GDT, constants) (line 19) |
| `.data` | 4KB | Initialized data (line 23) |
| `.bss` | 4KB | Uninitialized data (page tables, stack, heap) (line 27) |
| `/DISCARD/` | -- | `.eh_frame`, `.eh_frame_hdr`, `.note.GNU-stack`, `.comment` (lines 32-37) |

Section ordering ensures the multiboot header appears within the first 32KB of the binary (required by GRUB).

---

## build.rs Linker Flags

Source: `kernel/seal-os/build.rs` lines 1-9

| Flag | Purpose |
|------|---------|
| `-T{manifest_dir}/linker.ld` | Use custom linker script (absolute path) (line 4) |
| `-no-pie` | Disable position-independent executable (kernel runs at fixed 1MB) (line 5) |
| `-z notext` | Allow `R_X86_64_32` relocations from 32-bit boot trampoline (boot.S `.code32` section uses 32-bit absolute addresses) (lines 7-8) |

The `-z notext` flag is critical: the `.code32` trampoline in boot.S uses 32-bit absolute addresses (`lea pml4, %edi` etc.) which generate `R_X86_64_32` relocations. Without this flag, the linker rejects them as text relocations.

---

## Boot Stack

Source: `boot.S` lines 117-120

A 64KB boot stack is allocated in `.bss`:
- `boot_stack_bottom`: base address (line 118)
- `.skip 65536`: 64KB of stack space (line 119)
- `boot_stack_top`: stack pointer target (grows downward) (line 120)

The 32-bit code sets `ESP = boot_stack_top` (line 17). After entering long mode, `RSP = boot_stack_top` (line 92). The multiboot info pointer is saved in ESI (line 14), then passed as RDI (first argument) to `_start` (line 95).
