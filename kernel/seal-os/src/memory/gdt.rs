// Seal OS -- Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Global Descriptor Table (GDT) setup for x86_64 with per-CPU TSS support.

use core::arch::asm;
use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicU16, Ordering};

use x86_64::instructions::segmentation::{CS, Segment};
use x86_64::instructions::tables::{lgdt, load_tss as ltr_instruction};
use x86_64::structures::gdt::SegmentSelector;
use x86_64::structures::tss::TaskStateSegment;
use x86_64::structures::DescriptorTablePointer;
use x86_64::VirtAddr;

use crate::cpu::PerCpu;
use crate::process::context_switch::KERNEL_STACK_SIZE;

// SAFETY: GDT is written exactly once during `init_gdt()` on the BSP before
// any other CPU accesses it. After that it is read-only except for TSS slots
// added by `init_tss_for_cpu()`, which runs sequentially on the BSP.
struct GdtArray(UnsafeCell<[u64; 128]>);
unsafe impl Sync for GdtArray {}

static GDT: GdtArray = GdtArray(UnsafeCell::new([0; 128]));

// SAFETY: TSS is written only via `set_kernel_stack()` which is called during
// context switches after the TSS has been initialized.
struct TssCell(UnsafeCell<TaskStateSegment>);
unsafe impl Sync for TssCell {}

static TSS: TssCell = TssCell(UnsafeCell::new(TaskStateSegment::new()));

pub static KERNEL_CODE_SELECTOR: AtomicU16 = AtomicU16::new(0);
pub static KERNEL_DATA_SELECTOR: AtomicU16 = AtomicU16::new(0);
pub static USER_CODE32_SELECTOR: AtomicU16 = AtomicU16::new(0);
pub static USER_DATA_SELECTOR: AtomicU16 = AtomicU16::new(0);
pub static USER_CODE_SELECTOR: AtomicU16 = AtomicU16::new(0);
pub static TSS_SELECTOR: AtomicU16 = AtomicU16::new(0);

static NEXT_GDT_SLOT: AtomicU16 = AtomicU16::new(6);

/// Initialise the GDT and reload segment registers.
pub fn init_gdt() {
    unsafe {
        let gdt = GDT.0.get();

        // Null
        (*gdt)[0] = 0;
        // Kernel code (64-bit, ring 0)
        (*gdt)[1] = 0x00209A0000000000;
        // Kernel data (ring 0)
        (*gdt)[2] = 0x0000920000000000;
        // User code 32-bit compat (ring 3)
        (*gdt)[3] = 0x00CF9A0000000000;
        // User data (ring 3)
        (*gdt)[4] = 0x00CF920000000000;
        // User code 64-bit (ring 3)
        (*gdt)[5] = 0x0020FA0000000000;

        KERNEL_CODE_SELECTOR.store(0x08, Ordering::SeqCst);
        KERNEL_DATA_SELECTOR.store(0x10, Ordering::SeqCst);
        USER_CODE32_SELECTOR.store(0x18 | 3, Ordering::SeqCst);
        USER_DATA_SELECTOR.store(0x20 | 3, Ordering::SeqCst);
        USER_CODE_SELECTOR.store(0x28 | 3, Ordering::SeqCst);
        TSS_SELECTOR.store(0x30, Ordering::SeqCst);

        let ptr = DescriptorTablePointer {
            limit: (core::mem::size_of::<[u64; 128]>() - 1) as u16,
            base: VirtAddr::new(gdt as u64),
        };
        lgdt(&ptr);

        CS::set_reg(SegmentSelector(KERNEL_CODE_SELECTOR.load(Ordering::SeqCst)));
        load_data_segments(SegmentSelector(KERNEL_DATA_SELECTOR.load(Ordering::SeqCst)));
    }
}

/// Set the kernel stack pointer used when entering ring 0 from ring 3.
pub unsafe fn set_kernel_stack(stack_top: u64) {
    (*TSS.0.get()).privilege_stack_table[0] = VirtAddr::new(stack_top);
}

/// Initialise the TSS for a CPU and allocate a GDT descriptor for it.
///
/// # Safety
/// Must be called exactly once per CPU before that CPU loads the TSS.
pub unsafe fn init_tss_for_cpu(per_cpu: &mut PerCpu) {
    let slot = NEXT_GDT_SLOT.fetch_add(2, Ordering::SeqCst) as usize;

    let tss_ptr = &per_cpu.tss as *const TaskStateSegment as u64;
    let limit = (core::mem::size_of::<TaskStateSegment>() - 1) as u64;

    // TSS descriptor (low 8 bytes)
    let base_low = tss_ptr & 0xFFFFFF;
    let base_mid = (tss_ptr >> 24) & 0xFF;
    let low = limit
        | (base_low << 16)
        | (0x89u64 << 40) // present, type = available 64-bit TSS
        | ((limit & 0xF0000) << 32)
        | (base_mid << 56);

    // TSS descriptor (high 8 bytes)
    let high = tss_ptr >> 32;

    let gdt = GDT.0.get();
    (*gdt)[slot] = low;
    (*gdt)[slot + 1] = high;

    // Stack pointers
    let stack_top = per_cpu.kernel_stack.as_ptr() as u64 + KERNEL_STACK_SIZE as u64;
    per_cpu.tss.privilege_stack_table[0] = VirtAddr::new(stack_top);

    // IST[0] for double-fault handler
    per_cpu.tss.interrupt_stack_table[0] = VirtAddr::new(stack_top);

    per_cpu.tss_selector = SegmentSelector((slot as u16) << 3);
}

/// Load a TSS selector via `ltr`.
///
/// # Safety
/// Selector must point to a valid, available TSS descriptor.
pub unsafe fn load_tss(selector: SegmentSelector) {
    ltr_instruction(selector);
}

unsafe fn load_data_segments(selector: SegmentSelector) {
    asm!(
        "mov ax, {sel:x}",
        "mov ds, ax",
        "mov es, ax",
        "mov fs, ax",
        "mov gs, ax",
        "mov ss, ax",
        sel = in(reg) selector.0,
        out("ax") _,
        options(preserves_flags, nomem),
    );
}
