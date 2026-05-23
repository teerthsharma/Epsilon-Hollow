// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Per-CPU data structures and accessors.

use alloc::vec::Vec;
use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use x86_64::registers::model_specific::Msr;
use x86_64::structures::gdt::SegmentSelector;
use x86_64::structures::tss::TaskStateSegment;

use crate::process::context_switch::{TaskContext, KERNEL_STACK_SIZE, detect_xsave, xsave_area_size};
use crate::process::scheduler::ManifoldScheduler;
use crate::process::task::Task;

pub mod smp;

pub const MAX_CPUS: usize = 32;

/// Per-CPU state. Must be page-aligned so it can be pointed to by GS base.
#[repr(C, align(4096))]
pub struct PerCpu {
    pub apic_id: u32,
    pub cpu_num: u32,
    pub current_task: *mut Task,
    pub idle_context: TaskContext,
    pub idle_xsave: Vec<u8>,
    pub kernel_stack: [u8; KERNEL_STACK_SIZE],
    pub double_fault_stack: [u8; 4096],
    pub tss: TaskStateSegment,
    pub scheduler: ManifoldScheduler,
    pub scheduler_lock: spin::Mutex<()>,
    pub ticks: u64,
    pub is_idle: bool,
    pub switching: bool,
    pub pending_reschedule: bool,
    pub ap_ready: bool,
    pub tss_selector: SegmentSelector,
}

// PerCpu contains Vecs (inside ManifoldScheduler) so it is not Copy.
// We back the static array with bytes and use ptr::write for initialization.
#[repr(C, align(4096))]
#[derive(Copy, Clone)]
struct PerCpuStorage {
    bytes: [u8; core::mem::size_of::<PerCpu>()],
}

// SAFETY: Each slot is written exactly once by the BSP before the corresponding
// CPU starts. After that, only that CPU accesses its own slot via gsbase.
struct PerCpuArray(UnsafeCell<[PerCpuStorage; MAX_CPUS]>);
unsafe impl Sync for PerCpuArray {}

static PER_CPU_ARRAY: PerCpuArray = PerCpuArray(UnsafeCell::new(
    [PerCpuStorage { bytes: [0; core::mem::size_of::<PerCpu>()] }; MAX_CPUS]
));

static PER_CPU_INITIALIZED: [AtomicBool; MAX_CPUS] = [
    AtomicBool::new(false), AtomicBool::new(false), AtomicBool::new(false), AtomicBool::new(false),
    AtomicBool::new(false), AtomicBool::new(false), AtomicBool::new(false), AtomicBool::new(false),
    AtomicBool::new(false), AtomicBool::new(false), AtomicBool::new(false), AtomicBool::new(false),
    AtomicBool::new(false), AtomicBool::new(false), AtomicBool::new(false), AtomicBool::new(false),
    AtomicBool::new(false), AtomicBool::new(false), AtomicBool::new(false), AtomicBool::new(false),
    AtomicBool::new(false), AtomicBool::new(false), AtomicBool::new(false), AtomicBool::new(false),
    AtomicBool::new(false), AtomicBool::new(false), AtomicBool::new(false), AtomicBool::new(false),
    AtomicBool::new(false), AtomicBool::new(false), AtomicBool::new(false), AtomicBool::new(false),
];

pub static CPU_COUNT: AtomicUsize = AtomicUsize::new(0);
pub static BSP_CPU_NUM: AtomicUsize = AtomicUsize::new(0);

/// Initialize the BSP's per-CPU data (CPU 0).
///
/// # Safety
/// Must be called exactly once on the BSP, after the heap is initialized.
pub unsafe fn init_bsp() {
    let cpu_num = 0;
    BSP_CPU_NUM.store(cpu_num, Ordering::SeqCst);

    detect_xsave();

    let base = PER_CPU_ARRAY.0.get() as *mut PerCpuStorage;
    let ptr = unsafe { (*base.add(cpu_num)).bytes.as_mut_ptr() as *mut PerCpu };
    ptr.write(PerCpu {
        apic_id: 0,
        cpu_num: cpu_num as u32,
        current_task: core::ptr::null_mut(),
        idle_context: TaskContext::zero(),
        idle_xsave: Vec::new(),
        kernel_stack: [0; KERNEL_STACK_SIZE],
        double_fault_stack: [0; 4096],
        tss: TaskStateSegment::new(),
        scheduler: ManifoldScheduler::new(),
        scheduler_lock: spin::Mutex::new(()),
        ticks: 0,
        is_idle: false,
        switching: false,
        pending_reschedule: false,
        ap_ready: true,
        tss_selector: SegmentSelector(0),
    });

    let per_cpu = &mut *ptr;
    crate::memory::gdt::init_tss_for_cpu(per_cpu);

    // Allocate aligned idle XSAVE area
    let xsave_size = xsave_area_size();
    let idle_xsave = vec![0u8; xsave_size + 64];
    let aligned = ((idle_xsave.as_ptr() as usize) + 63) & !63;
    per_cpu.idle_xsave = idle_xsave;
    per_cpu.idle_context.xsave_ptr = aligned as *mut u8;

    // GS base -> PerCpu
    Msr::new(0xC000_0101).write(ptr as u64);
    PER_CPU_INITIALIZED[cpu_num].store(true, Ordering::SeqCst);
    CPU_COUNT.fetch_add(1, Ordering::SeqCst);
}

/// Allocate and initialize per-CPU data for an AP.
///
/// # Safety
/// Must be called on the BSP before the AP is started.
pub fn alloc_ap_cpu(apic_id: u32, cpu_num: u32) -> &'static mut PerCpu {
    assert!((cpu_num as usize) < MAX_CPUS);
    assert!(cpu_num != 0, "CPU 0 is reserved for BSP");

    unsafe {
        assert!(
            !PER_CPU_INITIALIZED[cpu_num as usize].load(Ordering::SeqCst),
            "AP CPU {} already allocated",
            cpu_num
        );

        let base = PER_CPU_ARRAY.0.get() as *mut PerCpuStorage;
        let ptr = unsafe { (*base.add(cpu_num as usize)).bytes.as_mut_ptr() as *mut PerCpu };
        ptr.write(PerCpu {
            apic_id,
            cpu_num,
            current_task: core::ptr::null_mut(),
            idle_context: TaskContext::zero(),
            idle_xsave: Vec::new(),
            kernel_stack: [0; KERNEL_STACK_SIZE],
            double_fault_stack: [0; 4096],
            tss: TaskStateSegment::new(),
            scheduler: ManifoldScheduler::new(),
            scheduler_lock: spin::Mutex::new(()),
            ticks: 0,
            is_idle: true,
            switching: false,
            pending_reschedule: false,
            ap_ready: false,
            tss_selector: SegmentSelector(0),
        });

        let per_cpu = &mut *ptr;
        crate::memory::gdt::init_tss_for_cpu(per_cpu);

        // Allocate aligned idle XSAVE area
        let xsave_size = xsave_area_size();
        let idle_xsave = vec![0u8; xsave_size + 64];
        let aligned = ((idle_xsave.as_ptr() as usize) + 63) & !63;
        per_cpu.idle_xsave = idle_xsave;
        per_cpu.idle_context.xsave_ptr = aligned as *mut u8;

        PER_CPU_INITIALIZED[cpu_num as usize].store(true, Ordering::SeqCst);
        CPU_COUNT.fetch_add(1, Ordering::SeqCst);

        per_cpu
    }
}

/// Get a mutable reference to the current CPU's PerCpu struct via GS base.
///
/// # Safety
/// GS base must have been initialized to point to a valid PerCpu.
pub unsafe fn this_cpu() -> &'static mut PerCpu {
    let gsbase = Msr::new(0xC000_0101).read();
    &mut *(gsbase as *mut PerCpu)
}

/// Return the logical CPU number of the current processor.
pub fn current_cpu_num() -> u32 {
    unsafe { this_cpu().cpu_num }
}

/// Get a mutable reference to a specific CPU's PerCpu struct.
///
/// # Safety
/// The CPU must have been initialized.
pub unsafe fn per_cpu(cpu_num: u32) -> Option<&'static mut PerCpu> {
    let idx = cpu_num as usize;
    if idx >= MAX_CPUS || !PER_CPU_INITIALIZED[idx].load(Ordering::SeqCst) {
        return None;
    }
    let base = PER_CPU_ARRAY.0.get() as *mut PerCpuStorage;
    Some(&mut *(unsafe { (*base.add(idx)).bytes.as_mut_ptr() as *mut PerCpu }))
}

/// Return the APIC ID for a given logical CPU number, if initialized.
pub fn apic_id_for_cpu(cpu_num: u32) -> Option<u32> {
    unsafe {
        let idx = cpu_num as usize;
        if idx < MAX_CPUS && PER_CPU_INITIALIZED[idx].load(Ordering::SeqCst) {
            let base = PER_CPU_ARRAY.0.get() as *const PerCpuStorage;
            let ptr = unsafe { (*base.add(idx)).bytes.as_ptr() as *const PerCpu };
            Some((*ptr).apic_id)
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use super::*;
    use crate::{test_assert, test_assert_eq};
    use crate::testing::TestResult;

    fn test_smp_basic() -> TestResult {
        test_assert_eq!(MAX_CPUS, 32);
        test_assert_eq!(BSP_CPU_NUM.load(Ordering::SeqCst), 0);
        // CPU_COUNT may be 0 if init_bsp() has not been called in the test
        // environment, but it must never exceed MAX_CPUS.
        test_assert!(CPU_COUNT.load(Ordering::SeqCst) <= MAX_CPUS, "cpu_count <= MAX_CPUS");
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test("smp::smp_basic", test_smp_basic);
    }
}
