// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! SMAP (Supervisor Mode Access Prevention) and SMEP (Supervisor Mode Execution
//! Prevention) enforcement.
//!
//! SMEP prevents the kernel from executing user pages.
//! SMAP prevents the kernel from reading/writing user pages unless explicitly
//! allowed via `stac`/`clac`.

use x86_64::registers::control::{Cr4, Cr4Flags};

/// Upper limit of user address space (non-canonical gap starts here).
const USER_SPACE_LIMIT: u64 = 0x0000_8000_0000_0000;

/// Check whether the CPU supports SMEP/SMAP via CPUID leaf 7.
fn has_smep_smap() -> bool {
    let (smep, smap) = unsafe {
        let leaf7 = core::arch::x86_64::__cpuid_count(7, 0);
        let smep = (leaf7.ebx & (1 << 7)) != 0;
        let smap = (leaf7.ebx & (1 << 20)) != 0;
        (smep, smap)
    };
    smep && smap
}

/// Enable SMEP and SMAP in CR4 if the CPU supports them.
///
/// # Safety
/// Must be called once during early boot.
pub unsafe fn enable_smap_smep() {
    if !has_smep_smap() {
        return;
    }
    let mut cr4 = Cr4::read_raw();
    cr4 |= 1 << 20; // CR4.SMEP
    cr4 |= 1 << 21; // CR4.SMAP
    unsafe {
        Cr4::write_raw(cr4);
    }
}

/// Validate that a pointer range lies entirely in user address space.
fn is_user_ptr(ptr: *const u8, len: usize) -> bool {
    let addr = ptr as u64;
    if addr == 0 || addr >= USER_SPACE_LIMIT {
        return false;
    }
    let end = addr.saturating_add(len as u64);
    end <= USER_SPACE_LIMIT
}

/// Page-fault handler integration stub for SMAP/SMEP violations.
///
/// In a full implementation the #PF handler would inspect the error code and
/// call this when a supervisor access to a user page was attempted while
/// EFLAGS.AC was clear.
pub fn handle_user_access_fault() {
    // TODO: integrate with IDT #PF handler.
    // For now this is a placeholder.
}

/// Safely copy `len` bytes from user space into `kernel_buf`.
///
/// Uses `stac` to temporarily allow access to user pages under SMAP,
/// then `clac` to restore protection.
///
/// # Safety
/// Caller must ensure `user_ptr` is valid for read. This function only checks
/// that the pointer resides in the user address space.
pub unsafe fn copy_from_user(kernel_buf: &mut [u8], user_ptr: *const u8) -> Result<(), ()> {
    if !is_user_ptr(user_ptr, kernel_buf.len()) {
        return Err(());
    }
    unsafe {
        // stac — set AC flag, allowing supervisor access to user pages
        core::arch::asm!(".byte 0x0f, 0x01, 0xcb", options(nomem, nostack));
        core::ptr::copy_nonoverlapping(
            user_ptr,
            kernel_buf.as_mut_ptr(),
            kernel_buf.len(),
        );
        // clac — clear AC flag, re-enabling SMAP protection
        core::arch::asm!(".byte 0x0f, 0x01, 0xca", options(nomem, nostack));
    }
    Ok(())
}

/// Safely copy `len` bytes from `kernel_buf` to user space.
///
/// Uses `stac`/`clac` around the copy.
///
/// # Safety
/// Caller must ensure `user_ptr` is valid for write.
pub unsafe fn copy_to_user(user_ptr: *mut u8, kernel_buf: &[u8]) -> Result<(), ()> {
    if !is_user_ptr(user_ptr, kernel_buf.len()) {
        return Err(());
    }
    unsafe {
        core::arch::asm!(".byte 0x0f, 0x01, 0xcb", options(nomem, nostack));
        core::ptr::copy_nonoverlapping(
            kernel_buf.as_ptr(),
            user_ptr,
            kernel_buf.len(),
        );
        core::arch::asm!(".byte 0x0f, 0x01, 0xca", options(nomem, nostack));
    }
    Ok(())
}

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use super::*;
    use crate::testing::TestResult;
    use crate::test_assert;

    fn test_user_ptr_validation() -> TestResult {
        test_assert!(!is_user_ptr(core::ptr::null(), 1));
        test_assert!(!is_user_ptr(0x0000_8000_0000_0000 as *const u8, 1));
        test_assert!(is_user_ptr(0x1000 as *const u8, 1));
        test_assert!(!is_user_ptr(0x7fff_ffff_ffff as *const u8, 2)); // crosses boundary
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test("security::smap_smep_user_ptr", test_user_ptr_validation);
    }
}
