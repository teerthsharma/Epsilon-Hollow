// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Aether-Lang runtime bridge — connects SealShell to the AEGIS interpreter.
//! Registers kernel callbacks so Aether-Lang `fs`, `process`, `net`,
//! `graphics`, `window`, `ui`, and `input` modules call real OS functions.

pub mod stdlib;

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::slice;
use spin::Mutex;

pub struct AetherRuntime {
    interpreter: aether_lang::Interpreter,
}

impl AetherRuntime {
    pub fn new() -> Self {
        unsafe {
            aether_lang::register_kernel_callbacks(
                aether_lang::FsCallbacks {
                    read: Some(aether_fs_read),
                    write: Some(aether_fs_write),
                    exists: Some(aether_fs_exists),
                    mkdir: Some(aether_fs_mkdir),
                },
                aether_lang::ProcessCallbacks {
                    pid: Some(aether_process_pid),
                    exit: Some(aether_process_exit),
                },
                aether_lang::NetCallbacks {
                    local_ip: Some(aether_net_local_ip),
                    has_nic: Some(aether_net_has_nic),
                },
                aether_lang::HwCallbacks {
                    mmio_read32: Some(aether_mmio_read32),
                    mmio_write32: Some(aether_mmio_write32),
                    pci_read_config: Some(aether_pci_read_config),
                },
            );
        }
        let mut interpreter = aether_lang::Interpreter::new();
        interpreter.gfx_callbacks = aether_lang::GraphicsCallbacks {
            clear: Some(aether_gfx_clear),
            fill_rect: Some(aether_gfx_fill_rect),
            fill_rounded_rect: Some(aether_gfx_fill_rounded_rect),
            draw_text: Some(aether_gfx_draw_text),
            draw_line: Some(aether_gfx_draw_line),
            draw_circle: Some(aether_gfx_draw_circle),
            set_pixel: Some(aether_gfx_set_pixel),
            gradient_v: Some(aether_gfx_gradient_v),
            glow_rect: Some(aether_gfx_glow_rect),
        };
        interpreter.win_callbacks = aether_lang::WindowCallbacks {
            create: Some(aether_win_create),
            close: Some(aether_win_close),
            set_title: Some(aether_win_set_title),
            focus: Some(aether_win_focus),
            width: Some(aether_win_width),
            height: Some(aether_win_height),
            set_dirty: Some(aether_win_set_dirty),
        };
        interpreter.input_callbacks = aether_lang::InputCallbacks {
            mouse_x: Some(aether_input_mouse_x),
            mouse_y: Some(aether_input_mouse_y),
            mouse_pressed: Some(aether_input_mouse_pressed),
            key_pressed: Some(aether_input_key_pressed),
        };
        Self { interpreter }
    }

    pub fn execute_source(&mut self, source: &str) -> Result<String, String> {
        let mut parser = aether_lang::Parser::new(source);
        let program = parser
            .parse()
            .map_err(|e| format!("parse error at {}:{}: {}", e.line, e.column, e.message))?;
        let value = self
            .interpreter
            .execute(&program)
            .map_err(|e| format!("{:?}", e))?;
        Ok(format!("{:?}", value))
    }

    pub fn execute_file(&mut self, name: &str, source: &str) -> Result<String, String> {
        self.execute_source(source)
            .map_err(|e| format!("in '{}': {}", name, e))
    }

    pub fn is_ready(&self) -> bool {
        true
    }

    pub fn set_current_window(&mut self, id: u32) {
        self.interpreter.current_window_id = id;
    }

    pub fn set_mouse_state(&mut self, x: i32, y: i32, pressed: bool, clicked: bool) {
        unsafe {
            aether_lang::AETHER_INPUT_STATE.mouse_x = x;
            aether_lang::AETHER_INPUT_STATE.mouse_y = y;
            aether_lang::AETHER_INPUT_STATE.mouse_pressed = pressed;
            aether_lang::AETHER_INPUT_STATE.mouse_clicked = clicked;
        }
    }

    pub fn set_key(&mut self, key: &str) {
        let bytes = key.as_bytes();
        let len = bytes.len().min(8);
        let mut last_key = [0u8; 8];
        last_key[..len].copy_from_slice(&bytes[..len]);
        unsafe {
            aether_lang::AETHER_INPUT_STATE.last_key = last_key;
        }
    }

    pub fn call_render(&mut self) -> Result<String, String> {
        let value = self
            .interpreter
            .call_function("render")
            .map_err(|e| format!("{:?}", e))?;
        Ok(format!("{:?}", value))
    }
}

impl Default for AetherRuntime {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Global compositor pointer for window callbacks
// ═══════════════════════════════════════════════════════════════════════════════

static AETHER_COMPOSITOR: core::sync::atomic::AtomicPtr<crate::wm::compositor::Compositor> =
    core::sync::atomic::AtomicPtr::new(core::ptr::null_mut());

#[derive(Clone, Copy)]
struct AetherMmioRange {
    base: u64,
    len: u64,
}

static AETHER_MMIO_RANGES: Mutex<Vec<AetherMmioRange>> = Mutex::new(Vec::new());

/// Set the global compositor pointer for window operations.
/// # Safety
/// Must be called once after the compositor is created, with a pointer that
/// remains valid for the lifetime of the desktop session.
pub unsafe fn set_compositor_ptr(ptr: *mut crate::wm::compositor::Compositor) {
    AETHER_COMPOSITOR.store(ptr, core::sync::atomic::Ordering::SeqCst);
}

/// Mint a hardware capability for Aether-Lang MMIO callbacks.
///
/// Driver code should call this only for a BAR or device window it owns. Aether
/// scripts pass raw offsets today, but those raw integers gain authority only
/// when they fall inside one of these kernel-approved ranges.
pub fn register_aether_mmio_range(base: u64, len: u64) -> bool {
    if len == 0 || base.checked_add(len).is_none() {
        return false;
    }
    AETHER_MMIO_RANGES
        .lock()
        .push(AetherMmioRange { base, len });
    true
}

fn aether_mmio_allowed(addr: u64, width: u64) -> bool {
    if width == 0 || addr % width != 0 {
        return false;
    }
    let end = match addr.checked_add(width) {
        Some(end) => end,
        None => return false,
    };
    AETHER_MMIO_RANGES.lock().iter().any(|range| {
        let range_end = match range.base.checked_add(range.len) {
            Some(end) => end,
            None => return false,
        };
        addr >= range.base && end <= range_end
    })
}

// ═══════════════════════════════════════════════════════════════════════════════
// Kernel callbacks — called by Aether-Lang interpreter via function pointers.
// ═══════════════════════════════════════════════════════════════════════════════

/// Ceiling on any single string an `.aether` script hands across the FFI
/// boundary (fs path, window title, on-screen text). Matches POSIX
/// `PATH_MAX` — the largest string this boundary legitimately needs to
/// accept is a filesystem path; titles and draw-text strings are always far
/// shorter in practice, so one ceiling covers every caller without a second
/// invented number.
const AETHER_STR_MAX: usize = 4096;

/// Convert an interpreter-supplied `(ptr, len)` into a validated `&str`.
///
/// Every `extern "C"` callback that receives a string from the Aether-Lang
/// interpreter must route through here before touching the bytes: `ptr` may
/// be null or wild, `len` may be unbounded, and the bytes are not guaranteed
/// to be valid UTF-8. Returns `None` — a defined refusal — instead of ever
/// producing a `&str` that would violate `from_utf8_unchecked`'s safety
/// contract.
///
/// # Safety
/// `ptr` must be valid for reads of `len` bytes when non-null (the FFI
/// caller's contract, same as `slice::from_raw_parts`). This function only
/// guarantees it will not read past a null pointer, past `AETHER_STR_MAX`
/// bytes, or interpret non-UTF-8 bytes as `&str`.
unsafe fn aether_str<'a>(ptr: *const u8, len: usize) -> Option<&'a str> {
    if ptr.is_null() || len > AETHER_STR_MAX {
        return None;
    }
    core::str::from_utf8(slice::from_raw_parts(ptr, len)).ok()
}

extern "C" fn aether_fs_read(
    path: *const u8,
    path_len: usize,
    buf: *mut u8,
    buf_len: usize,
) -> isize {
    let Some(path) = (unsafe { aether_str(path, path_len) }) else {
        return -1;
    };
    crate::fs::vfs::with_vfs(|vfs| {
        let handle = match vfs.lookup(path) {
            Ok(h) => h,
            Err(_) => return -1,
        };
        let mut tmp = vec![0u8; buf_len];
        match vfs.read(handle, &mut tmp, 0) {
            Ok(n) => {
                let n = n.min(buf_len);
                unsafe {
                    core::ptr::copy_nonoverlapping(tmp.as_ptr(), buf, n);
                }
                n as isize
            }
            Err(_) => -1,
        }
    })
}

extern "C" fn aether_fs_write(
    path: *const u8,
    path_len: usize,
    data: *const u8,
    data_len: usize,
) -> isize {
    let Some(path) = (unsafe { aether_str(path, path_len) }) else {
        return -1;
    };
    crate::fs::vfs::with_vfs(|vfs| {
        let handle = match vfs.lookup(path) {
            Ok(h) => h,
            Err(_) => return -1,
        };
        let buf = unsafe { slice::from_raw_parts(data, data_len) };
        match vfs.write(handle, buf, 0) {
            Ok(n) => n as isize,
            Err(_) => -1,
        }
    })
}

extern "C" fn aether_fs_exists(path: *const u8, path_len: usize) -> bool {
    let Some(path) = (unsafe { aether_str(path, path_len) }) else {
        return false;
    };
    crate::fs::vfs::with_vfs(|vfs| vfs.lookup(path).is_ok())
}

extern "C" fn aether_fs_mkdir(path: *const u8, path_len: usize) -> isize {
    let Some(path) = (unsafe { aether_str(path, path_len) }) else {
        return -1;
    };
    crate::fs::vfs::with_vfs(|vfs| match vfs.mkdir(path) {
        Ok(_) => 0,
        Err(_) => -1,
    })
}

extern "C" fn aether_process_pid() -> u64 {
    crate::process::scheduler::current_task_id()
}

extern "C" fn aether_process_exit(_code: i32) -> ! {
    loop {
        core::hint::spin_loop();
    }
}

extern "C" fn aether_net_local_ip(buf: *mut u8, buf_len: usize) -> isize {
    let ip = crate::net::local_ip();
    let s = format!("{}.{}.{}.{}", ip[0], ip[1], ip[2], ip[3]);
    let bytes = s.as_bytes();
    let n = bytes.len().min(buf_len);
    unsafe {
        core::ptr::copy_nonoverlapping(bytes.as_ptr(), buf, n);
    }
    n as isize
}

extern "C" fn aether_net_has_nic() -> bool {
    crate::drivers::net::has_nic()
}

extern "C" fn aether_mmio_read32(addr: u64) -> u32 {
    if !aether_mmio_allowed(addr, 4) {
        return 0;
    }
    // SAFETY: The address is 4-byte aligned and contained in a kernel-minted
    // MMIO range registered by the driver that owns the device window.
    unsafe { core::ptr::read_volatile(addr as *const u32) }
}

extern "C" fn aether_mmio_write32(addr: u64, val: u32) {
    if !aether_mmio_allowed(addr, 4) {
        return;
    }
    // SAFETY: The address is 4-byte aligned and contained in a kernel-minted
    // MMIO range registered by the driver that owns the device window.
    unsafe { core::ptr::write_volatile(addr as *mut u32, val) }
}

extern "C" fn aether_pci_read_config(bus: u8, slot: u8, func: u8, offset: u8) -> u32 {
    crate::drivers::pci::pci_read32(bus, slot, func, offset)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Graphics callbacks — delegate to htek.rs primitives
// ═══════════════════════════════════════════════════════════════════════════════

extern "C" fn aether_gfx_clear(win_id: u32, color: u32) {
    unsafe {
        let comp = AETHER_COMPOSITOR.load(core::sync::atomic::Ordering::SeqCst);
        if comp.is_null() {
            return;
        }
        if let Some(win) = (*comp).window_mut(win_id) {
            crate::graphics::htek::fill_solid(
                win,
                0,
                0,
                win.client_width(),
                win.client_height(),
                color,
            );
        }
    }
}

extern "C" fn aether_gfx_fill_rect(win_id: u32, x: u32, y: u32, w: u32, h: u32, color: u32) {
    unsafe {
        let comp = AETHER_COMPOSITOR.load(core::sync::atomic::Ordering::SeqCst);
        if comp.is_null() {
            return;
        }
        if let Some(win) = (*comp).window_mut(win_id) {
            crate::graphics::htek::fill_solid(win, x, y, w, h, color);
        }
    }
}

extern "C" fn aether_gfx_fill_rounded_rect(
    win_id: u32,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    r: u32,
    color: u32,
) {
    unsafe {
        let comp = AETHER_COMPOSITOR.load(core::sync::atomic::Ordering::SeqCst);
        if comp.is_null() {
            return;
        }
        if let Some(win) = (*comp).window_mut(win_id) {
            crate::graphics::htek::fill_rounded_rect(win, x, y, w, h, r, color);
        }
    }
}

extern "C" fn aether_gfx_draw_text(
    win_id: u32,
    x: u32,
    y: u32,
    text: *const u8,
    text_len: usize,
    color: u32,
) {
    let Some(text) = (unsafe { aether_str(text, text_len) }) else {
        return;
    };
    unsafe {
        let comp = AETHER_COMPOSITOR.load(core::sync::atomic::Ordering::SeqCst);
        if comp.is_null() {
            return;
        }
        if let Some(win) = (*comp).window_mut(win_id) {
            crate::graphics::htek::render_text_small(win, x, y, text, color);
        }
    }
}

extern "C" fn aether_gfx_draw_line(win_id: u32, x0: i32, y0: i32, x1: i32, y1: i32, color: u32) {
    unsafe {
        let comp = AETHER_COMPOSITOR.load(core::sync::atomic::Ordering::SeqCst);
        if comp.is_null() {
            return;
        }
        if let Some(win) = (*comp).window_mut(win_id) {
            crate::graphics::htek::draw_aa_line(win, x0, y0, x1, y1, color, 255);
        }
    }
}

extern "C" fn aether_gfx_draw_circle(win_id: u32, cx: u32, cy: u32, r: u32, color: u32) {
    unsafe {
        let comp = AETHER_COMPOSITOR.load(core::sync::atomic::Ordering::SeqCst);
        if comp.is_null() {
            return;
        }
        if let Some(win) = (*comp).window_mut(win_id) {
            crate::graphics::htek::draw_circle_filled(win, cx, cy, r, color);
        }
    }
}

extern "C" fn aether_gfx_set_pixel(win_id: u32, x: u32, y: u32, color: u32) {
    unsafe {
        let comp = AETHER_COMPOSITOR.load(core::sync::atomic::Ordering::SeqCst);
        if comp.is_null() {
            return;
        }
        if let Some(win) = (*comp).window_mut(win_id) {
            win.set_client_pixel(x, y, color);
        }
    }
}

extern "C" fn aether_gfx_gradient_v(
    win_id: u32,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    top: u32,
    bottom: u32,
) {
    unsafe {
        let comp = AETHER_COMPOSITOR.load(core::sync::atomic::Ordering::SeqCst);
        if comp.is_null() {
            return;
        }
        if let Some(win) = (*comp).window_mut(win_id) {
            crate::graphics::htek::fill_gradient_v(win, x, y, w, h, top, bottom);
        }
    }
}

extern "C" fn aether_gfx_glow_rect(
    win_id: u32,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    spread: u32,
    color: u32,
) {
    unsafe {
        let comp = AETHER_COMPOSITOR.load(core::sync::atomic::Ordering::SeqCst);
        if comp.is_null() {
            return;
        }
        if let Some(win) = (*comp).window_mut(win_id) {
            crate::graphics::htek::glow_rect(win, x, y, w, h, spread, color);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Window callbacks — delegate to compositor
// ═══════════════════════════════════════════════════════════════════════════════

extern "C" fn aether_win_create(
    title: *const u8,
    title_len: usize,
    width: u32,
    height: u32,
) -> u32 {
    let Some(title) = (unsafe { aether_str(title, title_len) }) else {
        return 0;
    };
    unsafe {
        let comp = AETHER_COMPOSITOR.load(core::sync::atomic::Ordering::SeqCst);
        if comp.is_null() {
            return 0;
        }
        (*comp).create_window(title, 100, 100, width, height)
    }
}

extern "C" fn aether_win_close(id: u32) {
    unsafe {
        let comp = AETHER_COMPOSITOR.load(core::sync::atomic::Ordering::SeqCst);
        if comp.is_null() {
            return;
        }
        if let Some(win) = (*comp).window_mut(id) {
            use crate::wm::window::WindowState;
            win.state = WindowState::Closed;
        }
    }
}

extern "C" fn aether_win_set_title(id: u32, title: *const u8, title_len: usize) {
    let Some(title) = (unsafe { aether_str(title, title_len) }) else {
        return;
    };
    unsafe {
        let comp = AETHER_COMPOSITOR.load(core::sync::atomic::Ordering::SeqCst);
        if comp.is_null() {
            return;
        }
        if let Some(win) = (*comp).window_mut(id) {
            win.title = String::from(title);
            win.render_decorations();
        }
    }
}

extern "C" fn aether_win_focus(id: u32) {
    unsafe {
        let comp = AETHER_COMPOSITOR.load(core::sync::atomic::Ordering::SeqCst);
        if comp.is_null() {
            return;
        }
        (*comp).focus_window(id);
    }
}

extern "C" fn aether_win_width(id: u32) -> u32 {
    unsafe {
        let comp = AETHER_COMPOSITOR.load(core::sync::atomic::Ordering::SeqCst);
        if comp.is_null() {
            return 0;
        }
        if let Some(win) = (*comp).window_mut(id) {
            win.client_width()
        } else {
            0
        }
    }
}

extern "C" fn aether_win_height(id: u32) -> u32 {
    unsafe {
        let comp = AETHER_COMPOSITOR.load(core::sync::atomic::Ordering::SeqCst);
        if comp.is_null() {
            return 0;
        }
        if let Some(win) = (*comp).window_mut(id) {
            win.client_height()
        } else {
            0
        }
    }
}

extern "C" fn aether_win_set_dirty(id: u32) {
    unsafe {
        let comp = AETHER_COMPOSITOR.load(core::sync::atomic::Ordering::SeqCst);
        if comp.is_null() {
            return;
        }
        if let Some(win) = (*comp).window_mut(id) {
            win.dirty = true;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Input callbacks — read from global input state
// ═══════════════════════════════════════════════════════════════════════════════

extern "C" fn aether_input_mouse_x() -> i32 {
    unsafe { aether_lang::AETHER_INPUT_STATE.mouse_x }
}

extern "C" fn aether_input_mouse_y() -> i32 {
    unsafe { aether_lang::AETHER_INPUT_STATE.mouse_y }
}

extern "C" fn aether_input_mouse_pressed() -> bool {
    unsafe { aether_lang::AETHER_INPUT_STATE.mouse_pressed }
}

extern "C" fn aether_input_key_pressed(buf: *mut u8, buf_len: usize) -> isize {
    unsafe {
        let key = aether_lang::AETHER_INPUT_STATE.last_key;
        let n = key
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(key.len())
            .min(buf_len);
        core::ptr::copy_nonoverlapping(key.as_ptr(), buf, n);
        n as isize
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "test-mode")]
pub mod tests {
    use super::*;
    use crate::testing::TestResult;
    use crate::{test_assert, test_assert_eq};

    /// RED: every `extern "C"` fs/gfx/window callback in this file used to
    /// call `core::str::from_utf8_unchecked(slice::from_raw_parts(ptr, len))`
    /// directly on interpreter-supplied `(ptr, len)` pairs — no null check,
    /// no length cap, no UTF-8 validation, despite `from_utf8_unchecked`'s
    /// safety contract requiring valid UTF-8. A null pointer dereferences
    /// immediately; invalid UTF-8 produces a `&str` that violates its own
    /// invariant (undefined behaviour). `aether_str` is the guard every one
    /// of those call sites now routes through — this test proves a null
    /// pointer yields a defined `None`, never a dereference.
    fn test_aether_str_rejects_null_pointer() -> TestResult {
        let result = unsafe { aether_str(core::ptr::null(), 4) };
        test_assert!(result.is_none(), "null pointer must be refused, not read");
        // Zero length makes the null pointer's dereference moot for
        // `slice::from_raw_parts`, so it is covered separately too: a null
        // pointer must be refused regardless of len.
        let result_zero_len = unsafe { aether_str(core::ptr::null(), 0) };
        test_assert!(result_zero_len.is_none());
        TestResult::Pass
    }

    /// RED: `path_len`/`text_len`/`title_len` were trusted verbatim from the
    /// interpreter with no upper bound, so a corrupted or malicious length
    /// value drove `slice::from_raw_parts` to read arbitrarily far past
    /// whatever the pointer actually backs. `AETHER_STR_MAX` (POSIX
    /// `PATH_MAX`) caps it.
    fn test_aether_str_rejects_over_length() -> TestResult {
        let data = vec![b'a'; AETHER_STR_MAX + 1];
        let result = unsafe { aether_str(data.as_ptr(), data.len()) };
        test_assert!(
            result.is_none(),
            "length beyond AETHER_STR_MAX must be refused"
        );
        TestResult::Pass
    }

    /// RED: `from_utf8_unchecked` is documented UB on non-UTF-8 input, and
    /// nothing in this file validated the bytes first — a script could pass
    /// arbitrary bytes (e.g. a lone continuation byte) as a "path" and the
    /// resulting `&str` would already violate its own invariant before any
    /// kernel code inspected it.
    fn test_aether_str_rejects_invalid_utf8() -> TestResult {
        let bad = [0xFFu8, 0xFE, 0x00];
        let result = unsafe { aether_str(bad.as_ptr(), bad.len()) };
        test_assert!(result.is_none(), "invalid UTF-8 must be refused");
        TestResult::Pass
    }

    fn test_aether_str_accepts_valid_input() -> TestResult {
        let text = "/seal/etc/config.toml";
        let result = unsafe { aether_str(text.as_ptr(), text.len()) };
        test_assert_eq!(result, Some(text));
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test(
            "lang::aether_str_rejects_null_pointer",
            test_aether_str_rejects_null_pointer,
        );
        crate::testing::register_test(
            "lang::aether_str_rejects_over_length",
            test_aether_str_rejects_over_length,
        );
        crate::testing::register_test(
            "lang::aether_str_rejects_invalid_utf8",
            test_aether_str_rejects_invalid_utf8,
        );
        crate::testing::register_test(
            "lang::aether_str_accepts_valid_input",
            test_aether_str_accepts_valid_input,
        );
    }
}
