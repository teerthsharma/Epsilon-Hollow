// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! UEFI entry point — pure Rust, zero assembly.
//! UEFI firmware loads this as a PE/COFF binary in 64-bit long mode.
//! We query GOP for framebuffer, then exit boot services and run the kernel.

use uefi::boot::MemoryType;
use uefi::mem::memory_map::MemoryMap;
use uefi::prelude::*;
use uefi::proto::console::gop::{GraphicsOutput, PixelFormat};
use uefi::proto::loaded_image::LoadedImage;
use uefi::table::cfg::ACPI2_GUID;

use super::boot_info::{BootInfo, MemoryDescriptor, MAX_MEMORY_DESCRIPTORS};
#[cfg(not(test))]
use crate::graphics::framebuffer::Framebuffer;
use crate::serial_println;

/// Bytes per pixel that `Framebuffer` can render for a GOP pixel format, or
/// `None` if the format is unrenderable and the mode must be refused.
///
/// `Framebuffer::put_pixel` stores a whole `u32` per pixel, so only the two
/// formats UEFI defines as 32-bit are usable:
///
/// - `Rgb` and `Bgr` are 24-bit colour plus a reserved byte, always 4 bytes.
/// - `Bitmask` carries explicit channel masks and may be any width; RGB565 is
///   the common case. Its scanline stride is `stride * 2`, so the `stride * 4`
///   pitch below would be double the truth and every write from the middle of
///   the frame onward would land past the end of the framebuffer.
/// - `BltOnly` has no CPU-visible linear buffer at all. It is refused here
///   rather than downstream: `GraphicsOutput::frame_buffer` asserts on it, so
///   reaching that call with a `BltOnly` mode panics during boot instead of
///   returning the null address the caller checks for.
const fn gop_format_bytes_per_pixel(format: PixelFormat) -> Option<u32> {
    match format {
        PixelFormat::Rgb | PixelFormat::Bgr => Some(4),
        PixelFormat::Bitmask | PixelFormat::BltOnly => None,
    }
}

// The UEFI build evaluates these; the crate has no host test target.
const _: () = assert!(matches!(
    gop_format_bytes_per_pixel(PixelFormat::Rgb),
    Some(4)
));
const _: () = assert!(matches!(
    gop_format_bytes_per_pixel(PixelFormat::Bgr),
    Some(4)
));
const _: () = assert!(gop_format_bytes_per_pixel(PixelFormat::Bitmask).is_none());
const _: () = assert!(gop_format_bytes_per_pixel(PixelFormat::BltOnly).is_none());

const GOP_MIN_WIDTH: usize = 640;
const GOP_MIN_HEIGHT: usize = 480;
const GOP_MAX_BOOT_WIDTH: usize = 1024;
const GOP_MAX_BOOT_HEIGHT: usize = 768;

pub fn run() -> Status {
    // Initialize serial port FIRST — before any fallible UEFI operation.
    // If anything panics below, the panic handler can use serial output.
    crate::drivers::serial::init();
    serial_println!("Seal OS — UEFI entry reached, serial online");

    // Initialize UEFI helpers (logger, allocator)
    if let Err(e) = uefi::helpers::init() {
        serial_println!("[WARN] UEFI helpers init failed: {:?}", e.status());
    }

    // Search UEFI config tables for ACPI 2.0 RSDP
    let acpi_rsdp = uefi::system::with_config_table(|tables| {
        let mut rsdp = 0u64;
        for table in tables {
            if table.guid == ACPI2_GUID {
                rsdp = table.address as u64;
                serial_println!("[BOOT] ACPI 2.0 RSDP found @ {:#X}", rsdp);
                break;
            }
        }
        rsdp
    });

    // Query GOP for framebuffer; fall back to Bochs VBE dispi on VirtualBox.
    let boot_info = get_framebuffer();

    // Get kernel image base and size before exiting boot services
    let image_handle = uefi::boot::image_handle();
    let (kernel_base, kernel_size) =
        match uefi::boot::open_protocol_exclusive::<LoadedImage>(image_handle) {
            Ok(img) => {
                let (base, size) = img.info();
                (base as u64, size)
            }
            Err(_) => {
                serial_println!("[WARN] Failed to get LoadedImage protocol");
                (0, 0)
            }
        };

    // Exit boot services — after this, UEFI is gone and we own the machine.
    // uefi-rs 0.32 already retries once internally on INVALID_PARAMETER
    // (VirtualBox quirk), matching the Linux kernel behaviour.
    let memory_map = unsafe { uefi::boot::exit_boot_services(MemoryType::LOADER_DATA) };

    serial_println!("Seal OS — UEFI boot complete, boot services exited");

    if boot_info.fb_addr != 0 {
        serial_println!(
            "[BOOT] Framebuffer: {}x{}x{}bpp @ {:#X}, pitch={}",
            boot_info.fb_width,
            boot_info.fb_height,
            boot_info.fb_bpp,
            boot_info.fb_addr,
            boot_info.fb_pitch
        );
    } else {
        serial_println!("[BOOT] No framebuffer available");
    }

    // Copy memory map descriptors into BootInfo
    let mut descriptors = [MemoryDescriptor {
        phys_start: 0,
        page_count: 0,
        ty: 0,
    }; MAX_MEMORY_DESCRIPTORS];
    let mut len = 0;
    for desc in memory_map.entries() {
        if len < MAX_MEMORY_DESCRIPTORS {
            descriptors[len] = MemoryDescriptor {
                phys_start: desc.phys_start,
                page_count: desc.page_count,
                ty: desc.ty.0,
            };
            len += 1;
        }
    }

    let boot_info = BootInfo {
        fb_addr: boot_info.fb_addr,
        fb_width: boot_info.fb_width,
        fb_height: boot_info.fb_height,
        fb_pitch: boot_info.fb_pitch,
        fb_bpp: boot_info.fb_bpp,
        kernel_base,
        kernel_size,
        memory_map: descriptors,
        memory_map_len: len,
        acpi_rsdp,
    };

    // Hand off to kernel
    crate::kernel_main(&boot_info);
}

fn get_framebuffer() -> BootInfo {
    // Try UEFI GOP first.
    if let Some(gop_info) = try_gop_framebuffer() {
        return gop_info;
    }

    // GOP failed — try Bochs VBE dispi (VirtualBox fallback).
    serial_println!("[BOOT] GOP unavailable — probing Bochs VBE dispi...");
    if let Some(vbe_info) = try_vbe_framebuffer() {
        return vbe_info;
    }

    serial_println!("[BOOT] No graphics available — falling back to serial-only");
    empty_boot_info()
}

fn empty_boot_info() -> BootInfo {
    BootInfo {
        fb_addr: 0,
        fb_width: 0,
        fb_height: 0,
        fb_pitch: 0,
        fb_bpp: 0,
        kernel_base: 0,
        kernel_size: 0,
        memory_map: [MemoryDescriptor {
            phys_start: 0,
            page_count: 0,
            ty: 0,
        }; MAX_MEMORY_DESCRIPTORS],
        memory_map_len: 0,
        acpi_rsdp: 0,
    }
}

fn try_gop_framebuffer() -> Option<BootInfo> {
    let gop_handle = uefi::boot::get_handle_for_protocol::<GraphicsOutput>().ok()?;
    let mut gop = uefi::boot::open_protocol_exclusive::<GraphicsOutput>(gop_handle).ok()?;

    // Pick a sane boot resolution. Some firmware exposes huge virtual modes
    // (for example 7680x4320 in VirtualBox); building a full back buffer there
    // can stall early boot before the theorem gate.
    let mut best_mode = None;
    let mut best_size = (0usize, 0usize);
    for mode in gop.modes() {
        let info = mode.info();
        // A mode the framebuffer cannot render is not a candidate at all.
        if gop_format_bytes_per_pixel(info.pixel_format()).is_none() {
            continue;
        }
        let (w, h) = info.resolution();
        if best_mode.is_none() || gop_mode_is_better((w, h), best_size) {
            best_size = (w, h);
            best_mode = Some(mode);
        }
    }

    if let Some(mode) = best_mode {
        let (w, h) = mode.info().resolution();
        if let Err(e) = gop.set_mode(&mode) {
            serial_println!("[BOOT] GOP set_mode({}x{}) failed: {:?}", w, h, e.status());
            return None;
        }
        serial_println!(
            "[BOOT] GOP mode set to {}x{} (area={})",
            w,
            h,
            w.saturating_mul(h)
        );
    } else {
        serial_println!("[BOOT] No GOP mode with a renderable pixel format found");
        return None;
    }

    let mode_info = gop.current_mode_info();
    let (width, height) = mode_info.resolution();
    let stride = mode_info.stride();
    // Re-check the format of the mode the firmware actually ended up in, before
    // touching the frame buffer: `gop.frame_buffer()` panics on `BltOnly`.
    let Some(bytes_per_pixel) = gop_format_bytes_per_pixel(mode_info.pixel_format()) else {
        serial_println!(
            "[BOOT] GOP current mode has unrenderable pixel format {:?}",
            mode_info.pixel_format()
        );
        return None;
    };
    // The firmware reports pixels per scanline, never bytes; the byte stride is
    // that count times the width the format defines.
    let pitch = stride.saturating_mul(bytes_per_pixel as usize);
    let bpp = (bytes_per_pixel * 8) as u8;
    let mut fb = gop.frame_buffer();
    let fb_addr = fb.as_mut_ptr() as u64;
    let fb_size = fb.size();

    // Reject clearly invalid framebuffers.
    if fb_addr == 0 || width == 0 || height == 0 {
        serial_println!(
            "[BOOT] GOP returned invalid framebuffer (addr={:#X}, {}x{})",
            fb_addr,
            width,
            height
        );
        return None;
    }

    // The kernel writes `pitch * height` bytes. `fb_size` is the only bound the
    // firmware measures for us, and UEFI 2.10 12.9 defines it as the size
    // needed to support the active mode, so it should never be the smaller of
    // the two.
    // ponytail: reported only, not enforced — no OVMF build was available to
    // confirm the value it reports, and refusing the mode on an unverified
    // number would risk the boot path. Once one boot log shows this line absent
    // and `needed <= fb_size` holds on real firmware, make this a `return None`
    // alongside the other rejections above.
    let needed = pitch.saturating_mul(height);
    if fb_size != 0 && needed > fb_size {
        serial_println!(
            "[WARN] GOP framebuffer smaller than the mode needs: {}x{} pitch {} needs {} bytes, firmware reports {}",
            width,
            height,
            pitch,
            needed,
            fb_size
        );
    }

    if width < 640 || height < 480 {
        serial_println!(
            "[WARN] GOP framebuffer resolution {}x{} is smaller than expected",
            width,
            height
        );
    }

    #[cfg(not(test))]
    {
        let fb_test =
            unsafe { Framebuffer::new(fb_addr, width as u32, height as u32, pitch as u32, bpp) };
        if fb_test.is_ready() {
            serial_println!("[BOOT] Framebuffer accessibility test passed");
        } else {
            serial_println!(
                "[WARN] Framebuffer accessibility test failed — GOP mode may be incompatible"
            );
        }
    }

    Some(BootInfo {
        fb_addr,
        fb_width: width as u32,
        fb_height: height as u32,
        fb_pitch: pitch as u32,
        fb_bpp: bpp,
        kernel_base: 0,
        kernel_size: 0,
        memory_map: [MemoryDescriptor {
            phys_start: 0,
            page_count: 0,
            ty: 0,
        }; MAX_MEMORY_DESCRIPTORS],
        memory_map_len: 0,
        acpi_rsdp: 0,
    })
}

/// Bochs VBE dispi fallback — used when VirtualBox GOP fails or is missing.
/// Sets 1024×768×32 via port I/O and returns the linear framebuffer address.
fn gop_mode_class((w, h): (usize, usize)) -> u8 {
    if w >= GOP_MIN_WIDTH
        && h >= GOP_MIN_HEIGHT
        && w <= GOP_MAX_BOOT_WIDTH
        && h <= GOP_MAX_BOOT_HEIGHT
    {
        2
    } else if w >= GOP_MIN_WIDTH && h >= GOP_MIN_HEIGHT {
        1
    } else {
        0
    }
}

fn gop_mode_is_better(candidate: (usize, usize), current: (usize, usize)) -> bool {
    let candidate_class = gop_mode_class(candidate);
    let current_class = gop_mode_class(current);
    if candidate_class != current_class {
        return candidate_class > current_class;
    }

    let candidate_area = candidate.0.saturating_mul(candidate.1);
    let current_area = current.0.saturating_mul(current.1);
    match candidate_class {
        2 => candidate_area > current_area,
        1 => candidate_area < current_area,
        _ => candidate_area > current_area,
    }
}

#[cfg(test)]
mod tests {
    use super::gop_mode_is_better;

    #[test]
    fn prefers_largest_mode_inside_boot_cap() {
        assert!(gop_mode_is_better((1024, 768), (800, 600)));
        assert!(!gop_mode_is_better((1280, 1024), (1024, 768)));
    }

    #[test]
    fn falls_back_to_smallest_usable_large_mode() {
        assert!(gop_mode_is_better((2560, 1440), (7680, 4320)));
        assert!(!gop_mode_is_better((7680, 4320), (2560, 1440)));
    }
}

fn try_vbe_framebuffer() -> Option<BootInfo> {
    const VBE_INDEX: u16 = 0x1CE;
    const VBE_DATA: u16 = 0x1CF;

    unsafe fn inw(port: u16) -> u16 {
        let value: u16;
        core::arch::asm!("in ax, dx", out("ax") value, in("dx") port, options(nomem, nostack));
        value
    }

    unsafe fn outw(port: u16, value: u16) {
        core::arch::asm!("out dx, ax", in("dx") port, in("ax") value, options(nomem, nostack));
    }

    unsafe {
        outw(VBE_INDEX, 0x00); // ID register
        let id = inw(VBE_DATA);
        if id != 0xB0C5 && id != 0xB0C4 {
            serial_println!("[VBE] Bochs VBE not detected (id={:#X})", id);
            return None;
        }
        serial_println!("[VBE] Bochs VBE detected (id={:#X})", id);

        // Set 1024×768×32
        outw(VBE_INDEX, 0x01);
        outw(VBE_DATA, 1024); // XRES
        outw(VBE_INDEX, 0x02);
        outw(VBE_DATA, 768); // YRES
        outw(VBE_INDEX, 0x03);
        outw(VBE_DATA, 32); // BPP
        outw(VBE_INDEX, 0x04);
        outw(VBE_DATA, 0x41); // ENABLE + linear framebuffer

        // Try to read the linear framebuffer address (registers 0x0D/0x0E).
        // Some implementations return it split across two 16-bit registers.
        outw(VBE_INDEX, 0x0D);
        let fb_lo = inw(VBE_DATA) as u32;
        outw(VBE_INDEX, 0x0E);
        let fb_hi = inw(VBE_DATA) as u32;
        let fb_addr = ((fb_hi << 16) | fb_lo) as u64;

        let fb_addr = if fb_addr != 0 {
            serial_println!("[VBE] Linear framebuffer @ {:#X}", fb_addr);
            fb_addr
        } else {
            // VirtualBox / Bochs standard default.
            serial_println!(
                "[VBE] Using default framebuffer address {:#X}",
                0xE000_0000u64
            );
            0xE000_0000
        };

        let width = 1024u32;
        let height = 768u32;
        let pitch = width * 4;
        let bpp = 32u8;

        // Skip the accessibility test here: in UEFI context the VBE framebuffer
        // may not be mapped yet, but our kernel identity-maps the first 16 GiB
        // so it will be accessible after virt::init().

        Some(BootInfo {
            fb_addr,
            fb_width: width,
            fb_height: height,
            fb_pitch: pitch,
            fb_bpp: bpp,
            kernel_base: 0,
            kernel_size: 0,
            memory_map: [MemoryDescriptor {
                phys_start: 0,
                page_count: 0,
                ty: 0,
            }; MAX_MEMORY_DESCRIPTORS],
            memory_map_len: 0,
            acpi_rsdp: 0,
        })
    }
}
