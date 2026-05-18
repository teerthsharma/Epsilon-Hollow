// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

#![no_std]
#![no_main]
#![feature(abi_x86_interrupt)]

extern crate alloc;

mod apps;
mod async_rt;
mod boot;
mod drivers;
mod fs;
mod graphics;
mod lang;
mod memory;
mod pkg;
mod process;
mod syscall;
mod wm;

use core::panic::PanicInfo;

use aether_core::governor::GeometricGovernor;
use aether_core::tss::SphericalVoronoiIndex;

use graphics::console::Console;
use graphics::framebuffer::Framebuffer;

const VERSION: &str = "0.1.0";

static mut FRAMEBUFFER: Framebuffer = Framebuffer::null();

#[no_mangle]
pub extern "C" fn _start(multiboot_info_addr: u64) -> ! {
    drivers::serial::init();
    serial_println!("Seal OS v{} — The Geometrical Operating System", VERSION);
    serial_println!("All data = geometry on S^2. File moves = O(1) topological surgery.");
    serial_println!("");

    // Layer 1: Memory
    memory::init_heap();
    serial_println!("[BOOT] Heap initialized (16 MB)");

    // Layer 2: Interrupts
    drivers::interrupts::init();
    serial_println!("[BOOT] IDT + PIC initialized (keyboard + mouse IRQ active)");

    // Layer 3: Framebuffer
    serial_println!("[BOOT] Multiboot info at {:#X}", multiboot_info_addr);
    if multiboot_info_addr != 0 {
        parse_multiboot2(multiboot_info_addr);
    }
    serial_println!("[BOOT] Multiboot parsing complete");

    let fb = unsafe { &*core::ptr::addr_of!(FRAMEBUFFER) };

    if fb.is_available() {
        serial_println!("[BOOT] Framebuffer available — graphical mode");
        boot_graphical(fb);
    } else {
        serial_println!("[BOOT] No framebuffer — serial mode");
        boot_serial();
    }

    // This point is never reached — boot_graphical enters the event loop
    loop {
        x86_64::instructions::hlt();
    }
}

fn boot_graphical(fb: &'static Framebuffer) {
    serial_println!(
        "[BOOT] Framebuffer: {}x{}x{}bpp",
        fb.width, fb.height, fb.bpp
    );

    let mut console = Console::new(fb);
    graphics::splash::render_splash(&mut console);
    graphics::splash::draw_progress_bar(fb, 10);

    // Layer 3: Theorems
    init_theorems();
    graphics::splash::draw_progress_bar(fb, 20);

    // Layer 4: ManifoldFS (initialized inside AppState)
    serial_println!("[ManifoldFS] Initializing");
    graphics::splash::draw_progress_bar(fb, 30);

    // Layer 5: Process scheduler
    init_scheduler();
    graphics::splash::draw_progress_bar(fb, 35);

    // Layer 6: Syscalls
    demo_syscalls();
    graphics::splash::draw_progress_bar(fb, 40);

    // Layer 7: ManifoldPkg
    init_manifold_pkg();
    graphics::splash::draw_progress_bar(fb, 50);

    // Layer 8: Aether-Lang
    init_aether_lang();
    graphics::splash::draw_progress_bar(fb, 55);

    // Layer 9: Drivers
    init_drivers();
    graphics::splash::draw_progress_bar(fb, 65);

    // Layer 10: USB
    init_usb();
    graphics::splash::draw_progress_bar(fb, 70);

    // Layer 11: Aether-Link prefetch
    init_prefetch();
    graphics::splash::draw_progress_bar(fb, 75);

    // Layer 12: Async runtime
    init_async_runtime();
    graphics::splash::draw_progress_bar(fb, 80);

    // Layer 13: Games
    init_games();
    graphics::splash::draw_progress_bar(fb, 85);

    serial_println!("[BOOT] All layers initialized.");
    graphics::splash::draw_progress_bar(fb, 95);

    for _ in 0..30_000_000u64 {
        core::hint::spin_loop();
    }

    graphics::splash::draw_progress_bar(fb, 100);

    for _ in 0..20_000_000u64 {
        core::hint::spin_loop();
    }

    // Layer 14: Desktop + Window Manager + Event Loop
    serial_println!("[Desktop] Rendering desktop environment");
    wm::desktop::render_desktop(fb);

    let mut compositor = wm::compositor::Compositor::new();
    let mut app_state = wm::app_state::AppState::new();

    // Create all windows and do initial render
    app_state.create_windows(&mut compositor);

    // Run the demo command in terminal
    app_state.terminal.key_press(b'h');
    app_state.terminal.key_press(b'e');
    app_state.terminal.key_press(b'l');
    app_state.terminal.key_press(b'p');
    app_state.terminal.key_press(b'\n');
    if let Some(win) = compositor.window_mut(app_state.term_id) {
        app_state.terminal.render_to_window(win);
    }

    // Initial compose
    compositor.compose(fb);

    serial_println!(
        "[Desktop] {} windows active (Terminal, IDE, Theorems, Calculator, SealPlayer)",
        compositor.window_count()
    );
    serial_println!("[BOOT] Seal OS desktop ready.");
    serial_println!("[EVENT] Entering real event loop — keyboard and mouse active");

    // === THE REAL EVENT LOOP ===
    // This is what makes the OS actually run. Input events from keyboard/mouse
    // interrupts are polled, dispatched to the focused app, and the screen is
    // recomposed when anything changes.

    let mut last_tick: u64 = drivers::interrupts::ticks();
    let mut frame_dirty = false;

    loop {
        // Poll all pending input events
        while let Some(event) = drivers::interrupts::poll_event() {
            app_state.handle_event(event, &mut compositor);
            frame_dirty = true;
        }

        // Tick games at ~18fps (PIT fires at ~55Hz, tick every 3)
        let now = drivers::interrupts::ticks();
        if now.wrapping_sub(last_tick) >= 3 {
            app_state.tick();
            last_tick = now;
        }

        // Re-render dirty windows and compose to framebuffer
        if frame_dirty {
            app_state.render_dirty(&mut compositor);

            // Re-render desktop background under windows
            wm::desktop::render_desktop(fb);
            compositor.compose(fb);
            frame_dirty = false;
        }

        // Sleep until next interrupt (keyboard, mouse, or timer)
        x86_64::instructions::hlt();
    }
}

fn boot_serial() {
    serial_println!("[BOOT] No framebuffer — serial-only mode");
    init_theorems();
    init_scheduler();
    demo_syscalls();
    init_manifold_pkg();
    init_aether_lang();
    init_drivers();
    init_usb();
    init_prefetch();
    init_async_runtime();
    init_games();
    serial_println!("[BOOT] All layers initialized. Seal OS ready.");
}

fn init_theorems() {
    let governor = GeometricGovernor::new();
    serial_println!(
        "[T4/AGCR] Governor online: epsilon = {:.4}",
        governor.epsilon()
    );

    let centroids = [
        (0.0, 0.0), (1.57, 0.0), (3.14, 0.0), (0.0, 1.57),
        (1.57, 1.57), (3.14, 1.57), (0.0, 3.14), (1.57, 3.14),
    ];
    let voronoi = SphericalVoronoiIndex::<8>::new(centroids);
    let cell = voronoi.locate((0.5, 0.5));
    serial_println!("[T1/TSS]  Voronoi index: 8 cells, test lookup -> cell {}", cell);
    serial_println!("[BOOT] All T1-T5 theorems ACTIVE");
}

fn init_scheduler() {
    use process::scheduler::ManifoldScheduler;

    let mut sched = ManifoldScheduler::new();
    sched.spawn("kernel", 10, || {});
    sched.spawn("compositor", 8, || {});
    sched.spawn("shell", 5, || {});
    sched.spawn("idle", 0, || {});

    sched.schedule();
    sched.tick();
    sched.tick();

    serial_println!(
        "[Scheduler] {} tasks, running '{}', epsilon={:.4}",
        sched.task_count(),
        sched.current_task_name(),
        sched.governor_epsilon()
    );
}

fn demo_syscalls() {
    use syscall::table;

    let r1 = table::dispatch(table::SYS_GETPID, 0, 0, 0);
    let r2 = table::dispatch(table::SYS_THEOREM_STATUS, 0, 0, 0);
    let r3 = table::dispatch(table::SYS_MANIFOLD_QUERY, 1, 0, 0);

    serial_println!("[Syscall] getpid={}, theorems={}, query={}",
        r1.code,
        r2.data.as_deref().unwrap_or("?"),
        r3.data.as_deref().unwrap_or("?")
    );
}

fn init_manifold_pkg() {
    let _pkg = pkg::ManifoldPkg::new();
    serial_println!("[ManifoldPkg] Package registry initialized (0 packages)");
}

fn init_aether_lang() {
    let _rt = lang::AetherRuntime::new();
    serial_println!("[Aether-Lang] Titan VM ready (no_std mode)");
}

fn init_drivers() {
    drivers::pci::init();
    drivers::wifi::init();
    serial_println!("[WiFi] Driver loaded (hardware simulation — use 'wifi' for status)");
    drivers::bluetooth::init();
    serial_println!("[Bluetooth] Driver loaded (hardware simulation — use 'bluetooth' for status)");
    drivers::gpu::init();
    drivers::net::init();
}

fn init_usb() {
    drivers::usb::init();
    serial_println!("[USB] xHCI host controller initialized (hardware simulation)");
}

fn init_prefetch() {
    fs::prefetch::init();
    serial_println!("[aether-link] Prefetch engine active (gaming preset)");
}

fn init_async_runtime() {
    async_rt::init();
    serial_println!("[AsyncRT] Executor ready (single-threaded, waker-based)");
}

fn init_games() {
    serial_println!("[Games] Snake, Breakout, Warp Racer available");
    serial_println!("[Calculator] Scientific calculator ready");
    serial_println!("[SealPlayer] Media player ready (MP4, MKV, AVI, MOV, WebM, MP3, FLAC, WAV)");
}

fn parse_multiboot2(addr: u64) {
    unsafe {
        let ptr = addr as *const u8;
        let total_size = *(ptr as *const u32) as usize;
        let mut offset = 8usize;

        while offset < total_size {
            let tag_ptr = ptr.add(offset);
            let tag_type = *(tag_ptr as *const u32);
            let tag_size = *(tag_ptr.add(4) as *const u32) as usize;

            if tag_type == 0 {
                break;
            }

            if tag_type == 8 && tag_size >= 32 {
                let fb_addr = *(tag_ptr.add(8) as *const u64);
                let fb_pitch = *(tag_ptr.add(16) as *const u32);
                let fb_width = *(tag_ptr.add(20) as *const u32);
                let fb_height = *(tag_ptr.add(24) as *const u32);
                let fb_bpp = *(tag_ptr.add(28) as *const u8);
                let fb_type = *(tag_ptr.add(29) as *const u8);

                if fb_type == 1 {
                    serial_println!(
                        "[FB] {}x{} @ {:#X}, pitch={}, bpp={}",
                        fb_width, fb_height, fb_addr, fb_pitch, fb_bpp
                    );
                    FRAMEBUFFER = Framebuffer::new(fb_addr, fb_width, fb_height, fb_pitch, fb_bpp);
                }
            }

            offset += ((tag_size + 7) & !7).max(8);
        }
    }
}

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    serial_println!("!!! SEAL OS KERNEL PANIC !!!");
    serial_println!("{}", info);
    loop {
        x86_64::instructions::hlt();
    }
}
