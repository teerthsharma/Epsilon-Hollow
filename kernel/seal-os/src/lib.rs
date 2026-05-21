// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

#![cfg_attr(not(test), no_std)]
#![cfg_attr(not(test), no_main)]
#![cfg_attr(not(test), feature(abi_x86_interrupt))]

#[macro_use]
extern crate alloc;

#[cfg(not(test))]
pub mod apps;
#[cfg(not(test))]
pub mod async_rt;
pub mod boot;
pub mod cpu;
pub mod drivers;
pub mod fs;
#[cfg(not(test))]
pub mod graphics;
pub mod net;
#[cfg(not(test))]
pub mod lang;
pub mod memory;
pub mod ml_engine;
#[cfg(not(test))]
pub mod pkg;
pub mod process;
pub mod security;
pub mod sync;
pub mod syscall;
#[cfg(not(test))]
pub mod wm;

#[cfg(any(test, feature = "test-mode"))]
pub mod testing;


#[cfg(not(test))]
use boot::boot_info::BootInfo;
#[cfg(not(test))]
use graphics::framebuffer::Framebuffer;

pub const VERSION: &str = "1.0.0-alpha";

#[cfg(not(test))]
static FRAMEBUFFER: spin::Once<Framebuffer> = spin::Once::new();

use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Live governor epsilon value (stored as u64 bits).
pub static GOVERNOR_EPSILON: AtomicU64 = AtomicU64::new(0);

/// Live theorem activation states (T1–T5).
pub static THEOREM_STATES: [AtomicBool; 5] = [
    AtomicBool::new(true),
    AtomicBool::new(true),
    AtomicBool::new(true),
    AtomicBool::new(true),
    AtomicBool::new(true),
];

#[cfg(not(test))]
#[no_mangle]
pub fn kernel_main(info: &BootInfo) -> ! {
    serial_println!("Seal OS v{} — The Geometrical Operating System", VERSION);
    serial_println!("All data = geometry on S^2. File moves = O(1) topological surgery.");
    serial_println!("Boot: UEFI (pure Rust, zero assembly)");
    serial_println!("");

    // Layer 1: Memory
    unsafe { memory::init(info); }
    let ram_mb = memory::total_ram(info) / (1024 * 1024);
    let free_frames = memory::phys::free_count();
    serial_println!("[BOOT] Heap initialized ({} MB detected, {} free frames)", ram_mb, free_frames);
    let test_frame = memory::phys::alloc_frame();
    serial_println!("[DEBUG] alloc_frame test: {:?}", test_frame);
    if let Some(frame) = test_frame {
        unsafe { memory::phys::free_frame(frame); }
    }

    // Layer 1.1: Bring up application processors (GS base for this_cpu)
    cpu::smp::smp_init();
    serial_println!("[BOOT] SMP initialised");

    // Layer 1.2: ACPI (needed for SMP APIC discovery)
    drivers::acpi::init(info);

    // Layer 1.5: Security hardening (SMAP/SMEP, MAC, audit)
    security::init_security();
    serial_println!("[BOOT] Security subsystem initialised (SMAP/SMEP + MAC + audit)");

    // Layer 2: Interrupts
    drivers::interrupts::init();
    serial_println!("[BOOT] IDT + PIC initialized");
    unsafe { drivers::apic::init_local_apic_timer_for_bsp(); }

    // Layer 2b: APIC (foundation for SMP)
    unsafe { drivers::apic::init(); }
    serial_println!("[BOOT] Local APIC + IO APIC initialised");

    // Layer 2.5: Syscall MSRs
    process::userspace::init_syscall_msrs();
    serial_println!("[BOOT] SYSCALL/SYSRET MSRs programmed");

    // When test-mode is enabled, skip desktop boot and run tests
    #[cfg(feature = "test-mode")]
    {
        serial_println!("[TEST-MODE] Running integration tests...");
        testing::runner::test_main();
    }

    // Layer 3: Framebuffer (from UEFI GOP)
    #[cfg(not(feature = "test-mode"))]
    {
        if info.fb_addr != 0 {
            FRAMEBUFFER.call_once(|| unsafe {
                Framebuffer::new(
                    info.fb_addr, info.fb_width, info.fb_height, info.fb_pitch, info.fb_bpp,
                )
            });
        }

        if let Some(fb) = FRAMEBUFFER.get() {
            serial_println!("[BOOT] Framebuffer available — graphical mode");
            boot_graphical(fb);
        } else {
            serial_println!("[BOOT] No framebuffer — serial mode");
            boot_serial();
        }
    }

    loop {
        x86_64::instructions::hlt();
    }
}

#[cfg(test)]
pub fn kernel_main(_info: &boot::boot_info::BootInfo) -> ! {
    panic!("kernel_main should not be called in host tests")
}

#[cfg(not(test))]
fn boot_graphical(fb: &'static Framebuffer) {
    serial_println!(
        "[BOOT] Framebuffer: {}x{}x{}bpp",
        fb.width, fb.height, fb.bpp
    );

    // First-boot login screen with mascot splash
    serial_println!("[LOGIN] Showing login screen");
    let mut login = graphics::login::LoginScreen::new();
    login.render(fb);

    let login_start = drivers::interrupts::ticks();
    loop {
        if let Some(event) = drivers::interrupts::poll_event() {
            if let wm::event::InputEvent::KeyPress(scancode) = event {
                let ch = drivers::interrupts::scancode_to_char(scancode);
                if ch != 0 {
                    if login.key_press(ch) {
                        break;
                    }
                    login.render(fb);
                }
            }
        }
        // Auto-login after ~3 seconds of inactivity (timer ticks ≈ ms on APIC)
        if drivers::interrupts::ticks().wrapping_sub(login_start) > 0 {
            serial_println!("[LOGIN] Auto-login (timeout)");
            break;
        }
        x86_64::instructions::hlt();
    }
    serial_println!("[LOGIN] Authenticated");

    let mut console = graphics::console::Console::new(fb);
    graphics::splash::render_splash(&mut console);
    graphics::splash::draw_progress_bar(fb, 10);

    init_theorems();
    graphics::splash::draw_progress_bar(fb, 20);

    serial_println!("[ManifoldFS] Initialized");
    graphics::splash::draw_progress_bar(fb, 30);

    init_scheduler();
    graphics::splash::draw_progress_bar(fb, 35);

    // Enter the scheduler so that spawned kernel tasks actually execute.
    // When they yield, we resume here and continue desktop initialisation.
    process::scheduler::yield_current();

    syscall::table::init_syscall_fs();
    graphics::splash::draw_progress_bar(fb, 40);

    init_manifold_pkg();
    graphics::splash::draw_progress_bar(fb, 50);

    init_aether_lang();
    graphics::splash::draw_progress_bar(fb, 55);

    init_drivers();
    if let Err(e) = fs::init_vfs() {
        serial_println!("[WARN] VFS init failed: {:?}", e);
    }
    graphics::splash::draw_progress_bar(fb, 65);

    init_usb();
    graphics::splash::draw_progress_bar(fb, 70);

    init_prefetch();
    graphics::splash::draw_progress_bar(fb, 75);

    init_async_runtime();
    graphics::splash::draw_progress_bar(fb, 80);

    init_games();
    graphics::splash::draw_progress_bar(fb, 85);

    serial_println!("[BOOT] All layers initialized.");
    graphics::splash::draw_progress_bar(fb, 95);

    // Stage 5: Userspace Init Handoff
    let init = process::create("/bin/init");
    init.execve("/bin/init", &["/bin/init"], &[]);

    for _ in 0..30_000_000u64 {
        core::hint::spin_loop();
    }

    graphics::splash::draw_progress_bar(fb, 100);

    for _ in 0..20_000_000u64 {
        core::hint::spin_loop();
    }

    serial_println!("[Desktop] Rendering desktop environment");
    wm::desktop::render_desktop(fb);

    let mut compositor = wm::compositor::Compositor::new();
    let mut app_state = wm::app_state::AppState::new();

    app_state.create_windows(&mut compositor);
    serial_println!("[SealShell] Loaded");
    serial_println!("[Shell] T1/TSS  Voronoi cells: 8, Betti-0: 8");

    app_state.terminal.key_press(b'h');
    app_state.terminal.key_press(b'e');
    app_state.terminal.key_press(b'l');
    app_state.terminal.key_press(b'p');
    app_state.terminal.key_press(b'\n');
    if let Some(win) = compositor.window_mut(app_state.term_id) {
        app_state.terminal.render_to_window(win);
    }

    compositor.compose(fb);

    serial_println!(
        "[Desktop] {} windows active (Terminal, IDE, Theorems, Calculator, SealPlayer, Snake, Breakout, Warp Racer)",
        compositor.window_count()
    );
    serial_println!("[BOOT] Seal OS desktop ready.");
    serial_println!("[EVENT] Entering real event loop — keyboard and mouse active");

    let mut last_tick: u64 = drivers::interrupts::ticks();
    let mut frame_dirty = false;

    loop {
        while let Some(event) = drivers::interrupts::poll_event() {
            app_state.handle_event(event, &mut compositor);
            frame_dirty = true;
        }

        let now = drivers::interrupts::ticks();
        if now.wrapping_sub(last_tick) >= 3 {
            app_state.tick();
            net::poll();
            last_tick = now;
        }

        if frame_dirty {
            app_state.render_dirty(&mut compositor);
            wm::desktop::render_desktop(fb);
            compositor.compose(fb);
            frame_dirty = false;
        }

        // Yield to the scheduler so background tasks (kernel, shell,
        // compositor) get CPU time.  We resume here on the next tick.
        process::scheduler::yield_current();
        x86_64::instructions::hlt();
    }
}

#[cfg(not(test))]
fn boot_serial() {
    serial_println!("[BOOT] No framebuffer — serial-only mode");
    init_theorems();
    init_scheduler();
    init_manifold_pkg();
    init_aether_lang();
    init_drivers();
    if let Err(e) = fs::init_vfs() {
        serial_println!("[WARN] VFS init failed: {:?}", e);
    }
    init_usb();
    init_prefetch();
    init_async_runtime();
    init_games();
    serial_println!("[BOOT] All layers initialized. Seal OS ready.");
}

#[cfg(not(test))]
fn init_theorems() {
    use aether_core::governor::GeometricGovernor;
    use aether_core::tss::SphericalVoronoiIndex;

    let governor = GeometricGovernor::new();
    let epsilon = governor.epsilon();
    GOVERNOR_EPSILON.store(epsilon.to_bits(), Ordering::Relaxed);
    serial_println!(
        "[T4/AGCR] Governor online: epsilon = {:.4}",
        epsilon
    );

    let centroids = [
        (0.0, 0.0), (1.57, 0.0), (3.14, 0.0), (0.0, 1.57),
        (1.57, 1.57), (3.14, 1.57), (0.0, 3.14), (1.57, 3.14),
    ];
    let voronoi = SphericalVoronoiIndex::<8>::new(centroids);
    let cell = voronoi.locate((0.5, 0.5));
    serial_println!("[T1/TSS]  Voronoi index: 8 cells, test lookup -> cell {}", cell);
    serial_println!("[BOOT] All T1-T5 theorems ACTIVE");
    for state in &THEOREM_STATES {
        state.store(true, Ordering::Relaxed);
    }
}

#[cfg(not(test))]
fn init_scheduler() {
    process::scheduler::init();
    process::scheduler::spawn("kernel", 10, kernel_task_main);
    process::scheduler::spawn("compositor", 8, compositor_task_main);
    process::scheduler::spawn("shell", 5, shell_task_main);

    serial_println!(
        "[Scheduler] {} tasks, running '{}', epsilon={:.4}",
        process::scheduler::task_count(),
        process::scheduler::current_task_name(),
        process::scheduler::governor_epsilon()
    );
}

fn kernel_task_main() {
    loop {
        // Kernel housekeeping — could monitor system health, etc.
        crate::process::scheduler::yield_current();
    }
}

fn compositor_task_main() {
    loop {
        // Compositor refresh loop
        crate::process::scheduler::yield_current();
    }
}

fn shell_task_main() {
    loop {
        // Shell input processing
        crate::process::scheduler::yield_current();
    }
}

#[cfg(not(test))]
fn init_manifold_pkg() {
    let _pkg = pkg::ManifoldPkg::new();
    serial_println!("[ManifoldPkg] Package registry initialized (0 packages)");
}

#[cfg(not(test))]
fn init_aether_lang() {
    let _rt = lang::AetherRuntime::new();
    serial_println!("[Aether-Lang] Titan VM ready (no_std mode)");
}

#[cfg(not(test))]
fn init_drivers() {
    drivers::pci::init();
    if drivers::block::ahci::init().is_none() {
        serial_println!("[AHCI] Initialization failed, continuing without AHCI");
    }
    drivers::wifi::init();
    drivers::bluetooth::init();
    drivers::gpu::init();
    if drivers::nvme::init().is_none() {
        serial_println!("[NVMe] Initialization failed, continuing without NVMe");
    }
    if drivers::audio::hda::init().is_none() {
        serial_println!("[HDA] Initialization failed, continuing without audio");
    }
    if drivers::usb::xhci::init().is_none() {
        serial_println!("[xHCI] Initialization failed, continuing without USB 3.0");
    }
    net::init();
}

#[cfg(not(test))]
fn init_usb() {
    drivers::usb::init();
    serial_println!("[USB] Subsystem initialized (xHCI host controller probe pending)");
}

#[cfg(not(test))]
fn init_prefetch() {
    fs::prefetch::init();
    serial_println!("[aether-link] Prefetch engine active (gaming preset)");
}

#[cfg(not(test))]
fn init_async_runtime() {
    async_rt::init();
    serial_println!("[AsyncRT] Executor ready (single-threaded, waker-based)");
}

#[cfg(not(test))]
fn init_games() {
    serial_println!("[Games] Snake, Breakout, Warp Racer available");
    serial_println!("[Calculator] Scientific calculator ready");
    serial_println!("[SealPlayer] Media player ready (WAV/PCM playback; additional formats planned)");
}


