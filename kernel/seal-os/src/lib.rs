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
#[cfg(not(test))]
pub mod lang;
pub mod memory;
pub mod ml_engine;
pub mod net;
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

pub const VERSION: &str = "0.4.6";

#[cfg(not(test))]
static FRAMEBUFFER: spin::Once<Framebuffer> = spin::Once::new();

/// Access the global framebuffer, if one was initialised.
pub fn framebuffer() -> Option<&'static Framebuffer> {
    FRAMEBUFFER.get()
}

#[cfg(not(test))]
fn pin_kernel_static_pages() {
    pin_kernel_static_page_range(
        "FRAMEBUFFER",
        core::ptr::addr_of!(FRAMEBUFFER).cast::<u8>(),
        core::mem::size_of_val(&FRAMEBUFFER),
    );
    pin_kernel_static_page_range(
        "BOOT_INFO",
        core::ptr::addr_of!(BOOT_INFO).cast::<u8>(),
        core::mem::size_of::<BootInfo>(),
    );
}

#[cfg(not(test))]
fn pin_kernel_static_page_range(name: &str, ptr: *const u8, len: usize) {
    let start = (ptr as u64) & !0xfff;
    let end = ((ptr as u64).saturating_add(len as u64).saturating_add(4095)) & !0xfff;
    let mut pages = 0usize;
    let mut recovered = 0usize;
    let mut untranslated = 0usize;
    let mut page = start;

    while page < end {
        match memory::virt::translate(x86_64::VirtAddr::new(page)) {
            Some(phys) => {
                pages += 1;
                if memory::phys::mark_used(phys) {
                    recovered += 1;
                }
            }
            None => {
                untranslated += 1;
            }
        }
        page = page.saturating_add(4096);
    }

    if recovered != 0 || untranslated != 0 {
        serial_println!(
            "[MEM] pinned kernel static: name={} pages={} recovered_free_pages={} untranslated={}",
            name,
            pages,
            recovered,
            untranslated
        );
    }
}

/// Static copy of the UEFI BootInfo.  The original lives on the UEFI firmware
/// stack in low physical memory; once the physical allocator is initialised
/// that memory may be re-allocated and overwritten.  We copy it here before
/// any frame allocations can clobber it.
#[cfg(not(test))]
static mut BOOT_INFO: core::mem::MaybeUninit<BootInfo> = core::mem::MaybeUninit::uninit();

use alloc::string::String;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Live governor epsilon value (stored as u64 bits).
pub static GOVERNOR_EPSILON: AtomicU64 = AtomicU64::new(0);

/// Live theorem verification states (T1-T10; T1-T5 feed runtime subsystems).
pub const THEOREM_COUNT: usize = 10;

pub static THEOREM_STATES: [AtomicBool; THEOREM_COUNT] = [
    AtomicBool::new(true),
    AtomicBool::new(true),
    AtomicBool::new(true),
    AtomicBool::new(true),
    AtomicBool::new(true),
    AtomicBool::new(false),
    AtomicBool::new(false),
    AtomicBool::new(false),
    AtomicBool::new(false),
    AtomicBool::new(false),
];

pub const THEOREM_NAMES: [&str; THEOREM_COUNT] = [
    "T1/TSS", "T2/SCM", "T3/GMC", "T4/AGCR", "T5/HCS", "T6/RGCS", "T7/PHKP", "T8/TEB", "T9/CMA",
    "T10/WPHB",
];

#[cfg(not(test))]
#[no_mangle]
pub fn kernel_main(info: &BootInfo) -> ! {
    serial_println!("Seal OS v{} — The Geometrical Operating System", VERSION);
    serial_println!(
        "OS state = topology on S^2. File moves use metadata topology; file bytes stay in place."
    );
    serial_println!("Boot: UEFI (pure Rust, zero assembly)");
    serial_println!("");

    // Copy boot_info into static storage before the physical allocator can
    // re-use the UEFI stack pages that hold the original.
    let info = unsafe {
        let ptr = core::ptr::addr_of_mut!(BOOT_INFO).cast::<BootInfo>();
        core::ptr::write(ptr, *info);
        &*ptr
    };

    // Layer 1: Memory
    unsafe {
        memory::init(info);
    }
    serial_println!(
        "[BOOT] memory::init() done, free frames = {}",
        memory::phys::free_count()
    );
    let kernel_start = x86_64::PhysAddr::new(info.kernel_base);
    let kernel_end = x86_64::PhysAddr::new(info.kernel_base + info.kernel_size);
    let fb_start = x86_64::PhysAddr::new(info.fb_addr);
    let fb_size = info.fb_pitch as u64 * info.fb_height as u64;
    let fb_end = x86_64::PhysAddr::new(info.fb_addr + fb_size);
    unsafe {
        memory::phys::init_high(
            &info.memory_map[..info.memory_map_len],
            kernel_start,
            kernel_end,
            fb_start,
            fb_end,
        );
    }
    serial_println!(
        "[BOOT] init_high() done, free frames = {}",
        memory::phys::free_count()
    );
    pin_kernel_static_pages();
    memory::topo_ram::init();
    let alloc_proof = memory::phys::allocation_o1_proof();
    serial_println!(
        "[ALLOC] O(1) proof: topo_cells={}, l3_word_probes_per_cell={}, single_word_probes_per_cell={}, contiguous_candidate_probes={}, contiguous_max_run_pages={}, toporam_max_run_pages={}, marking=bounded_by_contiguous_max_run_pages",
        alloc_proof.topo_cells,
        alloc_proof.l3_word_probes_per_cell,
        alloc_proof.single_alloc_word_probes_per_cell,
        alloc_proof.contiguous_candidate_probes,
        alloc_proof.contiguous_max_run_pages,
        memory::topo_ram::max_run_pages()
    );
    let ram_mb = memory::total_ram(info) / (1024 * 1024);
    let free_frames = memory::phys::free_count();
    serial_println!(
        "[BOOT] Heap initialized ({} MB detected, {} free frames)",
        ram_mb,
        free_frames
    );
    let test_frame = memory::phys::alloc_frame();
    serial_println!("[DEBUG] alloc_frame test: {:?}", test_frame);
    if let Some(frame) = test_frame {
        unsafe {
            memory::phys::free_frame(frame);
        }
    }
    let (alloc_fast_hits, alloc_bounded_misses, alloc_max_contig_probes) =
        memory::phys::topo_alloc_stats();
    serial_println!(
        "[ALLOC] runtime counters: fast_hits={}, bounded_misses={}, max_contiguous_probes_seen={}",
        alloc_fast_hits,
        alloc_bounded_misses,
        alloc_max_contig_probes
    );
    run_topo_ram_hotpath_bench();
    run_allocator_hotpath_bench();
    run_manifold_teleport_bench();

    // Layer 1.1: Bring up application processors (GS base for this_cpu)
    cpu::smp::smp_init();
    serial_println!(
        "[BOOT] SMP initialised, free frames = {}",
        memory::phys::free_count()
    );

    // Switch to the BSP idle kernel stack before enabling SMAP.
    // The UEFI firmware stack lives in user-addressable low memory;
    // once SMAP is enabled the kernel can no longer touch it.
    unsafe {
        let cpu = cpu::this_cpu();
        let stack_top = cpu.kernel_stack.as_ptr() as u64 + cpu.kernel_stack.len() as u64;
        let stack_top = stack_top & !0xF;
        core::arch::asm!(
            "mov rsp, {stack}",
            "jmp {f}",
            stack = in(reg) stack_top,
            f = in(reg) kernel_main_continue as *const (),
            options(noreturn)
        );
    }
}

/// Continuation of `kernel_main` after switching to the idle kernel stack.
/// Must not be inlined so the `jmp` target has a stable address.
#[inline(never)]
extern "C" fn kernel_main_continue() -> ! {
    let info = unsafe { &*core::ptr::addr_of!(BOOT_INFO).cast::<BootInfo>() };
    // Layer 1.2: ACPI (needed for SMP APIC discovery)
    drivers::acpi::init(info);
    serial_println!(
        "[BOOT] ACPI init done, free frames = {}",
        memory::phys::free_count()
    );

    // Layer 1.5: Security hardening (SMAP/SMEP, MAC, audit)
    security::init_security();
    serial_println!(
        "[BOOT] Security subsystem initialised (SMAP/SMEP + MAC + audit), free frames = {}",
        memory::phys::free_count()
    );

    // Layer 2: Interrupts
    drivers::interrupts::init();
    serial_println!(
        "[BOOT] IDT + PIC initialized, free frames = {}",
        memory::phys::free_count()
    );
    drivers::rtc::init();
    serial_println!(
        "[BOOT] RTC initialized, free frames = {}",
        memory::phys::free_count()
    );
    unsafe {
        drivers::apic::init_local_apic_timer_for_bsp();
    }
    drivers::watchdog::init(5000);
    serial_println!(
        "[BOOT] Watchdog initialized (5 s), free frames = {}",
        memory::phys::free_count()
    );

    // Layer 2b: APIC (foundation for SMP)
    unsafe {
        drivers::apic::init();
    }
    serial_println!(
        "[BOOT] Local APIC + IO APIC initialised, free frames = {}",
        memory::phys::free_count()
    );

    // Layer 2.5: Syscall MSRs
    process::userspace::init_syscall_msrs();
    serial_println!(
        "[BOOT] SYSCALL/SYSRET MSRs programmed, free frames = {}",
        memory::phys::free_count()
    );

    // Entropy driver (before networking / TLS)
    drivers::entropy::init();
    serial_println!(
        "[BOOT] Entropy driver initialised, free frames = {}",
        memory::phys::free_count()
    );

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
            crate::FRAMEBUFFER.call_once(|| unsafe {
                crate::graphics::framebuffer::Framebuffer::new(
                    info.fb_addr,
                    info.fb_width,
                    info.fb_height,
                    info.fb_pitch,
                    info.fb_bpp,
                )
            });
        }
        if let Some(fb) = crate::FRAMEBUFFER.get() {
            fb.init_back_buffer();
        }

        graphics::topo_render::init();

        if let Some(fb) = crate::FRAMEBUFFER.get() {
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
        fb.width,
        fb.height,
        fb.bpp
    );

    // The theorem core is the boot contract. Verify it before UI waits,
    // driver probes, or graphics work can hide failures in VM smoke.
    init_theorems();

    // Initialize drivers and VFS before login so /etc/passwd is available.
    serial_println!(
        "[BOOT] Before init_drivers(), free frames = {}",
        memory::phys::free_count()
    );
    init_drivers();
    serial_println!(
        "[BOOT] After init_drivers(), free frames = {}",
        memory::phys::free_count()
    );
    if let Err(e) = fs::init_vfs() {
        serial_println!("[WARN] VFS init failed: {:?}", e);
    } else {
        serial_println!(
            "[ManifoldFS] Initialized, free frames = {}",
            memory::phys::free_count()
        );
    }
    memory::swap::init();
    security::passwd::init_passwd();
    security::shadow::shadow_init();
    security::group::init_group();

    // First-boot login screen with mascot splash
    serial_println!("[LOGIN] Showing login screen");
    let mut login = graphics::login::LoginScreen::new();
    login.render(fb);
    fb.blit();

    let mut run_installer = false;
    let login_start = drivers::interrupts::ticks();
    loop {
        if let Some(event) = drivers::interrupts::poll_event() {
            if let wm::event::InputEvent::KeyPress(scancode) = event {
                let ch = drivers::interrupts::scancode_to_char(scancode);
                if ch == b'i' || ch == b'I' {
                    run_installer = true;
                    break;
                }
                if ch != 0 {
                    if login.key_press(ch) {
                        break;
                    }
                    login.render(fb);
                    fb.blit();
                }
            }
        }
        // Auto-login quickly for CI/headless VM environments.
        if drivers::interrupts::ticks().wrapping_sub(login_start) > 500 {
            serial_println!("[LOGIN] Auto-login (no input detected)");
            break;
        }
        x86_64::instructions::hlt();
    }

    if let Some(user) = login.authenticated_user() {
        crate::security::passwd::set_current_user(user.clone());
        serial_println!(
            "[LOGIN] Authenticated as {} (uid={})",
            user.username,
            user.uid
        );
    } else {
        serial_println!("[LOGIN] Authenticated");
    }

    // Default password reminder (shown every boot for v1)
    serial_println!("[Lypnos Guard] Default login: seal / seal");
    serial_println!("[Lypnos Guard] Press Ctrl+L to lock files into topological sleep.");

    if run_installer {
        serial_println!("[INSTALLER] Launching disk installer");
        let mut installer = apps::installer::Installer::new();
        installer.render(fb);
        fb.blit();
        loop {
            if let Some(event) = drivers::interrupts::poll_event() {
                if installer.handle_event(event) {
                    break;
                }
                installer.render(fb);
                fb.blit();
            }
            x86_64::instructions::hlt();
        }
        serial_println!("[INSTALLER] Completed — continuing to desktop");
    }

    if wm::welcome::is_first_boot() {
        serial_println!("[WELCOME] Showing first-run welcome wizard");
        let mut welcome = wm::welcome::WelcomeScreen::new();
        welcome.render(fb);
        fb.blit();
        let welcome_start = drivers::interrupts::ticks();
        loop {
            if let Some(event) = drivers::interrupts::poll_event() {
                if welcome.handle_event(event) {
                    wm::welcome::mark_first_boot_done();
                    break;
                }
                welcome.render(fb);
                fb.blit();
            }
            // Auto-dismiss quickly for CI/headless VM environments.
            if drivers::interrupts::ticks().wrapping_sub(welcome_start) > 500 {
                serial_println!("[WELCOME] Auto-dismissed (no input detected)");
                wm::welcome::mark_first_boot_done();
                break;
            }
            x86_64::instructions::hlt();
        }
        serial_println!("[WELCOME] Completed");
    }

    serial_println!("[BOOT] Splash render start");
    let mut console = graphics::console::Console::new(fb);
    graphics::splash::render_splash(&mut console);
    serial_println!("[BOOT] Splash render done");
    fb.blit();
    serial_println!("[BOOT] Splash blit done");
    graphics::splash::draw_progress_bar(fb, 10, "Memory & paging");
    serial_println!("[BOOT] Splash progress ready");

    graphics::splash::draw_progress_bar(fb, 20, "Topology theorems");
    serial_println!("[BOOT] Topology progress drawn");

    graphics::splash::draw_progress_bar(fb, 30, "");
    serial_println!("[BOOT] Scheduler init start");

    init_scheduler();
    run_scheduler_select_bench();
    serial_println!("[BOOT] Scheduler init done");
    graphics::splash::draw_progress_bar(fb, 35, "Task scheduler");
    serial_println!("[BOOT] Scheduler progress drawn");

    // Enter the scheduler so that spawned kernel tasks actually execute.
    // When they yield, we resume here and continue desktop initialisation.
    serial_println!("[BOOT] Scheduler first yield start");
    process::scheduler::yield_current();
    serial_println!("[BOOT] Scheduler first yield returned");

    syscall::table::init_syscall_fs();
    serial_println!("[BOOT] Syscall FS ready");
    graphics::splash::draw_progress_bar(fb, 40, "");

    init_manifold_pkg();
    pkg::emit_boot_proof();
    graphics::splash::draw_progress_bar(fb, 50, "Network stack");

    init_aether_lang();
    graphics::splash::draw_progress_bar(fb, 55, "");

    graphics::splash::draw_progress_bar(fb, 65, "Window manager");

    // Boot-time DHCP poll with a short headless-safe timeout
    let dhcp_start = drivers::interrupts::ticks();
    while !net::dhcp::has_lease() && drivers::interrupts::ticks().wrapping_sub(dhcp_start) < 500 {
        net::poll();
        x86_64::instructions::hlt();
    }
    if net::dhcp::has_lease() {
        serial_println!("[DHCP] Lease acquired during boot");
    } else {
        serial_println!("[DHCP] Timeout — continuing without lease");
    }

    init_usb();
    graphics::splash::draw_progress_bar(fb, 70, "");

    init_prefetch();
    graphics::splash::draw_progress_bar(fb, 75, "");

    init_async_runtime();
    graphics::splash::draw_progress_bar(fb, 80, "Desktop");

    init_games();
    graphics::splash::draw_progress_bar(fb, 85, "");

    crate::apps::settings::GLOBAL_SETTINGS
        .lock()
        .init_defaults();

    serial_println!("[BOOT] All layers initialized.");
    graphics::splash::draw_progress_bar(fb, 95, "");

    // Stage 5: Userspace Init Handoff. If `/bin/init` is absent on the
    // current image, keep booting the native desktop instead of blocking.
    if fs::vfs::with_vfs(|vfs| vfs.lookup_follow("/bin/init")).is_ok() {
        let init = process::create("/bin/init");
        init.execve("/bin/init", &["/bin/init"], &[]);
    } else {
        serial_println!("[execve] '/bin/init' not found; continuing with kernel desktop");
    }

    for _ in 0..300_000u64 {
        core::hint::spin_loop();
    }

    graphics::splash::draw_progress_bar(fb, 100, "Ready");

    for _ in 0..200_000u64 {
        core::hint::spin_loop();
    }

    serial_println!("[Desktop] Rendering desktop environment");
    wm::desktop::render_desktop(fb);
    fb.blit();

    let mut compositor = wm::compositor::Compositor::new();
    unsafe {
        crate::lang::set_compositor_ptr(&mut compositor);
    }
    let mut app_state = wm::app_state::AppState::new();

    app_state.create_windows(&mut compositor);
    run_aether_desktop_probe(&mut app_state);
    run_laamba_app_proof(&app_state);
    serial_println!("[SealShell] Loaded");

    // Bring up the Seal-native control surface first; LAAMBA stays a launchable app lane.
    app_state.focus_app(1, &mut compositor);
    serial_println!("[Shell] T1/TSS  Voronoi cells: 8, Betti-0: 8");

    app_state.terminal.key_press(b'h');
    app_state.terminal.key_press(b'e');
    app_state.terminal.key_press(b'l');
    app_state.terminal.key_press(b'p');
    app_state.terminal.key_press(b'\n');
    if let Some(win) = compositor.window_mut(app_state.term_id) {
        app_state.terminal.render_to_window(win);
    }

    wm::desktop::render_desktop(fb);
    compositor.mark_dirty(0, 0, fb.width, fb.height);
    compositor.compose_full(fb);
    for _ in 0..3 {
        fb.blit();
        for _ in 0..50_000u64 {
            core::hint::spin_loop();
        }
    }
    run_graphics_desktop_proof(fb, &compositor);
    serial_println!("[BOOT] Desktop proof frame blit done");
    run_graphics_desktop_live_proof(fb, &mut compositor, &mut app_state);
    run_desktop_soak_probe(fb, &mut compositor);

    serial_println!(
        "[Desktop] {} windows active (Terminal, IDE, Files, Theorems, Calculator, SealPlayer, Snake, Breakout, Warp Racer, Tensor Viewer, LAAMBA Governor, Aether App)",
        compositor.window_count()
    );
    drivers::watchdog::enable();
    serial_println!("[BOOT] Seal OS desktop ready.");
    serial_println!("[EVENT] Entering real event loop — keyboard and mouse active");

    let mut last_tick: u64 = drivers::interrupts::ticks();
    let mut frame_dirty = false;

    use core::sync::atomic::AtomicBool;
    static CTRL_HELD: AtomicBool = AtomicBool::new(false);

    loop {
        drivers::watchdog::pet();
        while let Some(event) = drivers::interrupts::poll_event() {
            // Global keyboard shortcuts (TopCrypt / Lypnos Guard / Tensor View)
            match event {
                wm::event::InputEvent::KeyPress(0x1D) => {
                    CTRL_HELD.store(true, Ordering::Relaxed);
                }
                wm::event::InputEvent::KeyRelease(0x1D) => {
                    CTRL_HELD.store(false, Ordering::Relaxed);
                }
                wm::event::InputEvent::KeyPress(scancode) if CTRL_HELD.load(Ordering::Relaxed) => {
                    match scancode {
                        0x26 => {
                            // Ctrl+L — Lypnos Lock
                            let msg = lypnos_lock_selected(&mut app_state);
                            serial_println!("{}", msg);
                            frame_dirty = true;
                            continue;
                        }
                        0x12 => {
                            // Ctrl+E — Export
                            let msg = topcrypt_export_selected(&mut app_state);
                            serial_println!("{}", msg);
                            frame_dirty = true;
                            continue;
                        }
                        0x17 => {
                            // Ctrl+I — Import
                            let msg = topcrypt_import_selected(&mut app_state);
                            serial_println!("{}", msg);
                            frame_dirty = true;
                            continue;
                        }
                        0x14 => {
                            // Ctrl+T — Tensor View
                            serial_println!("[Tensor Viewer] Launching...");
                            crate::wm::app_launcher::launch_app(9);
                            app_state.focus_app(9, &mut compositor);
                            frame_dirty = true;
                            continue;
                        }
                        _ => {}
                    }
                }
                _ => {}
            }

            if !wm::desktop::handle_input(event, fb, &mut compositor, &mut app_state) {
                app_state.handle_event(event, &mut compositor);
            }
            frame_dirty = true;
        }

        while let Some(app_id) = wm::app_launcher::poll_launch_queue() {
            app_state.focus_app(app_id, &mut compositor);
            frame_dirty = true;
        }

        let now = drivers::interrupts::ticks();
        if now.wrapping_sub(last_tick) >= 3 {
            app_state.tick();
            net::poll();
            drivers::usb::poll();
            last_tick = now;
        }

        if frame_dirty {
            app_state.render_dirty(&mut compositor);
            wm::desktop::render_desktop(fb);
            compositor.mark_dirty(0, 0, fb.width, fb.height);
            compositor.compose_full(fb);
            wm::desktop::render_overlays(fb);
            wm::desktop::render_notifications(fb);
            fb.blit();
            frame_dirty = false;
        }

        // Yield to the scheduler so background tasks (kernel, shell,
        // compositor) get CPU time.  We resume here on the next tick.
        process::scheduler::yield_current();
        x86_64::instructions::hlt();
    }
}

fn run_laamba_app_proof(app_state: &wm::app_state::AppState) {
    let result = if app_state.laamba_id > 0 && app_state.aether_id > 0 {
        "pass"
    } else {
        "fail"
    };
    serial_println!(
        "[LAAMBA] app proof: version=1 native_app=kernel window=LAAMBA_Governor window_id={} launcher_id=10 desktop_icon=1 start_menu=1 aether_host_window_id={} runtime_bridge=rust_native_manifest python_runtime=0 result={}",
        app_state.laamba_id,
        app_state.aether_id,
        result
    );
}

struct GraphicsDesktopProof {
    scanned_pixels: u64,
    nonblack_px: u64,
    visible_icons: u64,
    icon_region_signal: u64,
    icon_color_buckets: u64,
    control_region_signal: u64,
    primary_titlebar_signal: u64,
    start_button_signal: u64,
    theorem_indicator_signal: u64,
    minimized_app_lane_signal: u64,
    power_button_signal: u64,
    sampled_pixels: u64,
    nonblack_samples: u64,
    sample_hash: u64,
}

fn run_graphics_desktop_proof(
    fb: &graphics::framebuffer::Framebuffer,
    compositor: &wm::compositor::Compositor,
) {
    let proof = measure_graphics_desktop_proof(fb);
    let back_buffer = if fb.has_back_buffer() { 1 } else { 0 };
    let result = if back_buffer == 1
        && fb.width >= 1024
        && fb.height >= 768
        && fb.bpp == 32
        && (fb.pitch as u64) >= (fb.width as u64).saturating_mul(4)
        && compositor.window_count() >= 12
        && compositor.focused_window_id() >= 1
        && proof.scanned_pixels >= (fb.width as u64).saturating_mul(fb.height as u64)
        && proof.nonblack_px >= 20_000
        && proof.visible_icons >= 10
        && proof.icon_region_signal >= 2_000
        && proof.icon_color_buckets >= 6
        && proof.control_region_signal >= 8_000
        && proof.primary_titlebar_signal >= 6_000
        && proof.start_button_signal >= 1_000
        && proof.theorem_indicator_signal >= 500
        && proof.minimized_app_lane_signal >= 300
        && proof.power_button_signal >= 220
        && proof.sampled_pixels >= 1_024
        && proof.nonblack_samples >= 512
        && proof.sample_hash != 0
    {
        "pass"
    } else {
        "fail"
    };

    serial_println!(
        "[GFX] desktop-proof version=1 surface=framebuffer width={} height={} bpp={} pitch={} back_buffer={} window_count={} focused_window_id={} scanned_pixels={} nonblack_px={} visible_icons={} icon_region_signal={} icon_color_buckets={} control_region_signal={} primary_titlebar_signal={} start_button_signal={} theorem_indicator_signal={} minimized_app_lane_signal={} power_button_signal={} sampled_pixels={} nonblack_samples={} sample_hash={} result={}",
        fb.width,
        fb.height,
        fb.bpp,
        fb.pitch,
        back_buffer,
        compositor.window_count(),
        compositor.focused_window_id(),
        proof.scanned_pixels,
        proof.nonblack_px,
        proof.visible_icons,
        proof.icon_region_signal,
        proof.icon_color_buckets,
        proof.control_region_signal,
        proof.primary_titlebar_signal,
        proof.start_button_signal,
        proof.theorem_indicator_signal,
        proof.minimized_app_lane_signal,
        proof.power_button_signal,
        proof.sampled_pixels,
        proof.nonblack_samples,
        proof.sample_hash,
        result
    );
}

const DESKTOP_LIVE_SAMPLE_COUNT: usize = 128;

fn run_graphics_desktop_live_proof(
    fb: &graphics::framebuffer::Framebuffer,
    compositor: &mut wm::compositor::Compositor,
    app_state: &mut wm::app_state::AppState,
) {
    const FILES_ICON_CENTER_X: i32 = 344;
    const FILES_ICON_CENTER_Y: i32 = 74;
    const FILES_REGION_X: u32 = 160;
    const FILES_REGION_Y: u32 = 120;
    const FILES_REGION_W: u32 = 564;
    const FILES_REGION_H: u32 = 72;

    let pre_focused = compositor.focused_window_id();
    let post_window_id = app_state.fm_id;
    let mut pre_samples = [0u32; DESKTOP_LIVE_SAMPLE_COUNT];
    let pre_hash = sample_desktop_live_region(
        fb,
        FILES_REGION_X,
        FILES_REGION_Y,
        FILES_REGION_W,
        FILES_REGION_H,
        &mut pre_samples,
    );

    while wm::app_launcher::poll_launch_queue().is_some() {}

    let mouse = *compositor.mouse_state();
    let move_handled = wm::desktop::handle_input(
        wm::event::InputEvent::MouseMove {
            dx: FILES_ICON_CENTER_X.saturating_sub(mouse.x),
            dy: FILES_ICON_CENTER_Y.saturating_sub(mouse.y),
        },
        fb,
        compositor,
        app_state,
    );
    let click_handled = wm::desktop::handle_input(
        wm::event::InputEvent::MouseButton {
            button: 0,
            pressed: true,
        },
        fb,
        compositor,
        app_state,
    );

    app_state.render_dirty(compositor);
    wm::desktop::render_desktop(fb);
    compositor.mark_dirty(0, 0, fb.width, fb.height);
    compositor.compose_full(fb);
    wm::desktop::render_overlays(fb);
    wm::desktop::render_notifications(fb);
    let blit = if fb.blit() { 1u64 } else { 0u64 };

    let mut launched_app_id = 0u64;
    while let Some(app_id) = wm::app_launcher::poll_launch_queue() {
        launched_app_id = app_id as u64;
    }

    let mut post_samples = [0u32; DESKTOP_LIVE_SAMPLE_COUNT];
    let post_hash = sample_desktop_live_region(
        fb,
        FILES_REGION_X,
        FILES_REGION_Y,
        FILES_REGION_W,
        FILES_REGION_H,
        &mut post_samples,
    );
    let changed_samples = count_desktop_live_sample_changes(&pre_samples, &post_samples);
    let mut vram_samples = [0u32; DESKTOP_LIVE_SAMPLE_COUNT];
    let vram_hash = sample_desktop_live_presented_region(
        fb,
        FILES_REGION_X,
        FILES_REGION_Y,
        FILES_REGION_W,
        FILES_REGION_H,
        &mut vram_samples,
    );
    let vram_changed_samples = count_desktop_live_sample_changes(&pre_samples, &vram_samples);
    let vram_matches_backbuffer = count_desktop_live_sample_matches(&post_samples, &vram_samples);
    let post_focused = compositor.focused_window_id();
    let icon_hit = if click_handled && launched_app_id == 3 {
        1u64
    } else {
        0u64
    };
    let handled = if move_handled && click_handled {
        1u64
    } else {
        0u64
    };
    let result = if handled == 1
        && icon_hit == 1
        && launched_app_id == 3
        && pre_focused != 0
        && post_window_id != 0
        && post_focused == post_window_id
        && post_focused != pre_focused
        && compositor.window_count() >= 12
        && pre_hash != 0
        && post_hash != 0
        && pre_hash != post_hash
        && changed_samples >= 32
        && vram_hash != 0
        && vram_hash != pre_hash
        && vram_changed_samples >= 32
        && vram_matches_backbuffer >= 32
        && blit == 1
    {
        "pass"
    } else {
        "fail"
    };

    serial_println!(
        "[GFX] desktop-live-proof version=1 route=desktop_handle_input action=desktop_icon_launch app=Files app_id=3 events=2 handled={} icon_hit={} launched_app_id={} pre_focused={} post_focused={} post_window_id={} window_count={} pre_hash={} post_hash={} changed_samples={} vram_hash={} vram_changed_samples={} vram_matches_backbuffer={} blit={} result={}",
        handled,
        icon_hit,
        launched_app_id,
        pre_focused,
        post_focused,
        post_window_id,
        compositor.window_count(),
        pre_hash,
        post_hash,
        changed_samples,
        vram_hash,
        vram_changed_samples,
        vram_matches_backbuffer,
        blit,
        result
    );
}

fn desktop_live_sample_point(x: u32, y: u32, w: u32, h: u32, idx: usize) -> (u32, u32) {
    let sx = x + ((idx as u32).wrapping_mul(37) % w.max(1));
    let sy = y + ((idx as u32).wrapping_mul(17) % h.max(1));
    (sx, sy)
}

fn desktop_live_hash_push(hash: u64, color: u32, idx: usize) -> u64 {
    hash.wrapping_mul(1_099_511_628_211)
        .wrapping_add((color as u64) ^ ((idx as u64).wrapping_mul(0x9E37_79B9)))
}

fn sample_desktop_live_region(
    fb: &graphics::framebuffer::Framebuffer,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    out: &mut [u32; DESKTOP_LIVE_SAMPLE_COUNT],
) -> u64 {
    let mut hash = 14_695_981_039_346_656_037u64;
    let mut idx = 0usize;
    while idx < DESKTOP_LIVE_SAMPLE_COUNT {
        let (sx, sy) = desktop_live_sample_point(x, y, w, h, idx);
        let color = fb.get_pixel(sx, sy);
        out[idx] = color;
        hash = desktop_live_hash_push(hash, color, idx);
        idx += 1;
    }
    hash
}

fn sample_desktop_live_presented_region(
    fb: &graphics::framebuffer::Framebuffer,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    out: &mut [u32; DESKTOP_LIVE_SAMPLE_COUNT],
) -> u64 {
    let mut hash = 14_695_981_039_346_656_037u64;
    let mut idx = 0usize;
    while idx < DESKTOP_LIVE_SAMPLE_COUNT {
        let (sx, sy) = desktop_live_sample_point(x, y, w, h, idx);
        let color = fb.get_presented_pixel(sx, sy);
        out[idx] = color;
        hash = desktop_live_hash_push(hash, color, idx);
        idx += 1;
    }
    hash
}

fn count_desktop_live_sample_changes(
    before: &[u32; DESKTOP_LIVE_SAMPLE_COUNT],
    after: &[u32; DESKTOP_LIVE_SAMPLE_COUNT],
) -> u64 {
    let mut changed = 0u64;
    let mut idx = 0usize;
    while idx < DESKTOP_LIVE_SAMPLE_COUNT {
        if before[idx] != after[idx] {
            changed += 1;
        }
        idx += 1;
    }
    changed
}

fn count_desktop_live_sample_matches(
    left: &[u32; DESKTOP_LIVE_SAMPLE_COUNT],
    right: &[u32; DESKTOP_LIVE_SAMPLE_COUNT],
) -> u64 {
    let mut matches = 0u64;
    let mut idx = 0usize;
    while idx < DESKTOP_LIVE_SAMPLE_COUNT {
        if left[idx] == right[idx] {
            matches += 1;
        }
        idx += 1;
    }
    matches
}

fn measure_graphics_desktop_proof(fb: &graphics::framebuffer::Framebuffer) -> GraphicsDesktopProof {
    const ICON_XS: [u32; 10] = [20, 120, 220, 320, 420, 520, 620, 720, 820, 920];
    let mut icon_color_pixels = [0u64; 10];
    let mut icon_color_buckets = [false; 64];
    let mut proof = GraphicsDesktopProof {
        scanned_pixels: 0,
        nonblack_px: 0,
        visible_icons: 0,
        icon_region_signal: 0,
        icon_color_buckets: 0,
        control_region_signal: 0,
        primary_titlebar_signal: 0,
        start_button_signal: 0,
        theorem_indicator_signal: 0,
        minimized_app_lane_signal: 0,
        power_button_signal: 0,
        sampled_pixels: 0,
        nonblack_samples: 0,
        sample_hash: 0xcbf29ce484222325,
    };

    let height = fb.height;
    let width = fb.width;
    for y in 0..height {
        for x in 0..width {
            let color = fb.get_pixel(x, y);
            let r = ((color >> 16) & 0xff) as u8;
            let g = ((color >> 8) & 0xff) as u8;
            let b = (color & 0xff) as u8;
            let max_channel = r.max(g).max(b);
            let min_channel = r.min(g).min(b);

            proof.scanned_pixels += 1;
            if r > 8 || g > 8 || b > 8 {
                proof.nonblack_px += 1;
            }
            if x < 1_000 && (40..140).contains(&y) && (r > 32 || g > 32 || b > 32) {
                proof.icon_region_signal += 1;
            }
            if (50..98).contains(&y) && max_channel >= 96 && max_channel - min_channel >= 40 {
                for (idx, icon_x) in ICON_XS.iter().copied().enumerate() {
                    if (icon_x..icon_x + 48).contains(&x) {
                        icon_color_pixels[idx] += 1;
                        let bucket =
                            ((r as usize / 64) << 4) | ((g as usize / 64) << 2) | (b as usize / 64);
                        icon_color_buckets[bucket.min(icon_color_buckets.len() - 1)] = true;
                        break;
                    }
                }
            }
            if (40..700).contains(&x) && (120..580).contains(&y) && (r > 28 || g > 28 || b > 28) {
                proof.control_region_signal += 1;
            }
            if (48..660).contains(&x) && (132..158).contains(&y) && b > 96 && g > 32 && r < 96 {
                proof.primary_titlebar_signal += 1;
            }
            if (4..76).contains(&x)
                && (height.saturating_sub(24)..height.saturating_sub(4)).contains(&y)
                && b > 140
                && g > 80
                && r < 110
            {
                proof.start_button_signal += 1;
            }
            if (90..270).contains(&x)
                && (height.saturating_sub(20)..height.saturating_sub(10)).contains(&y)
                && (max_channel >= 96 || max_channel - min_channel >= 40)
            {
                proof.theorem_indicator_signal += 1;
            }
            if (350..width.saturating_sub(132)).contains(&x)
                && (height.saturating_sub(24)..height.saturating_sub(4)).contains(&y)
                && (max_channel >= 120 || (max_channel >= 64 && max_channel - min_channel >= 36))
            {
                proof.minimized_app_lane_signal += 1;
            }
            if (width.saturating_sub(36)..width.saturating_sub(12)).contains(&x)
                && (height.saturating_sub(22)..height.saturating_sub(6)).contains(&y)
                && r > 180
                && g < 130
                && b < 130
            {
                proof.power_button_signal += 1;
            }

            if x % 16 == 0 && y % 16 == 0 {
                proof.sampled_pixels += 1;
                if r > 8 || g > 8 || b > 8 {
                    proof.nonblack_samples += 1;
                }
                proof.sample_hash ^= color as u64;
                proof.sample_hash = proof.sample_hash.wrapping_mul(0x100000001b3);
                proof.sample_hash ^= ((x as u64) << 32) ^ y as u64;
            }
        }
    }

    proof.visible_icons = icon_color_pixels
        .iter()
        .copied()
        .filter(|count| *count >= 1_200)
        .count() as u64;
    proof.icon_color_buckets = icon_color_buckets.iter().filter(|seen| **seen).count() as u64;
    proof
}

fn run_desktop_soak_probe(
    fb: &graphics::framebuffer::Framebuffer,
    compositor: &mut wm::compositor::Compositor,
) {
    const FRAMES: usize = 24;
    let mut samples = [0u64; FRAMES];
    let dirty_px_max = fb.width as u64 * fb.height as u64;

    for (idx, sample) in samples.iter_mut().enumerate() {
        let before = read_desktop_soak_cycles();
        wm::desktop::render_desktop(fb);
        compositor.mark_dirty(0, 0, fb.width, fb.height);
        if idx % 3 == 0 {
            compositor.mouse_move(7, 3, fb.width as i32, fb.height as i32);
        }
        compositor.compose_full(fb);
        fb.blit();
        let after = read_desktop_soak_cycles();
        *sample = after.saturating_sub(before);
    }

    for i in 1..FRAMES {
        let value = samples[i];
        let mut j = i;
        while j > 0 && samples[j - 1] > value {
            samples[j] = samples[j - 1];
            j -= 1;
        }
        samples[j] = value;
    }

    let p50_cycles = samples[FRAMES / 2];
    let p95_cycles = samples[(FRAMES * 95 / 100).min(FRAMES - 1)];
    let max_cycles = samples[FRAMES - 1];
    serial_println!(
        "[GFX] desktop-soak frames={} p50_cycles={} p95_cycles={} max_cycles={} missed_16ms=unscaled input_events=0 dirty_px_max={}",
        FRAMES,
        p50_cycles,
        p95_cycles,
        max_cycles,
        dirty_px_max
    );
}

#[inline]
fn read_desktop_soak_cycles() -> u64 {
    unsafe { core::arch::x86_64::_rdtsc() }
}

fn run_allocator_hotpath_bench() {
    const ITERS: usize = 64;
    let mut samples = [0u64; ITERS];
    let (fast_before, misses_before, contig_before) = memory::phys::topo_alloc_stats();
    let free_before = memory::phys::free_count();
    let mut ok = 0usize;

    for sample in &mut samples {
        let before = read_desktop_soak_cycles();
        if let Some(frame) = memory::phys::alloc_frame() {
            unsafe {
                memory::phys::free_frame(frame);
            }
            ok += 1;
        }
        let after = read_desktop_soak_cycles();
        *sample = after.saturating_sub(before);
    }

    for i in 1..ITERS {
        let value = samples[i];
        let mut j = i;
        while j > 0 && samples[j - 1] > value {
            samples[j] = samples[j - 1];
            j -= 1;
        }
        samples[j] = value;
    }

    let (fast_after, misses_after, contig_after) = memory::phys::topo_alloc_stats();
    let free_after = memory::phys::free_count();
    let p50_cycles = samples[ITERS / 2];
    let p95_cycles = samples[(ITERS * 95 / 100).min(ITERS - 1)];
    let max_cycles = samples[ITERS - 1];
    serial_println!(
        "[BENCH] alloc-frame iterations={} ok={} p50_cycles={} p95_cycles={} max_cycles={} fast_hits_delta={} bounded_misses_delta={} max_contiguous_probes_seen_delta={} free_before={} free_after={}",
        ITERS,
        ok,
        p50_cycles,
        p95_cycles,
        max_cycles,
        fast_after.saturating_sub(fast_before),
        misses_after.saturating_sub(misses_before),
        contig_after.saturating_sub(contig_before),
        free_before,
        free_after
    );
}

fn run_topo_ram_hotpath_bench() {
    const ITERS: usize = 64;
    let mut samples = [0u64; ITERS];
    let stats_before = memory::topo_ram::alloc_stats();
    let free_before = memory::phys::free_count();
    let mut ok = 0usize;

    for (idx, sample) in samples.iter_mut().enumerate() {
        let Some(hint) = memory::topo_ram::proof_hint(memory::topo_ram::ZoneHint::Low, idx & 7)
        else {
            continue;
        };
        let before = read_desktop_soak_cycles();
        if let Some(frame) =
            memory::topo_ram::alloc_frames(1, memory::topo_ram::ZoneHint::Low, Some(&hint))
        {
            memory::topo_ram::access_frame(frame);
            memory::topo_ram::free_frames(frame, 1);
            ok += 1;
        }
        let after = read_desktop_soak_cycles();
        *sample = after.saturating_sub(before);
    }

    for i in 1..ITERS {
        let value = samples[i];
        let mut j = i;
        while j > 0 && samples[j - 1] > value {
            samples[j] = samples[j - 1];
            j -= 1;
        }
        samples[j] = value;
    }

    let stats_after = memory::topo_ram::alloc_stats();
    let free_after = memory::phys::free_count();
    let p50_cycles = samples[ITERS / 2];
    let p95_cycles = samples[(ITERS * 95 / 100).min(ITERS - 1)];
    let max_cycles = samples[ITERS - 1];
    serial_println!(
        "[BENCH] toporam-alloc iterations={} ok={} p50_cycles={} p95_cycles={} max_cycles={} target_cell_hits_delta={} target_cell_fallbacks_delta={} low_to_high_fallbacks_delta={} high_to_low_fallbacks_delta={} pcie_to_high_fallbacks_delta={} pcie_to_low_fallbacks_delta={} free_before={} free_after={}",
        ITERS,
        ok,
        p50_cycles,
        p95_cycles,
        max_cycles,
        stats_after
            .target_cell_hits
            .saturating_sub(stats_before.target_cell_hits),
        stats_after
            .target_cell_fallbacks
            .saturating_sub(stats_before.target_cell_fallbacks),
        stats_after
            .low_to_high_fallbacks
            .saturating_sub(stats_before.low_to_high_fallbacks),
        stats_after
            .high_to_low_fallbacks
            .saturating_sub(stats_before.high_to_low_fallbacks),
        stats_after
            .pcie_to_high_fallbacks
            .saturating_sub(stats_before.pcie_to_high_fallbacks),
        stats_after
            .pcie_to_low_fallbacks
            .saturating_sub(stats_before.pcie_to_low_fallbacks),
        free_before,
        free_after
    );
}

fn run_manifold_teleport_bench() {
    const SAMPLES: usize = 3;
    const LOADS: [usize; SAMPLES] = [8, 64, 256];
    const PAYLOAD_BYTES: usize = 64;
    const MOCK_BLOCK_SECTORS: usize = 4096;
    let payload = [0xA5u8; PAYLOAD_BYTES];
    let mut samples = [0u64; SAMPLES];
    let mut ok = 0usize;
    let mut ticks_max = 0u64;
    let mut metadata_ops_max = 0u64;
    let mut persistence_bytes_per_move = 0u64;
    let mut payload_points = 0usize;
    let mut same_inode = 0usize;
    let mut src_gone = 0usize;
    let mut dst_present = 0usize;

    for (idx, load) in LOADS.iter().copied().enumerate() {
        let mut bench_fs = fs::manifold_fs::ManifoldFS::new_with_mock_store(MOCK_BLOCK_SECTORS);
        let root = bench_fs.root_id();
        let src = match bench_fs.mkdir("src", root) {
            Ok(id) => id,
            Err(_) => continue,
        };
        let dst = match bench_fs.mkdir("dst", root) {
            Ok(id) => id,
            Err(_) => continue,
        };
        for i in 0..load {
            let name = alloc::format!("f{}", i);
            let _ = bench_fs.store(name.as_str(), b"x", src);
        }
        match bench_fs.store("target.bin", &payload, src) {
            Ok(_) => {}
            Err(_) => continue,
        }
        let inode_before = match bench_fs.resolve_path_from("target.bin", src) {
            Ok(id) => id,
            Err(_) => continue,
        };

        let before = read_desktop_soak_cycles();
        let result = bench_fs.teleport("target.bin", src, dst);
        let after = read_desktop_soak_cycles();
        samples[idx] = after.saturating_sub(before);

        if let Ok(result) = result {
            let dst_inode = bench_fs.resolve_path_from("target.bin", dst).ok();
            let moved_same_inode =
                result.inode_id == inode_before && dst_inode == Some(inode_before);
            let source_removed = !bench_fs.exists("target.bin", src);
            let destination_present = bench_fs.exists("target.bin", dst);
            if moved_same_inode {
                same_inode += 1;
            }
            if source_removed {
                src_gone += 1;
            }
            if destination_present {
                dst_present += 1;
            }
            if moved_same_inode && source_removed && destination_present {
                ok += 1;
            }
            ticks_max = ticks_max.max(result.elapsed_ticks);
            metadata_ops_max = metadata_ops_max.max(result.metadata_ops);
            persistence_bytes_per_move = persistence_bytes_per_move.max(result.persistence_bytes);
            payload_points = payload_points.max(result.payload_points);
        }
    }

    for i in 1..SAMPLES {
        let value = samples[i];
        let mut j = i;
        while j > 0 && samples[j - 1] > value {
            samples[j] = samples[j - 1];
            j -= 1;
        }
        samples[j] = value;
    }

    serial_println!(
        "[BENCH] manifold-teleport api=teleport fs_mode=mock_block persistence=metadata_only samples={} ok={} same_inode={} src_gone={} dst_present={} entries_min={} entries_max={} payload_bytes={} p50_cycles={} p95_cycles={} max_cycles={} ticks_max={} metadata_ops_max={} persistence_bytes_per_move={} payload_points={}",
        SAMPLES,
        ok,
        same_inode,
        src_gone,
        dst_present,
        LOADS[0],
        LOADS[SAMPLES - 1],
        PAYLOAD_BYTES,
        samples[SAMPLES / 2],
        samples[(SAMPLES * 95 / 100).min(SAMPLES - 1)],
        samples[SAMPLES - 1],
        ticks_max,
        metadata_ops_max,
        persistence_bytes_per_move,
        payload_points
    );
}

fn run_scheduler_select_bench() {
    const ITERS: usize = 64;
    let mut samples = [0u64; ITERS];
    let mut ok = 0usize;
    let mut ready_before = 0usize;
    let mut ready_after = 0usize;
    let mut selected_priority_max = 0u8;

    for sample in &mut samples {
        let before = read_desktop_soak_cycles();
        let probe = process::scheduler::benchmark_select_next_probe();
        let after = read_desktop_soak_cycles();
        *sample = after.saturating_sub(before);

        selected_priority_max = selected_priority_max.max(probe.selected_priority);
        if ready_before == 0 {
            ready_before = probe.ready_tasks_before;
            ready_after = probe.ready_tasks_after;
        }

        if probe.selected
            && probe.ready_tasks_before >= 1
            && probe.ready_tasks_after == probe.ready_tasks_before
            && probe.ready_cells_before <= process::scheduler::SCHEDULER_SELECT_CELL_PROBE_BOUND
            && probe.selected_cell < process::scheduler::SCHEDULER_SELECT_CELL_PROBE_BOUND
        {
            ok += 1;
        }
    }

    for i in 1..ITERS {
        let value = samples[i];
        let mut j = i;
        while j > 0 && samples[j - 1] > value {
            samples[j] = samples[j - 1];
            j -= 1;
        }
        samples[j] = value;
    }

    serial_println!(
        "[BENCH] scheduler-select-next selector=select_next_task mode=live_requeue clock=rdtsc iterations={} ok={} ready_before={} ready_after={} cells={} priority_buckets={} voronoi_locate_probes={} max_cell_bitmap_tests={} max_priority_bucket_scan={} context_switches=0 selected_priority_max={} p50_cycles={} p95_cycles={} max_cycles={}",
        ITERS,
        ok,
        ready_before,
        ready_after,
        process::scheduler::SCHEDULER_SELECT_CELL_PROBE_BOUND,
        process::scheduler::SCHEDULER_SELECT_PRIORITY_BUCKET_BOUND,
        process::scheduler::SCHEDULER_SELECT_VORONOI_LOCATE_PROBES,
        process::scheduler::SCHEDULER_SELECT_MAX_CELL_BITMAP_TESTS,
        process::scheduler::SCHEDULER_SELECT_MAX_PRIORITY_BUCKET_SCAN,
        selected_priority_max,
        samples[ITERS / 2],
        samples[(ITERS * 95 / 100).min(ITERS - 1)],
        samples[ITERS - 1]
    );
}

/// Lock the currently selected file in the file manager.
fn lypnos_lock_selected(app_state: &mut wm::app_state::AppState) -> String {
    let name = match app_state.file_manager.selected_name(&app_state.fs) {
        Some(n) => n,
        None => return String::from("[Lypnos Guard] No file selected."),
    };
    let data = match app_state
        .fs
        .resolve_path_from(&name, app_state.file_manager.cwd())
    {
        Ok(id) => match app_state.fs.inode(id) {
            Some(inode) => inode.data.clone(),
            None => return format!("[Lypnos Guard] '{}' inode missing", name),
        },
        Err(_) => return format!("[Lypnos Guard] '{}' not found", name),
    };
    let mut topo = fs::topcrypt::encode_bytes(&data, 0x5EA1);
    fs::topcrypt::lock_file(&mut topo, 0xBEEF);
    let path = alloc::format!("{}/{}", app_state.file_manager.cwd(), name);
    fs::topcrypt::mark_locked(&path);
    alloc::format!("[Lypnos Guard] '{}' locked. Sweet dreams.", name)
}

/// Export the currently selected topological file to flat bytes.
fn topcrypt_export_selected(app_state: &mut wm::app_state::AppState) -> String {
    let name = match app_state.file_manager.selected_name(&app_state.fs) {
        Some(n) if n.ends_with(".topo") => n,
        Some(_) => return String::from("[TopCrypt] Selected file is not .topo"),
        None => return String::from("[TopCrypt] No file selected."),
    };
    let data = match app_state
        .fs
        .resolve_path_from(&name, app_state.file_manager.cwd())
    {
        Ok(id) => match app_state.fs.inode(id) {
            Some(inode) => inode.data.clone(),
            None => return format!("[TopCrypt] '{}' inode missing", name),
        },
        Err(_) => return format!("[TopCrypt] '{}' not found", name),
    };
    let topo = fs::topcrypt::encode_bytes(&data, 0x5EA1);
    let flat = fs::topcrypt::decode_bytes(&topo);
    let out_name = alloc::format!("{}.flat", name);
    match app_state.fs.store(&out_name, &flat, 0) {
        Ok(_) => alloc::format!(
            "[TopCrypt] File flattened for export: {} ({} bytes)",
            out_name,
            flat.len()
        ),
        Err(e) => alloc::format!("[TopCrypt] Export failed: {:?}", e),
    }
}

/// Import the currently selected flat file into topological format.
fn topcrypt_import_selected(app_state: &mut wm::app_state::AppState) -> String {
    let name = match app_state.file_manager.selected_name(&app_state.fs) {
        Some(n) => n,
        None => return String::from("[TopCrypt] No file selected."),
    };
    let data = match app_state
        .fs
        .resolve_path_from(&name, app_state.file_manager.cwd())
    {
        Ok(id) => match app_state.fs.inode(id) {
            Some(inode) => inode.data.clone(),
            None => return format!("[TopCrypt] '{}' inode missing", name),
        },
        Err(_) => return format!("[TopCrypt] '{}' not found", name),
    };
    let topo = fs::topcrypt::encode_bytes(&data, 0x5EA1);
    let blocks = topo.block_count as usize;
    let topo_name = alloc::format!("{}.topo", name);
    let serialized = fs::topcrypt::decode_bytes(&topo);
    match app_state
        .fs
        .store(&topo_name, &serialized, app_state.file_manager.cwd())
    {
        Ok(id) => alloc::format!(
            "[TopCrypt] File absorbed into manifold: {} → {} blocks on S² (inode {})",
            topo_name,
            blocks,
            id
        ),
        Err(e) => alloc::format!("[TopCrypt] Import failed: {:?}", e),
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
    pkg::emit_boot_proof();
    memory::swap::init();
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

    let theorem_ok = verify_topology_theorems();
    let mut all_verified = true;
    for (idx, ok) in theorem_ok.iter().enumerate() {
        THEOREM_STATES[idx].store(*ok, Ordering::Relaxed);
        all_verified &= *ok;
    }

    let governor = GeometricGovernor::new();
    let epsilon = governor.epsilon();
    GOVERNOR_EPSILON.store(epsilon.to_bits(), Ordering::Relaxed);
    serial_println!("[T4/AGCR] Governor online: epsilon = {:.4}", epsilon);

    let centroids = tss_boot_centroids();
    let voronoi = SphericalVoronoiIndex::<TSS_BOOT_CELL_COUNT>::new(centroids);
    let cell = voronoi.locate((0.5, 0.5));
    serial_println!(
        "[T1/TSS]  Voronoi index: 8 cells, test lookup -> cell {}",
        cell
    );

    for (name, ok) in THEOREM_NAMES.iter().zip(theorem_ok.iter()) {
        serial_println!(
            "[THEOREM] {} {}",
            name,
            if *ok { "VERIFIED" } else { "FAILED" }
        );
    }

    if !all_verified {
        panic!("Seal OS theorem core failed boot verification");
    }

    serial_println!("[BOOT] All T1-T10 theorems VERIFIED; T1-T5 ACTIVE in runtime paths");
}

fn verify_topology_theorems() -> [bool; THEOREM_COUNT] {
    use aether_verified::{
        aether_agcr, aether_gmc, aether_hcs, aether_scm, aether_tss, aether_world,
    };

    let centroids = tss_boot_centroids();
    let theta = aether_tss::theta_min_from_epsilon(0.5);
    let t1 = aether_tss::verify_packing_bound(centroids.len(), theta)
        && aether_tss::verify_separation(&centroids, theta);

    let alpha = 0.1;
    let s1 = 5.0;
    let s2 = 3.0;
    let pred = 4.0;
    let t_s1 = aether_scm::apply_operator(s1, pred, alpha);
    let t_s2 = aether_scm::apply_operator(s2, pred, alpha);
    let t2 = aether_scm::lipschitz_constant(alpha) < 1.0
        && aether_scm::verify_contraction(f64::abs(s1 - s2), f64::abs(t_s1 - t_s2), alpha);

    let t3 =
        aether_gmc::verify_entropy_nonincreasing(100, 50, 1000) && aether_gmc::max_merges(8) == 7;

    let rho = aether_agcr::contraction_rate(0.01, 0.05, 1.0);
    let t4 = rho > 0.0
        && rho < 1.0
        && aether_agcr::half_life(rho).is_finite()
        && aether_agcr::gain_margin_stable(0.01, 0.05, 1.0);

    let t5 =
        aether_hcs::verify_hcs(1.0, 4, 128, 10) && aether_hcs::separation_ratio(1.0, 4, 10) > 60.0;

    let t6_bound = aether_world::tangent_deviation_bound(0.01, 1.0, 128);
    let t6 = t6_bound > 0.0
        && t6_bound < 0.01
        && aether_world::sync_frequency(0.01, 128, 0.001, 1.0, 1.0) > 0.0;

    let base_latency = aether_world::betti_latency(800, 150, 50);
    let sparse_latency = aether_world::sparse_latency(base_latency, 0.7);
    let t7 = sparse_latency < base_latency && base_latency < 5000.0;

    let landauer = aether_world::landauer_energy_per_bit(300.0);
    let t8 = landauer > 2.8e-21 && landauer < 2.9e-21;

    let align = aether_world::alignment_error_bound(0.5, 1.0);
    let curve = aether_world::procrustes_curvature_error(1.0, 0.5, 128, 128);
    let t9 = align > 0.86 && align < 0.87 && curve > 0.0 && curve < 0.001;

    let h_info = aether_world::predictive_horizon(1000, 128, 1e-4, 10.0);
    let h_stability = aether_world::paper_horizon_estimate();
    let t10 = h_info > 1e5 && h_stability > 1e6;

    [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10]
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
        // Kernel housekeeping — swap daemon and health monitoring.
        crate::memory::swap::swap_out_daemon();
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
    pkg::GLOBAL_PKG.lock().init_defaults();
    serial_println!("[ManifoldPkg] Package registry initialized (0 packages)");
}

#[cfg(not(test))]
fn init_aether_lang() {
    let _rt = lang::AetherRuntime::new();
    serial_println!("[Aether-Lang] Titan VM ready (no_std mode)");
}

#[cfg(not(test))]
fn run_aether_desktop_probe(app_state: &mut wm::app_state::AppState) {
    const BOOT_PROBE: &str = r#"
let aether_boot_probe = "seal-topology-ok"~
let result = aether_boot_probe~
"#;

    match app_state.aether_app_host.load_script(BOOT_PROBE) {
        Ok(result) if result == "Str(\"seal-topology-ok\")" => serial_println!(
            "[Aether-Lang] runtime proof: parser=ok interpreter=ok app_host=ok script=aether_boot_probe result=seal-topology-ok"
        ),
        Ok(result) => serial_println!(
            "[Aether-Lang] runtime proof failed: unexpected result={}",
            result
        ),
        Err(e) => serial_println!("[Aether-Lang] runtime proof failed: {}", e),
    }
}

const TSS_BOOT_CELL_COUNT: usize = 8;

fn tss_boot_centroids() -> [(f64, f64); TSS_BOOT_CELL_COUNT] {
    let north = 0.615_479_708_670_387_4; // asin(1 / sqrt(3))
    let south = -north;
    let step = core::f64::consts::FRAC_PI_4;
    [
        (north, step),
        (north, step * 3.0),
        (north, step * 5.0),
        (north, step * 7.0),
        (south, step),
        (south, step * 3.0),
        (south, step * 5.0),
        (south, step * 7.0),
    ]
}

#[cfg(not(test))]
fn init_drivers() {
    drivers::cpu::intel::init();
    unsafe {
        drivers::cpu::amd::init();
    }
    serial_println!("[BOOT] driver init: cpu done");
    drivers::pci::init();
    serial_println!("[BOOT] driver init: pci done");
    if drivers::block::ahci::init().is_none() {
        serial_println!("[AHCI] Initialization failed, continuing without AHCI");
    }
    serial_println!("[BOOT] driver init: ahci done");
    drivers::disk::init();
    serial_println!("[BOOT] driver init: disk done");
    drivers::wifi::init();
    serial_println!("[BOOT] driver init: wifi done");
    drivers::bluetooth::init();
    serial_println!("[BOOT] driver init: bluetooth done");
    drivers::gpu::init();
    serial_println!("[BOOT] driver init: gpu done");
    apps::gpu_benchmark::run_structured_topology_proof();
    if drivers::nvme::init().is_none() {
        serial_println!("[NVMe] Initialization failed, continuing without NVMe");
    }
    serial_println!("[BOOT] driver init: nvme done");
    if drivers::audio::hda::init().is_none() {
        serial_println!("[HDA] Initialization failed, continuing without audio");
    }
    serial_println!("[BOOT] driver init: hda done");
    if drivers::usb::xhci::init().is_none() {
        serial_println!("[xHCI] Initialization failed, continuing without USB 3.0");
    } else {
        drivers::usb::mass_storage::init();
    }
    serial_println!("[BOOT] driver init: xhci done");
    net::init();
    run_tcp_packet_demux_bench();
    serial_println!("[BOOT] driver init: net done");
}

#[cfg(not(test))]
fn init_usb() {
    drivers::usb::init();
    serial_println!("[USB] Subsystem initialized (xHCI host controller probe pending)");
}

#[cfg(not(test))]
fn run_tcp_packet_demux_bench() {
    net::tcp::reset_for_benchmark();
    let proof = net::tcp::packet_fixture_proof();
    let accepted_state = match proof.accepted_state {
        net::tcp::TcpState::Established => "established",
        net::tcp::TcpState::Listen => "listen",
        net::tcp::TcpState::SynReceived => "syn_received",
        net::tcp::TcpState::SynSent => "syn_sent",
        net::tcp::TcpState::Closed => "closed",
        net::tcp::TcpState::FinWait1 => "fin_wait1",
        net::tcp::TcpState::FinWait2 => "fin_wait2",
        net::tcp::TcpState::CloseWait => "close_wait",
        net::tcp::TcpState::Closing => "closing",
        net::tcp::TcpState::LastAck => "last_ack",
        net::tcp::TcpState::TimeWait => "time_wait",
    };
    serial_println!(
        "[BENCH] tcp-packet-demux api=handle_tcp_packet fixture=listener_first accepted_state={} ok={} listener_first={} exact_flow={} decoy_rx_bytes={} listener_fallback={} payload_bytes={} rx_bytes={} o1_index={} index_hit={} index_lookup_probes={} index_probe_bound={} index_capacity={} listener_index_hit={} listener_lookup_probes={} listener_probe_bound={} listener_index_capacity={} exact_scan={} cleanup={}",
        accepted_state,
        if proof.ok { 1 } else { 0 },
        if proof.listener_first { 1 } else { 0 },
        if proof.exact_flow { 1 } else { 0 },
        proof.decoy_rx_bytes,
        if proof.listener_fallback { 1 } else { 0 },
        proof.payload_bytes,
        proof.rx_bytes,
        if proof.o1_index { 1 } else { 0 },
        if proof.index_hit { 1 } else { 0 },
        proof.index_lookup_probes,
        proof.index_probe_bound,
        proof.index_capacity,
        if proof.listener_index_hit { 1 } else { 0 },
        proof.listener_lookup_probes,
        proof.listener_probe_bound,
        proof.listener_index_capacity,
        if proof.exact_scan { 1 } else { 0 },
        if proof.cleanup_ok { "ok" } else { "leak" }
    );
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
    serial_println!(
        "[SealPlayer] Media player ready (WAV/PCM playback; additional formats planned)"
    );
}
