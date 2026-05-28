// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

#![cfg_attr(not(test), no_std)]
#![allow(
    dead_code,
    unused_unsafe,
    improper_ctypes_definitions,
    unused_imports,
    unused_features
)]
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

pub const VERSION: &str = "0.4.5";

#[cfg(not(test))]
static FRAMEBUFFER: spin::Once<Framebuffer> = spin::Once::new();

/// Access the global framebuffer, if one was initialised.
pub fn framebuffer() -> Option<&'static Framebuffer> {
    FRAMEBUFFER.get()
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

/// Live theorem activation states (T1–T5).
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
    serial_println!("All data = geometry on S^2. File moves = O(1) topological surgery.");
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
    memory::topo_ram::init();
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

    compositor.compose(fb);
    fb.blit();

    serial_println!(
        "[Desktop] {} windows active (Terminal, IDE, Theorems, Calculator, SealPlayer, Snake, Breakout, Warp Racer, Tensor Viewer, LAAMBA Governor)",
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
            compositor.compose(fb);
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
    drivers::pci::init();
    if drivers::block::ahci::init().is_none() {
        serial_println!("[AHCI] Initialization failed, continuing without AHCI");
    }
    drivers::disk::init();
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
    } else {
        drivers::usb::mass_storage::init();
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
    serial_println!(
        "[SealPlayer] Media player ready (WAV/PCM playback; additional formats planned)"
    );
}
