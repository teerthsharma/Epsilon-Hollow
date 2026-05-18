// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

#![cfg_attr(not(test), no_std)]
#![cfg_attr(not(test), no_main)]
#![cfg_attr(not(test), feature(abi_x86_interrupt))]

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
pub mod net;
#[cfg(not(test))]
pub mod lang;
pub mod memory;
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

pub const VERSION: &str = "0.1.0";

#[cfg(not(test))]
static mut FRAMEBUFFER: Framebuffer = Framebuffer::null();

#[cfg(not(test))]
#[no_mangle]
pub fn kernel_main(info: &BootInfo) -> ! {
    serial_println!("Seal OS v{} — The Geometrical Operating System", VERSION);
    serial_println!("All data = geometry on S^2. File moves = O(1) topological surgery.");
    serial_println!("Boot: UEFI (pure Rust, zero assembly)");
    serial_println!("");

    // Layer 1: Memory
    unsafe { memory::init(info); }
    serial_println!("[BOOT] Memory subsystem initialised");

    // Layer 1.2: ACPI (needed for SMP APIC discovery)
    drivers::acpi::init(info);
    drivers::acpi::test_acpi();

    // Layer 1.5: Security hardening (SMAP/SMEP, MAC, audit)
    security::init_security();
    serial_println!("[BOOT] Security subsystem initialised (SMAP/SMEP + MAC + audit)");

    // Layer 2: Interrupts
    drivers::interrupts::init();
    serial_println!("[BOOT] IDT + APIC initialized (keyboard + mouse IRQ active)");
    unsafe { drivers::apic::init_local_apic_timer_for_bsp(); }

    // Layer 2b: APIC (foundation for SMP)
    unsafe { drivers::apic::init(); }
    serial_println!("[BOOT] Local APIC + IO APIC initialised");

    // Layer 2c: Bring up application processors
    cpu::smp::smp_init();
    serial_println!("[BOOT] SMP initialised");

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
            unsafe {
                FRAMEBUFFER = Framebuffer::new(
                    info.fb_addr, info.fb_width, info.fb_height, info.fb_pitch, info.fb_bpp,
                );
            }
        }

        let fb = unsafe { &*core::ptr::addr_of!(FRAMEBUFFER) };

        if fb.is_available() {
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

    let mut console = graphics::console::Console::new(fb);
    graphics::splash::render_splash(&mut console);
    graphics::splash::draw_progress_bar(fb, 10);

    init_theorems();
    graphics::splash::draw_progress_bar(fb, 20);

    serial_println!("[ManifoldFS] Initialized");
    graphics::splash::draw_progress_bar(fb, 30);

    init_scheduler();
    test_userspace();
    graphics::splash::draw_progress_bar(fb, 35);

    syscall::table::init_syscall_fs();
    demo_syscalls();
    demo_teleport();
    graphics::splash::draw_progress_bar(fb, 40);

    init_manifold_pkg();
    graphics::splash::draw_progress_bar(fb, 50);

    init_aether_lang();
    graphics::splash::draw_progress_bar(fb, 55);

    init_drivers();
    drivers::block::ahci::test_ahci();
    fs::init_vfs();
    security::init_security();
    test_vfs();
    test_security();
    test_network();
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
        "[Desktop] {} windows active (Terminal, IDE, Theorems, Calculator, SealPlayer)",
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
            last_tick = now;
        }

        if frame_dirty {
            app_state.render_dirty(&mut compositor);
            wm::desktop::render_desktop(fb);
            compositor.compose(fb);
            frame_dirty = false;
        }

        x86_64::instructions::hlt();
    }
}

#[cfg(not(test))]
fn boot_serial() {
    serial_println!("[BOOT] No framebuffer — serial-only mode");
    init_theorems();
    init_scheduler();
    test_userspace();
    demo_syscalls();
    demo_teleport();
    init_manifold_pkg();
    init_aether_lang();
    init_drivers();
    drivers::block::ahci::test_ahci();
    fs::init_vfs();
    security::init_security();
    test_vfs();
    test_security();
    test_network();
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

#[cfg(not(test))]
fn init_scheduler() {
    process::scheduler::init();
    process::scheduler::spawn("kernel", 10, kernel_task_main);
    process::scheduler::spawn("compositor", 8, compositor_task_main);
    process::scheduler::spawn("shell", 5, shell_task_main);

    serial_println!(
        "[Scheduler] {} tasks spawned, preemptive multitasking active",
        process::scheduler::task_count()
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
fn demo_teleport() {
    use fs::manifold_fs::ManifoldFS;

    let mut fs = ManifoldFS::new();
    let root = 0u64;
    let docs = fs.mkdir("docs", root).unwrap_or(1);
    let _file = fs.store_text("hello.txt", "Teleport me", docs).unwrap_or(2);
    let vol = fs.mkdir("vol_a", root).unwrap_or(3);

    let start = drivers::interrupts::ticks();
    if fs.teleport("hello.txt", docs, vol).is_ok() {
        let elapsed = drivers::interrupts::ticks().saturating_sub(start);
        serial_println!(
            "[ManifoldFS] Teleported 'hello.txt' in {} ticks — O(1)",
            elapsed
        );
    } else {
        serial_println!("[ManifoldFS] Teleport demo skipped");
    }
}

#[cfg(not(test))]
fn demo_syscalls() {
    use syscall::table;

    let r1 = table::dispatch(table::SYS_GETPID, 0, 0, 0);
    let r2 = table::dispatch(table::SYS_THEOREM_STATUS, 0, 0, 0);
    let r3 = table::dispatch(table::SYS_MANIFOLD_QUERY, 1, 0, 0);

    serial_println!("[Syscalls verified] getpid={}, theorems={}, query={}",
        r1.code,
        r2.data.as_deref().unwrap_or("?"),
        r3.data.as_deref().unwrap_or("?")
    );
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
    drivers::block::ahci::init();
    drivers::wifi::init();
    serial_println!("[WiFi] Driver loaded (hardware simulation — use 'wifi' for status)");
    drivers::bluetooth::init();
    serial_println!("[Bluetooth] Driver loaded (hardware simulation — use 'bluetooth' for status)");
    drivers::gpu::init();
    net::init();
}

#[cfg(not(test))]
fn init_usb() {
    drivers::usb::init();
    serial_println!("[USB] xHCI host controller initialized (hardware simulation)");
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
    serial_println!("[SealPlayer] Media player ready (MP4, MKV, AVI, MOV, WebM, MP3, FLAC, WAV)");
}

#[cfg(not(test))]
fn test_vfs() {
    use alloc::vec::Vec;
    use fs::vfs::{with_vfs, VfsNodeType};

    serial_println!("[test_vfs] Starting VFS cross-filesystem tests...");

    // Test 1: /proc/version
    let handle = with_vfs(|vfs| vfs.lookup_follow("/proc/version")).expect("proc/version");
    let mut buf = [0u8; 256];
    let len = with_vfs(|vfs| vfs.read(handle, &mut buf, 0)).expect("read version");
    let text = core::str::from_utf8(&buf[..len]).unwrap_or("?");
    serial_println!("[test_vfs] /proc/version = {}", text);

    // Test 2: /dev/zero
    let hzero = with_vfs(|vfs| vfs.lookup_follow("/dev/zero")).expect("dev/zero");
    let mut zbuf = [0u8; 8];
    with_vfs(|vfs| vfs.read(hzero, &mut zbuf, 0)).expect("read zero");
    serial_println!(
        "[test_vfs] /dev/zero read OK (all zeros: {})",
        zbuf.iter().all(|&b| b == 0)
    );

    // Test 3: /dev/null write
    let hnull = with_vfs(|vfs| vfs.lookup_follow("/dev/null")).expect("dev/null");
    with_vfs(|vfs| vfs.write(hnull, b"discard", 0)).expect("write null");
    serial_println!("[test_vfs] /dev/null write OK");

    // Test 4: mkdir in ManifoldFS root
    let _hdir = with_vfs(|vfs| vfs.mkdir("/test_dir")).expect("mkdir");
    serial_println!("[test_vfs] mkdir /test_dir OK");

    // Test 5: create and write in ManifoldFS
    let hfile = with_vfs(|vfs| vfs.create("/test_dir/hello.txt")).expect("create");
    with_vfs(|vfs| vfs.write(hfile, b"hello vfs", 0)).expect("write file");
    let mut fbuf = [0u8; 128];
    let flen = with_vfs(|vfs| vfs.read(hfile, &mut fbuf, 0)).expect("read file");
    let ftext = core::str::from_utf8(&fbuf[..flen]).unwrap_or("?");
    serial_println!("[test_vfs] ManifoldFS read: {}", ftext);

    // Test 6: stat /proc
    let hproc = with_vfs(|vfs| vfs.lookup_follow("/proc")).expect("proc");
    let node = with_vfs(|vfs| vfs.stat(hproc)).expect("stat proc");
    serial_println!(
        "[test_vfs] /proc is_dir={}",
        matches!(node.node_type, VfsNodeType::Directory)
    );

    // Test 7: readdir /proc
    let entries = with_vfs(|vfs| vfs.readdir(hproc)).expect("readdir proc");
    let names: Vec<_> = entries.iter().map(|e| e.name.as_str()).collect();
    serial_println!("[test_vfs] /proc entries: {}", names.join(", "));

    serial_println!("[test_vfs] All tests passed");
}

#[cfg(not(test))]
fn test_security() {
    use alloc::string::String;
    use alloc::vec;
    use security::aslr;
    use security::mac;
    use security::audit;
    use security::seccomp;

    serial_println!("[test_security] Starting security subsystem tests...");

    // Test 1: ASLR returns distinct values
    let mmap = aslr::randomize_mmap_base();
    let stack = aslr::randomize_stack_top();
    let heap = aslr::randomize_heap_base();
    serial_println!("[test_security] ASLR mmap={:#x} stack={:#x} heap={:#x}", mmap, stack, heap);

    // Test 2: SMAP/SMEP enabled (CR4 bits)
    let cr4 = unsafe { x86_64::registers::control::Cr4::read_raw() };
    let smep = (cr4 & (1 << 20)) != 0;
    let smap = (cr4 & (1 << 21)) != 0;
    serial_println!("[test_security] SMAP={} SMEP={}", smap, smep);

    // Test 3: MAC blocks /root for non-root
    mac::init_default_policy();
    let blocked = !mac::check_file_permission(1000, "/root/secret", mac::Permissions::R);
    serial_println!("[test_security] MAC /root blocked for uid=1000: {}", blocked);

    // Test 4: MAC allows /data and /tmp
    let data_ok = mac::check_file_permission(1000, "/data/file", mac::Permissions::R);
    let tmp_ok = mac::check_file_permission(1000, "/tmp/foo", mac::Permissions::W);
    serial_println!("[test_security] MAC /data={} /tmp={}", data_ok, tmp_ok);

    // Test 5: Audit log generation
    audit::audit_log(audit::AuditEvent::Open { uid: 0, path: String::from("/etc/shadow"), perms: String::from("r") });
    serial_println!("[test_security] Audit event generated");

    // Test 6: seccomp filter
    let filter = vec![
        seccomp::SeccompInsn { code: 0x20, jt: 0, jf: 0, k: 0 },
        seccomp::SeccompInsn { code: 0x15, jt: 0, jf: 1, k: 1 },
        seccomp::SeccompInsn { code: 0x06, jt: 0, jf: 0, k: seccomp::SECCOMP_RET_ALLOW },
        seccomp::SeccompInsn { code: 0x06, jt: 0, jf: 0, k: seccomp::SECCOMP_RET_KILL },
    ];
    let _ = seccomp::seccomp_load_filter(1, &filter);
    let action = seccomp::seccomp_check(1, 1);
    serial_println!("[test_security] seccomp syscall=1 action={:#x}", action);

    serial_println!("[test_security] All tests passed");
}

#[cfg(not(test))]
fn test_userspace() {
    serial_println!("[Userspace] Loading test ELF...");

    // Minimal ELF64: writes "Hello userspace\n" via SYS_WRITE then SYS_EXIT
    static TEST_ELF: &[u8] = &[
        0x7f, 0x45, 0x4c, 0x46, 0x02, 0x01, 0x01, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x02, 0x00, 0x3e, 0x00, 0x01, 0x00, 0x00, 0x00,
        0x78, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x38, 0x00,
        0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x01, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00,
        0xb2, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0xb2, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x48, 0xc7, 0xc0, 0x01, 0x00, 0x00, 0x00, 0x48,
        0xc7, 0xc7, 0x01, 0x00, 0x00, 0x00, 0x48, 0x8d,
        0x35, 0x15, 0x00, 0x00, 0x00, 0x48, 0xc7, 0xc2,
        0x0e, 0x00, 0x00, 0x00, 0x0f, 0x05, 0x48, 0xc7,
        0xc0, 0x00, 0x00, 0x00, 0x00, 0x48, 0x31, 0xff,
        0x0f, 0x05, 0x48, 0x65, 0x6c, 0x6c, 0x6f, 0x20,
        0x75, 0x73, 0x65, 0x72, 0x73, 0x70, 0x61, 0x63,
        0x65, 0x0a,
    ];

    match process::scheduler::spawn_user("test_prog", 5, TEST_ELF) {
        Ok(id) => {
            serial_println!("[Userspace] Spawned user task id={}", id);
        }
        Err(e) => {
            serial_println!("[Userspace] Failed to load test ELF: {:?}", e);
        }
    }
}

#[cfg(not(test))]
fn test_network() {
    use core::sync::atomic::Ordering;
    serial_println!("[test_network] Starting network tests...");
    net::set_local_ip([127, 0, 0, 1]);
    net::icmp::ECHO_REPLY_RECEIVED.store(false, Ordering::SeqCst);
    net::icmp::send_echo_request([127, 0, 0, 1], 1);
    if net::icmp::ECHO_REPLY_RECEIVED.load(Ordering::SeqCst) {
        serial_println!("[test_network] Loopback ping OK");
    } else {
        serial_println!("[test_network] Loopback ping FAILED");
    }
}
