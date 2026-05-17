// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

#![no_std]
#![no_main]
#![feature(abi_x86_interrupt)]

extern crate alloc;

mod apps;
mod boot;
mod drivers;
mod fs;
mod graphics;
mod memory;
mod process;
mod syscall;
mod wm;

use core::panic::PanicInfo;

use aether_core::governor::GeometricGovernor;
use aether_core::tss::SphericalVoronoiIndex;

use graphics::console::Console;
use graphics::framebuffer::Framebuffer;

const VERSION: &str = "1.0.0-alpha";

static mut FRAMEBUFFER: Framebuffer = Framebuffer::null();

#[no_mangle]
pub extern "C" fn _start(multiboot_info_addr: u64) -> ! {
    drivers::serial::init();
    serial_println!("Seal OS v{} — The Geometrical Operating System", VERSION);
    serial_println!("All data = geometry on S^2. File moves = O(1) topological surgery.");
    serial_println!("");

    // Layer 1: Memory
    memory::init_heap();
    serial_println!("[BOOT] Heap initialized (4 MB)");

    // Layer 2: Interrupts
    drivers::interrupts::init();
    serial_println!("[BOOT] IDT + PIC initialized");

    // Layer 3: Framebuffer
    if multiboot_info_addr != 0 {
        parse_multiboot2(multiboot_info_addr);
    }

    let fb = unsafe { &*core::ptr::addr_of!(FRAMEBUFFER) };

    if fb.is_available() {
        boot_graphical(fb);
    } else {
        boot_serial();
    }

    serial_println!("[BOOT] Entering event loop");

    loop {
        x86_64::instructions::hlt();
    }
}

fn boot_graphical(fb: &'static Framebuffer) {
    serial_println!(
        "[BOOT] Framebuffer: {}x{}x{}bpp",
        fb.width, fb.height, fb.bpp
    );

    // Boot splash
    let mut console = Console::new(fb);
    graphics::splash::render_splash(&mut console);
    graphics::splash::draw_progress_bar(fb, 20);

    // Layer 3: Theorems
    init_theorems();
    graphics::splash::draw_progress_bar(fb, 40);

    // Layer 4: ManifoldFS
    demo_manifold_fs();
    graphics::splash::draw_progress_bar(fb, 55);

    // Layer 5: Process scheduler
    init_scheduler();
    graphics::splash::draw_progress_bar(fb, 65);

    // Layer 6: Syscalls
    demo_syscalls();
    graphics::splash::draw_progress_bar(fb, 75);

    // Layer 7: Shell
    demo_shell();
    graphics::splash::draw_progress_bar(fb, 85);

    serial_println!("[BOOT] All layers initialized.");
    graphics::splash::draw_progress_bar(fb, 95);

    // Pause on splash
    for _ in 0..30_000_000u64 {
        core::hint::spin_loop();
    }

    graphics::splash::draw_progress_bar(fb, 100);

    for _ in 0..20_000_000u64 {
        core::hint::spin_loop();
    }

    // Layer 8+9: Desktop + Window Manager
    render_desktop_environment(fb);
}

fn boot_serial() {
    serial_println!("[BOOT] No framebuffer — serial-only mode");
    init_theorems();
    demo_manifold_fs();
    init_scheduler();
    demo_syscalls();
    demo_shell();
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

fn demo_manifold_fs() {
    use fs::manifold_fs::ManifoldFS;

    serial_println!("[ManifoldFS] Initializing");
    let mut mfs = ManifoldFS::new();

    let vol_a = mfs.mkdir("vol_a", 0).unwrap();
    let vol_b = mfs.mkdir("vol_b", 0).unwrap();

    mfs.store_text("hello.txt", "Hello from Seal OS!", vol_a).unwrap();
    mfs.store_text("kernel.rs", "fn _start() -> ! { loop {} }", vol_a).unwrap();
    mfs.store_text("theorem.md", "T1-T5 all active in kernel", vol_a).unwrap();

    let result = mfs.teleport("hello.txt", vol_a, vol_b).unwrap();
    serial_println!(
        "[ManifoldFS] Teleported 'hello.txt' ({} bytes) in {} ticks — O(1)",
        result.original_size, result.elapsed_ticks
    );

    let stats = mfs.stats();
    serial_println!(
        "[ManifoldFS] {} files, {} dirs, governor e={:.4}",
        stats.total_files, stats.total_dirs, stats.governor_epsilon
    );
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

fn demo_shell() {
    use apps::shell::Shell;

    let mut shell = Shell::new();

    // Populate FS
    shell.execute("mkdir docs");
    shell.execute("touch readme.txt Welcome to Seal OS");
    shell.execute("touch notes.txt Topological file surgery");

    let output = shell.execute("theorems");
    for line in output.lines().take(3) {
        serial_println!("[Shell] {}", line);
    }

    let output = shell.execute("uname");
    serial_println!("[Shell] {}", output.lines().next().unwrap_or(""));
}

fn render_desktop_environment(fb: &'static Framebuffer) {
    serial_println!("[Desktop] Rendering desktop environment");

    // Layer 9: Desktop wallpaper
    wm::desktop::render_desktop(fb);

    // Layer 8: Window Manager
    let mut compositor = wm::compositor::Compositor::new();

    // Create windows for built-in apps
    let term_id = compositor.create_window("Terminal", 40, 40, 600, 400);
    let ide_id = compositor.create_window("Seal IDE", 120, 80, 640, 440);
    let tv_id = compositor.create_window("Theorems", 500, 60, 420, 340);

    // Layer 10a: Terminal
    let mut terminal = apps::terminal::Terminal::new(600, 400);
    terminal.key_press(b'h');
    terminal.key_press(b'e');
    terminal.key_press(b'l');
    terminal.key_press(b'p');
    terminal.key_press(b'\n');

    if let Some(win) = compositor.window_mut(term_id) {
        terminal.render_to_window(win);
    }

    // Layer 10b: Seal IDE
    let ide = apps::seal_ide::SealIde::new();
    if let Some(win) = compositor.window_mut(ide_id) {
        ide.render_to_window(win);
    }

    // Layer 10d: Theorem Viewer
    let viewer = apps::theorem_viewer::TheoremViewer::new();
    if let Some(win) = compositor.window_mut(tv_id) {
        viewer.render_to_window(win);
    }

    // Compose all windows to framebuffer
    compositor.compose(fb);

    serial_println!("[Desktop] {} windows active", compositor.window_count());
    serial_println!("[BOOT] Seal OS desktop ready.");
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
