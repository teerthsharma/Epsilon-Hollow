// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! App launcher — spawns kernel tasks for desktop apps.

use alloc::vec::Vec;
use spin::Mutex;

static LAUNCH_QUEUE: Mutex<Vec<u8>> = Mutex::new(Vec::new());

pub fn launch_app(app_id: u8) {
    crate::serial_println!("[Launcher] launch_app id={}", app_id);
    LAUNCH_QUEUE.lock().push(app_id);
    let entry: fn() = match app_id {
        1 => crate::apps::terminal::main,
        2 => crate::apps::seal_ide::main,
        3 => crate::apps::file_manager::main,
        4 => crate::apps::calculator::main,
        5 => crate::apps::theorem_viewer::main,
        6 => crate::apps::snake::main,
        7 => crate::apps::breakout::main,
        8 => crate::apps::warp_racer::main,
        9 => crate::apps::tensor_viewer::main,
        10 => crate::apps::laamba_governor::main,
        11 => crate::apps::aether_app_host::main,
        _ => return,
    };
    crate::process::scheduler::spawn("app", 5, entry);
}

pub fn poll_launch_queue() -> Option<u8> {
    LAUNCH_QUEUE.lock().pop()
}
