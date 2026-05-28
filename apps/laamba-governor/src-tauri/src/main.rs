// Console visible in release so errors are seen
// #![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::fs::OpenOptions;
use std::io::Write;
use std::panic;

fn main() {
    let _ = std::fs::create_dir_all("logs");

    // Log startup
    let _ = OpenOptions::new()
        .create(true)
        .append(true)
        .open("logs/app.log")
        .and_then(|mut f| {
            f.write_all(format!("[START] {}\n", chrono::Local::now()).as_bytes())
        });

    panic::set_hook(Box::new(|info| {
        let msg = format!("[PANIC] {}\n", info);
        let _ = OpenOptions::new()
            .create(true)
            .append(true)
            .open("logs/crash.log")
            .and_then(|mut f| f.write_all(msg.as_bytes()));
        eprintln!("{}", msg);
    }));

    laamba_governor::run()
}
