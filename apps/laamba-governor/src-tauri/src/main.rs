// Console visible in release so errors are seen
// #![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::panic;

fn main() {
    if std::env::args().nth(1).as_deref() == Some("--laamba-native-engine") {
        let args: Vec<String> = std::env::args().skip(2).collect();
        let engine_id = args.first().map(String::as_str).unwrap_or("native");
        let dataset = std::env::var("LAAMBA_DATASET_PATH").ok();
        let mut stdin = String::new();
        let _ = std::io::stdin().read_to_string(&mut stdin);
        let params = if stdin.trim().is_empty() {
            serde_json::Value::Null
        } else {
            serde_json::from_str(&stdin).unwrap_or_else(|_| serde_json::Value::String(stdin))
        };
        let result = laamba_governor::native::engine_command(engine_id, dataset.as_deref(), params);
        println!("{}", result);
        return;
    }

    let _ = std::fs::create_dir_all("logs");

    // Log startup
    let _ = OpenOptions::new()
        .create(true)
        .append(true)
        .open("logs/app.log")
        .and_then(|mut f| f.write_all(format!("[START] {}\n", chrono::Local::now()).as_bytes()));

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
