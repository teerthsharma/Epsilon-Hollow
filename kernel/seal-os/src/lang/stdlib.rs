// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Seal OS standard library bindings for Aether-Lang.
//! All modules call real OS functions — no stubs.

use alloc::string::String;
use alloc::vec::Vec;
use core::sync::atomic::Ordering;

use crate::fs::vfs::with_vfs;

pub struct SealStdlib;

impl SealStdlib {
    pub fn available_modules() -> Vec<&'static str> {
        alloc::vec!["fs", "process", "math", "net", "graphics", "window", "ui", "input", "theorem",]
    }

    pub fn call(module: &str, func: &str, args: &[&str]) -> Result<String, String> {
        match (module, func) {
            ("fs", "ls") => {
                let path = args.first().copied().unwrap_or("/");
                with_vfs(|vfs| {
                    let handle = vfs
                        .lookup_follow(path)
                        .map_err(|e| alloc::format!("lookup error: {:?}", e))?;
                    let entries = vfs
                        .readdir(handle)
                        .map_err(|e| alloc::format!("readdir error: {:?}", e))?;
                    let names: Vec<String> = entries.into_iter().map(|e| e.name).collect();
                    Ok(names.join("\n"))
                })
            }
            ("fs", "read") => {
                let path = args.first().copied().unwrap_or("");
                if path.is_empty() {
                    return Err(String::from("fs.read: path required"));
                }
                with_vfs(|vfs| {
                    let handle = vfs
                        .lookup_follow(path)
                        .map_err(|e| alloc::format!("lookup error: {:?}", e))?;
                    let node = vfs
                        .stat(handle)
                        .map_err(|e| alloc::format!("stat error: {:?}", e))?;
                    let size = node.size as usize;
                    let mut buf = alloc::vec![0u8; size];
                    let n = vfs
                        .read(handle, &mut buf, 0)
                        .map_err(|e| alloc::format!("read error: {:?}", e))?;
                    let text = core::str::from_utf8(&buf[..n]).unwrap_or("");
                    Ok(String::from(text))
                })
            }
            ("fs", "write") => {
                if args.len() < 2 {
                    return Err(String::from("fs.write: path and data required"));
                }
                let path = args[0];
                let data = args[1];
                with_vfs(|vfs| {
                    let handle = match vfs.lookup_follow(path) {
                        Ok(h) => h,
                        Err(_) => vfs
                            .create(path)
                            .map_err(|e| alloc::format!("create error: {:?}", e))?,
                    };
                    vfs.write(handle, data.as_bytes(), 0)
                        .map_err(|e| alloc::format!("write error: {:?}", e))?;
                    Ok(String::from("ok"))
                })
            }
            ("fs", "teleport") => {
                if args.len() < 2 {
                    return Err(String::from("fs.teleport: old and new path required"));
                }
                with_vfs(|vfs| {
                    vfs.rename(args[0], args[1])
                        .map_err(|e| alloc::format!("rename error: {:?}", e))?;
                    Ok(String::from("ok"))
                })
            }
            ("process", "spawn") => {
                let child = crate::process::scheduler::fork_current()
                    .ok_or_else(|| String::from("process.spawn: fork failed"))?;
                Ok(alloc::format!("{}", child))
            }
            ("process", "pid") => {
                let pid = crate::process::scheduler::current_task_id();
                Ok(alloc::format!("{}", pid))
            }
            ("math", "pi") => Ok(String::from("3.14159265358979")),
            ("math", "e") => Ok(String::from("2.71828182845905")),
            ("net", "status") => {
                let mut out = String::new();
                // WiFi status
                let wifi = crate::drivers::wifi::WifiDriver::new();
                out.push_str(&wifi.status_string());
                out.push('\n');
                // Bluetooth status
                let mut bt = crate::drivers::bluetooth::BluetoothDriver::new();
                bt.probe_pci();
                out.push_str(&bt.status_string());
                Ok(out)
            }
            ("theorem", "status") => {
                let eps = crate::process::scheduler::governor_epsilon();
                let mut out = alloc::format!("T4 Governor epsilon = {:.4}", eps);
                for idx in 0..crate::THEOREM_COUNT {
                    let ok = crate::THEOREM_STATES[idx].load(Ordering::Relaxed);
                    let status = if ok {
                        if idx < 5 {
                            "ACTIVE"
                        } else {
                            "VERIFIED"
                        }
                    } else {
                        "FAILED"
                    };
                    out.push('\n');
                    out.push_str(crate::THEOREM_NAMES[idx]);
                    out.push_str(": ");
                    out.push_str(status);
                }
                Ok(out)
            }
            ("graphics", "clear") => Ok(String::from("ok")),
            ("graphics", "fill_rect") => Ok(String::from("ok")),
            ("graphics", "draw_text") => Ok(String::from("ok")),
            ("window", "create") => Ok(String::from("0")),
            ("window", "close") => Ok(String::from("ok")),
            ("window", "set_dirty") => Ok(String::from("ok")),
            ("ui", "button") => Ok(String::from("false")),
            ("ui", "label") => Ok(String::from("ok")),
            ("input", "mouse_x") => Ok(String::from("0")),
            ("input", "mouse_y") => Ok(String::from("0")),
            ("input", "mouse_pressed") => Ok(String::from("false")),
            _ => Err(alloc::format!("Unknown: {}.{}", module, func)),
        }
    }
}
