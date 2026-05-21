// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Seal OS standard library bindings for Aether-Lang (fs, process, net stubs).

use alloc::string::String;
use alloc::vec::Vec;

pub struct SealStdlib;

impl SealStdlib {
    pub fn available_modules() -> Vec<&'static str> {
        alloc::vec![
            "fs",
            "process",
            "math",
            "net",
            "graphics",
            "theorem",
        ]
    }

    pub fn call(module: &str, func: &str) -> Result<String, String> {
        match (module, func) {
            ("fs", "ls") => Err(String::from("fs.ls — not implemented")),
            ("fs", "read") => Err(String::from("fs.read — not implemented")),
            ("fs", "write") => Err(String::from("fs.write — not implemented")),
            ("fs", "teleport") => Err(String::from("fs.teleport — not implemented")),
            ("process", "spawn") => Err(String::from("process.spawn — not implemented")),
            ("process", "pid") => Err(String::from("process.pid — not implemented")),
            ("math", "pi") => Ok(String::from("3.14159265358979")),
            ("math", "e") => Ok(String::from("2.71828182845905")),
            ("net", "status") => Err(String::from("net.status — not implemented")),
            ("theorem", "status") => Err(String::from("theorem.status — not implemented")),
            _ => Err(alloc::format!("Unknown: {}.{}", module, func)),
        }
    }
}
