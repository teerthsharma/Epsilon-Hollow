// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Aether-Lang runtime bridge — connects SealShell to the AEGIS interpreter.
//! Registers kernel callbacks so Aether-Lang `fs`, `process`, and `net`
//! modules call real OS functions instead of returning stubs.

pub mod stdlib;

use alloc::format;
use alloc::string::String;
use core::slice;

use aether_lang::HwCallbacks;

pub struct AetherRuntime {
    interpreter: aether_lang::Interpreter,
}

impl AetherRuntime {
    pub fn new() -> Self {
        unsafe {
            aether_lang::register_kernel_callbacks(
                aether_lang::FsCallbacks {
                    read: Some(aether_fs_read),
                    write: Some(aether_fs_write),
                    exists: Some(aether_fs_exists),
                    mkdir: Some(aether_fs_mkdir),
                },
                aether_lang::ProcessCallbacks {
                    pid: Some(aether_process_pid),
                    exit: Some(aether_process_exit),
                },
                aether_lang::NetCallbacks {
                    local_ip: Some(aether_net_local_ip),
                    has_nic: Some(aether_net_has_nic),
                },
                aether_lang::HwCallbacks {
                    mmio_read32: Some(aether_mmio_read32),
                    mmio_write32: Some(aether_mmio_write32),
                    pci_read_config: Some(aether_pci_read_config),
                },
            );
        }
        Self {
            interpreter: aether_lang::Interpreter::new(),
        }
    }

    pub fn execute_source(&mut self, source: &str) -> Result<String, String> {
        let mut parser = aether_lang::Parser::new(source);
        let program = parser.parse().map_err(|e| format!("parse error: {}", e.message))?;
        let value = self.interpreter.execute(&program).map_err(|e| format!("{:?}", e))?;
        Ok(format!("{:?}", value))
    }

    pub fn execute_file(&mut self, name: &str, source: &str) -> Result<String, String> {
        self.execute_source(source)
            .map_err(|e| format!("in '{}': {}", name, e))
    }

    pub fn is_ready(&self) -> bool {
        true
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Kernel callbacks — called by Aether-Lang interpreter via function pointers.
// ═══════════════════════════════════════════════════════════════════════════════

extern "C" fn aether_fs_read(
    path: *const u8,
    path_len: usize,
    buf: *mut u8,
    buf_len: usize,
) -> isize {
    let path = unsafe { core::str::from_utf8_unchecked(slice::from_raw_parts(path, path_len)) };
    crate::fs::vfs::with_vfs(|vfs| {
        let handle = match vfs.lookup(path) {
            Ok(h) => h,
            Err(_) => return -1,
        };
        let mut tmp = vec![0u8; buf_len];
        match vfs.read(handle, &mut tmp, 0) {
            Ok(n) => {
                let n = n.min(buf_len);
                unsafe {
                    core::ptr::copy_nonoverlapping(tmp.as_ptr(), buf, n);
                }
                n as isize
            }
            Err(_) => -1,
        }
    })
}

extern "C" fn aether_fs_write(
    path: *const u8,
    path_len: usize,
    data: *const u8,
    data_len: usize,
) -> isize {
    let path = unsafe { core::str::from_utf8_unchecked(slice::from_raw_parts(path, path_len)) };
    crate::fs::vfs::with_vfs(|vfs| {
        let handle = match vfs.lookup(path) {
            Ok(h) => h,
            Err(_) => return -1,
        };
        let buf = unsafe { slice::from_raw_parts(data, data_len) };
        match vfs.write(handle, buf, 0) {
            Ok(n) => n as isize,
            Err(_) => -1,
        }
    })
}

extern "C" fn aether_fs_exists(path: *const u8, path_len: usize) -> bool {
    let path = unsafe { core::str::from_utf8_unchecked(slice::from_raw_parts(path, path_len)) };
    crate::fs::vfs::with_vfs(|vfs| vfs.lookup(path).is_ok())
}

extern "C" fn aether_fs_mkdir(path: *const u8, path_len: usize) -> isize {
    let path = unsafe { core::str::from_utf8_unchecked(slice::from_raw_parts(path, path_len)) };
    crate::fs::vfs::with_vfs(|vfs| match vfs.mkdir(path) {
        Ok(_) => 0,
        Err(_) => -1,
    })
}

extern "C" fn aether_process_pid() -> u64 {
    crate::process::scheduler::current_task_id()
}

extern "C" fn aether_process_exit(_code: i32) -> ! {
    loop {
        core::hint::spin_loop();
    }
}

extern "C" fn aether_net_local_ip(buf: *mut u8, buf_len: usize) -> isize {
    let ip = crate::net::local_ip();
    let s = format!("{}.{}.{}.{}", ip[0], ip[1], ip[2], ip[3]);
    let bytes = s.as_bytes();
    let n = bytes.len().min(buf_len);
    unsafe {
        core::ptr::copy_nonoverlapping(bytes.as_ptr(), buf, n);
    }
    n as isize
}

extern "C" fn aether_net_has_nic() -> bool {
    crate::drivers::net::has_nic()
}

extern "C" fn aether_mmio_read32(addr: u64) -> u32 {
    unsafe { core::ptr::read_volatile(addr as *const u32) }
}

extern "C" fn aether_mmio_write32(addr: u64, val: u32) {
    unsafe { core::ptr::write_volatile(addr as *mut u32, val) }
}

extern "C" fn aether_pci_read_config(bus: u8, slot: u8, func: u8, offset: u8) -> u32 {
    crate::drivers::pci::pci_read32(bus, slot, func, offset)
}
