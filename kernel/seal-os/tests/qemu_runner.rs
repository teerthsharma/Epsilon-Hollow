use std::process::{Command, Stdio};
use std::io::{BufRead, BufReader, Write};
use std::time::{Duration, Instant};

pub struct Qemu {
    cmd: Command,
    timeout: Duration,
}

impl Qemu {
    pub fn new() -> Self {
        let mut cmd = Command::new("qemu-system-x86_64");
        cmd.arg("-m").arg("1G");
        cmd.arg("-bios").arg("OVMF.fd"); // Just an example, maybe not needed if provided via kernel arg
        cmd.arg("-nographic");
        
        Self {
            cmd,
            timeout: Duration::from_secs(30),
        }
    }

    pub fn kernel(mut self, path: &str) -> Self {
        self.cmd.arg("-drive").arg(format!("format=raw,file=fat:rw:{}", path));
        self
    }

    pub fn isa_debug_exit(mut self) -> Self {
        self.cmd.arg("-device").arg("isa-debug-exit,iobase=0x501,iosize=0x04");
        self
    }
    
    pub fn timeout(mut self, secs: u64) -> Self {
        self.timeout = Duration::from_secs(secs);
        self
    }

    pub fn start(&mut self) -> bool {
        self.cmd.stdout(Stdio::piped());
        self.cmd.stderr(Stdio::piped());
        
        println!("[QEMU] Starting...");
        let mut child = self.cmd.spawn().expect("Failed to start QEMU");
        
        let stdout = child.stdout.take().unwrap();
        let reader = BufReader::new(stdout);
        
        let start_time = Instant::now();
        let mut success = false;
        
        for line in reader.lines() {
            if start_time.elapsed() > self.timeout {
                println!("[QEMU] Timeout reached!");
                let _ = child.kill();
                return false;
            }
            
            let line = line.unwrap_or_default();
            println!("[QEMU] {}", line);
            
            if line.contains("ALL TESTS PASSED") {
                success = true;
            }
            if line.contains("TEST FAILED") {
                success = false;
            }
        }
        
        let status = child.wait().expect("Failed to wait on QEMU");
        // isa-debug-exit returns (code << 1) | 1. If success=0, it returns 1.
        if status.code() == Some(1) || success {
            true
        } else {
            false
        }
    }
}

#[test]
fn test_qemu_runner() {
    let mut qemu = Qemu::new()
        // Assuming we build it before running this test
        .kernel("target/x86_64-unknown-uefi/release")
        .isa_debug_exit()
        .timeout(30);
    
    assert!(qemu.start(), "QEMU integration tests failed");
}
