#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
compile_error!("ubuntu-alloc-bench requires x86/x86_64 rdtsc cycle counters");

use std::fs;

const DEFAULT_ITERS: usize = 64;
const PAGE_BYTES: usize = 4096;

fn main() {
    let iterations = requested_iterations().max(DEFAULT_ITERS);
    let mut samples = Vec::with_capacity(iterations);
    let mut ok = 0usize;

    for seed in 0..iterations {
        let start = read_cycles();
        {
            let mut page = vec![0u8; PAGE_BYTES].into_boxed_slice();
            page[0] = seed.to_le_bytes()[0];
            page[PAGE_BYTES - 1] = seed.wrapping_mul(31).to_le_bytes()[0];
            std::hint::black_box(page.as_mut_ptr());
        }
        let end = read_cycles();
        samples.push(end.saturating_sub(start));
        ok += 1;
    }

    samples.sort_unstable();
    let p50 = percentile(&samples, 50);
    let p95 = percentile(&samples, 95);
    let max = samples.last().copied().unwrap_or(0);
    let os_release = detect_os_release();

    println!(
        "[UBUNTU-BENCH] alloc-frame os={} version_id={} kernel={} iterations={iterations} ok={ok} bytes={PAGE_BYTES} backend=rust-std-box-page-touch-drop clock=rdtsc p50_cycles={p50} p95_cycles={p95} max_cycles={max}",
        os_release.id, os_release.version_id, os_release.kernel
    );
}

fn requested_iterations() -> usize {
    std::env::args()
        .nth(1)
        .and_then(|arg| arg.parse::<usize>().ok())
        .unwrap_or(DEFAULT_ITERS)
}

fn percentile(samples: &[u64], percent: usize) -> u64 {
    if samples.is_empty() {
        return 0;
    }
    let last = samples.len() - 1;
    let idx = (last * percent) / 100;
    samples[idx]
}

struct OsRelease {
    id: String,
    version_id: String,
    kernel: String,
}

fn detect_os_release() -> OsRelease {
    if cfg!(target_os = "windows") {
        return OsRelease {
            id: String::from("windows"),
            version_id: String::from("unknown"),
            kernel: String::from("windows"),
        };
    }
    let Ok(text) = fs::read_to_string("/etc/os-release") else {
        return OsRelease {
            id: String::from("unknown"),
            version_id: String::from("unknown"),
            kernel: kernel_release(),
        };
    };
    let mut id = String::from("unknown");
    let mut version_id = String::from("unknown");
    for line in text.lines() {
        if let Some(value) = line.strip_prefix("ID=") {
            id = value.trim_matches('"').to_ascii_lowercase();
        }
        if let Some(value) = line.strip_prefix("VERSION_ID=") {
            version_id = value.trim_matches('"').to_ascii_lowercase();
        }
    }
    OsRelease {
        id,
        version_id,
        kernel: kernel_release(),
    }
}

fn kernel_release() -> String {
    fs::read_to_string("/proc/sys/kernel/osrelease").map_or_else(
        |_| String::from("unknown"),
        |text| metric_token(text.trim()),
    )
}

fn metric_token(value: &str) -> String {
    let mut token = String::new();
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '+' | '-') {
            token.push(ch.to_ascii_lowercase());
        } else {
            token.push('_');
        }
    }
    if token.is_empty() {
        String::from("unknown")
    } else {
        token
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
fn read_cycles() -> u64 {
    // SAFETY: `_rdtsc` is a serializable CPU counter read on the current x86_64 core.
    unsafe { core::arch::x86_64::_rdtsc() }
}

#[cfg(target_arch = "x86")]
#[allow(unsafe_code)]
fn read_cycles() -> u64 {
    // SAFETY: `_rdtsc` is a serializable CPU counter read on the current x86 core.
    unsafe { core::arch::x86::_rdtsc() }
}
