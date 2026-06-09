use std::process::Command;

fn main() {
    // UEFI target uses PE/COFF format — no custom linker script needed

    // Shader compilation pipeline
    // Try to compile OpenCL C sources to GCN ISA binaries.  If ROCm is
    // not available on the host, fall back to stub binaries so that
    // `include_bytes!` in shader_binaries.rs still resolves.
    let _out_dir = std::env::var("OUT_DIR").unwrap_or_default();
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default();
    let script_path = std::path::Path::new(&manifest_dir)
        .join("..")
        .join("..")
        .join("scripts")
        .join("compile_shaders.py");
    let shaders_dir = std::path::Path::new(&manifest_dir)
        .join("src")
        .join("drivers")
        .join("gpu")
        .join("shaders");

    // Only run the compile step when the .cl sources are newer than the
    // generated .bin files, or when the .bin files are missing.
    let mut needs_compile = false;
    if let Ok(entries) = std::fs::read_dir(&shaders_dir) {
        for entry in entries {
            if let Ok(entry) = entry {
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("cl") {
                    let bin_path = path.with_extension("bin");
                    let cl_mtime = std::fs::metadata(&path).and_then(|m| m.modified());
                    let bin_mtime = std::fs::metadata(&bin_path).and_then(|m| m.modified());
                    if bin_mtime.is_err() || cl_mtime.unwrap() > bin_mtime.unwrap() {
                        needs_compile = true;
                        break;
                    }
                }
            }
        }
    }

    if needs_compile {
        let has_clang = Command::new("clang")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        let has_llvm_objcopy = Command::new("llvm-objcopy")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);

        if has_clang && has_llvm_objcopy {
            println!("cargo:warning=Compiling GCN shaders with ROCm/LLVM...");
            let status = Command::new("python3")
                .arg(&script_path)
                .arg(&shaders_dir)
                .arg(&shaders_dir)
                .status();
            if let Ok(st) = status {
                if !st.success() {
                    println!(
                        "cargo:warning=Shader compilation failed; build may use stale binaries."
                    );
                }
            }
        } else {
            println!("cargo:warning=ROCm not found; generating stub shader binaries...");
            let status = Command::new("python3")
                .arg(&script_path)
                .arg("--stubs")
                .arg(&shaders_dir)
                .arg(&shaders_dir)
                .status();
            if let Ok(st) = status {
                if !st.success() {
                    println!("cargo:warning=Stub generation failed.");
                }
            }
        }
    }

    // Re-run build.rs if any .cl source changes.
    println!("cargo:rerun-if-changed={}", shaders_dir.display());
    if let Ok(entries) = std::fs::read_dir(&shaders_dir) {
        for entry in entries {
            if let Ok(entry) = entry {
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("cl") {
                    println!("cargo:rerun-if-changed={}", path.display());
                }
            }
        }
    }
}
