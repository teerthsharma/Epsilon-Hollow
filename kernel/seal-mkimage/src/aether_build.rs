// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

#![allow(dead_code)]

//! Aether Build Driver — Stream 6 Scaffolding
//!
//! Invokes the Aether bootstrap compiler (`aether-cli`) to compile kernel
//! Aether sources (`.ae` / `.aether`) into object images suitable for
//! embedding in the Seal OS boot image.
//!
//! This is **Phase 0** of the self-hosting roadmap: the build driver itself is
//! written in Rust and shells out to the Aether bootstrap compiler. Future
//! phases will replace the Rust shim with Aether-native compilation.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Default directories scanned for Aether kernel sources.
pub const DEFAULT_AETHER_SOURCE_DIRS: &[&str] = &[
    "kernel/aether/Aether-Lang/examples",
    "kernel/aether/aether-link/src",
    "apps/laamba-governor/native",
];

/// Extension patterns recognised as Aether source.
pub const AETHER_EXTENSIONS: &[&str] = &["ae", "aether"];

/// Output subdirectory under `target/` where Aether object images are staged.
pub const AETHER_BUILD_DIR: &str = "target/aether-build";

/// Metadata written alongside each compiled object image.
#[derive(Debug, Clone)]
pub struct AetherObjectMeta {
    pub source_path: PathBuf,
    pub source_hash: String,
    pub compiled_at: String,
    pub bootstrap_version: String,
}

/// A single compiled Aether object image.
pub struct AetherObject {
    pub meta: AetherObjectMeta,
    /// Bytecode payload (TitanVM OpCode sequence) or placeholder for Phase 1.
    pub payload: Vec<u8>,
}

/// Build configuration for the Aether driver.
pub struct AetherBuildConfig {
    pub source_dirs: Vec<PathBuf>,
    pub output_dir: PathBuf,
    pub project_root: PathBuf,
    /// When `true`, run `aether check` before accepting a source file.
    pub verify_syntax: bool,
}

impl Default for AetherBuildConfig {
    fn default() -> Self {
        let project_root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        Self {
            source_dirs: DEFAULT_AETHER_SOURCE_DIRS
                .iter()
                .map(|p| project_root.join(p))
                .collect(),
            output_dir: project_root.join(AETHER_BUILD_DIR),
            project_root,
            verify_syntax: true,
        }
    }
}

/// Discover all Aether source files under the configured directories.
pub fn discover_sources(config: &AetherBuildConfig) -> Vec<PathBuf> {
    let mut sources = Vec::new();
    for dir in &config.source_dirs {
        if !dir.is_dir() {
            continue;
        }
        let Ok(entries) = fs::read_dir(dir) else {
            continue;
        };
        for entry in entries.filter_map(Result::ok) {
            let path = entry.path();
            if let Some(ext) = path.extension() {
                let ext = ext.to_string_lossy();
                if AETHER_EXTENSIONS.contains(&ext.as_ref()) && path.is_file() {
                    sources.push(path);
                }
            }
        }
    }
    sources
}

/// Run the Aether bootstrap compiler (`cargo run -p aether-cli`) in check mode.
/// Returns `Ok(())` if syntax is valid.
pub fn bootstrap_check(project_root: &Path, source: &Path) -> Result<(), String> {
    let output = Command::new("cargo")
        .arg("run")
        .arg("-p")
        .arg("aether-cli")
        .arg("--")
        .arg("check")
        .arg(source)
        .current_dir(project_root)
        .output()
        .map_err(|e| format!("failed to spawn aether-cli: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("aether check failed for {}: {}", source.display(), stderr));
    }
    Ok(())
}

/// Compile a single Aether source into an object image.
///
/// Phase 0 implementation:
/// 1. Verify syntax via bootstrap compiler.
/// 2. Compute source hash for incremental-build tracking.
/// 3. Write a placeholder `.aeo` payload (full bytecode serialization in Phase 1).
pub fn compile_source(
    config: &AetherBuildConfig,
    source: &Path,
) -> Result<AetherObject, String> {
    if config.verify_syntax {
        bootstrap_check(&config.project_root, source)?;
    }

    let source_bytes = fs::read(source).map_err(|e| format!("read error: {e}"))?;
    let source_hash = md5_hash(&source_bytes)
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect::<String>();

    // TODO(Phase 1): Invoke Aether bootstrap compiler in compile-to-bytecode mode
    // and serialize the TitanVM OpCode vector into the payload.
    let payload = build_placeholder_payload(source, &source_hash);

    let meta = AetherObjectMeta {
        source_path: source.to_path_buf(),
        source_hash,
        compiled_at: iso_timestamp(),
        bootstrap_version: String::from("0.1.0"),
    };

    Ok(AetherObject { meta, payload })
}

/// Compile all discovered sources and write object images to the output dir.
pub fn build_all(config: &AetherBuildConfig) -> Result<HashMap<PathBuf, AetherObject>, String> {
    fs::create_dir_all(&config.output_dir)
        .map_err(|e| format!("cannot create output dir: {e}"))?;

    let sources = discover_sources(config);
    if sources.is_empty() {
        eprintln!("[aether-build] warning: no Aether sources found in {:?}", config.source_dirs);
    }

    let mut artifacts = HashMap::new();
    for src in &sources {
        let obj = compile_source(config, src)?;
        let out_name = format!(
            "{}.aeo",
            src.file_stem().unwrap_or_default().to_string_lossy()
        );
        let out_path = config.output_dir.join(&out_name);
        fs::write(&out_path, &obj.payload)
            .map_err(|e| format!("write error for {}: {e}", out_path.display()))?;
        println!(
            "[aether-build] {} → {} (hash {})",
            src.display(),
            out_path.display(),
            &obj.meta.source_hash[..8]
        );
        artifacts.insert(src.clone(), obj);
    }

    // Write build manifest for dependency tracking.
    write_manifest(config, &artifacts)?;
    Ok(artifacts)
}

/// Write a JSON manifest describing the current build state.
fn write_manifest(
    config: &AetherBuildConfig,
    artifacts: &HashMap<PathBuf, AetherObject>,
) -> Result<(), String> {
    let manifest_path = config.output_dir.join("aether-build-manifest.json");
    let mut lines = vec![
        "{".to_string(),
        format!("  \"version\": 1,"),
        format!("  \"timestamp\": \"{}\",", iso_timestamp()),
        format!("  \"phases\": {{"),
        format!("    \"current\": 0,"),
        format!("    \"description\": \"Rust driver + Aether bootstrap compiler\","),
        format!("  }},"),
        format!("  \"artifacts\": ["),
    ];

    let mut first = true;
    for (src, obj) in artifacts {
        let comma = if first { "" } else { "," };
        first = false;
        lines.push(format!(
            "{}    {{ \"source\": \"{}\", \"hash\": \"{}\", \"size\": {} }}",
            comma,
            src.display(),
            obj.meta.source_hash,
            obj.payload.len()
        ));
    }

    lines.push("  ]".to_string());
    lines.push("}".to_string());

    fs::write(&manifest_path, lines.join("\n"))
        .map_err(|e| format!("manifest write error: {e}"))?;
    println!("[aether-build] manifest → {}", manifest_path.display());
    Ok(())
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn md5_hash(data: &[u8]) -> [u8; 16] {
    // Simple MD5-like digest using djb2 + checksum for scaffolding.
    // Replace with a proper md5 crate or ring::digest in production.
    let mut h: u32 = 5381;
    let mut checksum: u32 = 0;
    for &b in data {
        h = h.wrapping_mul(33).wrapping_add(b as u32);
        checksum = checksum.wrapping_add(b as u32);
    }
    let mut out = [0u8; 16];
    out[0..4].copy_from_slice(&h.to_le_bytes());
    out[4..8].copy_from_slice(&checksum.to_le_bytes());
    out[8..12].copy_from_slice(&(data.len() as u32).to_le_bytes());
    // Fill remainder with a simple xor-fold
    let mut xor_fold: u32 = 0;
    for chunk in data.chunks(4) {
        let mut word = [0u8; 4];
        word[..chunk.len()].copy_from_slice(chunk);
        xor_fold ^= u32::from_le_bytes(word);
    }
    out[12..16].copy_from_slice(&xor_fold.to_le_bytes());
    out
}

fn iso_timestamp() -> String {
    use std::time::SystemTime;
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    // Epoch seconds as placeholder timestamp; replace with chrono in production.
    format!("{}", now)
}

/// Compute a placeholder payload containing a minimal header + source hash.
fn build_placeholder_payload(source: &Path, hash: &str) -> Vec<u8> {
    let header = b"AEO\x00"; // Aether Object magic
    let name = source.file_stem().unwrap_or_default().to_string_lossy();
    let mut payload = Vec::new();
    payload.extend_from_slice(header);
    payload.extend_from_slice(&(name.len() as u32).to_le_bytes());
    payload.extend_from_slice(name.as_bytes());
    payload.extend_from_slice(&(hash.len() as u32).to_le_bytes());
    payload.extend_from_slice(hash.as_bytes());
    payload.extend_from_slice(b"\x00\x00\x00\x00"); // reserved flags
    payload
}
