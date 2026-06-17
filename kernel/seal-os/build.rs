fn main() {
    // UEFI target uses PE/COFF format, so no custom linker script is needed.
    // GPU kernels are embedded only when real checked-in ISA blobs exist.
    // The build never fabricates shader stubs.
    let out_dir = std::env::var("OUT_DIR").unwrap_or_default();
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default();
    let shaders_dir = std::path::Path::new(&manifest_dir)
        .join("src")
        .join("drivers")
        .join("gpu")
        .join("shaders");
    let generated = std::path::Path::new(&out_dir).join("generated_shader_binaries.rs");

    let shader_bins = [
        (
            "VORONOI_ASSIGN_GCN",
            "voronoi_assign",
            "voronoi_assign.bin",
            1u32,
            4u32,
        ),
        ("JL_PROJECT_GCN", "jl_project", "jl_project.bin", 1u32, 4u32),
        (
            "SPECTRAL_STEP_GCN",
            "spectral_step",
            "spectral_step.bin",
            1u32,
            2u32,
        ),
        ("S2_DISTANCE_GCN", "s2_distance", "s2_distance.bin", 2u32, 8u32),
    ];

    let mut statics = String::new();
    let mut entries = String::new();
    for (ident, name, file, min_waves, preferred_waves) in shader_bins {
        let path = shaders_dir.join(file);
        println!("cargo:rerun-if-changed={}", path.display());
        if path.exists() {
            let path_lit = path.display().to_string().replace('\\', "\\\\");
            statics.push_str(&format!(
                "pub static {ident}: &[u8] = include_bytes!(r#\"{path_lit}\"#);\n"
            ));
            entries.push_str(&format!(
                "    KernelMeta {{ name: \"{name}\", binary: {ident}, code_size_bytes: {ident}.len(), min_waves: {min_waves}, preferred_waves: {preferred_waves} }},\n"
            ));
        } else {
            println!(
                "cargo:warning=Real GPU shader binary missing: {}; hardware dispatch will report kernel_not_found.",
                path.display()
            );
        }
    }

    let generated_source = format!("{statics}\npub const KERNELS: &[KernelMeta] = &[\n{entries}];\n");
    std::fs::write(&generated, generated_source).expect("write generated GPU shader binary table");

    // Re-run build.rs if any shader source changes.
    println!("cargo:rerun-if-changed={}", shaders_dir.display());
    if let Ok(entries) = std::fs::read_dir(&shaders_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("cl") {
                println!("cargo:rerun-if-changed={}", path.display());
            }
        }
    }
}
