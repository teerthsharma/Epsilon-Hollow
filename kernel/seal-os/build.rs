fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    println!("cargo:rerun-if-changed=linker.ld");
    println!("cargo:rustc-link-arg-bins=-T{}/linker.ld", manifest_dir);
    println!("cargo:rustc-link-arg-bins=-no-pie");
    // Allow R_X86_64_32 relocations from 32-bit boot trampoline (boot.S .code32)
    println!("cargo:rustc-link-arg-bins=-z");
    println!("cargo:rustc-link-arg-bins=notext");
}
