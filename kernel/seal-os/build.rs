fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    println!("cargo:rerun-if-changed=linker.ld");
    println!("cargo:rustc-link-arg=-T{}/linker.ld", manifest_dir);
}
