# ManifoldPkg — Native Package Manager

ManifoldPkg is Seal OS's package manager. Packages are ManifoldFS inodes; dependencies are resolved via Voronoi cell lookup.

## Usage

```
install numpy        # Install a package
remove numpy         # Remove a package
packages             # List installed packages
update               # Update all packages
```

## Architecture

```
install numpy
    |
    v
ManifoldPkg resolves "numpy" via Voronoi lookup
    |
    v
Package = ManifoldFS inode:
  - manifest (name, version, deps, carrier)
  - payload (compiled binary or source)
  - carrier type (aether, rust, python, c, js)
    |
    v
Carrier compiles/interprets the package
    |
    v
Installed into /packages/<name>/ in ManifoldFS
```

## Carriers

Each carrier bridges a language ecosystem into Seal OS:

| Carrier | Language | Status |
|---|---|---|
| `aether` | Aether-Lang (native) | Active |
| `rust` | Rust crates | Active |
| `python` | Python (pip) | Active |
| `c` | C libraries | Stub (v0.2.0) |
| `js` | JavaScript | Stub (v0.2.0) |

## Package Manifest

```toml
[package]
name = "numpy"
version = "1.0.0"
carrier = "python"
description = "Numerical computing"
deps = ["math-core"]
```

## Dependency Resolution

Dependencies are mapped to Voronoi cells on S^2. Resolution is a geometric nearest-neighbor lookup, giving O(1) amortized dependency discovery for packages within the same topological neighborhood.
