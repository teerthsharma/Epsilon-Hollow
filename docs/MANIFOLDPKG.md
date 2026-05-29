# ManifoldPkg — Native Package Manager

ManifoldPkg is Seal OS's package manager. Packages are ManifoldFS inodes; dependencies are resolved via Voronoi cell lookup.

## Usage

```
install topo-tensor        # Install a native Aether package
remove topo-tensor         # Remove a package
packages                   # List installed packages
update                     # Update all packages
```

## Architecture

```
install topo-tensor
    |
    v
ManifoldPkg resolves "topo-tensor" via Voronoi lookup
    |
    v
Package = ManifoldFS inode:
  - manifest (name, version, deps, carrier)
  - payload (compiled binary or source)
  - carrier type (aether, rust)
    |
    v
Carrier compiles/interprets the package
    |
    v
Installed into /packages/<name>/ in ManifoldFS
```

## Carriers

Each carrier bridges native Seal OS code into ManifoldFS:

| Carrier | Language | Status |
|---|---|---|
| `aether` | Aether-Lang (native) | Active |
| `rust` | Rust crates | Active |

## Package Manifest

```toml
[package]
name = "topo-tensor"
version = "1.0.0"
carrier = "aether"
description = "Topological tensor movement"
deps = ["math-core"]
```

## Dependency Resolution

Dependencies are mapped to Voronoi cells on S^2. Resolution first narrows to a geometric neighborhood; an O(1) dependency-discovery claim requires a capped bucket or benchmark gate.
