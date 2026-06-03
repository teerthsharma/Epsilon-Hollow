# ManifoldPkg — Native Package Manager

ManifoldPkg is Seal OS's package manager. `.eph` packages carry a manifest, signature bytes, and file payloads. The boot proof parses an embedded `.eph`, installs it through the real package manager, extracts a file into VFS, lists the manifest, removes the registry entry, and emits `[ManifoldPkg] proof`.

## Usage

```
install topo-tensor.eph    # Install a local .eph package from the current shell folder
install topo-tensor        # Fetch a signed .eph from the configured registry when networking is available
remove topo-tensor         # Remove a package
packages                   # List installed packages
update                     # Report registry refresh status
```

## Architecture

```
install topo-tensor.eph
    |
    v
SealShell reads raw .eph bytes from ManifoldFS
    |
    v
ManifoldPkg parses the package:
  - manifest (name, version, deps, carrier)
  - signature bytes
  - file payloads
    |
    v
Dependency resolver checks installed package graph
    |
    v
Files are extracted through VFS and the manifest is registered
```

## Carriers

Carrier metadata records the intended runtime for a package. The package proof covers parse/install/extract/list/remove, not executing arbitrary package entrypoints.

| Carrier | Language | Status |
|---|---|---|
| `aether` | Aether-Lang (native) | Manifest carrier |
| `rust` | Rust crates | Manifest carrier |

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

## Proof Gate

`seal-mkimage --check-theorem-log` requires a unique marker:

```text
[ManifoldPkg] proof version=1 source=embedded_eph parse=ok install=ok extract=ok list=ok remove=ok ... metadata_only=0 ... result=pass
```

Remote registry fixtures and signed release-package gates are still pending.
