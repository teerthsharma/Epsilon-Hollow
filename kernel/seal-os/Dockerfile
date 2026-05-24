# Seal OS — Build & Smoke-Test Environment
# Replicates the CI kernel-build + kernel-image + kernel-qemu-smoke pipeline.
#
# Build:
#   docker build -f kernel/seal-os/Dockerfile -t seal-os-smoke ..
# Run smoke test:
#   docker run --rm seal-os-smoke
# Run interactively:
#   docker run --rm -it seal-os-smoke bash

FROM rust:1.85-bookworm AS builder

# Install nightly toolchain + components required for UEFI bare-metal build
RUN rustup toolchain install nightly --component rust-src,llvm-tools-preview
RUN rustup target add x86_64-unknown-uefi --toolchain nightly

WORKDIR /build

# Copy workspace root manifest so path dependencies can resolve workspace keys
COPY Cargo.toml Cargo.lock ./

# Copy dependencies (must match paths in kernel/seal-os/Cargo.toml)
COPY kernel/epsilon/epsilon/crates/aether-core /build/kernel/epsilon/epsilon/crates/aether-core
COPY kernel/aether/Aether-Lang/crates/aether-lang /build/kernel/aether/Aether-Lang/crates/aether-lang

# Copy seal-os kernel
COPY kernel/seal-os /build/kernel/seal-os

# Build kernel (release)
RUN cd /build/kernel/seal-os && cargo +nightly build --release

# Copy seal-mkimage tool
COPY kernel/seal-mkimage /build/kernel/seal-mkimage

# Build mkimage tool (stable)
RUN cd /build/kernel/seal-mkimage && cargo +stable build --release

# Create UEFI disk image
RUN cd /build/kernel/seal-mkimage && cargo +stable run --release

# Runtime stage with QEMU + OVMF
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    qemu-system-x86 \
    ovmf \
    && rm -rf /var/lib/apt/lists/*

# Copy the built disk image from builder
COPY --from=builder /build/kernel/seal-os/target/x86_64-unknown-uefi/release/seal-os.img /seal-os.img

# Verify image exists
RUN test -f /seal-os.img && echo "Disk image ready"

ENV IMG=/seal-os.img
ENV OVMF=/usr/share/OVMF/OVMF_CODE_4M.fd

# Default: run QEMU smoke test (non-interactive, serial to stdout)
ENTRYPOINT ["qemu-system-x86_64", \
    "-machine", "q35", \
    "-drive", "if=pflash,format=raw,readonly=on,file=/usr/share/OVMF/OVMF_CODE_4M.fd", \
    "-drive", "file=/seal-os.img,format=raw", \
    "-nographic", \
    "-m", "4G", \
    "-no-reboot", \
    "-no-shutdown"]
