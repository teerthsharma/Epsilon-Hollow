# Seal OS Build + Run Environment
# Builds the kernel, creates ISO, runs in QEMU with X11 forwarding

FROM rust:latest AS builder

# Install nightly + bare-metal target
RUN rustup toolchain install nightly --component rust-src llvm-tools-preview

# Install GRUB and xorriso for ISO creation
RUN apt-get update && apt-get install -y --no-install-recommends \
    grub-pc-bin \
    grub-common \
    xorriso \
    mtools \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy aether-core dependency
COPY kernel/epsilon/epsilon/crates/aether-core /build/kernel/epsilon/epsilon/crates/aether-core

# Copy seal-os kernel
COPY kernel/seal-os /build/kernel/seal-os

# Build kernel
WORKDIR /build/kernel/seal-os
RUN cargo +nightly build --release

# Create ISO
RUN mkdir -p iso/boot/grub \
    && cp target/x86_64-unknown-none/release/seal-os iso/boot/kernel.bin \
    && cp boot/grub/grub.cfg iso/boot/grub/grub.cfg \
    && grub-mkrescue -o /build/seal-os.iso iso/

# Runtime image with QEMU
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    qemu-system-x86 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/seal-os.iso /seal-os.iso

ENV DISPLAY=:0

ENTRYPOINT ["qemu-system-x86_64", \
    "-cdrom", "/seal-os.iso", \
    "-serial", "stdio", \
    "-m", "512M", \
    "-vga", "std", \
    "-no-reboot", \
    "-no-shutdown"]
