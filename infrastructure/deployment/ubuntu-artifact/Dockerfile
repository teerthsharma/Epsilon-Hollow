# Ubuntu 24.04 benchmark artifact builder
# Produces a bootable UEFI ISO that auto-runs 5 honest-comparison benchmarks.

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libc6-dev \
    linux-image-generic \
    grub-pc-bin \
    grub-efi-amd64-bin \
    xorriso \
    mtools \
    dosfstools \
    busybox-static \
    cpio \
    gzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY workloads.c bench_runner.sh build.sh ./

RUN chmod +x build.sh bench_runner.sh && ./build.sh

# The ISO is emitted to /build/ubuntu-bench.iso
