# Epsilon-Hollow v0.5 — The Geometrical Operating System
# All data stored as geometry on S². File moves = O(1) topological surgery.

FROM rust:1.85-bookworm AS builder

WORKDIR /build

# Copy workspace files
COPY Cargo.toml Cargo.lock ./
COPY kernel/aether/aether-verified ./kernel/aether/aether-verified
COPY kernel/epsilon/epsilon ./kernel/epsilon/epsilon

# Build release binary
RUN cargo build --release --manifest-path kernel/epsilon/epsilon/Cargo.toml -p epsilon-os \
    && strip kernel/epsilon/epsilon/target/release/epsilon-os

# Runtime image
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/kernel/epsilon/epsilon/target/release/epsilon-os /usr/local/bin/epsilon-os

# Create two volumes to demonstrate cross-volume O(1) teleportation
RUN mkdir -p /data/vol_a /data/vol_b

VOLUME ["/data/vol_a", "/data/vol_b"]

ENV RUST_LOG=info

ENTRYPOINT ["epsilon-os"]
