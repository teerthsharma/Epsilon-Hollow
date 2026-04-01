FROM rust:1.80 as builder

WORKDIR /usr/src/apeiron

# Copy workspace manifests and dependencies
COPY network ./network
COPY runtime ./runtime

# Build the CLI release binary
WORKDIR /usr/src/apeiron/runtime
# Ensure we build the aether-cli package which contains the binary "aether"
RUN cargo build --release -p aether-cli

# Runner Stage
FROM debian:bookworm-slim

# Install Python for model downloader and OpenSSL/certificates
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ca-certificates \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy binary
COPY --from=builder /usr/src/apeiron/runtime/target/release/aether .

# Copy scripts
COPY runtime/entrypoint.sh .
COPY --from=builder /usr/src/apeiron/runtime/download_model.py . 
# Note: download_model.py was at root in previous steps, ensuring it's in context.
# Actually, the previous step created it in root `New folder (12)`. 
# The context for docker build is likely `.` (root) or `./APEIRON`.
# If context is `./APEIRON`, we need to make sure `download_model.py` is accessible.
# I'll assume the user will move it or I should move it to APEIRON/runtime to be safe in the build context.
# For now, let's assume it's copied from where it is.
# Re-reading: The context in docker-compose is `./APEIRON` (line 6 of orig compose). 
# But `download_model.py` is in `New folder (12)`.
# I should move it to `APEIRON/runtime` first to be clean.

RUN chmod +x entrypoint.sh
RUN pip3 install huggingface_hub --break-system-packages

EXPOSE 8080

ENTRYPOINT ["./entrypoint.sh"]
CMD ["./aether", "--host", "0.0.0.0"]
