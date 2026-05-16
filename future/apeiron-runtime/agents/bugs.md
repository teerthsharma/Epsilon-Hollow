# System Bug Report & Technical Debt
**Date**: 2026-01-26
**Component**: Apeiron Runtime & Agent Architecture

## 🔴 Critical Issues

### 1. Hardware Mismatch (GPU vs CPU)
- **File**: `apeiron-runtime/src/llm_engine.rs` vs `docker-compose.yml`
- **Description**: The Rust code explicitly forces `Device::Cpu` in `LLMEngine::new()`. However, `docker-compose.yml` requests NVIDIA GPU resources.
- **Impact**: The container might fail to start on non-NVIDIA machines due to the reservation, or simply waste resources. The code will **not** utilize the GPU even if available.
- **Fix Required**: Make `Device` configurable via environment variable (`AETHER_DEVICE=cuda/cpu`).

### 2. Runtime Stability (Panics)
- **File**: `apeiron-runtime/src/main.rs`, `llm_engine.rs`
- **Description**: Heavy use of `.unwrap()` and `expect()`.
    - If `start_memory.json` is malformed -> Panic.
    - If Model generation fails (e.g., OOM) -> Panic.
- **Impact**: The entire `apeiron-kernel` service will crash and restart loops if an edge case is hit.
- **Fix Required**: Implement proper `Result<T, E>` error handling and graceful degradation (e.g., fallback to rule-based if LLM fails).

## ⚠️ Logic & Heuristics

### 3. Naive Correction Trigger
- **File**: `apeiron-runtime/src/main.rs`
- **Description**: The detection logic checks for substrings: `["actually", "no,", "wrong"]`.
    - **False Positive**: User says "No, I agree with you." -> Detected as Correction -> Red Pulse -> System rewrites memory.
- **Impact**: Data poisoning of the Akashic Records with incorrect "facts".
- **Fix Required**: Use a lightweight classifier (BERT/Zero-shot) or prompt the LLM to classify the intent ("Is this a correction?").

### 4. Persistence Concurrency
- **File**: `apeiron-runtime/src/main.rs`
- **Description**: `self.save()` writes the entire JSON file on every correction.
- **Impact**: 
    - **Performance**: High I/O latency on large memory files.
    - **Race Conditions**: No file locking. If multiple threads were to write (currently guarded by Mutex, but susceptible to scaling issues), data could be corrupted.
- **Fix Required**: Append-only log (WAL) or SQLite buffer.

## 📦 Build & Dependency

### 5. Dependency Hell (`rand` vs `candle`)
- **Description**: `candle-core` relies on specific `rand` versions. Mixing versions causes trait bound errors (`Distribution<f16>`).
- **Status**: Currently patched by bumping to `0.8.2`, but fragile. Future updates must align `candle`, `tokenizers`, and `fastrand` versions carefully.

### 6. Model Download Reliability
- **File**: `entrypoint.sh` / `download_model.py`
- **Description**: Depends on `huggingface_hub` and public internet.
- **Impact**: Container will loop infinitely or fail if HuggingFace is down or rate-limited.
- **Fix Required**: Bundle model in a separate data image or use a local mirror.
