# Epsilon IDE Engine: The Potato PC Edition (2GB RAM / 2GB VRAM)

The **Epsilon IDE Engine (Potato PC Edition)** is a production-grade AI coding assistant designed to perform the "impossible": running a high-performance Large Language Model (LLM) on hardware with as little as **2GB of System RAM and 2GB of Video RAM (VRAM)**. 

By replacing the traditional, "heavy" AI stack—which typically requires 10-15GB of memory—with a coordinated suite of seven lightweight technologies, this engine operates with a total footprint of approximately **520MB RAM and 390MB VRAM**, leaving massive headroom for the operating system and complex coding tasks.

---

## 🏗 Main System Architecture

The engine utilizes the **SEAL Architecture**, simplified for extreme constraints. Unlike high-end versions that use a multi-model "swap-deck," the Potato Edition employs a **Single Shared Model** strategy to minimize memory overhead.

```text
  [ USER INTERFACE ]          [ ORCHESTRATION ]             [ ENGINE CORE ]
  +----------------+      +-----------------------+      +-----------------------+
  |  VS Code / IDE | <--> |  AETHER LINK (Lite)   | <--> |   BITNET.CPP SERVER   |
  +----------------+      | (Async Stdin/Stdout)  |      | (Qwen 0.5B Ternary)   |
                          +----------+------------+      +----------+------------+
                                     |                              |
                          +----------v------------+      +----------v------------+
                          |   PICOCLAW AGENTS     |      |  MEMORY & HARDWARE    |
                          | (Serial Priority Bus) |      | - INT8 KV Cache       |
                          +----------+------------+      | - Sparse Attention    |
                                     |                   | - 4GB SSD Swap File   |
                          +----------v------------+      +----------+------------+
                          |  CLARA CONTEXT ORACLE |                 |
                          | (Disk-Backed TF-IDF)  | <---------------+
                          +-----------------------+
```

### The "Impossible" Solution Stack
1.  **BitNet 1.58-bit:** Compresses the model weights by **8×** (988MB down to 120MB) by using ternary values.
2.  **bitnet.cpp:** A lean C++ inference engine that replaces the 800MB+ PyTorch framework.
3.  **tinygrad:** A minimalist ML framework (50MB idle) used for managing the KV cache.
4.  **Sparse Attention:** Reduces memory growth from quadratic $O(n^2)$ to linear $O(n \cdot k)$, shrinking the cache from 440MB to 80MB.
5.  **PicoClaw Micro-Agents:** Four specialized agents that decompose tasks; **three out of four use zero AI inference**, saving massive compute.
6.  **Clara Context Oracle:** Replaces heavy neural embeddings (500MB) with classical **TF-IDF search** (20MB).
7.  **SSD Swap Space:** A 4GB safety net that prevents hard crashes if RAM usage spikes.

---

## 🛠 Component Sub-Architectures

### 1. BitNet: Ternary Weight Packing
Traditional models store weights as 16-bit floats (FP16). BitNet uses **ternary values {-1, 0, +1}**, which mathematically require only **1.58 bits** per weight.

**Packing Architecture:**
*   **Values:** -1, 0, and +1 are encoded as 2-bit pairs (e.g., `00`, `01`, `10`).
*   **Storage:** Four ternary weights are packed into a **single 8-bit byte**, achieving extreme density.
*   **Compute:** This eliminates the "Multiply Bottleneck." Since weights are -1, 0, or 1, the CPU only performs **additions, subtractions, or skips**.

### 2. PicoClaw: Serial Micro-Agent Pipeline
To stay within 2GB, agents execute **serially** (one at a time) rather than concurrently, ensuring only one model context exists in VRAM at any moment.

```text
[INTENT] -> [ROUTER] -> [RECALL] -> [CODER] -> [CRITIC] -> [TOKEN STREAM]
               |           |           |           |
            Keyword      TF-IDF      BitNet      Syntax
             Match       Search     Inference    Check
            (No AI)     (No AI)    (1 Model Call) (No AI)
```
*   **Router:** Uses a dictionary lookup to classify intent (e.g., `write` $\rightarrow$ `CODE_GEN`) in <1ms.
*   **Recall:** Queries the **Clara Oracle** for relevant code snippets using cosine similarity.
*   **Coder:** The **only agent** that calls the LLM. It assembles a tight prompt (<300 tokens) to save memory.
*   **Critic:** Validates code using **Tree-sitter AST parsing** to catch syntax errors without needing a second LLM pass.

### 3. Clara Context Oracle: Disk-Backed Memory
Clara provides "codebase memory" without a neural model. It converts project files into **384-dimensional TF-IDF vectors**.
*   **Search:** Uses the `sqlite-vec` extension to perform cosine distance calculations directly in SQL.
*   **Efficiency:** The index lives on the SSD. Only the current query and top-3 results ever touch RAM (~20MB total).

---

## 📊 Complete Memory Budget (Hard Limits)

Every megabyte is accounted for to prevent **CUDA Out Of Memory (OOM)** errors, which can crash GPU drivers.

| Pool | Component | Size (MB) | Note |
| :--- | :--- | :--- | :--- |
| **VRAM** | BitNet Weights | 120 MB | Qwen 0.5B Ternary |
| **VRAM** | KV Cache (INT8) | 80 MB | 512-token context |
| **VRAM** | Activations | 50 MB | Temporary compute space |
| **VRAM** | Driver Overhead | 100 MB | Fixed OS cost |
| **Total VRAM** | | **350-390 MB** | **~1.7 GB Free** |
| | | | |
| **RAM** | Linux OS (Minimal) | 200 MB | Ubuntu Server (No GUI) |
| **RAM** | Python / bitnet.cpp | 110 MB | No PyTorch bloat |
| **RAM** | Agents / Buffers | 210 MB | Working memory |
| **Total RAM** | | **~520 MB** | **~1.5 GB Free** |

---

## 🚀 Data Flow: Keyboard to Ghost Text

1.  **Capture:** VS Code captures 200 characters of cursor context and sends a JSON request via **Aether Link**.
2.  **Routing:** The **Router Agent** identifies the task type (e.g., `CODE_GEN`) in microseconds.
3.  **Retrieval:** The **Recall Agent** pulls top-3 relevant code snippets from the **Clara Oracle**.
4.  **Inference:** The **Coder Agent** sends a compact prompt to `bitnet.cpp`. The LLM generates tokens at ~40 tok/s.
5.  **Validation:** The **Critic Agent** uses Tree-sitter to ensure the code is syntactically valid.
6.  **Streaming:** Valid tokens flow through a **Named Pipe** to VS Code, appearing instantly as **Ghost Text**.

---

## 🛠 Installation & OS Hardening

To run on a 2GB system, the OS must be "stripped lean" to free up RAM.
1.  **OS:** Install **Ubuntu Server 22.04** (saves 500MB RAM over Desktop).
2.  **Hardening:** Disable Bluetooth, CUPS (printing), and the Display Manager.
3.  **Swap:** Create a **4GB swap file on an SSD** (never an HDD) and set `vm.swappiness=10` to ensure swap is only used for "cold" data.
4.  **Build:** Compile the C++ extensions (`vram_guard.cpp` and `token_pipe.cpp`) to handle hardware-level locks and zero-copy token delivery.