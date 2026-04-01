# Epsilon IDE Engine v2
**Three-Tier Local AI — Fast · Balanced · Deep**

A fully local AI coding assistant with automatic model routing.  
No internet. No API keys. Runs on your hardware.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)
![CUDA](https://img.shields.io/badge/CUDA-12.4-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Part of SEAL](https://img.shields.io/badge/Part%20of-SEAL%20Project-purple?style=flat-square)

---

## ⚠️ WARNINGS & DISCLAIMERS

**READ THIS BEFORE INSTALLING OR USING THIS SOFTWARE**

### Hardware Risks
- **GPU VRAM exhaustion**: Loading models can consume 1-20 GB VRAM. Exceeding capacity may crash your system or damage hardware.
- **Thermal stress**: Continuous GPU inference generates significant heat. Ensure adequate cooling.
- **RAM consumption**: Deep tier requires 20+ GB RAM. Attempting to load on insufficient memory may freeze your system.
- **Disk space**: Model downloads require 3-25 GB. Ensure sufficient free space before running setup.

### Software Risks
- **Experimental software**: This is research-grade code. Expect bugs, crashes, and breaking changes.
- **File system access**: The engine can read/write files in your project directory. Review generated code before execution.
- **No warranty**: This software is provided AS-IS. The author is not responsible for data loss, system damage, or any other issues.
- **CUDA compatibility**: Requires CUDA 12+. Mismatched versions may cause silent failures or system instability.

### Model Behavior
- **Unfiltered outputs**: Local models may generate incorrect, biased, or inappropriate code. Always review output.
- **No guarantees**: Generated code may contain bugs, security vulnerabilities, or licensing issues. Test thoroughly.
- **Context limitations**: Models work within limited context windows. Long files may be truncated.

### Security & Privacy
- **Telegram bot risks**: If enabled, your bot token grants access to your system. Keep it secret. Anyone with the token can send prompts to your machine.
- **Network isolation**: While models run locally, setup scripts download models from the internet. Review `setup.sh` before execution.
- **Code execution**: Generated code runs with your user permissions. Malicious prompts could theoretically generate harmful code.

### Legal
- **Model licenses**: Downloaded models (Qwen, DeepSeek) have their own licenses. Review before commercial use.
- **Generated code ownership**: Unclear IP status of AI-generated code. Consult legal counsel for commercial projects.
- **Compliance**: You are responsible for ensuring your use complies with local laws and regulations.

### System Requirements Reality Check
- **Minimum is not optimal**: Listed minimums will result in slow, frustrating experience.
- **Deep tier requires serious hardware**: 20+ GB RAM, fast SSD. Not suitable for laptops.
- **WSL2 on Windows**: Adds overhead. Native Linux strongly recommended.

**BY USING THIS SOFTWARE, YOU ACKNOWLEDGE THESE RISKS AND AGREE THAT THE AUTHOR BEARS NO LIABILITY FOR ANY DAMAGES, DATA LOSS, OR ISSUES ARISING FROM ITS USE.**

---

## What is new in v2

| Feature | v1 | v2 |
|---|---|---|
| Model | BitNet 2B (general) | Qwen2.5-Coder (code specialist) |
| Inference | CPU only — 4 tok/s | GPU — 30-80 tok/s |
| Model tiers | 1 | 3 (fast / balanced / deep) |
| Context window | 512 tokens | 2048 tokens |
| Conversation memory | None | Persistent JSON |
| File writing | None | Creates files on disk |
| Telegram | None | Full bot integration |
| Auto routing | None | Complexity scoring |

---

## The three tiers
```
Request → ROUTER scores complexity (1-10)
               │
     ┌─────────┼──────────┐
     ▼         ▼          ▼
  score 0-2  score 3-6  score 7-10
     │         │          │
  FAST       BALANCED    DEEP
  1.5B       7B          33B
  ~1GB VRAM  ~4GB VRAM   CPU/SSD
  1-2s       5-15s       30-120s
```

**⚠️ WARNING**: Only one model is active at a time. VRAM is freed when switching tiers, but the transition takes 5-10 seconds. During model loading, the engine will not respond to requests.

---

## Quick start
```bash
# One command — sets up everything
# WARNING: Downloads 3+ GB of models. Review setup.sh first.
bash setup.sh

# Start the engine
source venv/bin/activate
python3 backend/main.py

# Test it
echo '{"prompt": "write a binary search function"}' \
  | python3 backend/main.py 2>/dev/null
```

---

## Installation (manual)

### Requirements
- Ubuntu 22.04 or WSL2 on Windows 10/11
- Python 3.11
- NVIDIA GPU with CUDA 12+ (optional but strongly recommended)
- **3 GB free disk space minimum** (25 GB if downloading all tiers)
- **8 GB RAM minimum** (20+ GB recommended for deep tier)
- **4 GB VRAM minimum** (8+ GB recommended)

### Step 1 — System dependencies
```bash
sudo apt update
sudo apt install -y git cmake build-essential clang lld \
    python3.11 python3.11-venv python3-pip pybind11-dev
```

### Step 2 — CUDA toolkit (for GPU inference)

**⚠️ WARNING**: This installs CUDA 12.4. If you have a different CUDA version, modify the repo URL accordingly. Mismatched CUDA versions can cause silent failures.

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update && sudo apt-get install -y cuda-toolkit-12-4
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
nvcc --version  # verify
```

### Step 3 — Clone and run setup

**⚠️ WARNING**: Review `setup.sh` before running. It downloads models from HuggingFace and modifies your Python environment.

```bash
git clone https://github.com/YOUR_USERNAME/epsilon-ide-v2.git
cd epsilon-ide-v2
bash setup.sh
```

The setup script handles everything else automatically.

---

## Usage

### Standard mode (VS Code / terminal)
```bash
python3 backend/main.py
```

Reads JSON from stdin, writes JSON to stdout:
```bash
echo '{"prompt": "write a quicksort function"}' \
  | python3 backend/main.py 2>/dev/null
```

### Telegram bot mode

**⚠️ SECURITY WARNING**: Your Telegram bot token grants remote access to this engine. Anyone with the token can execute prompts on your machine with your user permissions. Keep it secret. Revoke immediately if compromised.

```bash
# 1. Get a token from @BotFather on Telegram
# 2. Set it in config.yaml under telegram_token
python3 backend/main.py --telegram
```

### Both at once
```bash
python3 backend/main.py --all
```

---

## Forcing a specific tier

Prefix your prompt with a tier override:
```
use fast: write a hello world function
use balanced: create a complete FastAPI server with auth
use deep: design a microservices architecture for a payment system
```

---

## Configuration

Copy `config.yaml` (auto-generated by setup.sh) and edit:

| Key | Default | Description |
|---|---|---|
| `model_tier` | `auto` | `auto` scores complexity, or force `fast`/`balanced`/`deep` |
| `routing.fast_max` | `2` | Complexity scores 0-2 use fast tier |
| `routing.balanced_max` | `6` | Scores 3-6 use balanced tier |
| `idle_timeout` | `300` | Unload model after N seconds idle |
| `crawl_on_start` | `false` | Index project files on boot |
| `memory_turns` | `10` | How many conversation turns to remember |

---

## Downloading additional models

**⚠️ WARNING**: These downloads are large (4-20 GB each) and may take hours depending on your connection. Ensure sufficient disk space and stable internet.

```bash
# Balanced tier — Qwen2.5-Coder 7B (~4 GB)
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='Qwen/Qwen2.5-Coder-7B-Instruct-GGUF',
    filename='qwen2.5-coder-7b-instruct-q4_k_m.gguf',
    local_dir='BitNet/models/Qwen2.5-Coder-7B',
)
"

# Deep tier — DeepSeek-Coder 33B (~20 GB, CPU only)
# WARNING: Requires 20+ GB RAM and 20+ GB free disk space.
# Will cause system freeze if insufficient RAM. Only download if you have the resources.
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='TheBloke/deepseek-coder-33B-instruct-GGUF',
    filename='deepseek-coder-33b-instruct.Q4_K_M.gguf',
    local_dir='BitNet/models/DeepSeek-Coder-33B',
)
"
```

---

## Performance optimisations applied

- Flash attention (`--flash-attn`) — 15-20% faster on Ampere GPUs
- Temperature 0.0 for code generation — deterministic, slightly faster
- INT8 KV cache — halves attention memory vs float16
- Sparse attention top-64 — 8x less attention computation
- Idle watchdog — unloads model after 5 min idle to free VRAM
- Context caching — reuses system prompt KV state across requests

---

## Monitoring
```bash
# GPU usage (while engine runs)
watch -n 1 nvidia-smi

# Python process RAM
ps aux | grep "python3 backend/main.py"

# Real-time monitoring script
bash scripts/monitor.sh
```

---

## Known Issues & Troubleshooting

### CUDA out of memory
- **Symptom**: `RuntimeError: CUDA out of memory`
- **Fix**: Lower `max_context` in config.yaml, or force fast tier only

### Model fails to load
- **Symptom**: Engine hangs or crashes during model load
- **Fix**: Check `nvidia-smi` for VRAM usage. Close other GPU applications. Verify model files are not corrupted.

### Slow inference on CPU
- **Symptom**: Deep tier takes 2+ minutes per response
- **Fix**: This is expected. Deep tier runs on CPU. Consider using balanced tier instead.

### Telegram bot not responding
- **Symptom**: Bot shows online but doesn't reply
- **Fix**: Check `backend/main.py` logs. Verify token is correct in config.yaml. Ensure firewall allows outbound HTTPS.

### "Permission denied" errors
- **Symptom**: Cannot write files
- **Fix**: Ensure script has write permissions in project directory. Check SELinux/AppArmor if on hardened system.

---

## Project structure
```
epsilon-v2/
├── backend/
│   ├── tiers/
│   │   ├── model.py          # ModelServer — HTTP wrapper for llama-server
│   │   ├── model_manager.py  # TieredModelManager — async 3-tier management
│   │   └── router.py         # ModelRouter — complexity scoring + tier selection
│   ├── agents/
│   │   ├── orchestrator.py   # Six-agent pipeline
│   │   └── router.py         # Task classification + complexity scoring
│   ├── memory/
│   │   ├── conversation.py   # Persistent JSON conversation history
│   │   └── kv_cache.py       # INT8 sparse attention KV cache
│   ├── clara/
│   │   └── oracle.py         # TF-IDF code search (sqlite-vec)
│   ├── tools/
│   │   └── filesystem.py     # read_file, write_file, edit_file
│   ├── aether/
│   │   └── link.py           # stdin/stdout JSON bridge
│   ├── telegram/
│   │   └── bot.py            # Telegram bot integration
│   └── main.py               # Entry point
├── setup.sh                  # Automated setup script
├── requirements.txt          # Python dependencies
└── config.yaml               # Runtime configuration
```

---

## Roadmap

| Feature | Status |
|---|---|
| GPU inference (fast tier) | Done |
| Three-tier routing | Done |
| Conversation memory | Done |
| File writing tools | Done |
| Telegram bot | Done |
| VS Code extension | Planned |
| Word doc creation | Planned |
| Browser automation | Planned |
| Scheduled tasks | Planned |
| Remote desktop control | Planned |

---

## Part of SEAL

Epsilon IDE is the coding assistant component of **SEAL** — a personal AI runtime built to run entirely on local hardware.

---

## License

```
MIT License

Copyright (c) 2026 [Your Name / Organization]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### What This Means

**You can**:
- Use this software for any purpose (personal, commercial, research)
- Modify the source code
- Distribute copies
- Distribute modified versions
- Use in proprietary software

**You cannot**:
- Hold the author liable for any damages
- Claim warranty or support
- Use the author's name for endorsement without permission

**You must**:
- Include the above copyright notice and license text in all copies
- Acknowledge that the software is provided "as-is"

### Third-Party Licenses

This software uses models and libraries with their own licenses:
- **Qwen2.5-Coder**: Apache 2.0 (review at Qwen/Qwen2.5-Coder)
- **DeepSeek-Coder**: MIT (review at deepseek-ai/deepseek-coder)
- **llama.cpp**: MIT (review at ggerganov/llama.cpp)

You are responsible for complying with all applicable licenses.

---

## Contributing

Contributions are welcome, but understand:
- No guarantees your PR will be merged
- Code must follow existing style
- Test your changes on your own hardware first
- By contributing, you agree to license your contributions under MIT

---

## Support

**There is no official support channel.** This is a personal project shared for educational purposes.

- **Issues**: GitHub issues for bug reports only (no usage questions)
- **Pull requests**: Welcome, but review may be slow
- **Email support**: lovekeshanand6@gmail.com
- **Discord/Slack**: 1366730233570660442

**If this breaks your system, you're on your own.** That's why the warnings exist.

---

Built for people who refuse to let hardware limits stop them.

**Use at your own risk.**
