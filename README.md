# Invented by Teerth Sharma — Epsilon-Hollow

Technical setup and operations guide for the runnable paths in this repository.

This repository contains several experimental tracks (Python, Rust, Aether-Lang, runtime research). The instructions below focus on the paths that are currently executable from code in this repo:

- Python geometric core and theorem verification suite.
- FastAPI IDE backend with three local model tiers and Hugging Face downloads.
- Orchestrator entrypoint that can launch agent mode and/or backend mode.

## Runtime Components

| Component | Entrypoint | Notes |
|---|---|---|
| Geometric core agent | `kernel/epsilon/epsilon_core/main.py` | Standalone loop over `EpsilonHollowCore`. |
| Orchestrator | `infrastructure/orchestrator/main.py` | Modes: `agent`, `ide`, `full`. |
| IDE backend (FastAPI) | `kernel/epsilon/epsilon-ide/pentesting/backend/main.py` | API on `127.0.0.1:8742`. |
| Model downloader helper | `scripts/download_models.py` | Downloads backend model tiers to local disk. |
| One-click Ubuntu bootstrap | `ubuntu_one_click.sh` | Installs deps, creates venv, downloads LLMs, starts backend. |

## One-Click Ubuntu Setup

If you want a single file to do everything (configure + download LLMs + run backend):

```bash
chmod +x ubuntu_one_click.sh
./ubuntu_one_click.sh
```

Default behavior is fresh-machine bootstrap:

- Recreates `.venv`.
- Force reinstalls Python dependencies.
- Force re-downloads selected model tiers in parallel.
- Starts backend automatically and opens frontend automatically.
- Writes a full execution log to `logs/ubuntu_one_click_<timestamp>.log`.
- On failure, prints exact line number and failing command.
- On failure, attempts OpenClaw fallback provisioning and writes `launch_openclaw_debug.sh`.

What this script does:

1. Installs system packages (`python3`, `venv`, build tools, `git`, `curl`).
2. Creates or reuses `.venv` at repo root.
3. Installs runtime Python dependencies from `requirements/ubuntu-runtime.txt`.
4. Installs `torch` (CUDA wheel attempted when NVIDIA is detected).
5. Downloads model tiers used by backend into `kernel/epsilon/epsilon-ide/pentesting/backend/models`.
6. Exports backend runtime env and starts backend on `http://127.0.0.1:8742`.

### Script Options

```bash
./ubuntu_one_click.sh --help
```

Common combinations:

```bash
# Setup only, do not run server
./ubuntu_one_click.sh --no-run

# Setup server deps only, skip model downloads
./ubuntu_one_click.sh --skip-models --no-run

# Download only selected tiers
./ubuntu_one_click.sh --tiers foreman,logicgate --no-run

# Tune parallel model downloading
./ubuntu_one_click.sh --workers 3 --retries 2 --hf-workers 8

# Disable browser auto-open if running on headless server
./ubuntu_one_click.sh --no-open-frontend

# Override frontend URL and backend readiness timeout
./ubuntu_one_click.sh --frontend-url http://127.0.0.1:8742/ --ready-timeout 120

# Reuse existing virtualenv and avoid forced model re-download
./ubuntu_one_click.sh --reuse-venv --no-force-models --no-force-pip

# Set explicit workspace path exposed by backend
./ubuntu_one_click.sh --workspace /absolute/path/to/workspace
```

## Model Tiers Downloaded

The bootstrap and backend use these tier definitions:

| Tier | Hugging Face repo_id | Target directory |
|---|---|---|
| `foreman` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | `kernel/epsilon/epsilon-ide/pentesting/backend/models/tinyllama-1.1b` |
| `logicgate` | `Qwen/Qwen2.5-Coder-7B` | `kernel/epsilon/epsilon-ide/pentesting/backend/models/qwen2.5-coder-7b` |
| `architect` | `deepseek-ai/deepseek-coder-33b-instruct` | `kernel/epsilon/epsilon-ide/pentesting/backend/models/deepseek-coder-33b` |

Notes:

- Full snapshots can be large. Plan for high free disk headroom before downloading all tiers.
- To avoid hub throttling, set `HF_TOKEN` (or `HUGGINGFACE_HUB_TOKEN`) before running bootstrap.

## Manual Run After Setup

If you used `--no-run`, start backend manually:

```bash
source .venv/bin/activate
export EPSILON_WORKSPACE_ROOT="$(pwd)"
export EPSILON_DEV_MODE=1
python kernel/epsilon/epsilon-ide/pentesting/backend/main.py
```

### Quick Health Checks

```bash
curl -s http://127.0.0.1:8742/api/v1/status
curl -s http://127.0.0.1:8742/api/v1/workspace
curl -s http://127.0.0.1:8742/api/v1/claw/health
```

If bootstrap fails and OpenClaw fallback is enabled, run:

```bash
./launch_openclaw_debug.sh
```

## Orchestrator Usage

The orchestrator now honors `--port` and `--dev-mode` for IDE startup.

```bash
source .venv/bin/activate
python infrastructure/orchestrator/main.py --mode ide --port 8742 --dev-mode
```

Other modes:

```bash
python infrastructure/orchestrator/main.py --mode agent
python infrastructure/orchestrator/main.py --mode full --dev-mode
```

## Theorem Verification Suite

Run the 10-theorem Python verification script:

```bash
source .venv/bin/activate
python tests/verify_theorems.py
```

## Security and Runtime Controls

- `EPSILON_WORKSPACE_ROOT`: initial backend workspace root.
- `EPSILON_DEV_MODE=1`: enables DEV-only terminal endpoints in backend.
- `EPSILON_API_TOKEN`: optional bearer-token gate for mutating endpoints.

The backend can still change workspace at runtime via `POST /api/v1/workspace/open`.

## Important Boundaries

- Native C++ extensions in backend core are optional; Python fallbacks are present.
- Downloaded models are on disk; they are not loaded into RAM/VRAM until explicit load endpoints are called.
- Experimental Rust/Aether subprojects in this repo are not fully covered by `ubuntu_one_click.sh`.

## Reference Docs

- Research doc: `docs/research/MOTHER_OF_ALL_DOCS.md`
- Backend runtime: `kernel/epsilon/epsilon-ide/pentesting/backend/main.py`
- Bootstrap script: `ubuntu_one_click.sh`
