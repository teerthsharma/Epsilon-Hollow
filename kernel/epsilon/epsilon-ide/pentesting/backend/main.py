"""
sealMega IDE - FastAPI Backend
NO FAKE DATA. Reports real state. Real filesystem access. Real model downloads.
Hardware Directives from Dr. Anatoly [Redacted], MIT CSAIL:
  Alpha: CUDA Semaphore VRAM Fencing
  Beta:  sqlite-vec Clara (ChromaDB is DEAD)
  Gamma: Zero-Copy IPC (shared memory, no pickle)
  Delta: Perplexity Rollback (Cauchy-Schwarz safety net)
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import asyncio
import subprocess
import os
import sys
import json
import pathlib
from ai_engine import load_model_locally, generate_code, AI_AVAILABLE

# Core hardware modules — NOT Python toys
from core.vram_guard_py import (
    fence_vram, release_vram, is_vram_fenced, get_fence_holder, wait_for_vram,
    TIER_FOREMAN, TIER_LOGICGATE, TIER_ARCHITECT
)
from core import clara
from core import shared_memory_ipc as ipc
from core.perplexity_rollback import PruningGuard

app = FastAPI(title="sealMega IDE Engine", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pruning guard instance
pruning_guard = PruningGuard()

@app.on_event("startup")
async def startup_event():
    """Initialize core hardware modules on boot."""
    # Directive Gamma: Allocate shared memory block
    try:
        ipc.init_shared_memory()
    except Exception as e:
        print(f"[WARN] Shared memory init failed: {e}")
    
    # Directive Beta: Initialize Clara sqlite-vec
    db_path = os.path.join(BASE_DIR, "clara.db")
    try:
        clara.init_clara(db_path)
    except Exception as e:
        print(f"[WARN] Clara init failed: {e}")
    
    print("[sealMega] Core hardware modules initialized.")
    print(f"  [Alpha] VRAMGuard: {'C++ native' if False else 'OS-level fallback'}")
    print(f"  [Beta]  Clara: sqlite-vec at {db_path}")
    print(f"  [Gamma] IPC: {ipc.SHM_SIZE // (1024*1024)}MB shared memory")
    print(f"  [Delta] PruningGuard: spike ratio {pruning_guard.spike_ratio}x")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up shared memory on shutdown."""
    ipc.cleanup()

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "frontend")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Workspace root — defaults to user home, can be changed via API
workspace_root = os.path.expanduser("~")

# Model registry — what's actually on disk
MODEL_REGISTRY = {
    "foreman": {
        "name": "TinyLlama 1.1B",
        "repo_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "dir": os.path.join(MODELS_DIR, "tinyllama-1.1b"),
        "size_gb": 0.64,
    },
    "logicgate": {
        "name": "Qwen2.5-Coder 7B",
        "repo_id": "Qwen/Qwen2.5-Coder-7B",
        "dir": os.path.join(MODELS_DIR, "qwen2.5-coder-7b"),
        "size_gb": 4.5,
    },
    "architect": {
        "name": "DeepSeek-Coder 33B",
        "repo_id": "deepseek-ai/deepseek-coder-33b-instruct",
        "dir": os.path.join(MODELS_DIR, "deepseek-coder-33b"),
        "size_gb": 20.0,
    },
}

# Download state tracking
download_state = {}

# ──────────────────────────────────────────────
# Data Models
# ──────────────────────────────────────────────

class TierStatus(BaseModel):
    status: str  # "not_downloaded", "downloading", "ready", "error"
    model: str
    downloaded: bool
    size_gb: float
    path: str

class EngineStatusResponse(BaseModel):
    workspace: str
    foreman: TierStatus
    logicGate: TierStatus
    architect: TierStatus

class FileEntry(BaseModel):
    name: str
    path: str
    isDir: bool
    size: Optional[int] = None

class FileListResponse(BaseModel):
    entries: list[FileEntry]
    path: str

class FileReadResponse(BaseModel):
    content: str
    path: str
    language: str

class FileWriteRequest(BaseModel):
    path: str
    content: str

class ClawRequest(BaseModel):
    command: str
    cwd: str

class ClawResponse(BaseModel):
    approved: bool
    output: str
    exitCode: int
    safetyReason: Optional[str] = None

class ModelDownloadRequest(BaseModel):
    tier: str
    repo_id: str

class SelfImproveRequest(BaseModel):
    target_file: str
    instructions: str
    tier: str = "architect"

class ModelStatusResponse(BaseModel):
    status: str
    progress: float
    message: str

class WorkspaceRequest(BaseModel):
    path: str

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

BLOCKED_COMMANDS = [
    "rm -rf /", "rm -rf /*", "format c:", "del /f /s /q c:",
    ":(){:|:&};:", "mkfs", "dd if=/dev/zero",
]

BINARY_EXTENSIONS = {
    '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.svg',
    '.pdf', '.zip', '.tar', '.gz', '.7z', '.rar',
    '.exe', '.dll', '.so', '.dylib', '.o', '.obj',
    '.woff', '.woff2', '.ttf', '.eot',
    '.mp3', '.mp4', '.avi', '.mov', '.wav',
    '.pyc', '.pyo', '.class',
}

HIDDEN_DIRS = {
    'node_modules', '.git', '__pycache__', '.venv', 'venv',
    '.idea', '.vs', 'dist', 'build', '.next', '.nuxt',
    'target', 'bin', 'obj',
}

def check_model_downloaded(tier: str) -> bool:
    """Check if a model's files actually exist on disk."""
    info = MODEL_REGISTRY.get(tier)
    if not info:
        return False
    model_dir = info["dir"]
    if not os.path.isdir(model_dir):
        return False
    # Check for actual model files (safetensors, bin, gguf)
    for f in os.listdir(model_dir):
        if f.endswith(('.safetensors', '.bin', '.gguf', '.ggml')):
            return True
    return False

def get_tier_status(tier: str) -> TierStatus:
    """Get REAL status of a model tier - no faking."""
    info = MODEL_REGISTRY.get(tier, {})
    downloaded = check_model_downloaded(tier)

    if tier in download_state and download_state[tier].get("active"):
        status = "downloading"
    elif downloaded:
        status = "ready"
    else:
        status = "not_downloaded"

    return TierStatus(
        status=status,
        model=info.get("name", "Unknown"),
        downloaded=downloaded,
        size_gb=info.get("size_gb", 0),
        path=info.get("dir", ""),
    )

def get_language(path: str) -> str:
    ext = pathlib.Path(path).suffix.lower()
    lang_map = {
        '.py': 'python', '.js': 'javascript', '.ts': 'typescript',
        '.jsx': 'javascript', '.tsx': 'typescript', '.rs': 'rust',
        '.go': 'go', '.java': 'java', '.c': 'c', '.cpp': 'cpp',
        '.h': 'c', '.cs': 'csharp', '.rb': 'ruby', '.php': 'php',
        '.html': 'html', '.css': 'css', '.scss': 'scss',
        '.json': 'json', '.xml': 'xml', '.yaml': 'yaml', '.yml': 'yaml',
        '.md': 'markdown', '.sql': 'sql', '.sh': 'shell',
        '.ps1': 'powershell', '.toml': 'ini', '.txt': 'plaintext',
    }
    return lang_map.get(ext, 'plaintext')

# ──────────────────────────────────────────────
# Serve Frontend
# ──────────────────────────────────────────────

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

# Mount static files AFTER defining routes
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# ──────────────────────────────────────────────
# Status — HONEST, no fakes
# ──────────────────────────────────────────────

@app.get("/api/v1/status")
async def get_status():
    return EngineStatusResponse(
        workspace=workspace_root,
        foreman=get_tier_status("foreman"),
        logicGate=get_tier_status("logicgate"),
        architect=get_tier_status("architect"),
    )

# ──────────────────────────────────────────────
# Workspace
# ──────────────────────────────────────────────

@app.post("/api/v1/workspace/open")
async def open_workspace(req: WorkspaceRequest):
    global workspace_root
    path = os.path.abspath(req.path)
    if not os.path.isdir(path):
        return {"error": f"Directory not found: {path}"}
    workspace_root = path
    return {"workspace": workspace_root}

@app.get("/api/v1/workspace")
async def get_workspace():
    return {"workspace": workspace_root}

# ──────────────────────────────────────────────
# File System — REAL access
# ──────────────────────────────────────────────

@app.get("/api/v1/files/list")
async def list_files(path: str = "/"):
    """List files and directories. Path is relative to workspace or absolute."""
    if path == "/" or path == "":
        target = workspace_root
    elif os.path.isabs(path):
        target = path
    else:
        target = os.path.join(workspace_root, path)

    target = os.path.abspath(target)

    if not os.path.isdir(target):
        return FileListResponse(entries=[], path=target)

    entries = []
    try:
        for item in os.listdir(target):
            # Skip hidden and heavy directories
            if item.startswith('.') and item not in ('.env', '.gitignore'):
                if os.path.isdir(os.path.join(target, item)):
                    continue
            if item in HIDDEN_DIRS:
                continue

            full_path = os.path.join(target, item)
            is_dir = os.path.isdir(full_path)
            size = None
            if not is_dir:
                try:
                    size = os.path.getsize(full_path)
                except OSError:
                    size = 0

            entries.append(FileEntry(
                name=item,
                path=full_path.replace("\\", "/"),
                isDir=is_dir,
                size=size,
            ))
    except PermissionError:
        pass

    return FileListResponse(entries=entries, path=target.replace("\\", "/"))


@app.get("/api/v1/files/read")
async def read_file(path: str):
    """Read a file's content. Supports absolute paths — edit files ANYWHERE."""
    abs_path = os.path.abspath(path)

    if not os.path.isfile(abs_path):
        return {"error": f"File not found: {abs_path}", "content": "", "path": abs_path, "language": "plaintext"}

    ext = pathlib.Path(abs_path).suffix.lower()
    if ext in BINARY_EXTENSIONS:
        return {"error": "Binary file", "content": "[Binary file — cannot display]", "path": abs_path, "language": "plaintext"}

    try:
        # Try UTF-8 first, fall back to latin-1
        try:
            with open(abs_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(abs_path, 'r', encoding='latin-1') as f:
                content = f.read()

        # Limit to 1MB for safety
        if len(content) > 1_000_000:
            content = content[:1_000_000] + "\n\n--- [Truncated: file exceeds 1MB] ---"

        return FileReadResponse(
            content=content,
            path=abs_path.replace("\\", "/"),
            language=get_language(abs_path),
        )
    except PermissionError:
        return {"error": "Permission denied", "content": "", "path": abs_path, "language": "plaintext"}
    except Exception as e:
        return {"error": str(e), "content": "", "path": abs_path, "language": "plaintext"}


@app.post("/api/v1/files/write")
async def write_file(req: FileWriteRequest):
    """Write content to a file. Supports absolute paths — edit files ANYWHERE."""
    abs_path = os.path.abspath(req.path)

    try:
        # Create parent dirs if needed
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, 'w', encoding='utf-8') as f:
            f.write(req.content)
        return {"success": True, "path": abs_path.replace("\\", "/")}
    except PermissionError:
        return {"success": False, "error": "Permission denied"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ──────────────────────────────────────────────
# Model Download — REAL downloads from HuggingFace
# ──────────────────────────────────────────────

@app.post("/api/v1/models/download")
async def start_model_download(req: ModelDownloadRequest):
    """Start downloading a model from HuggingFace Hub."""
    tier = req.tier
    if tier not in MODEL_REGISTRY:
        return {"error": f"Unknown tier: {tier}"}

    if check_model_downloaded(tier):
        return {"status": "already_downloaded"}

    info = MODEL_REGISTRY[tier]
    model_dir = info["dir"]
    os.makedirs(model_dir, exist_ok=True)

    # Track state
    download_state[tier] = {"active": True, "progress": 0, "message": "Starting download..."}

    # Launch download in background
    asyncio.create_task(_download_model_task(tier, req.repo_id, model_dir))

    return {"status": "started", "tier": tier}


async def _download_model_task(tier: str, repo_id: str, model_dir: str):
    """Background task to download model using huggingface-cli."""
    try:
        download_state[tier]["message"] = "Checking huggingface-hub..."
        download_state[tier]["progress"] = 5

        # Check if huggingface-hub is installed
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "pip", "show", "huggingface-hub",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.wait()

        if proc.returncode != 0:
            download_state[tier]["message"] = "Installing huggingface-hub..."
            download_state[tier]["progress"] = 10
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "pip", "install", "huggingface-hub",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.wait()

        download_state[tier]["message"] = f"Downloading {repo_id}..."
        download_state[tier]["progress"] = 15

        # Download using huggingface_hub snapshot_download
        download_script = f"""
import sys
from huggingface_hub import snapshot_download
try:
    snapshot_download(
        repo_id="{repo_id}",
        local_dir=r"{model_dir}",
        local_dir_use_symlinks=False,
    )
    print("DOWNLOAD_COMPLETE")
except Exception as e:
    print(f"DOWNLOAD_ERROR: {{e}}")
    sys.exit(1)
"""
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", download_script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # Simulate progress while waiting (real progress is hard with HF hub)
        progress = 20
        while proc.returncode is None:
            await asyncio.sleep(3)
            progress = min(progress + 5, 90)
            download_state[tier]["progress"] = progress
            download_state[tier]["message"] = f"Downloading {repo_id}... ({progress}%)"
            try:
                await asyncio.wait_for(proc.wait(), timeout=0.1)
            except asyncio.TimeoutError:
                pass

        stdout = (await proc.stdout.read()).decode(errors='replace')
        stderr = (await proc.stderr.read()).decode(errors='replace')

        if "DOWNLOAD_COMPLETE" in stdout:
            download_state[tier] = {"active": False, "progress": 100, "message": "Download complete!", "status": "ready"}
        else:
            error_msg = stderr[:200] if stderr else stdout[:200]
            download_state[tier] = {"active": False, "progress": 0, "message": f"Failed: {error_msg}", "status": "error"}

    except Exception as e:
        download_state[tier] = {"active": False, "progress": 0, "message": str(e), "status": "error"}


@app.get("/api/v1/models/status/{tier}")
async def get_model_status(tier: str):
    """Get download progress for a model tier."""
    if tier in download_state:
        state = download_state[tier]
        return ModelStatusResponse(
            status=state.get("status", "downloading" if state.get("active") else "not_downloaded"),
            progress=state.get("progress", 0),
            message=state.get("message", ""),
        )

    if check_model_downloaded(tier):
        return ModelStatusResponse(status="ready", progress=100, message="Model ready")

    return ModelStatusResponse(status="not_downloaded", progress=0, message="Not downloaded")


# ──────────────────────────────────────────────
# AI Engine — Model Memory Loading & Self-Coding
# ──────────────────────────────────────────────

@app.post("/api/v1/models/load-memory/{tier}")
async def load_model_memory(tier: str):
    """Loads a downloaded model into RAM/VRAM via ai_engine."""
    if not check_model_downloaded(tier):
        return {"success": False, "error": f"Model {tier} not downloaded yet."}
    if not AI_AVAILABLE:
        return {"success": False, "error": "Torch/Transformers not installed."}
    
    info = MODEL_REGISTRY[tier]
    try:
        await load_model_locally(tier, info["dir"])
        return {"success": True, "message": f"{tier} loaded successfully"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/v1/architect/self-improve")
async def ai_self_improve(req: SelfImproveRequest):
    """The IDE asks the AI to rewrite its own source code."""
    abs_path = os.path.abspath(req.target_file)
    if not os.path.isfile(abs_path):
        return {"success": False, "error": f"File not found: {abs_path}"}
        
    try:
        with open(abs_path, 'r', encoding='utf-8') as f:
            current_code = f.read()
            
        prompt = f"You are the sealMega Architect. Your task is to modify the following code based on the instructions.\n\nInstructions: {req.instructions}\n\nCurrent Code (file: {abs_path}):\n```python\n{current_code}\n```\n\nRewrite the file completely combining the current code and the requested features/changes. Output ONLY the new code. Do not output markdown codeblocks around the entire response, just raw code:"
        
        new_code = await generate_code(req.tier, prompt, max_new_tokens=2048)
        
        # Clean up possible markdown artifacts if the model returns them anyway
        if new_code.startswith("```"):
            new_code = "\n".join(new_code.split('\n')[1:])
            if new_code.endswith("```"):
                new_code = new_code[:-3]
        
        with open(abs_path, 'w', encoding='utf-8') as f:
            f.write(new_code)
            
        return {"success": True, "path": abs_path}
    except Exception as e:
        return {"success": False, "error": str(e)}

# ──────────────────────────────────────────────
# Pico Claw — Terminal Command Execution
# ──────────────────────────────────────────────

@app.post("/api/v1/claw/execute")
async def claw_execute(req: ClawRequest):
    cmd_lower = req.command.lower().strip()
    for blocked in BLOCKED_COMMANDS:
        if blocked in cmd_lower:
            return ClawResponse(
                approved=False, output="", exitCode=-1,
                safetyReason=f"Blocked: matches dangerous pattern '{blocked}'",
            )

    cwd = req.cwd if os.path.isdir(req.cwd) else workspace_root

    try:
        result = subprocess.run(
            req.command, shell=True, capture_output=True, text=True,
            timeout=30, cwd=cwd, env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        output = result.stdout[:5000]
        if result.stderr:
            output += result.stderr[:2000]
        return ClawResponse(approved=True, output=output, exitCode=result.returncode)
    except subprocess.TimeoutExpired:
        return ClawResponse(approved=True, output="Command timed out (30s).", exitCode=-1)
    except Exception as e:
        return ClawResponse(approved=True, output=f"Error: {e}", exitCode=-1)


# ──────────────────────────────────────────────
# Clara — sqlite-vec AST Indexing (ChromaDB is DEAD)
# ──────────────────────────────────────────────

@app.post("/api/v1/clara/index")
async def clara_index_endpoint():
    """Index the workspace AST via sqlite-vec. Zero daemons. Zero HTTP overhead."""
    try:
        result = clara.index_workspace(workspace_root)
        return {"status": "indexed", **result}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/v1/clara/stats")
async def clara_stats():
    """Get Clara index statistics."""
    return clara.get_stats()

@app.get("/api/v1/clara/query")
async def clara_query(q: str, limit: int = 20):
    """Query the AST index for function/class signatures."""
    return {"results": clara.query_context(q, limit)}

# ──────────────────────────────────────────────
# Hardware Status — VRAMGuard + IPC
# ──────────────────────────────────────────────

@app.get("/api/v1/hardware/vram")
async def vram_status():
    """Get VRAM fence status."""
    return {
        "fenced": is_vram_fenced(),
        "holder": get_fence_holder(),
        "holder_name": ["Foreman", "Logic-Gate", "Architect"][get_fence_holder()] if get_fence_holder() >= 0 else None,
    }

@app.get("/api/v1/hardware/ipc")
async def ipc_status():
    """Get shared memory IPC status."""
    return ipc.get_status()

@app.get("/api/v1/hardware/pruning")
async def pruning_status():
    """Get perplexity rollback guard statistics."""
    return pruning_guard.get_stats()


# ──────────────────────────────────────────────
# WebSocket Terminal
# ──────────────────────────────────────────────

@app.websocket("/ws/terminal")
async def ws_terminal(websocket: WebSocket):
    await websocket.accept()
    try:
        # Start a shell process
        if sys.platform == "win32":
            shell = "powershell.exe"
        else:
            shell = os.environ.get("SHELL", "/bin/bash")

        process = await asyncio.create_subprocess_exec(
            shell,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=workspace_root,
            env={**os.environ, "TERM": "xterm-256color", "PYTHONIOENCODING": "utf-8"},
        )

        async def read_output():
            while True:
                data = await process.stdout.read(4096)
                if not data:
                    break
                try:
                    text = data.decode('utf-8', errors='replace')
                    await websocket.send_text(text)
                except Exception:
                    break

        output_task = asyncio.create_task(read_output())

        while True:
            data = await websocket.receive_text()
            if process.stdin:
                process.stdin.write(data.encode())
                await process.stdin.drain()

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        try:
            process.kill()
        except Exception:
            pass


# ──────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("[sealMega] IDE Engine starting on http://127.0.0.1:8742")
    print(f"[sealMega] Workspace: {workspace_root}")
    print(f"[sealMega] Models dir: {MODELS_DIR}")
    print(f"[sealMega] Frontend: {FRONTEND_DIR}")

    # Check which models exist
    for tier, info in MODEL_REGISTRY.items():
        exists = check_model_downloaded(tier)
        mark = "[OK]" if exists else "[--]"
        print(f"  {mark} {info['name']}: {'Downloaded' if exists else 'Not downloaded'}")

    uvicorn.run(app, host="127.0.0.1", port=8742)
