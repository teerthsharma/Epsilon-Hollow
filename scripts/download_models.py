#!/usr/bin/env python3
"""Download Epsilon-Hollow backend model tiers from Hugging Face."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "foreman": {
        "repo_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "subdir": "tinyllama-1.1b",
    },
    "logicgate": {
        "repo_id": "Qwen/Qwen2.5-Coder-7B",
        "subdir": "qwen2.5-coder-7b",
    },
    "architect": {
        "repo_id": "deepseek-ai/deepseek-coder-33b-instruct",
        "subdir": "deepseek-coder-33b",
    },
}

MODEL_FILE_SUFFIXES = (".safetensors", ".bin", ".gguf", ".ggml")


def has_model_files(model_dir: Path) -> bool:
    if not model_dir.exists() or not model_dir.is_dir():
        return False
    for file in model_dir.rglob("*"):
        if file.is_file() and file.suffix.lower() in MODEL_FILE_SUFFIXES:
            return True
    return False


def parse_tiers(raw_tiers: str) -> list[str]:
    if raw_tiers.strip().lower() == "all":
        return list(MODEL_REGISTRY.keys())

    tiers = [part.strip().lower() for part in raw_tiers.split(",") if part.strip()]
    if not tiers:
        raise ValueError("At least one tier must be provided.")

    unknown = [tier for tier in tiers if tier not in MODEL_REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown tier(s): {', '.join(unknown)}. Valid tiers: foreman, logicgate, architect"
        )

    return tiers


def download_tier(
    tier: str,
    backend_dir: Path,
    token: str | None,
    force: bool = False,
    hf_workers: int = 8,
    retries: int = 2,
) -> None:
    model_info = MODEL_REGISTRY[tier]
    repo_id = model_info["repo_id"]
    local_dir = backend_dir / "models" / model_info["subdir"]

    if force and local_dir.exists():
        print(f"[download] {tier}: removing existing directory {local_dir}")
        shutil.rmtree(local_dir)

    local_dir.mkdir(parents=True, exist_ok=True)

    if not force and has_model_files(local_dir):
        print(f"[download] {tier}: already present at {local_dir}")
        return

    attempts = retries + 1
    for attempt in range(1, attempts + 1):
        try:
            print(
                f"[download] {tier}: downloading {repo_id} -> {local_dir} "
                f"(attempt {attempt}/{attempts})"
            )
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
                token=token,
                max_workers=hf_workers,
                force_download=force,
            )
            print(f"[download] {tier}: complete")
            return
        except Exception:
            if attempt >= attempts:
                raise
            backoff_s = min(15, 2 * attempt)
            print(f"[download] {tier}: retrying in {backoff_s}s")
            time.sleep(backoff_s)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Epsilon-Hollow backend models")
    parser.add_argument(
        "--backend-dir",
        required=True,
        help="Path to kernel/epsilon/epsilon-ide/pentesting/backend",
    )
    parser.add_argument(
        "--tiers",
        default="all",
        help="Comma-separated model tiers (foreman,logicgate,architect) or 'all'",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force fresh snapshot download by clearing local tier directories first",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Parallel tier download workers (default: 3)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retries per tier on transient failure (default: 2)",
    )
    parser.add_argument(
        "--hf-workers",
        type=int,
        default=8,
        help="Per-tier Hugging Face download workers (default: 8)",
    )
    args = parser.parse_args()

    backend_dir = Path(args.backend_dir).resolve()
    if not backend_dir.exists():
        print(f"[error] backend directory does not exist: {backend_dir}", file=sys.stderr)
        return 2

    try:
        tiers = parse_tiers(args.tiers)
    except ValueError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        print("[download] using Hugging Face token from environment")

    tier_workers = max(1, min(args.workers, len(tiers)))
    retries = max(0, args.retries)
    hf_workers = max(1, args.hf_workers)
    print(
        f"[download] parallel tier workers={tier_workers}, retries={retries}, "
        f"hf_workers={hf_workers}"
    )

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=tier_workers) as executor:
        future_map = {
            executor.submit(
                download_tier,
                tier,
                backend_dir,
                token,
                args.force,
                hf_workers,
                retries,
            ): tier
            for tier in tiers
        }

        for future in as_completed(future_map):
            tier = future_map[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{tier}: {exc}")
                print(f"[download] {tier}: failed: {exc}", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)

    if failures:
        print("[error] model download failed for:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print("[download] all requested model tiers are ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
