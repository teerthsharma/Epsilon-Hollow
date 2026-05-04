# Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
# SPDX-License-Identifier: Epsilon-Hollow

import os
import sys
from huggingface_hub import hf_hub_download

def download_model():
    repo_id = "QuantFactory/Phi-3-mini-4k-instruct-GGUF"
    files = ["Phi-3-mini-4k-instruct.Q4_K_M.gguf", "tokenizer.json", "tokenizer_config.json"]
    local_dir = "./models"
    
    # Allow configuring mirror
    if os.environ.get("HF_ENDPOINT"):
        print(f"Using custom HF endpoint: {os.environ['HF_ENDPOINT']}")

    print(f"Downloading files from {repo_id}...")

    overall_success = True

    for filename in files:
        # Check if file already exists to avoid re-download
        final_path = os.path.join(local_dir, filename)
        if os.path.exists(final_path):
            print(f"File {filename} already exists at {final_path}. Skipping.")
            continue

        attempts = 0
        success = False
        while attempts < 3 and not success:
            attempts += 1
            try:
                path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir, local_dir_use_symlinks=False)
                print(f"Successfully downloaded {filename} to {path}")
                success = True
            except Exception as e:
                print(f"Attempt {attempts}/3 failed for {filename}: {e}")
                
                # Fallback on failure (simplified from original logic, focusing on reliability)
                if attempts == 3 and "tokenizer" in filename:
                     print(f"Trying fallback to microsoft/Phi-3-mini-4k-instruct...")
                     try:
                        path = hf_hub_download(repo_id="microsoft/Phi-3-mini-4k-instruct", filename=filename, local_dir=local_dir, local_dir_use_symlinks=False)
                        print(f"Successfully downloaded {filename} from official repo.")
                        success = True
                     except Exception as e2:
                        print(f"Fallback failed: {e2}")

        if not success:
            print(f"CRITICAL: Failed to download {filename} after 3 attempts.")
            overall_success = False
            break

    if not overall_success:
        print("Model download failed. Exiting.")
        sys.exit(1)

if __name__ == "__main__":
    download_model()
