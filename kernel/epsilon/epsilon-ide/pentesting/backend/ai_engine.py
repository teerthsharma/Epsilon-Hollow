import os
import asyncio

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

LOADED_MODELS = {}

async def load_model_locally(tier: str, model_path: str):
    """Loads a downloaded model into memory (VRAM/RAM)."""
    if not AI_AVAILABLE:
        raise Exception("Transformers/Torch not installed. Wait for installation to finish.")
    
    if tier in LOADED_MODELS:
        return True

    loop = asyncio.get_event_loop()
    
    def _load():
        device_map = "auto" if torch.cuda.is_available() else "cpu"
        
        # Determine 8-bit quantization based on size to fit in restricted VRAM
        load_in_8bit = True if ("33b" in model_path.lower() or "70b" in model_path.lower() or "7b" in model_path.lower()) else False
        
        print(f"[{tier}] Loading tokenizer from {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        
        print(f"[{tier}] Loading model to {device_map} (8-bit: {load_in_8bit})...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            device_map=device_map,
            load_in_8bit=load_in_8bit,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        LOADED_MODELS[tier] = pipe
        print(f"[{tier}] Model loaded successfully.")
        return True

    await loop.run_in_executor(None, _load)
    return True

async def generate_code(tier: str, prompt: str, max_new_tokens: int = 1024):
    """Generates code utilizing the loaded model."""
    if tier not in LOADED_MODELS:
        raise Exception(f"Model tier '{tier}' is not loaded in memory yet.")
        
    pipe = LOADED_MODELS[tier]
    
    loop = asyncio.get_event_loop()
    def _gen():
        outputs = pipe(
            prompt, 
            max_new_tokens=max_new_tokens, 
            do_sample=True, 
            temperature=0.1, 
            top_p=0.9, 
            return_full_text=False
        )
        return outputs[0]["generated_text"].strip()
        
    return await loop.run_in_executor(None, _gen)
