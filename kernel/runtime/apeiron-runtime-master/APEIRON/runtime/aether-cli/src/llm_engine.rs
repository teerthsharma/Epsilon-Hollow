// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow

use anyhow::{Error as E, Result};
use candle_core::{Device, Tensor};
use candle_core::quantized::gguf_file;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_llama as model;
use tokenizers::Tokenizer;
use std::path::{Path, PathBuf};
use std::io::Write;


pub struct LLMEngine {
    model_path: PathBuf,
    tokenizer: Option<Tokenizer>,
    logits_processor: LogitsProcessor,
    device: Device,
    is_mock: bool,
}

impl LLMEngine {
    pub fn new(model_path: &Path, tokenizer_path: &Path) -> Result<Self> {
        let device_env = std::env::var("AETHER_DEVICE").unwrap_or_else(|_| "cpu".to_string());
        
        let device = if device_env.to_lowercase() == "cuda" || device_env.to_lowercase() == "gpu" {
            println!("[LLM] Attempting to use CUDA Device...");
            Device::new_cuda(0).unwrap_or_else(|e| {
                eprintln!("[LLM] Warning: CUDA requested but failed: {}. Falling back to CPU.", e);
                Device::Cpu
            })
        } else {
            Device::Cpu
        };
        
        println!("[LLM] Using Device: {:?}", device);
        
        // Ensure paths exist
        let is_mock = !model_path.exists() || !tokenizer_path.exists();

        if is_mock {
            println!("[LLM] Model files not found. Initializing MOCK engine.");
            return Ok(Self {
                model_path: model_path.to_path_buf(),
                tokenizer: None,
                logits_processor: LogitsProcessor::new(42, Some(0.7), Some(0.9)),
                device,
                is_mock: true,
            });
        }

        println!("[LLM] Loading tokenizer from {:?}", tokenizer_path);
        let tokenizer_instance = Tokenizer::from_file(tokenizer_path).map_err(E::msg)?;
        
        Ok(Self {
            model_path: model_path.to_path_buf(),
            tokenizer: Some(tokenizer_instance),
            logits_processor: LogitsProcessor::new(42, Some(0.7), Some(0.9)), // Seed, Temp, Top-P
            device,
            is_mock: false,
        })
    }

    fn load_model(&self) -> Result<model::ModelWeights> {
        let mut file = std::fs::File::open(&self.model_path)?;
        let content = gguf_file::Content::read(&mut file)?;
        let model = model::ModelWeights::from_gguf(content, &mut file, &self.device)?;
        Ok(model)
    }

    pub fn generate(&mut self, prompt: &str, mut callback: impl FnMut(String)) -> Result<()> {
        if self.is_mock {
            // Simulate generation delay
            callback("Generating".to_string());
            // Simulate ~2 seconds of work
            for _ in 0..10 {
                std::thread::sleep(std::time::Duration::from_millis(200));
                callback(".".to_string());
            }
            callback(" Done.\n".to_string());
            return Ok(());
        }

        // Real generation
        let tokenizer = self.tokenizer.as_ref().unwrap();
        let mut model = self.load_model()?; 

        // Format prompt for Phi-3
        // <|user|>\n...\n<|end|>\n<|assistant|>\n
        let formatted_prompt = format!("<|user|>\n{}<|end|>\n<|assistant|>\n", prompt);
        
        let tokens = tokenizer.encode(formatted_prompt, true).map_err(E::msg)?;
        let mut token_ids = tokens.get_ids().to_vec();
        let max_tokens = 512;
        println!("[LLM] Starting generation for prompt ({} tokens)", token_ids.len());

        for i in 0..max_tokens {
            let (input, p) = if i == 0 {
                (Tensor::new(token_ids.as_slice(), &self.device)?.unsqueeze(0)?, 0)
            } else {
                let last_token = *token_ids.last().unwrap();
                (Tensor::new(&[last_token], &self.device)?.unsqueeze(0)?, token_ids.len() - 1)
            };

            let logits = model.forward(&input, p)?;
            let logits = logits.squeeze(0)?;
            
            // If it's the first pass (prompt processing), we only care about the last logit
            let logits = if i == 0 {
                logits.get(logits.dims()[0] - 1)?
            } else {
                logits
            };

            let next_token = self.logits_processor.sample(&logits)?;
            token_ids.push(next_token);

            if let Some(text) = tokenizer.decode(&[next_token], true).ok() {
                print!("{}", text);
                let _ = std::io::stdout().flush();
                callback(text);
            }

            if next_token == 32000 || next_token == 32007 {
                println!("\n[LLM] Generation finished (EOS)");
                break;
            }
        }

        Ok(())
    }
}
