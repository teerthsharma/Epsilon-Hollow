// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow

//! Multi-provider LLM bridge.
//!
//! Supports OpenAI (and compatible endpoints like Ollama, Together, Groq),
//! Google Gemini, and a no-op offline provider. Provider selection is via
//! environment variables — no API key required for core functionality.

use std::process::Command;

/// Errors from LLM calls.
#[derive(Debug)]
#[allow(dead_code)]
pub enum LlmError {
    NotConfigured,
    RequestFailed(String),
    ParseError(String),
}

impl std::fmt::Display for LlmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotConfigured => write!(f, "LLM not configured"),
            Self::RequestFailed(e) => write!(f, "request failed: {e}"),
            Self::ParseError(e) => write!(f, "parse error: {e}"),
        }
    }
}

/// Which LLM provider to use.
#[derive(Debug, Clone, PartialEq)]
pub enum Provider {
    OpenAi,
    Gemini,
    None,
}

/// Configuration for the LLM bridge.
#[derive(Debug, Clone)]
pub struct LlmConfig {
    pub provider: Provider,
    pub api_key: String,
    pub model: String,
    pub endpoint: String,
    pub max_tokens: u32,
}

impl LlmConfig {
    /// Build config from environment variables.
    ///
    /// Env vars checked:
    /// - `LLM_PROVIDER`: "openai", "gemini", or "none" (default: auto-detect from key)
    /// - `LLM_API_KEY`: universal key (fallback: `OPENAI_API_KEY` or `GEMINI_API_KEY`)
    /// - `LLM_MODEL`: model name (defaults per provider)
    /// - `LLM_ENDPOINT`: custom endpoint URL
    /// - `MINIMAX_API_KEY`: legacy compat (treated as OpenAI-compatible)
    pub fn from_env() -> Self {
        let provider_str = std::env::var("LLM_PROVIDER").unwrap_or_default();
        let api_key = Self::resolve_api_key(&provider_str);
        let provider = Self::resolve_provider(&provider_str, &api_key);
        let model = std::env::var("LLM_MODEL").unwrap_or_else(|_| match provider {
            Provider::OpenAi => "gpt-4o-mini".into(),
            Provider::Gemini => "gemini-2.0-flash".into(),
            Provider::None => String::new(),
        });
        let endpoint = std::env::var("LLM_ENDPOINT").unwrap_or_else(|_| match provider {
            Provider::OpenAi => "https://api.openai.com/v1/chat/completions".into(),
            Provider::Gemini => String::new(), // built dynamically with model name
            Provider::None => String::new(),
        });

        Self {
            provider,
            api_key,
            model,
            endpoint,
            max_tokens: 1024,
        }
    }

    /// Build config from explicit CLI args (overrides env).
    pub fn with_overrides(mut self, api_key: Option<String>, provider: Option<String>) -> Self {
        if let Some(key) = api_key {
            if !key.is_empty() {
                self.api_key = key;
            }
        }
        if let Some(p) = provider {
            self.provider = Self::resolve_provider(&p, &self.api_key);
        }
        // Re-detect if we got a key but provider was None
        if self.provider == Provider::None && !self.api_key.is_empty() {
            self.provider = if self.api_key.starts_with("sk-") {
                Provider::OpenAi
            } else if self.api_key.starts_with("AIza") {
                Provider::Gemini
            } else {
                Provider::OpenAi // default to OpenAI-compatible
            };
        }
        self
    }

    fn resolve_api_key(provider_str: &str) -> String {
        // Check universal key first
        if let Ok(key) = std::env::var("LLM_API_KEY") {
            if !key.is_empty() {
                return key;
            }
        }
        // Provider-specific keys
        match provider_str {
            "gemini" => std::env::var("GEMINI_API_KEY").unwrap_or_default(),
            "openai" => std::env::var("OPENAI_API_KEY").unwrap_or_default(),
            _ => {
                // Try all known key vars
                std::env::var("OPENAI_API_KEY")
                    .or_else(|_| std::env::var("GEMINI_API_KEY"))
                    .or_else(|_| std::env::var("MINIMAX_API_KEY"))
                    .unwrap_or_default()
            }
        }
    }

    fn resolve_provider(provider_str: &str, api_key: &str) -> Provider {
        match provider_str.to_lowercase().as_str() {
            "openai" => Provider::OpenAi,
            "gemini" => Provider::Gemini,
            "none" | "offline" => Provider::None,
            _ => {
                // Auto-detect from key prefix
                if api_key.starts_with("sk-") {
                    Provider::OpenAi
                } else if api_key.starts_with("AIza") {
                    Provider::Gemini
                } else if api_key.is_empty() {
                    Provider::None
                } else {
                    Provider::OpenAi // assume OpenAI-compatible
                }
            }
        }
    }

    #[allow(dead_code)]
    pub fn offline() -> Self {
        Self {
            provider: Provider::None,
            api_key: String::new(),
            model: String::new(),
            endpoint: String::new(),
            max_tokens: 1024,
        }
    }

    pub fn is_configured(&self) -> bool {
        self.provider != Provider::None && !self.api_key.is_empty()
    }

    pub fn provider_name(&self) -> &'static str {
        match self.provider {
            Provider::OpenAi => "OpenAI",
            Provider::Gemini => "Gemini",
            Provider::None => "offline",
        }
    }
}

/// The LLM bridge that dispatches to the configured provider.
pub struct LlmBridge {
    pub config: LlmConfig,
    offline_notice_shown: std::cell::Cell<bool>,
}

impl LlmBridge {
    pub fn new(config: LlmConfig) -> Self {
        Self {
            config,
            offline_notice_shown: std::cell::Cell::new(false),
        }
    }

    pub fn is_configured(&self) -> bool {
        self.config.is_configured()
    }

    /// Call the LLM with system + user messages. Returns None in offline mode.
    pub fn call(&self, system: &str, user: &str) -> Result<Option<String>, LlmError> {
        if !self.is_configured() {
            if !self.offline_notice_shown.get() {
                self.offline_notice_shown.set(true);
                eprintln!(
                    "  [info] No LLM configured. Core math active. Set LLM_API_KEY for synthesis."
                );
            }
            return Ok(None);
        }

        match self.config.provider {
            Provider::OpenAi => self.call_openai(system, user).map(Some),
            Provider::Gemini => self.call_gemini(system, user).map(Some),
            Provider::None => Ok(None),
        }
    }

    fn call_openai(&self, system: &str, user: &str) -> Result<String, LlmError> {
        let body = serde_json::json!({
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": self.config.max_tokens,
        });

        let response = self.curl_post(&self.config.endpoint, &body)?;
        let parsed: serde_json::Value =
            serde_json::from_str(&response).map_err(|e| LlmError::ParseError(e.to_string()))?;

        parsed["choices"][0]["message"]["content"]
            .as_str()
            .map(|s| s.to_string())
            .ok_or_else(|| {
                LlmError::ParseError(format!("unexpected response structure: {response}"))
            })
    }

    fn call_gemini(&self, system: &str, user: &str) -> Result<String, LlmError> {
        let endpoint = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
            self.config.model, self.config.api_key
        );

        let body = serde_json::json!({
            "system_instruction": {
                "parts": [{"text": system}]
            },
            "contents": [{
                "parts": [{"text": user}]
            }],
            "generationConfig": {
                "maxOutputTokens": self.config.max_tokens
            }
        });

        let response = self.curl_post_no_auth(&endpoint, &body)?;
        let parsed: serde_json::Value =
            serde_json::from_str(&response).map_err(|e| LlmError::ParseError(e.to_string()))?;

        parsed["candidates"][0]["content"]["parts"][0]["text"]
            .as_str()
            .map(|s| s.to_string())
            .ok_or_else(|| LlmError::ParseError(format!("unexpected Gemini response: {response}")))
    }

    fn curl_post(&self, endpoint: &str, body: &serde_json::Value) -> Result<String, LlmError> {
        let output = Command::new("curl")
            .args([
                "-s",
                "-X",
                "POST",
                endpoint,
                "-H",
                "Content-Type: application/json",
                "-H",
                &format!("Authorization: Bearer {}", self.config.api_key),
                "-d",
                &body.to_string(),
                "--max-time",
                "30",
            ])
            .output()
            .map_err(|e| LlmError::RequestFailed(format!("curl failed: {e}")))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(LlmError::RequestFailed(format!("curl error: {stderr}")));
        }

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    fn curl_post_no_auth(
        &self,
        endpoint: &str,
        body: &serde_json::Value,
    ) -> Result<String, LlmError> {
        let output = Command::new("curl")
            .args([
                "-s",
                "-X",
                "POST",
                endpoint,
                "-H",
                "Content-Type: application/json",
                "-d",
                &body.to_string(),
                "--max-time",
                "30",
            ])
            .output()
            .map_err(|e| LlmError::RequestFailed(format!("curl failed: {e}")))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(LlmError::RequestFailed(format!("curl error: {stderr}")));
        }

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }
}
