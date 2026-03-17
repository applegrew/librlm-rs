use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use crate::error::RlmError;
use crate::types::{CompletionResponse, Message, UsageInfo};

/// Trait for LLM backends. Implement this for custom providers.
#[async_trait]
pub trait LlmBackend: Send + Sync {
    /// The model name this backend uses.
    fn model_name(&self) -> &str;

    /// Send a completion request with the given messages.
    async fn completion(&self, messages: &[Message]) -> Result<CompletionResponse, RlmError>;
}

/// OpenAI-compatible API backend.
pub struct OpenAiBackend {
    model: String,
    api_key: String,
    base_url: String,
    client: Client,
    total_prompt_tokens: Arc<AtomicU32>,
    total_completion_tokens: Arc<AtomicU32>,
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: &'a [Message],
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
    model: String,
    usage: Option<ChatUsage>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatMessage,
}

#[derive(Deserialize)]
struct ChatMessage {
    content: Option<String>,
}

#[derive(Deserialize)]
struct ChatUsage {
    prompt_tokens: Option<u32>,
    completion_tokens: Option<u32>,
    total_tokens: Option<u32>,
}

impl OpenAiBackend {
    /// Create a new OpenAI-compatible backend.
    ///
    /// - `model`: Model name (e.g., "gpt-5", "gpt-5-mini")
    /// - `api_key`: API key for authentication
    /// - `base_url`: Optional base URL. Defaults to `https://api.openai.com/v1`.
    pub fn new(
        model: impl Into<String>,
        api_key: impl Into<String>,
        base_url: Option<String>,
    ) -> Self {
        Self {
            model: model.into(),
            api_key: api_key.into(),
            base_url: base_url.unwrap_or_else(|| "https://api.openai.com/v1".to_string()),
            client: Client::new(),
            total_prompt_tokens: Arc::new(AtomicU32::new(0)),
            total_completion_tokens: Arc::new(AtomicU32::new(0)),
        }
    }

    /// Get cumulative prompt tokens used.
    pub fn total_prompt_tokens(&self) -> u32 {
        self.total_prompt_tokens.load(Ordering::Relaxed)
    }

    /// Get cumulative completion tokens used.
    pub fn total_completion_tokens(&self) -> u32 {
        self.total_completion_tokens.load(Ordering::Relaxed)
    }
}

#[async_trait]
impl LlmBackend for OpenAiBackend {
    fn model_name(&self) -> &str {
        &self.model
    }

    async fn completion(&self, messages: &[Message]) -> Result<CompletionResponse, RlmError> {
        let url = format!("{}/chat/completions", self.base_url);

        let request = ChatRequest {
            model: &self.model,
            messages,
        };

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "unknown".to_string());
            return Err(RlmError::LlmError(format!("HTTP {}: {}", status, body)));
        }

        let chat_response: ChatResponse = response.json().await?;

        let content = chat_response
            .choices
            .first()
            .and_then(|c| c.message.content.clone())
            .unwrap_or_default();

        let usage = chat_response.usage.map_or(UsageInfo::default(), |u| {
            let prompt = u.prompt_tokens.unwrap_or(0);
            let completion = u.completion_tokens.unwrap_or(0);
            let total = u.total_tokens.unwrap_or(prompt + completion);
            UsageInfo {
                prompt_tokens: prompt,
                completion_tokens: completion,
                total_tokens: total,
            }
        });

        self.total_prompt_tokens
            .fetch_add(usage.prompt_tokens, Ordering::Relaxed);
        self.total_completion_tokens
            .fetch_add(usage.completion_tokens, Ordering::Relaxed);

        Ok(CompletionResponse {
            content,
            model: chat_response.model,
            usage,
        })
    }
}
