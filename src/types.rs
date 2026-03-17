use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Role in a conversation message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

/// A single message in a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
        }
    }
}

/// Token usage info from an LLM response.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UsageInfo {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

impl UsageInfo {
    pub fn accumulate(&mut self, other: &UsageInfo) {
        self.prompt_tokens += other.prompt_tokens;
        self.completion_tokens += other.completion_tokens;
        self.total_tokens += other.total_tokens;
    }
}

/// Response from an LLM completion call.
#[derive(Debug, Clone)]
pub struct CompletionResponse {
    pub content: String,
    pub model: String,
    pub usage: UsageInfo,
}

/// Result of executing code in the REPL.
#[derive(Debug, Clone)]
pub struct ReplResult {
    pub stdout: String,
    pub stderr: String,
    pub final_answer: Option<FinalAnswer>,
    pub execution_time: Duration,
}

/// How the final answer was signaled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinalAnswer {
    /// `FINAL(answer)` — the answer string directly.
    Direct(String),
    /// `FINAL_VAR(var_name)` — resolve the named variable from REPL state.
    Var(String),
}

/// A code block extracted from LLM output.
#[derive(Debug, Clone)]
pub struct CodeBlock {
    pub code: String,
    pub result: Option<ReplResult>,
}

/// The final result of an RLM completion.
#[derive(Debug, Clone)]
pub struct RlmCompletion {
    pub response: String,
    pub iterations: u32,
    pub total_usage: UsageInfo,
}
