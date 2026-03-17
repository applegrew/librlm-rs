use thiserror::Error;

/// All errors that can occur in the RLM library.
#[derive(Debug, Error)]
pub enum RlmError {
    #[error("LLM error: {0}")]
    LlmError(String),

    #[error("REPL error: {0}")]
    ReplError(String),

    #[error("REPL spawn error: {0}")]
    ReplSpawnError(String),

    #[error("Timeout exceeded: {0}")]
    TimeoutExceeded(String),

    #[error("Budget exceeded: {0}")]
    BudgetExceeded(String),

    #[error("Max iterations reached: {0}")]
    MaxIterationsReached(u32),

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Config error: {0}")]
    ConfigError(String),

    #[error("Bridge error: {0}")]
    BridgeError(String),

    #[error("HTTP error: {0}")]
    HttpError(String),

    #[error("JSON error: {0}")]
    JsonError(String),
}

impl From<reqwest::Error> for RlmError {
    fn from(e: reqwest::Error) -> Self {
        RlmError::HttpError(e.to_string())
    }
}

impl From<serde_json::Error> for RlmError {
    fn from(e: serde_json::Error) -> Self {
        RlmError::JsonError(e.to_string())
    }
}
