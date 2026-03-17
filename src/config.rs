use std::time::Duration;

/// Configuration for the RLM algorithm.
#[derive(Debug, Clone)]
pub struct RlmConfig {
    /// Maximum recursion depth for sub-RLM calls. Default: 1.
    pub max_depth: u32,
    /// Maximum iterations of the REPL loop. Default: 30.
    pub max_iterations: u32,
    /// Overall timeout for the entire RLM completion. Default: None (no timeout).
    pub max_timeout: Option<Duration>,
    /// Maximum characters of REPL output to include in metadata. Default: 20000.
    pub max_output_chars: usize,
    /// Optional custom system prompt override.
    pub system_prompt: Option<String>,
}

impl Default for RlmConfig {
    fn default() -> Self {
        Self {
            max_depth: 1,
            max_iterations: 30,
            max_timeout: None,
            max_output_chars: 20000,
            system_prompt: None,
        }
    }
}
