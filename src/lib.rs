//! # librlm
//!
//! Implementation of the Recursive Language Models (RLM) algorithm as described in
//! "Recursive Language Models" (Zhang, Kraska, Khattab — MIT CSAIL, Jan 2026).
//!
//! RLM enables LLMs to handle arbitrarily long prompts by treating them as part of
//! an external environment. The LLM interacts with the prompt through a persistent
//! Lua REPL, writing code to peek at, decompose, and recursively invoke sub-LLMs
//! over manageable chunks.
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use librlm::Rlm;
//!
//! # async fn example() -> Result<(), librlm::RlmError> {
//! let rlm = Rlm::builder()
//!     .root_model("gpt-5")
//!     .root_api_key("sk-...")
//!     .sub_model("gpt-5-mini")
//!     .max_iterations(30)
//!     .build()?;
//!
//! let result = rlm.completion("very long prompt...", Some("What is X?")).await?;
//! println!("{}", result.response);
//! # Ok(())
//! # }
//! ```

mod config;
mod error;
mod llm;
mod metadata;
mod parsing;
mod prompt;
mod repl;
mod rlm;
mod types;

pub use config::RlmConfig;
pub use error::RlmError;
pub use llm::{LlmBackend, OpenAiBackend};
pub use rlm::{Rlm, RlmBuilder};
pub use types::{
    CodeBlock, CompletionResponse, FinalAnswer, Message, ReplResult, RlmCompletion, Role,
    UsageInfo,
};
