use std::sync::Arc;

use tracing::{debug, info, warn};

use crate::config::RlmConfig;
use crate::error::RlmError;
use crate::llm::LlmBackend;
use crate::metadata::{format_repl_output, QueryMetadata};
use crate::parsing::{find_code_blocks, find_final_answer};
use crate::prompt::{build_system_prompt, build_user_message};
use crate::repl::LuaRepl;
use crate::types::{FinalAnswer, Message, RlmCompletion, UsageInfo};

/// The main RLM (Recursive Language Model) engine.
///
/// Implements Algorithm 1 from the RLM paper: the LLM interacts with a
/// long prompt through a persistent REPL, writing code to explore, decompose,
/// and recursively query sub-LLMs.
pub struct Rlm {
    config: RlmConfig,
    root_backend: Arc<dyn LlmBackend>,
    sub_backend: Arc<dyn LlmBackend>,
}

impl Rlm {
    /// Create a new RLM instance.
    pub fn new(
        config: RlmConfig,
        root_backend: Arc<dyn LlmBackend>,
        sub_backend: Option<Arc<dyn LlmBackend>>,
    ) -> Self {
        let sub = sub_backend.unwrap_or_else(|| root_backend.clone());
        Self {
            config,
            root_backend,
            sub_backend: sub,
        }
    }

    /// Create a builder for constructing an RLM instance.
    pub fn builder() -> RlmBuilder {
        RlmBuilder::default()
    }

    /// Run the RLM algorithm on the given prompt.
    ///
    /// - `prompt`: The (potentially very long) context to analyze
    /// - `query`: Optional specific question about the prompt
    pub async fn completion(
        &self,
        prompt: &str,
        query: Option<&str>,
    ) -> Result<RlmCompletion, RlmError> {
        self.completion_inner(prompt, query, 0).await
    }

    async fn completion_inner(
        &self,
        prompt: &str,
        query: Option<&str>,
        depth: u32,
    ) -> Result<RlmCompletion, RlmError> {
        // Base case: if at max depth, just do a plain LLM call
        if depth >= self.config.max_depth {
            info!(depth, "Max depth reached, falling back to plain LLM call");
            let truncated = if prompt.len() > self.config.max_output_chars {
                &prompt[..self.config.max_output_chars]
            } else {
                prompt
            };
            let content = match query {
                Some(q) => format!("{}\n\n{}", truncated, q),
                None => truncated.to_string(),
            };
            let messages = vec![Message::user(content)];
            let resp = self.root_backend.completion(&messages).await?;
            return Ok(RlmCompletion {
                response: resp.content,
                iterations: 0,
                total_usage: resp.usage,
            });
        }

        // Initialize REPL
        let repl = LuaRepl::new(self.sub_backend.clone())
            .map_err(|e| RlmError::ReplSpawnError(e.to_string()))?;
        repl.load_context(prompt)?;

        // Build initial history
        let metadata = QueryMetadata::new(prompt);
        let mut history: Vec<Message> = build_system_prompt(&metadata);
        history.push(build_user_message(query, &metadata));

        let mut total_usage = UsageInfo::default();
        let mut iterations = 0u32;

        // Main RLM loop
        loop {
            if iterations >= self.config.max_iterations {
                warn!(iterations, "Max iterations reached");
                // Ask LLM one last time for a forced answer
                history.push(Message::user(
                    "You have reached the maximum number of iterations. \
                     Please provide your best answer now using FINAL(\"your answer\")."
                        .to_string(),
                ));
                let resp = self.root_backend.completion(&history).await?;
                total_usage.accumulate(&resp.usage);

                // Try to extract final answer from forced response
                if let Some(fa) = find_final_answer(&resp.content) {
                    let answer = match fa {
                        FinalAnswer::Direct(s) => s,
                        FinalAnswer::Var(_) => resp.content.clone(),
                    };
                    return Ok(RlmCompletion {
                        response: answer,
                        iterations,
                        total_usage,
                    });
                }

                return Err(RlmError::MaxIterationsReached(iterations));
            }

            // Call root LLM
            debug!(iteration = iterations, "Calling root LLM");
            let llm_response = self.root_backend.completion(&history).await?;
            total_usage.accumulate(&llm_response.usage);
            let assistant_content = llm_response.content.clone();

            // Add assistant message to history
            history.push(Message::assistant(&assistant_content));

            // Extract code blocks
            let code_blocks = find_code_blocks(&assistant_content);

            if code_blocks.is_empty() {
                // No code blocks — check if there's a final answer in the text
                if let Some(fa) = find_final_answer(&assistant_content) {
                    let answer = match fa {
                        FinalAnswer::Direct(s) => s,
                        FinalAnswer::Var(name) => repl.get_variable(&name)?,
                    };
                    info!(iterations, "Final answer found in text");
                    return Ok(RlmCompletion {
                        response: answer,
                        iterations,
                        total_usage,
                    });
                }

                // No code and no final answer — prompt LLM to use the REPL
                history.push(Message::user(
                    "Please use the REPL (write code in ```repl``` blocks) to explore the context. \
                     Do not try to answer from memory."
                        .to_string(),
                ));
                iterations += 1;
                continue;
            }

            // Execute each code block
            let mut all_output = String::new();
            let mut found_final = None;

            for code in &code_blocks {
                debug!("Executing code block");
                let result = repl.execute_code(code);

                if !result.stdout.is_empty() {
                    all_output.push_str(&result.stdout);
                }
                if !result.stderr.is_empty() {
                    all_output.push_str(&format!("[Error]: {}\n", result.stderr));
                }

                if let Some(fa) = &result.final_answer {
                    found_final = Some(fa.clone());
                    break;
                }
            }

            // Check for final answer
            if let Some(fa) = found_final {
                let answer = match fa {
                    FinalAnswer::Direct(s) => s,
                    FinalAnswer::Var(name) => repl.get_variable(&name)?,
                };
                info!(iterations, "Final answer from REPL");
                return Ok(RlmCompletion {
                    response: answer,
                    iterations,
                    total_usage,
                });
            }

            // Also check for FINAL in stdout (in case print output contains it)
            if let Some(fa) = find_final_answer(&all_output) {
                let answer = match fa {
                    FinalAnswer::Direct(s) => s,
                    FinalAnswer::Var(name) => repl.get_variable(&name)?,
                };
                info!(iterations, "Final answer found in REPL output");
                return Ok(RlmCompletion {
                    response: answer,
                    iterations,
                    total_usage,
                });
            }

            // Append REPL output metadata to history
            let output_metadata =
                format_repl_output(&all_output, self.config.max_output_chars);
            history.push(Message::user(format!(
                "[REPL Output]:\n{}",
                output_metadata
            )));

            iterations += 1;
        }
    }
}

/// Builder for constructing an `Rlm` instance.
#[derive(Default)]
pub struct RlmBuilder {
    root_model: Option<String>,
    root_api_key: Option<String>,
    root_base_url: Option<String>,
    sub_model: Option<String>,
    sub_api_key: Option<String>,
    sub_base_url: Option<String>,
    root_backend: Option<Arc<dyn LlmBackend>>,
    sub_backend: Option<Arc<dyn LlmBackend>>,
    max_iterations: Option<u32>,
    max_depth: Option<u32>,
    max_output_chars: Option<usize>,
    max_timeout: Option<std::time::Duration>,
    system_prompt: Option<String>,
}

impl RlmBuilder {
    /// Set the root model name (e.g., "gpt-5").
    pub fn root_model(mut self, model: impl Into<String>) -> Self {
        self.root_model = Some(model.into());
        self
    }

    /// Set the root API key.
    pub fn root_api_key(mut self, key: impl Into<String>) -> Self {
        self.root_api_key = Some(key.into());
        self
    }

    /// Set the root base URL. Defaults to OpenAI's API.
    pub fn root_base_url(mut self, url: impl Into<String>) -> Self {
        self.root_base_url = Some(url.into());
        self
    }

    /// Set the sub-call model name (e.g., "gpt-5-mini").
    pub fn sub_model(mut self, model: impl Into<String>) -> Self {
        self.sub_model = Some(model.into());
        self
    }

    /// Set the sub-call API key. Defaults to root API key if not set.
    pub fn sub_api_key(mut self, key: impl Into<String>) -> Self {
        self.sub_api_key = Some(key.into());
        self
    }

    /// Set the sub-call base URL. Defaults to root base URL if not set.
    pub fn sub_base_url(mut self, url: impl Into<String>) -> Self {
        self.sub_base_url = Some(url.into());
        self
    }

    /// Set a custom root LLM backend (for non-OpenAI providers).
    pub fn root_backend(mut self, backend: Arc<dyn LlmBackend>) -> Self {
        self.root_backend = Some(backend);
        self
    }

    /// Set a custom sub-call LLM backend.
    pub fn sub_backend(mut self, backend: Arc<dyn LlmBackend>) -> Self {
        self.sub_backend = Some(backend);
        self
    }

    /// Set maximum iterations. Default: 30.
    pub fn max_iterations(mut self, n: u32) -> Self {
        self.max_iterations = Some(n);
        self
    }

    /// Set maximum recursion depth. Default: 1.
    pub fn max_depth(mut self, n: u32) -> Self {
        self.max_depth = Some(n);
        self
    }

    /// Set maximum output characters per REPL output. Default: 20000.
    pub fn max_output_chars(mut self, n: usize) -> Self {
        self.max_output_chars = Some(n);
        self
    }

    /// Set overall timeout for the RLM completion.
    pub fn max_timeout(mut self, d: std::time::Duration) -> Self {
        self.max_timeout = Some(d);
        self
    }

    /// Set a custom system prompt override.
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Build the `Rlm` instance.
    pub fn build(self) -> Result<Rlm, RlmError> {
        use crate::llm::OpenAiBackend;

        let root_api_key = self.root_api_key;
        let root_base_url = self.root_base_url;

        let root_backend: Arc<dyn LlmBackend> = match self.root_backend {
            Some(b) => b,
            None => {
                let model = self
                    .root_model
                    .ok_or_else(|| RlmError::ConfigError("root_model is required".into()))?;
                let api_key = root_api_key
                    .clone()
                    .ok_or_else(|| RlmError::ConfigError("root_api_key is required".into()))?;
                Arc::new(OpenAiBackend::new(model, api_key, root_base_url.clone()))
            }
        };

        let sub_backend: Option<Arc<dyn LlmBackend>> = match self.sub_backend {
            Some(b) => Some(b),
            None => {
                if let Some(model) = self.sub_model {
                    let api_key = self
                        .sub_api_key
                        .or_else(|| root_api_key.clone())
                        .ok_or_else(|| {
                            RlmError::ConfigError(
                                "sub_api_key is required when sub_model is set".into(),
                            )
                        })?;
                    let base_url = self.sub_base_url.or_else(|| root_base_url.clone());
                    Some(Arc::new(OpenAiBackend::new(model, api_key, base_url)))
                } else {
                    None
                }
            }
        };

        let config = RlmConfig {
            max_depth: self.max_depth.unwrap_or(1),
            max_iterations: self.max_iterations.unwrap_or(30),
            max_timeout: self.max_timeout,
            max_output_chars: self.max_output_chars.unwrap_or(20000),
            system_prompt: self.system_prompt,
        };

        Ok(Rlm::new(config, root_backend, sub_backend))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CompletionResponse, UsageInfo};
    use std::sync::atomic::{AtomicU32, Ordering};

    /// A mock LLM that returns scripted responses.
    struct ScriptedBackend {
        responses: Vec<String>,
        call_count: AtomicU32,
    }

    impl ScriptedBackend {
        fn new(responses: Vec<String>) -> Self {
            Self {
                responses,
                call_count: AtomicU32::new(0),
            }
        }
    }

    #[async_trait::async_trait]
    impl LlmBackend for ScriptedBackend {
        fn model_name(&self) -> &str {
            "scripted"
        }

        async fn completion(
            &self,
            _messages: &[Message],
        ) -> Result<CompletionResponse, RlmError> {
            let idx = self.call_count.fetch_add(1, Ordering::SeqCst) as usize;
            let content = if idx < self.responses.len() {
                self.responses[idx].clone()
            } else {
                self.responses.last().cloned().unwrap_or_default()
            };
            Ok(CompletionResponse {
                content,
                model: "scripted".to_string(),
                usage: UsageInfo {
                    prompt_tokens: 10,
                    completion_tokens: 20,
                    total_tokens: 30,
                },
            })
        }
    }

    #[tokio::test]
    async fn test_simple_final_direct() {
        let backend = Arc::new(ScriptedBackend::new(vec![
            // Iteration 1: explore context
            "Let me look at the context:\n```repl\nprint(string.sub(context, 1, 100))\n```"
                .to_string(),
            // Iteration 2: provide final answer
            "Based on what I see:\n```repl\nFINAL(\"The answer is 42\")\n```".to_string(),
        ]));

        let rlm = Rlm::new(
            RlmConfig {
                max_depth: 1,
                max_iterations: 10,
                ..Default::default()
            },
            backend.clone(),
            Some(backend),
        );

        let result = rlm
            .completion("This is a test context with the number 42", Some("What is the number?"))
            .await
            .unwrap();

        assert_eq!(result.response, "The answer is 42");
        assert_eq!(result.iterations, 1);
    }

    #[tokio::test]
    async fn test_final_var() {
        let backend = Arc::new(ScriptedBackend::new(vec![
            "```repl\nresult = \"computed value\"\nFINAL_VAR(\"result\")\n```".to_string(),
        ]));

        let rlm = Rlm::new(
            RlmConfig {
                max_depth: 1,
                max_iterations: 10,
                ..Default::default()
            },
            backend.clone(),
            Some(backend),
        );

        let result = rlm.completion("test prompt", None).await.unwrap();
        assert_eq!(result.response, "computed value");
    }

    #[tokio::test]
    async fn test_depth_zero_falls_back() {
        let backend = Arc::new(ScriptedBackend::new(vec![
            "Direct answer without REPL".to_string(),
        ]));

        let rlm = Rlm::new(
            RlmConfig {
                max_depth: 0,
                max_iterations: 10,
                ..Default::default()
            },
            backend.clone(),
            Some(backend),
        );

        let result = rlm.completion("test prompt", None).await.unwrap();
        assert_eq!(result.response, "Direct answer without REPL");
        assert_eq!(result.iterations, 0);
    }

    #[tokio::test]
    async fn test_no_code_blocks_prompts_repl_use() {
        let backend = Arc::new(ScriptedBackend::new(vec![
            // First response: no code blocks
            "I think the answer is maybe 42?".to_string(),
            // Second response: uses REPL
            "```repl\nFINAL(\"42\")\n```".to_string(),
        ]));

        let rlm = Rlm::new(
            RlmConfig {
                max_depth: 1,
                max_iterations: 10,
                ..Default::default()
            },
            backend.clone(),
            Some(backend),
        );

        let result = rlm
            .completion("test", Some("What number?"))
            .await
            .unwrap();
        assert_eq!(result.response, "42");
    }
}
