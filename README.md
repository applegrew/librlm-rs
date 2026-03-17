# librlm-rs

A Rust implementation of the **Recursive Language Models (RLM)** algorithm from ["Recursive Language Models"](https://arxiv.org/abs/2502.02116) (Zhang, Kraska, Khattab — MIT CSAIL, Jan 2026).

RLM enables LLMs to handle **arbitrarily long prompts** (10M+ tokens) by treating them as part of an external environment rather than feeding them directly into the context window. The LLM interacts with the prompt through a persistent REPL, writing code to explore, decompose, and recursively invoke sub-LLMs over manageable chunks.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    RLM Loop                         │
│                                                     │
│  ┌──────────┐    code     ┌──────────────────────┐  │
│  │ Root LLM │───────────▶│  Embedded Lua REPL    │  │
│  │ (e.g.    │◀───────────│                       │  │
│  │  GPT-5)  │  metadata   │  context = <prompt>  │  │
│  └──────────┘   (stdout)  │  llm_query() ─────┐  │  │
│                           │  re.*, fs.*, json.*│ │  │
│                           └───────────────────│──┘  │
│                                               │     │
│                           ┌───────────────────▼──┐  │
│                           │  Sub LLM             │  │
│                           │  (e.g. GPT-5-mini)   │  │
│                           └──────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

## How It Works

1. The (potentially very long) prompt is loaded into the REPL as a `context` variable — **not** into the LLM's context window
2. The LLM generates Lua code to explore the context (peek, search, chunk)
3. Code executes in the REPL; only metadata (length + prefix of output) goes back to the LLM
4. `llm_query()` enables recursive sub-LLM calls from within code
5. `FINAL("answer")` or `FINAL_VAR("var_name")` signals completion

## Quick Start

```rust
use librlm::Rlm;

#[tokio::main]
async fn main() -> Result<(), librlm::RlmError> {
    let rlm = Rlm::builder()
        .root_model("gpt-5")
        .root_api_key("sk-...")
        // Optional: cheaper model for sub-calls from within the REPL
        .sub_model("gpt-5-mini")
        .build()?;

    let result = rlm.completion(
        &std::fs::read_to_string("very_long_document.txt")?,
        Some("What are the key findings?"),
    ).await?;

    println!("{}", result.response);
    println!("Iterations: {}, Tokens: {}", result.iterations, result.total_usage.total_tokens);
    Ok(())
}
```

## Builder API

| Method | Required | Description |
|---|---|---|
| `root_model(name)` | Yes* | Root LLM model name |
| `root_api_key(key)` | Yes* | API key for root LLM |
| `root_base_url(url)` | No | Base URL (defaults to OpenAI) |
| `sub_model(name)` | No | Sub-call LLM model name |
| `sub_api_key(key)` | No | Sub-call API key (defaults to root) |
| `sub_base_url(url)` | No | Sub-call base URL (defaults to root) |
| `root_backend(impl)` | No | Custom `LlmBackend` trait object |
| `sub_backend(impl)` | No | Custom sub-call backend |
| `max_iterations(n)` | No | Max REPL loop iterations (default: 30) |
| `max_depth(n)` | No | Max recursion depth (default: 1) |
| `max_output_chars(n)` | No | Max REPL output chars in history (default: 20000) |
| `max_timeout(duration)` | No | Overall timeout |

\* Not required when using `root_backend()` instead.

## Custom LLM Backend

Implement the `LlmBackend` trait for non-OpenAI providers:

```rust
use librlm::{LlmBackend, Message, CompletionResponse, RlmError};
use async_trait::async_trait;

struct MyBackend;

#[async_trait]
impl LlmBackend for MyBackend {
    fn model_name(&self) -> &str { "my-model" }

    async fn completion(&self, messages: &[Message]) -> Result<CompletionResponse, RlmError> {
        // Your implementation here
        todo!()
    }
}
```

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `max_depth` | 1 | Recursion depth. 0 = plain LLM call, 1 = one level of REPL interaction |
| `max_iterations` | 30 | Maximum REPL loop iterations before forcing a final answer |
| `max_output_chars` | 20000 | Truncation limit for REPL output included in LLM history |
| `max_timeout` | None | Overall timeout for the completion |

## Prerequisites

- Rust toolchain (edition 2021+)
- No external runtime needed — the Lua VM (Luau) compiles from source into the binary

## Design Decisions

- **Embedded Lua REPL**: Uses `mlua` with the Luau feature. The Lua VM compiles from source — zero external dependencies. Rust-backed extensions (`re.*`, `fs.*`, `json.*`) compensate for Lua's minimal stdlib.
- **Two-LLM architecture**: Root LLM drives the main loop; optional cheaper sub-LLM handles `llm_query()` calls (matching the paper's GPT-5 + GPT-5-mini approach).
- **Pure library**: All configuration is programmatic. No config files, no environment variables.
- **Async API**: Built on tokio for non-blocking LLM HTTP calls.

## License

MIT
