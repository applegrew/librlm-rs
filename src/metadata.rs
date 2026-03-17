/// Metadata about a prompt/query for the LLM's awareness.
#[derive(Debug, Clone)]
pub struct QueryMetadata {
    /// Total length of the prompt in characters.
    pub total_length: usize,
    /// A short prefix of the prompt for context.
    pub prefix: String,
    /// Type of context (e.g., "text", "code", "mixed").
    pub context_type: String,
}

const PREFIX_LENGTH: usize = 200;

impl QueryMetadata {
    pub fn new(prompt: &str) -> Self {
        let total_length = prompt.len();
        let prefix = if prompt.len() <= PREFIX_LENGTH {
            prompt.to_string()
        } else {
            format!("{}...", &prompt[..PREFIX_LENGTH])
        };
        let context_type = infer_context_type(prompt);

        Self {
            total_length,
            prefix,
            context_type,
        }
    }

    /// Format metadata as a string for inclusion in LLM history.
    pub fn format(&self) -> String {
        format!(
            "[Context loaded: {} chars, type: {}]\nPreview: {}",
            self.total_length, self.context_type, self.prefix
        )
    }
}

/// Format REPL output metadata for inclusion in LLM history.
/// Only includes a prefix + length, not the full output.
pub fn format_repl_output(stdout: &str, max_chars: usize) -> String {
    if stdout.is_empty() {
        return "[No output]".to_string();
    }
    if stdout.len() <= max_chars {
        return stdout.to_string();
    }
    format!(
        "{}...\n[Output truncated: showing {}/{} chars]",
        &stdout[..max_chars],
        max_chars,
        stdout.len()
    )
}

fn infer_context_type(prompt: &str) -> String {
    let has_code_indicators = prompt.contains("def ")
        || prompt.contains("fn ")
        || prompt.contains("class ")
        || prompt.contains("function ")
        || prompt.contains("import ");
    let has_prose = prompt.len() > 500
        && prompt
            .chars()
            .filter(|c| c.is_alphabetic() || c.is_whitespace())
            .count() as f64
            / prompt.len() as f64
            > 0.8;

    if has_code_indicators && has_prose {
        "mixed".to_string()
    } else if has_code_indicators {
        "code".to_string()
    } else {
        "text".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_short_prompt_metadata() {
        let meta = QueryMetadata::new("Hello world");
        assert_eq!(meta.total_length, 11);
        assert_eq!(meta.prefix, "Hello world");
        assert_eq!(meta.context_type, "text");
    }

    #[test]
    fn test_long_prompt_metadata() {
        let prompt = "x".repeat(1000);
        let meta = QueryMetadata::new(&prompt);
        assert_eq!(meta.total_length, 1000);
        assert!(meta.prefix.ends_with("..."));
        assert_eq!(meta.prefix.len(), PREFIX_LENGTH + 3);
    }

    #[test]
    fn test_format_repl_output_short() {
        let output = "hello";
        assert_eq!(format_repl_output(output, 100), "hello");
    }

    #[test]
    fn test_format_repl_output_truncated() {
        let output = "x".repeat(200);
        let formatted = format_repl_output(&output, 50);
        assert!(formatted.contains("Output truncated"));
        assert!(formatted.contains("50/200"));
    }

    #[test]
    fn test_format_repl_output_empty() {
        assert_eq!(format_repl_output("", 100), "[No output]");
    }
}
