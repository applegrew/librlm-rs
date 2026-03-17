use regex::Regex;
use std::sync::LazyLock;

use crate::types::FinalAnswer;

static CODE_BLOCK_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)```repl\s*\n(.*?)```").unwrap());

static FINAL_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"FINAL\("([^"]*)"\)"#).unwrap());

static FINAL_VAR_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"FINAL_VAR\("([^"]*)"\)"#).unwrap());

/// Extract code from ` ```repl ``` ` fenced blocks.
pub fn find_code_blocks(text: &str) -> Vec<String> {
    CODE_BLOCK_RE
        .captures_iter(text)
        .map(|cap| cap[1].trim().to_string())
        .collect()
}

/// Detect `FINAL("...")` or `FINAL_VAR("...")` in REPL output text (not code blocks).
pub fn find_final_answer(text: &str) -> Option<FinalAnswer> {
    if let Some(cap) = FINAL_VAR_RE.captures(text) {
        return Some(FinalAnswer::Var(cap[1].to_string()));
    }
    if let Some(cap) = FINAL_RE.captures(text) {
        return Some(FinalAnswer::Direct(cap[1].to_string()));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_code_blocks_single() {
        let text = "Here is code:\n```repl\nprint(\"hello\")\n```\nDone.";
        let blocks = find_code_blocks(text);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0], "print(\"hello\")");
    }

    #[test]
    fn test_find_code_blocks_multiple() {
        let text = "```repl\nlocal x = 1\n```\ntext\n```repl\nlocal y = 2\n```";
        let blocks = find_code_blocks(text);
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0], "local x = 1");
        assert_eq!(blocks[1], "local y = 2");
    }

    #[test]
    fn test_find_code_blocks_none() {
        let text = "No code blocks here.\n```python\nx = 1\n```";
        let blocks = find_code_blocks(text);
        assert!(blocks.is_empty());
    }

    #[test]
    fn test_find_code_blocks_multiline() {
        let text = "```repl\nlocal x = 1\nlocal y = 2\nprint(x + y)\n```";
        let blocks = find_code_blocks(text);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0], "local x = 1\nlocal y = 2\nprint(x + y)");
    }

    #[test]
    fn test_find_final_direct() {
        let text = r#"Output: FINAL("the answer is 42")"#;
        let result = find_final_answer(text);
        assert_eq!(result, Some(FinalAnswer::Direct("the answer is 42".into())));
    }

    #[test]
    fn test_find_final_var() {
        let text = r#"FINAL_VAR("my_result")"#;
        let result = find_final_answer(text);
        assert_eq!(result, Some(FinalAnswer::Var("my_result".into())));
    }

    #[test]
    fn test_find_final_none() {
        let text = "Just some regular output";
        assert_eq!(find_final_answer(text), None);
    }

    #[test]
    fn test_final_var_takes_precedence() {
        let text = r#"FINAL("x") FINAL_VAR("y")"#;
        let result = find_final_answer(text);
        assert_eq!(result, Some(FinalAnswer::Var("y".into())));
    }
}
