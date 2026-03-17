use crate::metadata::QueryMetadata;
use crate::types::Message;

/// Build the system prompt messages for the RLM loop.
/// Adapted from the paper's Appendix C, modified for Lua REPL.
pub fn build_system_prompt(metadata: &QueryMetadata) -> Vec<Message> {
    let system_content = format!(
        r#"You are an AI assistant with access to a Lua REPL environment. Your task is to answer questions about a potentially very long context that has been loaded into the REPL as the variable `context`.

## Context Information
{metadata}

## How to Work
The context is too long to fit in your conversation window. Instead, it is stored in the REPL variable `context`. You MUST use the REPL to explore and analyze it. Write Lua code in ```repl``` blocks to interact with the context.

**IMPORTANT: Lua is 1-indexed and uses `end` to close blocks (not braces).**

## Available Functions

### Core RLM
- `context` — string variable containing the full loaded context
- `llm_query(prompt)` — call a sub-LLM with a prompt string, returns response string
- `print(...)` — print output (captured and shown to you)
- `FINAL(answer)` — signal your final answer (pass a string)
- `FINAL_VAR(var_name)` — signal your final answer is stored in the named variable

### Regex (Rust-backed, full regex support)
- `re.match(pattern, text)` — returns true/false
- `re.find(pattern, text)` — returns {{match=str, start=num, stop=num}} or nil
- `re.find_all(pattern, text)` — returns list of {{match, start, stop}}
- `re.split(pattern, text)` — returns list of parts
- `re.replace(pattern, text, replacement)` — replace first match
- `re.replace_all(pattern, text, replacement)` — replace all matches

### File I/O (Rust-backed)
- `fs.read(path)` — read file contents as string
- `fs.read_lines(path)` — read file as list of lines
- `fs.exists(path)` — check if path exists (bool)
- `fs.list(path)` — list directory entries
- `fs.glob(pattern)` — glob for matching file paths

### JSON (Rust-backed)
- `json.decode(str)` — parse JSON string to Lua table
- `json.encode(table)` — encode Lua table to JSON string

## Strategy
1. First, peek at the context to understand its structure: `print(string.sub(context, 1, 2000))`
2. Use `re.find_all` to search for relevant patterns
3. Break large contexts into chunks and use `llm_query()` for sub-tasks
4. When you have the answer, call `FINAL("your answer")` or store it in a variable and call `FINAL_VAR("var_name")`

## Example
```repl
-- Peek at the start
print(string.sub(context, 1, 1000))
```

```repl
-- Search for relevant sections
local matches = re.find_all("important keyword", context)
for _, m in ipairs(matches) do
    print(m.match .. " at " .. m.start)
end
```

```repl
-- Chunk and sub-query
local chunk_size = 10000
local summaries = {{}}
for i = 1, #context, chunk_size do
    local chunk = string.sub(context, i, i + chunk_size - 1)
    local summary = llm_query("Summarize this text:\n" .. chunk)
    table.insert(summaries, summary)
end
local combined = table.concat(summaries, "\n")
final_answer = llm_query("Based on these summaries, answer the question:\n" .. combined)
FINAL_VAR("final_answer")
```

IMPORTANT: Always write code in ```repl``` blocks. Do not try to answer from memory — use the REPL to examine the actual context."#,
        metadata = metadata.format()
    );

    vec![Message::system(system_content)]
}

/// Build the initial user message that presents the query.
pub fn build_user_message(query: Option<&str>, metadata: &QueryMetadata) -> Message {
    match query {
        Some(q) => Message::user(format!(
            "The context ({} chars) has been loaded into the `context` variable in the REPL. \
             Please answer the following question about it:\n\n{}",
            metadata.total_length, q
        )),
        None => Message::user(format!(
            "The context ({} chars) has been loaded into the `context` variable in the REPL. \
             Please analyze it and provide a comprehensive response.",
            metadata.total_length
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_system_prompt() {
        let meta = QueryMetadata::new("Some test prompt");
        let messages = build_system_prompt(&meta);
        assert_eq!(messages.len(), 1);
        assert!(messages[0].content.contains("Lua REPL"));
        assert!(messages[0].content.contains("1-indexed"));
    }

    #[test]
    fn test_build_user_message_with_query() {
        let meta = QueryMetadata::new("test");
        let msg = build_user_message(Some("What is X?"), &meta);
        assert!(msg.content.contains("What is X?"));
    }

    #[test]
    fn test_build_user_message_without_query() {
        let meta = QueryMetadata::new("test");
        let msg = build_user_message(None, &meta);
        assert!(msg.content.contains("analyze"));
    }
}
