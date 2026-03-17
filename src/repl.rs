use mlua::prelude::*;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::error::RlmError;
use crate::llm::LlmBackend;
use crate::types::{FinalAnswer, ReplResult};

/// Embedded Lua REPL with Rust-backed extensions.
pub struct LuaRepl {
    lua: Lua,
    stdout_buffer: Arc<Mutex<String>>,
    final_answer: Arc<Mutex<Option<FinalAnswer>>>,
}

impl LuaRepl {
    /// Create a new Lua REPL with the given sub-LLM backend for `llm_query()` calls.
    pub fn new(sub_backend: Arc<dyn LlmBackend>) -> Result<Self, RlmError> {
        let lua = Lua::new();
        let stdout_buffer = Arc::new(Mutex::new(String::new()));
        let final_answer: Arc<Mutex<Option<FinalAnswer>>> = Arc::new(Mutex::new(None));

        // Register print override
        {
            let buf = stdout_buffer.clone();
            let print_fn = lua
                .create_function(move |_, args: LuaMultiValue| {
                    let parts: Vec<String> = args
                        .into_iter()
                        .map(|v| match &v {
                            LuaValue::Nil => "nil".to_string(),
                            LuaValue::Boolean(b) => b.to_string(),
                            LuaValue::Integer(i) => i.to_string(),
                            LuaValue::Number(n) => n.to_string(),
                            LuaValue::String(s) => s.to_string_lossy().to_string(),
                            _ => format!("{:?}", v),
                        })
                        .collect();
                    let line = parts.join("\t");
                    let mut buffer = buf.lock().unwrap();
                    buffer.push_str(&line);
                    buffer.push('\n');
                    Ok(())
                })
                .map_err(|e| RlmError::ReplSpawnError(e.to_string()))?;
            lua.globals()
                .set("print", print_fn)
                .map_err(|e| RlmError::ReplSpawnError(e.to_string()))?;
        }

        // Register FINAL
        {
            let fa = final_answer.clone();
            let final_fn = lua
                .create_function(move |_, answer: String| {
                    let mut guard = fa.lock().unwrap();
                    *guard = Some(FinalAnswer::Direct(answer));
                    Ok(())
                })
                .map_err(|e| RlmError::ReplSpawnError(e.to_string()))?;
            lua.globals()
                .set("FINAL", final_fn)
                .map_err(|e| RlmError::ReplSpawnError(e.to_string()))?;
        }

        // Register FINAL_VAR
        {
            let fa = final_answer.clone();
            let final_var_fn = lua
                .create_function(move |_, var_name: String| {
                    let mut guard = fa.lock().unwrap();
                    *guard = Some(FinalAnswer::Var(var_name));
                    Ok(())
                })
                .map_err(|e| RlmError::ReplSpawnError(e.to_string()))?;
            lua.globals()
                .set("FINAL_VAR", final_var_fn)
                .map_err(|e| RlmError::ReplSpawnError(e.to_string()))?;
        }

        // Register llm_query
        {
            let backend = sub_backend.clone();
            let llm_fn = lua
                .create_function(move |_, prompt: String| {
                    let backend = backend.clone();
                    let result = tokio::task::block_in_place(|| {
                        tokio::runtime::Handle::current().block_on(async {
                            let messages = vec![crate::types::Message::user(prompt)];
                            backend.completion(&messages).await
                        })
                    });
                    match result {
                        Ok(resp) => Ok(resp.content),
                        Err(e) => Err(LuaError::external(e)),
                    }
                })
                .map_err(|e| RlmError::ReplSpawnError(e.to_string()))?;
            lua.globals()
                .set("llm_query", llm_fn)
                .map_err(|e| RlmError::ReplSpawnError(e.to_string()))?;
        }

        // Register re.* (regex functions backed by Rust)
        Self::register_regex(&lua)?;

        // Register fs.* (file I/O backed by Rust)
        Self::register_fs(&lua)?;

        // Register json.* (backed by serde_json)
        Self::register_json(&lua)?;

        Ok(Self {
            lua,
            stdout_buffer,
            final_answer,
        })
    }

    /// Load the context/prompt into the REPL as the `context` global variable.
    pub fn load_context(&self, prompt: &str) -> Result<(), RlmError> {
        self.lua
            .globals()
            .set("context", prompt.to_string())
            .map_err(|e| RlmError::ReplError(e.to_string()))
    }

    /// Execute Lua code and return the result.
    pub fn execute_code(&self, code: &str) -> ReplResult {
        // Clear stdout buffer
        {
            let mut buf = self.stdout_buffer.lock().unwrap();
            buf.clear();
        }
        // Clear final answer
        {
            let mut fa = self.final_answer.lock().unwrap();
            *fa = None;
        }

        let start = Instant::now();
        let exec_result = self.lua.load(code).exec();
        let execution_time = start.elapsed();

        let stdout = {
            let buf = self.stdout_buffer.lock().unwrap();
            buf.clone()
        };

        let final_answer = {
            let fa = self.final_answer.lock().unwrap();
            fa.clone()
        };

        match exec_result {
            Ok(()) => ReplResult {
                stdout,
                stderr: String::new(),
                final_answer,
                execution_time,
            },
            Err(e) => ReplResult {
                stdout,
                stderr: e.to_string(),
                final_answer,
                execution_time,
            },
        }
    }

    /// Get the string value of a global variable.
    pub fn get_variable(&self, name: &str) -> Result<String, RlmError> {
        let value: LuaValue = self
            .lua
            .globals()
            .get(name)
            .map_err(|e| RlmError::ReplError(e.to_string()))?;
        lua_value_to_string(&value)
    }

    /// Reset the REPL state.
    pub fn reset(&mut self) -> Result<(), RlmError> {
        // We can't truly reset mlua, but we can clear globals
        // For a full reset, create a new instance
        let mut buf = self.stdout_buffer.lock().unwrap();
        buf.clear();
        let mut fa = self.final_answer.lock().unwrap();
        *fa = None;
        Ok(())
    }

    fn register_regex(lua: &Lua) -> Result<(), RlmError> {
        let re_table = lua
            .create_table()
            .map_err(|e| RlmError::ReplSpawnError(e.to_string()))?;

        // re.match(pattern, text) -> bool
        let match_fn = lua
            .create_function(|_, (pattern, text): (String, String)| {
                match regex::Regex::new(&pattern) {
                    Ok(re) => Ok(re.is_match(&text)),
                    Err(e) => Err(LuaError::external(e)),
                }
            })
            .map_err(|e| RlmError::ReplSpawnError(e.to_string()))?;
        re_table
            .set("match", match_fn)
            .map_err(|e| RlmError::ReplSpawnError(e.to_string()))?;

        // re.find(pattern, text) -> {match, start, stop} or nil
        let find_fn = lua
            .create_function(|lua, (pattern, text): (String, String)| {
                match regex::Regex::new(&pattern) {
                    Ok(re) => match re.find(&text) {
                        Some(m) => {
                            let t = lua.create_table()?;
                            t.set("match", m.as_str().to_string())?;
                            t.set("start", m.start() + 1)?; // 1-indexed
                            t.set("stop", m.end())?;
                            Ok(LuaValue::Table(t))
                        }
                        None => Ok(LuaValue::Nil),
                    },
                    Err(e) => Err(LuaError::external(e)),
                }
            })
            .map_err(|e| RlmError::ReplSpawnError(e.to_string()))?;
        re_table
            .set("find", find_fn)
            .map_err(|e| RlmError::ReplSpawnError(e.to_string()))?;

        // re.find_all(pattern, text) -> [{match, start, stop}, ...]
        let find_all_fn = lua
            .create_function(|lua, (pattern, text): (String, String)| {
                match regex::Regex::new(&pattern) {
                    Ok(re) => {
                        let results = lua.create_table()?;
                        for (i, m) in re.find_iter(&text).enumerate() {
                            let t = lua.create_table()?;
                            t.set("match", m.as_str().to_string())?;
                            t.set("start", m.start() + 1)?;
                            t.set("stop", m.end())?;
                            results.set(i + 1, t)?;
                        }
                        Ok(results)
                    }
                    Err(e) => Err(LuaError::external(e)),
                }
            })
            .map_err(|e| RlmError::ReplSpawnError(e.to_string()))?;
        re_table
            .set("find_all", find_all_fn)
            .map_err(|e| RlmError::ReplSpawnError(e.to_string()))?;

        // re.split(pattern, text) -> [parts]
        let split_fn = lua
            .create_function(|lua, (pattern, text): (String, String)| {
                match regex::Regex::new(&pattern) {
                    Ok(re) => {
                        let results = lua.create_table()?;
                        for (i, part) in re.split(&text).enumerate() {
                            results.set(i + 1, part.to_string())?;
                        }
                        Ok(results)
                    }
                    Err(e) => Err(LuaError::external(e)),
                }
            })
            .map_err(|e| RlmError::ReplSpawnError(e.to_string()))?;
        re_table
            .set("split", split_fn)
            .map_err(|e| RlmError::ReplSpawnError(e.to_string()))?;

        // re.replace(pattern, text, replacement) -> string
        let replace_fn = lua
            .create_function(|_, (pattern, text, replacement): (String, String, String)| {
                match regex::Regex::new(&pattern) {
                    Ok(re) => Ok(re.replace(&text, replacement.as_str()).to_string()),
                    Err(e) => Err(LuaError::external(e)),
                }
            })
            .map_err(|e| RlmError::ReplSpawnError(e.to_string()))?;
        re_table
            .set("replace", replace_fn)
            .map_err(|e| RlmError::ReplSpawnError(e.to_string()))?;

        // re.replace_all(pattern, text, replacement) -> string
        let replace_all_fn = lua
            .create_function(|_, (pattern, text, replacement): (String, String, String)| {
                match regex::Regex::new(&pattern) {
                    Ok(re) => Ok(re.replace_all(&text, replacement.as_str()).to_string()),
                    Err(e) => Err(LuaError::external(e)),
                }
            })
            .map_err(|e| RlmError::ReplSpawnError(e.to_string()))?;
        re_table
            .set("replace_all", replace_all_fn)
            .map_err(|e| RlmError::ReplSpawnError(e.to_string()))?;

        lua.globals()
            .set("re", re_table)
            .map_err(|e| RlmError::ReplSpawnError(e.to_string()))?;

        Ok(())
    }

    fn register_fs(lua: &Lua) -> Result<(), RlmError> {
        let fs_table = lua
            .create_table()
            .map_err(|e| RlmError::ReplSpawnError(e.to_string()))?;

        // fs.read(path) -> string
        let read_fn = lua
            .create_function(|_, path: String| {
                std::fs::read_to_string(&path).map_err(LuaError::external)
            })
            .map_err(|e| RlmError::ReplSpawnError(e.to_string()))?;
        fs_table
            .set("read", read_fn)
            .map_err(|e| RlmError::ReplSpawnError(e.to_string()))?;

        // fs.read_lines(path) -> [lines]
        let read_lines_fn = lua
            .create_function(|lua, path: String| {
                let content = std::fs::read_to_string(&path).map_err(LuaError::external)?;
                let table = lua.create_table()?;
                for (i, line) in content.lines().enumerate() {
                    table.set(i + 1, line.to_string())?;
                }
                Ok(table)
            })
            .map_err(|e| RlmError::ReplSpawnError(e.to_string()))?;
        fs_table
            .set("read_lines", read_lines_fn)
            .map_err(|e| RlmError::ReplSpawnError(e.to_string()))?;

        // fs.exists(path) -> bool
        let exists_fn = lua
            .create_function(|_, path: String| Ok(std::path::Path::new(&path).exists()))
            .map_err(|e| RlmError::ReplSpawnError(e.to_string()))?;
        fs_table
            .set("exists", exists_fn)
            .map_err(|e| RlmError::ReplSpawnError(e.to_string()))?;

        // fs.list(path) -> [entry names]
        let list_fn = lua
            .create_function(|lua, path: String| {
                let table = lua.create_table()?;
                let entries = std::fs::read_dir(&path).map_err(LuaError::external)?;
                let mut i = 1;
                for entry in entries {
                    let entry = entry.map_err(LuaError::external)?;
                    if let Some(name) = entry.file_name().to_str() {
                        table.set(i, name.to_string())?;
                        i += 1;
                    }
                }
                Ok(table)
            })
            .map_err(|e| RlmError::ReplSpawnError(e.to_string()))?;
        fs_table
            .set("list", list_fn)
            .map_err(|e| RlmError::ReplSpawnError(e.to_string()))?;

        // fs.glob(pattern) -> [matching paths]
        let glob_fn = lua
            .create_function(|lua, pattern: String| {
                let table = lua.create_table()?;
                let mut i = 1;
                match glob::glob(&pattern) {
                    Ok(paths) => {
                        for entry in paths.flatten() {
                            if let Some(p) = entry.to_str() {
                                table.set(i, p.to_string())?;
                                i += 1;
                            }
                        }
                    }
                    Err(e) => return Err(LuaError::external(e)),
                }
                Ok(table)
            })
            .map_err(|e| RlmError::ReplSpawnError(e.to_string()))?;
        fs_table
            .set("glob", glob_fn)
            .map_err(|e| RlmError::ReplSpawnError(e.to_string()))?;

        lua.globals()
            .set("fs", fs_table)
            .map_err(|e| RlmError::ReplSpawnError(e.to_string()))?;

        Ok(())
    }

    fn register_json(lua: &Lua) -> Result<(), RlmError> {
        let json_table = lua
            .create_table()
            .map_err(|e| RlmError::ReplSpawnError(e.to_string()))?;

        // json.decode(str) -> Lua value
        let decode_fn = lua
            .create_function(|lua, s: String| {
                let value: serde_json::Value =
                    serde_json::from_str(&s).map_err(LuaError::external)?;
                json_to_lua(lua, &value)
            })
            .map_err(|e| RlmError::ReplSpawnError(e.to_string()))?;
        json_table
            .set("decode", decode_fn)
            .map_err(|e| RlmError::ReplSpawnError(e.to_string()))?;

        // json.encode(value) -> string
        let encode_fn = lua
            .create_function(|_, value: LuaValue| {
                let json_val = lua_to_json(&value).map_err(LuaError::external)?;
                serde_json::to_string(&json_val).map_err(LuaError::external)
            })
            .map_err(|e| RlmError::ReplSpawnError(e.to_string()))?;
        json_table
            .set("encode", encode_fn)
            .map_err(|e| RlmError::ReplSpawnError(e.to_string()))?;

        lua.globals()
            .set("json", json_table)
            .map_err(|e| RlmError::ReplSpawnError(e.to_string()))?;

        Ok(())
    }
}

fn json_to_lua(lua: &Lua, value: &serde_json::Value) -> LuaResult<LuaValue> {
    match value {
        serde_json::Value::Null => Ok(LuaValue::Nil),
        serde_json::Value::Bool(b) => Ok(LuaValue::Boolean(*b)),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(LuaValue::Integer(i as _))
            } else {
                Ok(LuaValue::Number(n.as_f64().unwrap_or(0.0)))
            }
        }
        serde_json::Value::String(s) => Ok(LuaValue::String(lua.create_string(s)?)),
        serde_json::Value::Array(arr) => {
            let table = lua.create_table()?;
            for (i, v) in arr.iter().enumerate() {
                table.set(i + 1, json_to_lua(lua, v)?)?;
            }
            Ok(LuaValue::Table(table))
        }
        serde_json::Value::Object(obj) => {
            let table = lua.create_table()?;
            for (k, v) in obj {
                table.set(k.as_str(), json_to_lua(lua, v)?)?;
            }
            Ok(LuaValue::Table(table))
        }
    }
}

fn lua_to_json(value: &LuaValue) -> Result<serde_json::Value, String> {
    match value {
        LuaValue::Nil => Ok(serde_json::Value::Null),
        LuaValue::Boolean(b) => Ok(serde_json::Value::Bool(*b)),
        LuaValue::Integer(i) => Ok(serde_json::json!(*i)),
        LuaValue::Number(n) => Ok(serde_json::json!(*n)),
        LuaValue::String(s) => Ok(serde_json::Value::String(
            s.to_str().map_err(|e| e.to_string())?.to_string(),
        )),
        LuaValue::Table(t) => {
            // Check if it's an array (sequential integer keys starting at 1)
            let len = t.raw_len();
            if len > 0 {
                let mut arr = Vec::new();
                for i in 1..=len {
                    let v: LuaValue = t.raw_get(i).map_err(|e| e.to_string())?;
                    arr.push(lua_to_json(&v)?);
                }
                Ok(serde_json::Value::Array(arr))
            } else {
                let mut map = serde_json::Map::new();
                for pair in t.clone().pairs::<String, LuaValue>() {
                    let (k, v) = pair.map_err(|e| e.to_string())?;
                    map.insert(k, lua_to_json(&v)?);
                }
                Ok(serde_json::Value::Object(map))
            }
        }
        _ => Err(format!("Cannot convert {:?} to JSON", value)),
    }
}

fn lua_value_to_string(value: &LuaValue) -> Result<String, RlmError> {
    match value {
        LuaValue::Nil => Ok("nil".to_string()),
        LuaValue::Boolean(b) => Ok(b.to_string()),
        LuaValue::Integer(i) => Ok(i.to_string()),
        LuaValue::Number(n) => Ok(n.to_string()),
        LuaValue::String(s) => s
            .to_str()
            .map(|s| s.to_string())
            .map_err(|e| RlmError::ReplError(e.to_string())),
        LuaValue::Table(_) => {
            let json = lua_to_json(value).map_err(RlmError::ReplError)?;
            Ok(serde_json::to_string_pretty(&json)
                .map_err(|e| RlmError::ReplError(e.to_string()))?)
        }
        _ => Ok(format!("{:?}", value)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CompletionResponse, Message, UsageInfo};
    use std::sync::Arc;

    struct MockBackend;

    #[async_trait::async_trait]
    impl LlmBackend for MockBackend {
        fn model_name(&self) -> &str {
            "mock"
        }
        async fn completion(&self, messages: &[Message]) -> Result<CompletionResponse, RlmError> {
            let last = messages.last().map(|m| m.content.as_str()).unwrap_or("");
            Ok(CompletionResponse {
                content: format!("Mock response to: {}", &last[..last.len().min(50)]),
                model: "mock".to_string(),
                usage: UsageInfo::default(),
            })
        }
    }

    fn make_repl() -> LuaRepl {
        LuaRepl::new(Arc::new(MockBackend)).unwrap()
    }

    #[test]
    fn test_print_capture() {
        let repl = make_repl();
        let result = repl.execute_code(r#"print("hello world")"#);
        assert_eq!(result.stdout.trim(), "hello world");
        assert!(result.stderr.is_empty());
    }

    #[test]
    fn test_variable_persistence() {
        let repl = make_repl();
        repl.execute_code("x = 42");
        let result = repl.execute_code("print(x)");
        assert_eq!(result.stdout.trim(), "42");
    }

    #[test]
    fn test_context_loading() {
        let repl = make_repl();
        repl.load_context("Hello, this is a test context.").unwrap();
        let result = repl.execute_code("print(string.sub(context, 1, 5))");
        assert_eq!(result.stdout.trim(), "Hello");
    }

    #[test]
    fn test_final_direct() {
        let repl = make_repl();
        let result = repl.execute_code(r#"FINAL("the answer")"#);
        assert_eq!(
            result.final_answer,
            Some(FinalAnswer::Direct("the answer".to_string()))
        );
    }

    #[test]
    fn test_final_var() {
        let repl = make_repl();
        repl.execute_code(r#"my_var = "computed result""#);
        let result = repl.execute_code(r#"FINAL_VAR("my_var")"#);
        assert_eq!(
            result.final_answer,
            Some(FinalAnswer::Var("my_var".to_string()))
        );
        let val = repl.get_variable("my_var").unwrap();
        assert_eq!(val, "computed result");
    }

    #[test]
    fn test_syntax_error() {
        let repl = make_repl();
        let result = repl.execute_code("this is not valid lua!!!");
        assert!(!result.stderr.is_empty());
    }

    #[test]
    fn test_regex_match() {
        let repl = make_repl();
        let result = repl.execute_code(
            r#"
            local m = re.match("hello", "say hello world")
            print(tostring(m))
        "#,
        );
        assert_eq!(result.stdout.trim(), "true");
    }

    #[test]
    fn test_regex_find_all() {
        let repl = make_repl();
        let result = repl.execute_code(
            r#"
            local matches = re.find_all("\\d+", "abc 123 def 456")
            for _, m in ipairs(matches) do
                print(m.match)
            end
        "#,
        );
        let lines: Vec<&str> = result.stdout.trim().lines().collect();
        assert_eq!(lines, vec!["123", "456"]);
    }

    #[test]
    fn test_regex_replace() {
        let repl = make_repl();
        let result = repl.execute_code(
            r#"
            local s = re.replace_all("\\d+", "a1b2c3", "X")
            print(s)
        "#,
        );
        assert_eq!(result.stdout.trim(), "aXbXcX");
    }

    #[test]
    fn test_json_roundtrip() {
        let repl = make_repl();
        let result = repl.execute_code(
            r#"
            local data = json.decode('{"name":"test","value":42}')
            print(data.name)
            print(data.value)
        "#,
        );
        let lines: Vec<&str> = result.stdout.trim().lines().collect();
        assert_eq!(lines, vec!["test", "42"]);
    }

    #[test]
    fn test_get_variable() {
        let repl = make_repl();
        repl.execute_code(r#"my_string = "hello""#);
        assert_eq!(repl.get_variable("my_string").unwrap(), "hello");

        repl.execute_code("my_num = 42");
        assert_eq!(repl.get_variable("my_num").unwrap(), "42");
    }
}
