//! `json.dumps()` compatible serialization (", " / ": " separators,
//! `ensure_ascii`), so requests are byte-identical to what pytapo sends.

use serde_json::Value;

pub fn dumps(v: &Value) -> String {
    let mut out = String::new();
    write_value(&mut out, v, false);
    out
}

/// `json.dumps(v, separators=(",", ":"))`
pub fn dumps_compact(v: &Value) -> String {
    let mut out = String::new();
    write_value(&mut out, v, true);
    out
}

fn write_value(out: &mut String, v: &Value, compact: bool) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => out.push_str(&n.to_string()),
        Value::String(s) => write_str(out, s),
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(if compact { "," } else { ", " });
                }
                write_value(out, x, compact);
            }
            out.push(']');
        }
        Value::Object(m) => {
            out.push('{');
            for (i, (k, x)) in m.iter().enumerate() {
                if i > 0 {
                    out.push_str(if compact { "," } else { ", " });
                }
                write_str(out, k);
                out.push_str(if compact { ":" } else { ": " });
                write_value(out, x, compact);
            }
            out.push('}');
        }
    }
}

fn write_str(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 || (c as u32) > 0x7e => {
                let mut buf = [0u16; 2];
                for unit in c.encode_utf16(&mut buf) {
                    out.push_str(&format!("\\u{:04x}", unit));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn matches_python() {
        let v = json!({"method": "login", "params": {"a": [1, 2], "b": "ž\"x", "c": true}});
        let esc = format!("{}u017e", '\\');
        assert_eq!(dumps(&v), format!(r#"{{"method": "login", "params": {{"a": [1, 2], "b": "{esc}\"x", "c": true}}}}"#));
        assert_eq!(dumps_compact(&v), format!(r#"{{"method":"login","params":{{"a":[1,2],"b":"{esc}\"x","c":true}}}}"#));
    }
}
