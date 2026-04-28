use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct InterfaceResult {
    pub ok: bool,
    pub raw: String,
}

impl InterfaceResult {
    pub fn status(&self) -> Option<String> {
        extract_json_string(&self.raw, "status")
    }

    pub fn message(&self) -> Option<String> {
        extract_json_string(&self.raw, "message")
    }

    pub fn run_id(&self) -> Option<String> {
        extract_json_string(&self.raw, "run_id")
    }
}

pub trait Dispatcher {
    fn dispatch(&self, command: &str) -> Result<InterfaceResult, String>;
}

#[derive(Debug, Clone)]
pub struct QueryRequest {
    pub run_id: Option<String>,
    pub command: String,
    pub payload: String,
}

impl QueryRequest {
    pub fn from_command_line(command: &str) -> Result<Self, String> {
        let (head, tail) = command
            .split_once('|')
            .ok_or_else(|| "command must contain '|'".to_string())?;
        let cmd = head.trim().to_ascii_lowercase();
        let payload = tail.trim().to_string();
        match cmd.as_str() {
            "search" | "fetch" | "learn" if payload.is_empty() => Err(format!("{cmd} requires payload")),
            "search" | "fetch" | "learn" | "status" | "quit" => Ok(Self { run_id: None, command: cmd, payload }),
            _ => Err(format!("invalid command: {cmd}")),
        }
    }

    pub fn to_json_line(&self) -> String {
        let mut fields = vec![
            format!("\"command\":\"{}\"", escape_json(&self.command)),
            format!("\"payload\":\"{}\"", escape_json(&self.payload)),
        ];
        if let Some(run_id) = &self.run_id {
            fields.push(format!("\"run_id\":\"{}\"", escape_json(run_id)));
        }
        format!("{{{}}}\n", fields.join(","))
    }
}

#[derive(Debug, Clone)]
pub enum Transport {
    Tcp { addr: String },
    #[cfg(unix)]
    Unix { path: String },
}

impl Transport {
    pub fn from_env() -> Self {
        #[cfg(unix)]
        {
            if let Ok(path) = std::env::var("AXIOM_INTERFACE_SOCKET") {
                if !path.trim().is_empty() {
                    return Transport::Unix { path };
                }
            }
        }
        let addr = std::env::var("AXIOM_INTERFACE_TCP").unwrap_or_else(|_| "127.0.0.1:8766".to_string());
        Transport::Tcp { addr }
    }
}

pub struct TcpDispatcher {
    addr: String,
    timeout: Duration,
}

impl TcpDispatcher {
    pub fn new(addr: impl Into<String>) -> Self {
        Self { addr: addr.into(), timeout: Duration::from_secs(5) }
    }
}

impl Dispatcher for TcpDispatcher {
    fn dispatch(&self, command: &str) -> Result<InterfaceResult, String> {
        let req = QueryRequest::from_command_line(command)?;
        let mut stream = TcpStream::connect(&self.addr).map_err(|e| e.to_string())?;
        stream.set_read_timeout(Some(self.timeout)).map_err(|e| e.to_string())?;
        stream.set_write_timeout(Some(self.timeout)).map_err(|e| e.to_string())?;
        stream.write_all(req.to_json_line().as_bytes()).map_err(|e| e.to_string())?;
        let mut reader = BufReader::new(stream);
        let mut buf = String::new();
        reader.read_line(&mut buf).map_err(|e| e.to_string())?;
        Ok(result_from_raw(buf))
    }
}

pub struct InterfaceDispatcher {
    transport: Transport,
    timeout: Duration,
}

impl InterfaceDispatcher {
    pub fn new(transport: Transport) -> Self {
        Self { transport, timeout: Duration::from_secs(5) }
    }

    pub fn from_env() -> Self {
        Self::new(Transport::from_env())
    }
}

impl Dispatcher for InterfaceDispatcher {
    fn dispatch(&self, command: &str) -> Result<InterfaceResult, String> {
        let req = QueryRequest::from_command_line(command)?;
        match &self.transport {
            Transport::Tcp { addr } => {
                let tcp = TcpDispatcher { addr: addr.clone(), timeout: self.timeout };
                tcp.dispatch(command)
            }
            #[cfg(unix)]
            Transport::Unix { path } => dispatch_unix(path, self.timeout, &req),
        }
    }
}

pub struct EchoDispatcher;

impl Dispatcher for EchoDispatcher {
    fn dispatch(&self, command: &str) -> Result<InterfaceResult, String> {
        Ok(InterfaceResult { ok: true, raw: format!("{{\"echo\":{:?}}}", command) })
    }
}

#[cfg(unix)]
fn dispatch_unix(path: &str, timeout: Duration, req: &QueryRequest) -> Result<InterfaceResult, String> {
    use std::os::unix::net::UnixStream;
    let mut stream = UnixStream::connect(path).map_err(|e| e.to_string())?;
    stream.set_read_timeout(Some(timeout)).map_err(|e| e.to_string())?;
    stream.set_write_timeout(Some(timeout)).map_err(|e| e.to_string())?;
    stream.write_all(req.to_json_line().as_bytes()).map_err(|e| e.to_string())?;
    let mut reader = BufReader::new(stream);
    let mut buf = String::new();
    reader.read_line(&mut buf).map_err(|e| e.to_string())?;
    Ok(result_from_raw(buf))
}

fn result_from_raw(raw: String) -> InterfaceResult {
    let status = extract_json_string(&raw, "status");
    let ok = matches!(status.as_deref(), Some("ok") | Some("accepted") | Some("empty"));
    InterfaceResult { ok, raw }
}

fn escape_json(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn extract_json_string(raw: &str, key: &str) -> Option<String> {
    let needle = format!("\"{}\"", key);
    let idx = raw.find(&needle)?;
    let rest = &raw[idx + needle.len()..];
    let colon = rest.find(':')?;
    let mut chars = rest[colon + 1..].chars().peekable();
    while matches!(chars.peek(), Some(c) if c.is_whitespace()) {
        chars.next();
    }
    if chars.next()? != '"' {
        return None;
    }
    let mut out = String::new();
    let mut escaped = false;
    for ch in chars {
        if escaped {
            match ch {
                '"' => out.push('"'),
                '\\' => out.push('\\'),
                'n' => out.push('\n'),
                'r' => out.push('\r'),
                't' => out.push('\t'),
                other => out.push(other),
            }
            escaped = false;
            continue;
        }
        if ch == '\\' {
            escaped = true;
            continue;
        }
        if ch == '"' {
            return Some(out);
        }
        out.push(ch);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn echo_dispatches() {
        let d = EchoDispatcher;
        let r = d.dispatch("status |").unwrap();
        assert!(r.ok);
    }

    #[test]
    fn query_request_json_escapes_payload() {
        let req = QueryRequest::from_command_line("search | rust \"docs\"").unwrap();
        let wire = req.to_json_line();
        assert!(wire.contains("\\\"docs\\\""));
        assert!(wire.ends_with('\n'));
    }

    #[test]
    fn parses_interface_result_status() {
        let result = result_from_raw("{\"run_id\":\"1\",\"status\":\"accepted\",\"message\":\"queued\"}\n".to_string());
        assert!(result.ok);
        assert_eq!(result.status().as_deref(), Some("accepted"));
        assert_eq!(result.message().as_deref(), Some("queued"));
    }
}
