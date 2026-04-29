#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    AxiomPrompt,
    Search(String),
    Fetch(String),
    Learn(String),
    Status,
    Quit,
}

impl Command {
    pub fn name(&self) -> &'static str {
        match self {
            Command::AxiomPrompt => "axiom",
            Command::Search(_) => "search",
            Command::Fetch(_) => "fetch",
            Command::Learn(_) => "learn",
            Command::Status => "status",
            Command::Quit => "quit",
        }
    }

    pub fn payload(&self) -> &str {
        match self {
            Command::Search(v) | Command::Fetch(v) | Command::Learn(v) => v,
            Command::AxiomPrompt | Command::Status | Command::Quit => "",
        }
    }

    pub fn to_line(&self) -> String {
        if matches!(self, Command::AxiomPrompt) {
            return "status |".to_string();
        }
        format!("{} | {}", self.name(), self.payload())
    }
}

pub fn parse_line(line: &str) -> Result<Command, String> {
    let stripped = line.trim();
    if stripped.eq_ignore_ascii_case("axiom") {
        return Ok(Command::AxiomPrompt);
    }
    let Some((head, tail)) = stripped.split_once('|') else {
        if stripped.is_empty() {
            return Err("command must contain text".to_string());
        }
        return Ok(Command::Search(stripped.to_string()));
    };
    let cmd = head.trim().to_ascii_lowercase();
    let payload = tail.trim().to_string();
    match cmd.as_str() {
        "search" if !payload.is_empty() => Ok(Command::Search(payload)),
        "fetch" if valid_fetch_payload(&payload) => Ok(Command::Fetch(payload)),
        "learn" if valid_domain_payload(&payload) => Ok(Command::Learn(payload)),
        "status" => Ok(Command::Status),
        "quit" => Ok(Command::Quit),
        _ => Err(format!("invalid command: {cmd}")),
    }
}

pub fn valid_fetch_payload(payload: &str) -> bool {
    payload.starts_with("http://") || payload.starts_with("https://")
}

pub fn valid_domain_payload(payload: &str) -> bool {
    let value = payload
        .trim()
        .trim_start_matches("http://")
        .trim_start_matches("https://")
        .split('/')
        .next()
        .unwrap_or("");
    !value.is_empty() && value.contains('.') && !value.chars().any(|c| c.is_whitespace())
}

pub fn completions(prefix: &str) -> Vec<&'static str> {
    let commands = ["search | ", "fetch | https://", "learn | ", "status |", "quit |"];
    commands
        .iter()
        .copied()
        .filter(|cmd| cmd.starts_with(prefix))
        .collect()
}

pub fn render_prompt(buffer: &str) -> String {
    format!("axiom> {buffer}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_search() {
        assert_eq!(parse_line("search | rust docs").unwrap(), Command::Search("rust docs".into()));
    }

    #[test]
    fn defaults_missing_pipe_to_search() {
        assert_eq!(parse_line("latest AI news").unwrap(), Command::Search("latest AI news".into()));
        assert_eq!(parse_line("axiom").unwrap(), Command::AxiomPrompt);
    }

    #[test]
    fn validates_fetch_and_domain_payloads() {
        assert!(parse_line("fetch | https://example.com").is_ok());
        assert!(parse_line("fetch | ftp://example.com").is_err());
        assert!(parse_line("learn | https://example.com/docs").is_ok());
        assert!(parse_line("learn | not a domain").is_err());
    }

    #[test]
    fn command_roundtrip_line() {
        let cmd = parse_line("search | rust").unwrap();
        assert_eq!(cmd.to_line(), "search | rust");
        assert!(completions("sta").contains(&"status |"));
    }
}
