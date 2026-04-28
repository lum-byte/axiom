use crate::logo::logo;
use crate::repl::render_prompt;

#[derive(Debug, Clone)]
pub struct UiState {
    pub input: String,
    pub history: Vec<String>,
    pub status: String,
    pub errors: Vec<String>,
    pub max_history: usize,
}

impl UiState {
    pub fn new() -> Self {
        Self { input: String::new(), history: Vec::new(), status: "cold".into(), errors: Vec::new(), max_history: 200 }
    }

    pub fn push_history(&mut self, line: impl Into<String>) {
        self.history.push(line.into());
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }
    }

    pub fn push_error(&mut self, err: impl Into<String>) {
        self.errors.push(err.into());
        if self.errors.len() > 20 {
            self.errors.remove(0);
        }
        self.status = "error".into();
    }

    pub fn set_status_from_result(&mut self, raw: &str) {
        if raw.contains("\"status\":\"ok\"") || raw.contains("\"status\": \"ok\"") {
            self.status = "ok".into();
        } else if raw.contains("\"status\":\"accepted\"") || raw.contains("\"status\": \"accepted\"") {
            self.status = "queued".into();
        } else if raw.contains("\"status\":\"empty\"") || raw.contains("\"status\": \"empty\"") {
            self.status = "learning".into();
        } else if raw.contains("\"status\":\"error\"") || raw.contains("\"status\": \"error\"") {
            self.status = "error".into();
        }
    }

    pub fn render(&self) -> String {
        let mut out = String::new();
        out.push_str(logo());
        out.push('\n');
        out.push_str(&format!("status: {}\n", self.status));
        if !self.errors.is_empty() {
            out.push_str("errors:\n");
            for item in self.errors.iter().rev().take(3).rev() {
                out.push_str("  ");
                out.push_str(item);
                out.push('\n');
            }
        }
        out.push_str("history:\n");
        for item in self.history.iter().rev().take(12).rev() {
            out.push_str("  ");
            out.push_str(item);
            out.push('\n');
        }
        out.push_str(&render_prompt(&self.input));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_prompt() {
        let ui = UiState::new();
        assert!(ui.render().contains("axiom>"));
    }

    #[test]
    fn tracks_errors_and_status() {
        let mut ui = UiState::new();
        ui.push_error("bad command");
        assert_eq!(ui.status, "error");
        ui.set_status_from_result("{\"status\":\"accepted\"}");
        assert_eq!(ui.status, "queued");
    }
}
