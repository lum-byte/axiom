mod dispatcher;
mod logo;
mod repl;
mod ui;

use dispatcher::{Dispatcher, EchoDispatcher, InterfaceDispatcher};
use repl::{parse_line, Command};
use std::io::{self, Write};
use ui::UiState;

fn main() {
    let echo = std::env::var("AXIOM_TUI_ECHO").ok().as_deref() == Some("1");
    let dispatcher: Box<dyn Dispatcher> = if echo {
        Box::new(EchoDispatcher)
    } else {
        Box::new(InterfaceDispatcher::from_env())
    };
    let mut ui = UiState::new();
    ui.status = "ready".to_string();
    println!("{}", ui.render());
    loop {
        print!("axiom> ");
        let _ = io::stdout().flush();
        let mut line = String::new();
        if io::stdin().read_line(&mut line).is_err() {
            break;
        }
        match parse_line(&line) {
            Ok(Command::Quit) => break,
            Ok(_) => match dispatcher.dispatch(line.trim()) {
                Ok(result) => {
                    ui.set_status_from_result(&result.raw);
                    ui.push_history(result.raw);
                    println!("{}", ui.render());
                }
                Err(err) => {
                    ui.push_error(format!("dispatch error: {err}"));
                    eprintln!("dispatch error: {err}");
                }
            },
            Err(err) => {
                ui.push_error(err.clone());
                eprintln!("{err}");
            }
        }
    }
}
