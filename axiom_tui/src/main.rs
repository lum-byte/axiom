#![allow(dead_code)]

mod dispatcher;
mod logo;
mod repl;
mod runtime_resolver;
mod ui;

use dispatcher::{Dispatcher, EchoDispatcher, InterfaceDispatcher};
use repl::{parse_line, Command};
use runtime_resolver::resolve_runtime;
use std::io::{self, Write};
use ui::UiState;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|arg| arg == "--resolve-runtime" || arg == "--check-runtime") {
        let deep = args.iter().any(|arg| arg == "--resolve-runtime-deep" || arg == "--deep");
        let resolution = resolve_runtime(deep);
        println!("{}", resolution.to_json());
        std::process::exit(if resolution.ok() { 0 } else { 1 });
    }
    if args.iter().any(|arg| arg == "--resolve-runtime-deep") {
        let resolution = resolve_runtime(true);
        println!("{}", resolution.to_json());
        std::process::exit(if resolution.ok() { 0 } else { 1 });
    }

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
            Ok(Command::AxiomPrompt) => {
                ui.status = "ready".to_string();
                println!("{}", ui.render());
            }
            Ok(command) => match dispatcher.dispatch(&command.to_line()) {
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
