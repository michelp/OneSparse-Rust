// Lisp REPL for rustsparse
//
// Interactive Read-Eval-Print Loop for the Lisp DSL

use rustsparse::lisp::completer::LispCompleter;
use rustsparse::lisp::eval::Evaluator;
use rustsparse::lisp::highlighter::LispHighlighter;
use rustsparse::lisp::validator::LispValidator;
use rustyline::completion::Completer;
use rustyline::error::ReadlineError;
use rustyline::highlight::{CmdKind, Highlighter};
use rustyline::hint::Hinter;
use rustyline::validate::{ValidationContext, ValidationResult, Validator};
use rustyline::{Editor, Helper};
use std::borrow::Cow;
use std::io::{self, BufRead};

/// Helper struct that combines all REPL features
struct ReplHelper {
    completer: LispCompleter,
    validator: LispValidator,
    highlighter: LispHighlighter,
}

impl ReplHelper {
    fn new() -> Self {
        Self {
            completer: LispCompleter::new(),
            validator: LispValidator::new(),
            highlighter: LispHighlighter::new(),
        }
    }
}

// Implement required traits for the helper
impl Completer for ReplHelper {
    type Candidate = <LispCompleter as Completer>::Candidate;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        ctx: &rustyline::Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Self::Candidate>)> {
        self.completer.complete(line, pos, ctx)
    }
}

impl Hinter for ReplHelper {
    type Hint = String;
}

impl Highlighter for ReplHelper {
    fn highlight<'l>(&self, line: &'l str, pos: usize) -> Cow<'l, str> {
        self.highlighter.highlight(line, pos)
    }

    fn highlight_char(&self, line: &str, pos: usize, kind: CmdKind) -> bool {
        self.highlighter.highlight_char(line, pos, kind)
    }
}

impl Validator for ReplHelper {
    fn validate(&self, ctx: &mut ValidationContext) -> rustyline::Result<ValidationResult> {
        self.validator.validate(ctx)
    }
}

// Implement the Helper trait (marker trait)
impl Helper for ReplHelper {}

fn main() {
    let mut eval = Evaluator::new();

    // Try to create an interactive editor with our custom helper
    let editor_result = Editor::<ReplHelper, _>::new();

    match editor_result {
        Ok(mut rl) => {
            // Set the helper with our custom completer
            rl.set_helper(Some(ReplHelper::new()));
            run_repl(&mut eval, &mut rl);
        }
        Err(_) => run_batch(&mut eval),
    }
}

fn run_repl(eval: &mut Evaluator, rl: &mut Editor<ReplHelper, rustyline::history::DefaultHistory>) {
    println!("RustSparse Lisp REPL");
    println!("Type (exit) or Ctrl-D to quit");
    println!("Use Ctrl-R to search history, Up/Down arrows to navigate");
    println!("Press TAB for completion");
    println!("Multi-line input supported - incomplete S-expressions will continue on next line");
    println!();

    // Try to load history from file
    let history_file = dirs::home_dir().map(|mut path| {
        path.push(".rustsparse_history");
        path
    });

    if let Some(ref path) = history_file {
        // Ignore errors when loading history (file may not exist yet)
        let _ = rl.load_history(path);
    }

    loop {
        let readline = rl.readline(">> ");

        match readline {
            Ok(line) => {
                let trimmed = line.trim();

                // Skip empty lines
                if trimmed.is_empty() {
                    continue;
                }

                // Add to history
                let _ = rl.add_history_entry(&line);

                // Check for exit
                if trimmed == "(exit)" || trimmed == "exit" {
                    break;
                }

                // Evaluate the line
                match eval.eval_program(trimmed) {
                    Ok(results) => {
                        for result in results {
                            println!("{}", result);
                        }
                    }
                    Err(e) => {
                        eprintln!("Error: {:?}", e);
                    }
                }
            }
            Err(ReadlineError::Interrupted) => {
                // Ctrl-C
                println!("^C");
                break;
            }
            Err(ReadlineError::Eof) => {
                // Ctrl-D
                println!();
                break;
            }
            Err(err) => {
                eprintln!("Error: {:?}", err);
                break;
            }
        }
    }

    // Save history on exit
    if let Some(ref path) = history_file {
        if let Err(e) = rl.save_history(path) {
            eprintln!("Warning: Failed to save history: {}", e);
        }
    }
}

fn run_batch(eval: &mut Evaluator) {
    let stdin = io::stdin();
    let mut input = String::new();

    // Read all input
    for line in stdin.lock().lines() {
        match line {
            Ok(l) => {
                input.push_str(&l);
                input.push('\n');
            }
            Err(e) => {
                eprintln!("Error reading input: {}", e);
                std::process::exit(1);
            }
        }
    }

    // Evaluate the entire input
    match eval.eval_program(&input) {
        Ok(results) => {
            for result in results {
                println!("{}", result);
            }
        }
        Err(e) => {
            eprintln!("Error: {:?}", e);
            std::process::exit(1);
        }
    }
}
