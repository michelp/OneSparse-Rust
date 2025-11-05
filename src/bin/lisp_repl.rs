// Lisp REPL for rustsparse
//
// Interactive Read-Eval-Print Loop for the Lisp DSL

use rustsparse::lisp::eval::Evaluator;
use std::io::{self, BufRead, Write};

fn main() {
    let mut eval = Evaluator::new();

    // Check if stdin is a TTY (interactive mode) or pipe (batch mode)
    if atty::is(atty::Stream::Stdin) {
        // Interactive mode
        run_repl(&mut eval);
    } else {
        // Batch mode: read from stdin
        run_batch(&mut eval);
    }
}

fn run_repl(eval: &mut Evaluator) {
    println!("RustSparse Lisp REPL");
    println!("Type (exit) or Ctrl-D to quit");
    println!();

    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();

    loop {
        print!("> ");
        io::stdout().flush().unwrap();

        match lines.next() {
            Some(Ok(line)) => {
                let trimmed = line.trim();

                // Check for exit
                if trimmed == "(exit)" || trimmed == "exit" {
                    break;
                }

                // Skip empty lines
                if trimmed.is_empty() {
                    continue;
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
            Some(Err(e)) => {
                eprintln!("Error reading input: {}", e);
                break;
            }
            None => {
                // EOF (Ctrl-D)
                println!();
                break;
            }
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
