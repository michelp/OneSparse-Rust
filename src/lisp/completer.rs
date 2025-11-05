// Tab completion for Lisp REPL
//
// Provides context-aware completion for function names, keywords, and variables

use rustyline::completion::{Completer, Pair};
use rustyline::Context;

/// Lisp completer for S-expression aware tab completion
pub struct LispCompleter {
    /// Built-in function names and keywords
    keywords: Vec<String>,
}

impl LispCompleter {
    /// Create a new Lisp completer with all built-in functions
    pub fn new() -> Self {
        let keywords = vec![
            // Special forms
            "defkernel".to_string(),
            "let".to_string(),

            // Semiring operations
            "plus-times".to_string(),
            "min-plus".to_string(),
            "max-times".to_string(),
            "or-and".to_string(),

            // Matrix operations
            "transpose".to_string(),
            "matmul".to_string(),
            "mxm".to_string(),
            "mxv".to_string(),
            "vxm".to_string(),

            // Element-wise operations
            "ewise-add".to_string(),
            "ewise-mult".to_string(),

            // Vector operations
            "vec-add".to_string(),
            "vec-mult".to_string(),

            // Unary apply operations
            "abs".to_string(),
            "neg".to_string(),
            "sqrt".to_string(),
            "exp".to_string(),
            "log".to_string(),

            // Reduction operations
            "reduce-row".to_string(),
            "reduce-col".to_string(),
            "reduce-vector".to_string(),

            // Format conversion
            "to-csr".to_string(),
            "to-csc".to_string(),
            "to-dense".to_string(),

            // Logging
            "set-log-level".to_string(),
            "get-log-level".to_string(),

            // Utility
            "exit".to_string(),
        ];

        Self { keywords }
    }

    /// Extract the word being completed from the line
    fn extract_word<'a>(&self, line: &'a str, pos: usize) -> (usize, &'a str) {
        // Find the start of the current word
        // A word starts after whitespace, opening paren, or at the beginning
        let word_start = line[..pos]
            .rfind(|c: char| c.is_whitespace() || c == '(' || c == ')')
            .map(|i| i + 1)
            .unwrap_or(0);

        let word = &line[word_start..pos];
        (word_start, word)
    }
}

impl Default for LispCompleter {
    fn default() -> Self {
        Self::new()
    }
}

impl Completer for LispCompleter {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Pair>)> {
        let (word_start, word) = self.extract_word(line, pos);

        // If word is empty or just whitespace, don't complete
        if word.is_empty() {
            return Ok((pos, Vec::new()));
        }

        // Find matching keywords
        let matches: Vec<Pair> = self.keywords
            .iter()
            .filter(|k| k.starts_with(word))
            .map(|k| Pair {
                display: k.clone(),
                replacement: k.clone(),
            })
            .collect();

        Ok((word_start, matches))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_completer_creation() {
        let completer = LispCompleter::new();
        assert!(!completer.keywords.is_empty());
    }

    #[test]
    fn test_complete_transpose() {
        let completer = LispCompleter::new();
        let (_, matches) = completer.complete("(tra", 4, &Context::new(&rustyline::history::DefaultHistory::new())).unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].replacement, "transpose");
    }

    #[test]
    fn test_complete_multiple_matches() {
        let completer = LispCompleter::new();
        let (_, matches) = completer.complete("(m", 2, &Context::new(&rustyline::history::DefaultHistory::new())).unwrap();
        // Should match: matmul, mxm, mxv, max-times, min-plus
        assert!(matches.len() >= 3);
        assert!(matches.iter().any(|p| p.replacement == "matmul"));
        assert!(matches.iter().any(|p| p.replacement == "mxm"));
        assert!(matches.iter().any(|p| p.replacement == "mxv"));
    }

    #[test]
    fn test_complete_after_space() {
        let completer = LispCompleter::new();
        let (start, matches) = completer.complete("(matmul a sq", 12, &Context::new(&rustyline::history::DefaultHistory::new())).unwrap();
        assert_eq!(start, 10); // Should start at "sq"
        assert!(matches.iter().any(|p| p.replacement == "sqrt"));
    }

    #[test]
    fn test_no_completion_for_empty() {
        let completer = LispCompleter::new();
        let (_, matches) = completer.complete("(", 1, &Context::new(&rustyline::history::DefaultHistory::new())).unwrap();
        assert_eq!(matches.len(), 0);
    }
}
