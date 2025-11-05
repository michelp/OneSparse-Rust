// Syntax highlighting for Lisp REPL
//
// Provides real-time syntax highlighting as you type

use rustyline::highlight::{CmdKind, Highlighter};
use std::borrow::Cow;

/// Lisp syntax highlighter
/// Highlights parentheses to show balanced/unbalanced state
pub struct LispHighlighter;

impl LispHighlighter {
    /// Create a new Lisp highlighter
    pub fn new() -> Self {
        Self
    }

    /// Check if parentheses are balanced
    fn check_balance(line: &str) -> (bool, i32) {
        let mut depth = 0;
        let mut min_depth = 0;

        for c in line.chars() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth < min_depth {
                        min_depth = depth;
                    }
                }
                _ => {}
            }
        }

        let has_unmatched_closing = min_depth < 0;
        let balanced = depth == 0 && !has_unmatched_closing;

        (balanced, depth)
    }
}

impl Default for LispHighlighter {
    fn default() -> Self {
        Self::new()
    }
}

impl Highlighter for LispHighlighter {
    fn highlight<'l>(&self, line: &'l str, _pos: usize) -> Cow<'l, str> {
        // If line is empty, no highlighting needed
        if line.trim().is_empty() {
            return Cow::Borrowed(line);
        }

        let (balanced, depth) = Self::check_balance(line);

        if balanced {
            // Green for balanced expressions
            Cow::Owned(format!("\x1b[32m{}\x1b[0m", line))
        } else if depth < 0 {
            // Red for unmatched closing parens
            Cow::Owned(format!("\x1b[31m{}\x1b[0m", line))
        } else {
            // Yellow for incomplete (unmatched opening parens)
            Cow::Owned(format!("\x1b[33m{}\x1b[0m", line))
        }
    }

    fn highlight_char(&self, line: &str, _pos: usize, _kind: CmdKind) -> bool {
        // Enable character-by-character highlighting
        !line.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_highlighter_creation() {
        let highlighter = LispHighlighter::new();
        let _ = highlighter; // Use it to avoid unused warning
    }

    #[test]
    fn test_balanced_expression() {
        let highlighter = LispHighlighter::new();
        let (balanced, depth) = LispHighlighter::check_balance("(+ 1 2)");
        assert!(balanced);
        assert_eq!(depth, 0);
    }

    #[test]
    fn test_unbalanced_opening() {
        let highlighter = LispHighlighter::new();
        let (balanced, depth) = LispHighlighter::check_balance("(+ 1 2");
        assert!(!balanced);
        assert_eq!(depth, 1);
    }

    #[test]
    fn test_unbalanced_closing() {
        let highlighter = LispHighlighter::new();
        let (balanced, depth) = LispHighlighter::check_balance("+ 1 2)");
        assert!(!balanced);
        assert_eq!(depth, -1);
    }

    #[test]
    fn test_nested_balanced() {
        let highlighter = LispHighlighter::new();
        let (balanced, depth) = LispHighlighter::check_balance("(defkernel foo [x] (+ x 1))");
        assert!(balanced);
        assert_eq!(depth, 0);
    }

    #[test]
    fn test_highlight_produces_output() {
        let highlighter = LispHighlighter::new();
        let highlighted = highlighter.highlight("(+ 1 2)", 0);
        // Should have ANSI codes
        assert!(highlighted.contains("\x1b["));
    }
}
