// Multi-line validation for Lisp REPL
//
// Validates S-expressions to support multi-line input with balanced parentheses

use rustyline::validate::{ValidationContext, ValidationResult, Validator};

/// Validator for Lisp S-expressions
/// Checks if parentheses are balanced and allows multi-line input
pub struct LispValidator;

impl LispValidator {
    /// Create a new Lisp validator
    pub fn new() -> Self {
        Self
    }

    /// Count the depth of nested parentheses
    /// Returns (depth, unmatched_closing) where:
    /// - depth > 0: there are unmatched opening parens
    /// - depth < 0: there are unmatched closing parens (count in unmatched_closing)
    fn check_parens(input: &str) -> (i32, i32) {
        let mut depth = 0;
        let mut min_depth = 0;

        for c in input.chars() {
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

        (depth, min_depth.abs())
    }
}

impl Default for LispValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl Validator for LispValidator {
    fn validate(&self, ctx: &mut ValidationContext) -> rustyline::Result<ValidationResult> {
        let input = ctx.input();

        // Empty input is valid (will be skipped by REPL)
        if input.trim().is_empty() {
            return Ok(ValidationResult::Valid(None));
        }

        let (depth, unmatched_closing) = Self::check_parens(input);

        if unmatched_closing > 0 {
            // Too many closing parens
            let plural = if unmatched_closing == 1 { "" } else { "s" };
            Ok(ValidationResult::Invalid(Some(
                format!("{} unmatched closing paren{}", unmatched_closing, plural)
            )))
        } else if depth > 0 {
            // Unmatched opening parens - need more input
            Ok(ValidationResult::Incomplete)
        } else {
            // Balanced parens
            Ok(ValidationResult::Valid(None))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validator_creation() {
        let validator = LispValidator::new();
        let _ = validator; // Use it to avoid unused warning
    }

    #[test]
    fn test_balanced_parens() {
        let validator = LispValidator::new();
        let (depth, unmatched) = LispValidator::check_parens("(+ 1 2)");
        assert_eq!(depth, 0);
        assert_eq!(unmatched, 0);
    }

    #[test]
    fn test_unmatched_opening() {
        let validator = LispValidator::new();
        let (depth, unmatched) = LispValidator::check_parens("(+ 1 2");
        assert_eq!(depth, 1);
        assert_eq!(unmatched, 0);
    }

    #[test]
    fn test_unmatched_closing() {
        let validator = LispValidator::new();
        let (depth, unmatched) = LispValidator::check_parens("+ 1 2)");
        assert_eq!(depth, -1);
        assert_eq!(unmatched, 1);
    }

    #[test]
    fn test_nested_balanced() {
        let validator = LispValidator::new();
        let (depth, unmatched) = LispValidator::check_parens("(defkernel foo [x] (+ x 1))");
        assert_eq!(depth, 0);
        assert_eq!(unmatched, 0);
    }

    #[test]
    fn test_nested_incomplete() {
        let validator = LispValidator::new();
        let (depth, unmatched) = LispValidator::check_parens("(defkernel foo [x] (+ x 1)");
        assert_eq!(depth, 1);
        assert_eq!(unmatched, 0);
    }

    #[test]
    fn test_multiple_expressions() {
        let validator = LispValidator::new();
        let (depth, unmatched) = LispValidator::check_parens("(+ 1 2) (* 3 4)");
        assert_eq!(depth, 0);
        assert_eq!(unmatched, 0);
    }
}
