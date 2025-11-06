// Lisp DSL for JIT-compiled kernels
//
// This module provides a Lisp-based domain-specific language for defining
// graph algorithms that compile to JIT-optimized kernels.
//
// Example usage:
//   (defkernel bfs [g u]
//     (or-and g u))
//
//   (bfs my_matrix)

pub mod ast;
pub mod compiler;
pub mod completer;
pub mod eval;
pub mod highlighter;
pub mod kernel;
pub mod parser;
pub mod types;
pub mod validator;

pub use ast::*;
pub use compiler::*;
pub use completer::*;
pub use eval::*;
pub use highlighter::*;
pub use kernel::*;
pub use parser::*;
pub use types::*;
pub use validator::*;
