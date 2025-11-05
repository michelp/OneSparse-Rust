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
pub mod parser;
pub mod types;
pub mod compiler;
pub mod kernel;
pub mod eval;

pub use ast::*;
pub use parser::*;
pub use types::*;
pub use compiler::*;
pub use kernel::*;
pub use eval::*;
