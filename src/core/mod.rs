// Core Layer: Rust Implementation Module
//
// Idiomatic Rust implementations of GraphBLAS data structures

pub mod error;
pub mod handles;
pub mod matrix;
pub mod vector;
pub mod scalar;
pub mod semiring;
pub mod monoid;
pub mod binary_op;
pub mod unary_op;
pub mod index_binary_op;
pub mod index_unary_op;

// Re-export commonly used types
pub use error::{GraphBlasError, Result};
pub use handles::HandleRegistry;
pub use matrix::Matrix;
pub use vector::Vector;
pub use scalar::Scalar;
pub use semiring::Semiring;
pub use monoid::Monoid;
pub use binary_op::BinaryOp;
pub use unary_op::UnaryOp;
pub use index_binary_op::IndexBinaryOp;
pub use index_unary_op::IndexUnaryOp;
