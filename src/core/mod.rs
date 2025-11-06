// Core Layer: Rust Implementation Module
//
// Idiomatic Rust implementations of GraphBLAS data structures

pub mod binary_op;
pub mod container;
pub mod error;
pub mod handles;
pub mod index_binary_op;
pub mod index_unary_op;
pub mod matrix;
pub mod monoid;
pub mod scalar;
pub mod semiring;
pub mod unary_op;
pub mod vector;

// Re-export commonly used types
pub use binary_op::BinaryOp;
pub use error::{GraphBlasError, Result};
pub use handles::HandleRegistry;
pub use index_binary_op::IndexBinaryOp;
pub use index_unary_op::IndexUnaryOp;
pub use matrix::Matrix;
pub use monoid::Monoid;
pub use scalar::Scalar;
pub use semiring::Semiring;
pub use unary_op::UnaryOp;
pub use vector::Vector;
