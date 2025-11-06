// FFI Layer: C API Module
//
// This module exposes the GraphBLAS C API

pub mod binary_op;
pub mod error;
pub mod index_binary_op;
pub mod index_unary_op;
pub mod matrix;
pub mod monoid;
pub mod scalar;
pub mod semiring;
pub mod types;
pub mod unary_op;
pub mod vector;

// Re-export commonly used types
pub use binary_op::GrB_BinaryOp;
pub use error::*;
pub use index_binary_op::GrB_IndexBinaryOp;
pub use index_unary_op::GrB_IndexUnaryOp;
pub use matrix::{GrB_Index, GrB_Matrix};
pub use monoid::GrB_Monoid;
pub use scalar::GrB_Scalar;
pub use semiring::GrB_Semiring;
pub use types::{
    GrB_BOOL, GrB_FP32, GrB_FP64, GrB_INT16, GrB_INT32, GrB_INT64, GrB_INT8, GrB_Type, GrB_UINT16,
    GrB_UINT32, GrB_UINT64, GrB_UINT8,
};
pub use unary_op::GrB_UnaryOp;
pub use vector::GrB_Vector;
