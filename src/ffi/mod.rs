// FFI Layer: C API Module
//
// This module exposes the GraphBLAS C API

pub mod error;
pub mod types;
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
pub use error::*;
pub use types::{GrB_Type, GrB_BOOL, GrB_INT8, GrB_INT16, GrB_INT32, GrB_INT64,
                GrB_UINT8, GrB_UINT16, GrB_UINT32, GrB_UINT64, GrB_FP32, GrB_FP64};
pub use matrix::{GrB_Matrix, GrB_Index};
pub use vector::GrB_Vector;
pub use scalar::GrB_Scalar;
pub use semiring::GrB_Semiring;
pub use monoid::GrB_Monoid;
pub use binary_op::GrB_BinaryOp;
pub use unary_op::GrB_UnaryOp;
pub use index_binary_op::GrB_IndexBinaryOp;
pub use index_unary_op::GrB_IndexUnaryOp;
