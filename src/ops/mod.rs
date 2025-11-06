// High-Level Operations Module
//
// Provides high-level GraphBLAS operations that use the JIT compilation pipeline.

pub mod apply;
pub mod descriptor;
pub mod ewise;
pub mod matmul;

// Re-exports
pub use apply::{apply_binary_left_matrix, apply_binary_right_matrix, apply_matrix, apply_vector};
pub use descriptor::{Descriptor, DescriptorField, DescriptorValue};
pub use ewise::{ewadd_matrix, ewadd_vector, ewmult_matrix, ewmult_vector};
pub use matmul::{mxm, mxv, vxm};
