// High-Level Operations Module
//
// Provides high-level GraphBLAS operations that use the JIT compilation pipeline.

pub mod descriptor;
pub mod matmul;
pub mod ewise;
pub mod apply;

// Re-exports
pub use descriptor::{Descriptor, DescriptorField, DescriptorValue};
pub use matmul::{mxm, mxv, vxm};
pub use ewise::{ewadd_matrix, ewmult_matrix, ewadd_vector, ewmult_vector};
pub use apply::{apply_matrix, apply_vector, apply_binary_left_matrix, apply_binary_right_matrix};
