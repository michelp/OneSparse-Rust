// RustSparse: Rust implementation of GraphBLAS API
//
// This library provides a Rust implementation of the GraphBLAS C API
// with link-time compatibility with SuiteSparse:GraphBLAS.
//
// Architecture:
// - Layer 1 (ffi): C API compatibility layer with opaque pointers
// - Layer 2 (core): FFI safety layer and Rust implementation
// - Layer 3 (types): Type system bridging runtime and compile-time types
// - JIT compilation: IR → Optimization → Code Generation → Execution

#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]

#[macro_use]
extern crate lazy_static;

// Public modules
pub mod types;
pub mod core;
pub mod ffi;

// JIT compilation infrastructure
pub mod ir;
pub mod compiler;
pub mod optimizer;
pub mod ops;

// Re-export commonly used items for convenience
pub use types::{GraphBLASType, TypeCode, TypeDescriptor};
pub use core::{GraphBlasError, Result};

// FFI exports are automatically available through #[no_mangle]
// functions in the ffi module
