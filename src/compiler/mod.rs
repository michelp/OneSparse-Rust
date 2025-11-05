// Compiler Module: JIT compilation infrastructure

pub mod backend;
pub mod cache;
pub mod cranelift_backend;

// Re-exports
pub use backend::{Backend, BackendFeature, CompiledFunction};
pub use cache::{CacheKey, CachedKernel, KernelCache};
pub use cranelift_backend::CraneliftBackend;
