// Compiler Backend Abstraction
//
// Trait for different JIT backends (Cranelift, LLVM, etc.)

use crate::core::error::Result;
use crate::ir::IRGraph;

/// Function pointer type for SpMV kernel
/// Arguments: (row_ptrs, col_indices, values, x, y, nrows)
type SpMVKernel = unsafe extern "C" fn(*const usize, *const usize, *const f64, *const f64, *mut f64, usize);

/// Thread-safe wrapper for function pointer
/// SAFETY: JIT-compiled functions are stateless and safe to call from any thread
#[derive(Clone, Copy)]
struct KernelPtr(*const u8);

unsafe impl Send for KernelPtr {}
unsafe impl Sync for KernelPtr {}

/// Compiled function that can be executed
pub struct CompiledFunction {
    /// Function pointer to compiled kernel
    kernel_ptr: Option<KernelPtr>,
    /// IR graph for reference
    pub(crate) _graph: Option<IRGraph>,
}

// Manual Clone impl since function pointers are Copy
impl Clone for CompiledFunction {
    fn clone(&self) -> Self {
        Self {
            kernel_ptr: self.kernel_ptr,
            _graph: self._graph.clone(),
        }
    }
}

impl CompiledFunction {
    /// Create a new compiled function (for testing)
    #[cfg(test)]
    pub fn new_stub() -> Self {
        Self {
            kernel_ptr: None,
            _graph: None
        }
    }

    /// Create a new compiled function (internal use)
    pub(crate) fn new() -> Self {
        Self {
            kernel_ptr: None,
            _graph: None
        }
    }

    /// Create compiled function with kernel pointer
    pub(crate) fn with_kernel(kernel_ptr: *const u8) -> Self {
        Self {
            kernel_ptr: Some(KernelPtr(kernel_ptr)),
            _graph: None,
        }
    }

    /// Execute the compiled function
    ///
    /// # Arguments
    /// * `inputs` - Pointers to input data
    /// * `outputs` - Pointers to output data
    ///
    /// # Returns
    /// Ok(()) if execution succeeded
    ///
    /// TODO: This is a basic implementation that only handles SpMV for now.
    /// A full implementation would need to handle different operation types and formats.
    pub fn execute(&self, inputs: &[*const ()], outputs: &[*mut ()]) -> Result<()> {
        use crate::core::matrix::SparseStorage;
        use crate::core::vector::Vector;

        // If no kernel, return early (stub)
        let Some(kernel_ptr) = self.kernel_ptr else {
            return Ok(());
        };

        // For SpMV: inputs[0] = matrix storage, inputs[1] = u.indices, inputs[2] = u.values
        // outputs[0] = output vector
        if inputs.len() >= 3 && outputs.len() >= 1 {
            // SAFETY: We trust that the pointers are valid from the calling code
            unsafe {
                // Cast input matrix storage pointer
                let matrix_storage = inputs[0] as *const SparseStorage<f64>;

                // Cast input vector pointers (indices and values)
                let _u_indices = inputs[1] as *const usize;
                let u_values = inputs[2] as *const f64;

                // Cast output vector pointer
                let output_vec = outputs[0] as *mut Vector<f64>;

                // Extract CSR data from matrix
                if let SparseStorage::CSR { row_ptrs, col_indices, values } = &*matrix_storage {
                    // Extract output vector data
                    let y_data = (*output_vec).values_mut();

                    // NOTE: Don't zero output here - the kernel handles initialization
                    // with the semiring identity value. Zeroing would break min-plus
                    // and other semirings that don't use 0 as identity.

                    let nrows = row_ptrs.len() - 1;

                    // Cast function pointer and call kernel
                    let kernel_fn: SpMVKernel = std::mem::transmute(kernel_ptr.0);
                    kernel_fn(
                        row_ptrs.as_ptr(),
                        col_indices.as_ptr(),
                        values.as_ptr(),
                        u_values,
                        y_data.as_mut_ptr(),
                        nrows,
                    );
                }
            }
        }

        Ok(())
    }

    /// Execute with graph reference (for debugging/interpretation)
    pub(crate) fn with_graph(graph: IRGraph) -> Self {
        Self {
            kernel_ptr: None,
            _graph: Some(graph)
        }
    }
}

/// JIT backend trait
pub trait Backend: Send + Sync {
    /// Compile an IR graph to executable code
    fn compile(&self, graph: &IRGraph) -> Result<CompiledFunction>;

    /// Check if backend supports a specific feature
    fn supports_feature(&self, feature: BackendFeature) -> bool;

    /// Get backend name
    fn name(&self) -> &str;
}

/// Backend features
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendFeature {
    /// SIMD vectorization
    SIMD,
    /// Multi-threading
    MultiThreading,
    /// GPU offload
    GPU,
}
