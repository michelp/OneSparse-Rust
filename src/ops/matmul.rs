// High-Level Matrix Multiplication Operations
//
// Provides high-level API for matrix multiplication that uses the JIT compilation
// pipeline: GraphBuilder → Optimization → Code Generation → Execution
//
// Operations:
// - mxm: Matrix-matrix multiply (C = A * B)
// - mxv: Matrix-vector multiply (w = A * u)
// - vxm: Vector-matrix multiply (w = u * A)

use crate::compiler::backend::Backend;
use crate::compiler::cache::{CacheKey, KernelCache};
use crate::compiler::cranelift_backend::CraneliftBackend;
use crate::core::error::{GraphBlasError, Result};
use crate::core::matrix::Matrix;
use crate::core::semiring::Semiring;
use crate::core::vector::Vector;
use crate::ir::builder::GraphBuilder;
use crate::ir::node::{BinaryOpKind, MonoidOp, ScalarValue, SemiringOp};
use crate::ir::types::ScalarType;
use crate::ir::Shape;
use crate::ops::descriptor::Descriptor;
use crate::optimizer::cse::CSEPass;
use crate::optimizer::format_select::FormatSelectionPass;
use crate::optimizer::fusion::FusionPass;
use crate::optimizer::pass::PassManager;
use crate::types::GraphBLASType;
use std::sync::Arc;

lazy_static::lazy_static! {
    /// Global kernel cache (100 MB default)
    static ref KERNEL_CACHE: KernelCache = KernelCache::new(100);
}

/// Matrix-matrix multiplication: C = A * B
///
/// # Arguments
/// * `c` - Output matrix (will be modified)
/// * `mask` - Optional mask matrix
/// * `a` - First input matrix
/// * `b` - Second input matrix
/// * `semiring` - Semiring for multiplication
/// * `desc` - Optional descriptor
pub fn mxm<T: GraphBLASType>(
    c: &mut Matrix<T>,
    mask: Option<&Matrix<bool>>,
    a: &Matrix<T>,
    b: &Matrix<T>,
    semiring: &Semiring<T>,
    desc: Option<&Descriptor>,
) -> Result<()> {
    // Get effective dimensions (considering transpose)
    let (a_rows, a_cols) = if let Some(d) = desc {
        if d.transpose_first {
            (a.ncols(), a.nrows())
        } else {
            (a.nrows(), a.ncols())
        }
    } else {
        (a.nrows(), a.ncols())
    };

    let (b_rows, b_cols) = if let Some(d) = desc {
        if d.transpose_second {
            (b.ncols(), b.nrows())
        } else {
            (b.nrows(), b.ncols())
        }
    } else {
        (b.nrows(), b.ncols())
    };

    // Validate dimensions
    if a_cols != b_rows {
        return Err(GraphBlasError::DimensionMismatch);
    }

    if c.nrows() != a_rows || c.ncols() != b_cols {
        return Err(GraphBlasError::DimensionMismatch);
    }

    // Check if mask dimensions match if present
    if let Some(m) = mask {
        if m.nrows() != c.nrows() || m.ncols() != c.ncols() {
            return Err(GraphBlasError::DimensionMismatch);
        }
    }

    // Build IR graph
    let mut builder = GraphBuilder::new();

    // Get scalar type
    let scalar_type = ScalarType::from_type_code(T::TYPE_CODE)
        .ok_or(GraphBlasError::InvalidValue)?;

    // Add inputs
    let a_shape = Shape::matrix(a.nrows(), a.ncols());
    let b_shape = Shape::matrix(b.nrows(), b.ncols());

    let a_node = builder.input_matrix("A", scalar_type, a_shape)?;
    let b_node = builder.input_matrix("B", scalar_type, b_shape)?;

    // Apply transpose if requested
    let (a_node, b_node) = if let Some(d) = desc {
        let a_final = if d.transpose_first {
            builder.transpose(a_node)?
        } else {
            a_node
        };

        let b_final = if d.transpose_second {
            builder.transpose(b_node)?
        } else {
            b_node
        };

        (a_final, b_final)
    } else {
        (a_node, b_node)
    };

    // Convert semiring to SemiringOp
    let semiring_op = SemiringOp {
        add_op: MonoidOp {
            binary_op: BinaryOpKind::Add, // TODO: Get from semiring
            identity: ScalarValue::from_type(scalar_type, 0.0),
        },
        mul_op: BinaryOpKind::Mul, // TODO: Get from semiring
    };

    // Create matmul operation
    let result = builder.matmul(a_node, b_node, semiring_op)?;

    // Mark as output
    builder.output(result)?;

    // Build graph
    let mut graph = builder.build();

    // Try to get cached kernel
    let cache_key = CacheKey::from_graph(&graph);
    let function = if let Some(cached_fn) = KERNEL_CACHE.get(&cache_key) {
        cached_fn
    } else {
        // Apply optimization passes
        let mut pass_manager = PassManager::new();
        pass_manager.add_pass(Box::new(FusionPass::new()));
        pass_manager.add_pass(Box::new(CSEPass::new()));
        pass_manager.add_pass(Box::new(FormatSelectionPass::new()));

        pass_manager.run_all(&mut graph)?;

        // Compile the graph
        let backend = CraneliftBackend::new()?;
        let compiled = backend.compile(&graph)?;

        // Cache the result (estimate 10KB per kernel)
        KERNEL_CACHE.insert(cache_key, compiled.clone(), 10240);

        Arc::new(compiled)
    };

    // Execute the compiled kernel
    // Extract sparse data from input matrices
    let a_storage = a.storage();
    let b_storage = b.storage();
    let c_storage = c.storage_mut();

    // Prepare input/output pointers for kernel execution
    // In a full implementation, these would point to the actual sparse arrays
    let inputs: Vec<*const ()> = vec![
        a_storage as *const _ as *const (),
        b_storage as *const _ as *const (),
    ];
    let outputs: Vec<*mut ()> = vec![
        c_storage as *mut _ as *mut (),
    ];

    // Call the compiled kernel
    // TODO: Real execution would:
    // 1. Convert storage format if needed (CSR/CSC/COO)
    // 2. Pass semiring operations to kernel
    // 3. Apply mask and descriptor settings
    // 4. Actually compute C = A * B using compiled native code
    function.execute(&inputs, &outputs)?;

    Ok(())
}

/// Matrix-vector multiplication: w = A * u
///
/// # Arguments
/// * `w` - Output vector (will be modified)
/// * `mask` - Optional mask vector
/// * `a` - Input matrix
/// * `u` - Input vector
/// * `semiring` - Semiring for multiplication
/// * `desc` - Optional descriptor
pub fn mxv<T: GraphBLASType>(
    w: &mut Vector<T>,
    mask: Option<&Vector<bool>>,
    a: &Matrix<T>,
    u: &Vector<T>,
    semiring: &Semiring<T>,
    desc: Option<&Descriptor>,
) -> Result<()> {
    // Validate dimensions
    if a.ncols() != u.size() {
        return Err(GraphBlasError::DimensionMismatch);
    }

    if w.size() != a.nrows() {
        return Err(GraphBlasError::DimensionMismatch);
    }

    // Check mask dimensions
    if let Some(m) = mask {
        if m.size() != w.size() {
            return Err(GraphBlasError::DimensionMismatch);
        }
    }

    // Build IR graph
    let mut builder = GraphBuilder::new();
    let scalar_type = ScalarType::from_type_code(T::TYPE_CODE)
        .ok_or(GraphBlasError::InvalidValue)?;

    let a_shape = Shape::matrix(a.nrows(), a.ncols());
    let u_shape = Shape::vector(u.size());

    let a_node = builder.input_matrix("A", scalar_type, a_shape)?;
    let u_node = builder.input_vector("u", scalar_type, u_shape)?;

    // Apply transpose if requested
    let a_node = if let Some(d) = desc {
        if d.transpose_first {
            builder.transpose(a_node)?
        } else {
            a_node
        }
    } else {
        a_node
    };

    // Convert semiring from core layer to IR layer
    // Use the semiring name to dispatch to the correct IR semiring
    let semiring_op = match semiring.name().as_ref() {
        "plus_times" => crate::ir::semirings::plus_times(scalar_type),
        "min_plus" => crate::ir::semirings::min_plus(scalar_type),
        "max_times" => crate::ir::semirings::max_times(scalar_type),
        "or_and" => crate::ir::semirings::or_and(),
        _ => {
            // Fallback to plus-times for unknown semirings
            crate::ir::semirings::plus_times(scalar_type)
        }
    };

    let result = builder.matvec(a_node, u_node, semiring_op)?;
    builder.output(result)?;

    let mut graph = builder.build();

    // Optimize and compile (similar to mxm)
    let cache_key = CacheKey::from_graph(&graph);
    let function = if let Some(cached_fn) = KERNEL_CACHE.get(&cache_key) {
        cached_fn
    } else {
        let mut pass_manager = PassManager::new();
        pass_manager.add_pass(Box::new(FusionPass::new()));
        pass_manager.add_pass(Box::new(CSEPass::new()));
        pass_manager.add_pass(Box::new(FormatSelectionPass::new()));
        pass_manager.run_all(&mut graph)?;

        let backend = CraneliftBackend::new()?;
        let compiled = backend.compile(&graph)?;
        KERNEL_CACHE.insert(cache_key, compiled.clone(), 10240);
        Arc::new(compiled)
    };

    // Execute the compiled kernel
    // Extract sparse data from inputs
    let a_storage = a.storage();

    // Prepare input/output pointers for kernel execution
    let inputs: Vec<*const ()> = vec![
        a_storage as *const _ as *const (),
        u.indices().as_ptr() as *const (),
        u.values().as_ptr() as *const (),
    ];
    let outputs: Vec<*mut ()> = vec![
        w as *mut Vector<T> as *mut (),  // Pass the whole vector for now
    ];

    // Call the compiled kernel
    // TODO: Real execution would compute w = A * u using compiled native code
    function.execute(&inputs, &outputs)?;

    Ok(())
}

/// Vector-matrix multiplication: w = u * A
///
/// # Arguments
/// * `w` - Output vector (will be modified)
/// * `mask` - Optional mask vector
/// * `u` - Input vector
/// * `a` - Input matrix
/// * `semiring` - Semiring for multiplication
/// * `desc` - Optional descriptor
pub fn vxm<T: GraphBLASType>(
    w: &mut Vector<T>,
    mask: Option<&Vector<bool>>,
    u: &Vector<T>,
    a: &Matrix<T>,
    semiring: &Semiring<T>,
    desc: Option<&Descriptor>,
) -> Result<()> {
    // Validate dimensions
    if u.size() != a.nrows() {
        return Err(GraphBlasError::DimensionMismatch);
    }

    if w.size() != a.ncols() {
        return Err(GraphBlasError::DimensionMismatch);
    }

    // Check mask dimensions
    if let Some(m) = mask {
        if m.size() != w.size() {
            return Err(GraphBlasError::DimensionMismatch);
        }
    }

    // Build IR graph
    let mut builder = GraphBuilder::new();
    let scalar_type = ScalarType::from_type_code(T::TYPE_CODE)
        .ok_or(GraphBlasError::InvalidValue)?;

    let u_shape = Shape::vector(u.size());
    let a_shape = Shape::matrix(a.nrows(), a.ncols());

    let u_node = builder.input_vector("u", scalar_type, u_shape)?;
    let a_node = builder.input_matrix("A", scalar_type, a_shape)?;

    // Apply transpose if requested
    let a_node = if let Some(d) = desc {
        if d.transpose_second {
            builder.transpose(a_node)?
        } else {
            a_node
        }
    } else {
        a_node
    };

    // Convert semiring from core layer to IR layer
    // Use the semiring name to dispatch to the correct IR semiring
    let semiring_op = match semiring.name().as_ref() {
        "plus_times" => crate::ir::semirings::plus_times(scalar_type),
        "min_plus" => crate::ir::semirings::min_plus(scalar_type),
        "max_times" => crate::ir::semirings::max_times(scalar_type),
        "or_and" => crate::ir::semirings::or_and(),
        _ => {
            // Fallback to plus-times for unknown semirings
            crate::ir::semirings::plus_times(scalar_type)
        }
    };

    let result = builder.vecmat(u_node, a_node, semiring_op)?;
    builder.output(result)?;

    let mut graph = builder.build();

    // Optimize and compile
    let cache_key = CacheKey::from_graph(&graph);
    let function = if let Some(cached_fn) = KERNEL_CACHE.get(&cache_key) {
        cached_fn
    } else {
        let mut pass_manager = PassManager::new();
        pass_manager.add_pass(Box::new(FusionPass::new()));
        pass_manager.add_pass(Box::new(CSEPass::new()));
        pass_manager.add_pass(Box::new(FormatSelectionPass::new()));
        pass_manager.run_all(&mut graph)?;

        let backend = CraneliftBackend::new()?;
        let compiled = backend.compile(&graph)?;
        KERNEL_CACHE.insert(cache_key, compiled.clone(), 10240);
        Arc::new(compiled)
    };

    // Execute the compiled kernel
    // Extract sparse data from inputs
    let a_storage = a.storage();

    // Prepare input/output pointers for kernel execution
    let inputs: Vec<*const ()> = vec![
        u.indices().as_ptr() as *const (),
        u.values().as_ptr() as *const (),
        a_storage as *const _ as *const (),
    ];
    let outputs: Vec<*mut ()> = vec![
        w as *mut Vector<T> as *mut (),  // Pass the whole vector for now
    ];

    // Call the compiled kernel
    // TODO: Real execution would compute w = u * A using compiled native code
    function.execute(&inputs, &outputs)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::semiring::Semiring;

    #[test]
    fn test_mxm_dimension_validation() {
        let mut c = Matrix::<f64>::new(10, 20).unwrap();
        let a = Matrix::<f64>::new(10, 15).unwrap();
        let b = Matrix::<f64>::new(15, 20).unwrap();

        let semiring = Semiring::plus_times().unwrap();

        // Should succeed with correct dimensions
        let result = mxm(&mut c, None, &a, &b, &semiring, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_mxm_dimension_mismatch() {
        let mut c = Matrix::<f64>::new(10, 20).unwrap();
        let a = Matrix::<f64>::new(10, 15).unwrap();
        let b = Matrix::<f64>::new(16, 20).unwrap(); // Wrong dimension

        let semiring = Semiring::plus_times().unwrap();

        let result = mxm(&mut c, None, &a, &b, &semiring, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_mxv_dimension_validation() {
        let mut w = Vector::<f64>::new(10).unwrap();
        let a = Matrix::<f64>::new(10, 20).unwrap();
        let u = Vector::<f64>::new(20).unwrap();

        let semiring = Semiring::plus_times().unwrap();

        let result = mxv(&mut w, None, &a, &u, &semiring, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_mxv_dimension_mismatch() {
        let mut w = Vector::<f64>::new(10).unwrap();
        let a = Matrix::<f64>::new(10, 20).unwrap();
        let u = Vector::<f64>::new(15).unwrap(); // Wrong dimension

        let semiring = Semiring::plus_times().unwrap();

        let result = mxv(&mut w, None, &a, &u, &semiring, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_vxm_dimension_validation() {
        let mut w = Vector::<f64>::new(20).unwrap();
        let u = Vector::<f64>::new(10).unwrap();
        let a = Matrix::<f64>::new(10, 20).unwrap();

        let semiring = Semiring::plus_times().unwrap();

        let result = vxm(&mut w, None, &u, &a, &semiring, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_vxm_dimension_mismatch() {
        let mut w = Vector::<f64>::new(20).unwrap();
        let u = Vector::<f64>::new(15).unwrap(); // Wrong dimension
        let a = Matrix::<f64>::new(10, 20).unwrap();

        let semiring = Semiring::plus_times().unwrap();

        let result = vxm(&mut w, None, &u, &a, &semiring, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_mxm_with_transpose_descriptor() {
        let mut c = Matrix::<f64>::new(10, 20).unwrap();
        let a = Matrix::<f64>::new(15, 10).unwrap(); // Will be transposed
        let b = Matrix::<f64>::new(15, 20).unwrap();

        let semiring = Semiring::plus_times().unwrap();
        let desc = Descriptor::with_transpose_first();

        let result = mxm(&mut c, None, &a, &b, &semiring, Some(&desc));
        assert!(result.is_ok());
    }
}
