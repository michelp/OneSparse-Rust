// High-Level Apply Operations
//
// Provides high-level API for apply operations that use the JIT compilation
// pipeline: GraphBuilder → Optimization → Code Generation → Execution
//
// Operations:
// - apply: Apply unary operation element-wise
// - apply_binary_left: Apply binary operation with scalar on left
// - apply_binary_right: Apply binary operation with scalar on right

use crate::compiler::backend::Backend;
use crate::compiler::cache::{CacheKey, KernelCache};
use crate::compiler::cranelift_backend::CraneliftBackend;
use crate::core::error::{GraphBlasError, Result};
use crate::core::matrix::Matrix;
use crate::core::vector::Vector;
use crate::ir::builder::GraphBuilder;
use crate::ir::node::{BinaryOpKind, ScalarValue, UnaryOpKind};
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
    /// Global kernel cache (shared with other ops)
    static ref KERNEL_CACHE: KernelCache = KernelCache::new(100);
}

/// Apply unary operation to matrix: C = op(A)
///
/// # Arguments
/// * `c` - Output matrix (will be modified)
/// * `mask` - Optional mask matrix
/// * `a` - Input matrix
/// * `op` - Unary operation to apply
/// * `desc` - Optional descriptor
pub fn apply_matrix<T: GraphBLASType>(
    c: &mut Matrix<T>,
    mask: Option<&Matrix<bool>>,
    a: &Matrix<T>,
    op: UnaryOpKind,
    desc: Option<&Descriptor>,
) -> Result<()> {
    // Validate dimensions
    if c.nrows() != a.nrows() || c.ncols() != a.ncols() {
        return Err(GraphBlasError::DimensionMismatch);
    }

    // Check mask dimensions
    if let Some(m) = mask {
        if m.nrows() != c.nrows() || m.ncols() != c.ncols() {
            return Err(GraphBlasError::DimensionMismatch);
        }
    }

    // Validate descriptor
    if let Some(d) = desc {
        d.validate_for_unary()?;
    }

    // Build IR graph
    let mut builder = GraphBuilder::new();
    let scalar_type =
        ScalarType::from_type_code(T::TYPE_CODE).ok_or(GraphBlasError::InvalidValue)?;

    let a_shape = Shape::matrix(a.nrows(), a.ncols());
    let a_node = builder.input_matrix("A", scalar_type, a_shape)?;

    // Create apply operation
    let result = builder.apply(a_node, op)?;
    builder.output(result)?;

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

        KERNEL_CACHE.insert(cache_key, compiled.clone(), 10240);
        Arc::new(compiled)
    };

    // Execute the compiled kernel
    let a_storage = a.storage();
    let c_storage = c.storage_mut();

    let inputs: Vec<*const ()> = vec![a_storage as *const _ as *const ()];
    let outputs: Vec<*mut ()> = vec![c_storage as *mut _ as *mut ()];

    // TODO: Real execution would compute C = op(A)
    function.execute(&inputs, &outputs)?;

    Ok(())
}

/// Apply unary operation to vector: w = op(u)
///
/// # Arguments
/// * `w` - Output vector (will be modified)
/// * `mask` - Optional mask vector
/// * `u` - Input vector
/// * `op` - Unary operation to apply
/// * `desc` - Optional descriptor
pub fn apply_vector<T: GraphBLASType>(
    w: &mut Vector<T>,
    mask: Option<&Vector<bool>>,
    u: &Vector<T>,
    op: UnaryOpKind,
    desc: Option<&Descriptor>,
) -> Result<()> {
    // Validate dimensions
    if w.size() != u.size() {
        return Err(GraphBlasError::DimensionMismatch);
    }

    // Check mask dimensions
    if let Some(m) = mask {
        if m.size() != w.size() {
            return Err(GraphBlasError::DimensionMismatch);
        }
    }

    // Validate descriptor
    if let Some(d) = desc {
        d.validate_for_unary()?;
    }

    // Build IR graph
    let mut builder = GraphBuilder::new();
    let scalar_type =
        ScalarType::from_type_code(T::TYPE_CODE).ok_or(GraphBlasError::InvalidValue)?;

    let u_shape = Shape::vector(u.size());
    let u_node = builder.input_vector("u", scalar_type, u_shape)?;

    let result = builder.apply(u_node, op)?;
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
    let inputs: Vec<*const ()> = vec![
        u.indices().as_ptr() as *const (),
        u.values().as_ptr() as *const (),
    ];
    let outputs: Vec<*mut ()> = vec![w as *mut Vector<T> as *mut ()];

    // TODO: Real execution would compute w = op(u)
    function.execute(&inputs, &outputs)?;

    Ok(())
}

/// Apply binary operation with scalar on left to matrix: C = scalar op A
///
/// # Arguments
/// * `c` - Output matrix (will be modified)
/// * `mask` - Optional mask matrix
/// * `scalar` - Scalar value on left
/// * `a` - Input matrix
/// * `op` - Binary operation to apply
/// * `desc` - Optional descriptor
pub fn apply_binary_left_matrix<T: GraphBLASType>(
    c: &mut Matrix<T>,
    mask: Option<&Matrix<bool>>,
    scalar: T,
    a: &Matrix<T>,
    op: BinaryOpKind,
    desc: Option<&Descriptor>,
) -> Result<()> {
    // Validate dimensions
    if c.nrows() != a.nrows() || c.ncols() != a.ncols() {
        return Err(GraphBlasError::DimensionMismatch);
    }

    // Check mask dimensions
    if let Some(m) = mask {
        if m.nrows() != c.nrows() || m.ncols() != c.ncols() {
            return Err(GraphBlasError::DimensionMismatch);
        }
    }

    // Validate descriptor
    if let Some(d) = desc {
        d.validate_for_unary()?;
    }

    // Build IR graph
    let mut builder = GraphBuilder::new();
    let scalar_type =
        ScalarType::from_type_code(T::TYPE_CODE).ok_or(GraphBlasError::InvalidValue)?;

    let a_shape = Shape::matrix(a.nrows(), a.ncols());
    let a_node = builder.input_matrix("A", scalar_type, a_shape)?;

    // Convert scalar to ScalarValue
    // TODO: Support all scalar types, not just f64
    let scalar_value = match scalar_type {
        ScalarType::Float64 => {
            ScalarValue::Float64(unsafe { *(&scalar as *const T as *const f64) })
        }
        _ => return Err(GraphBlasError::InvalidValue), // Only f64 supported for now
    };

    let result = builder.apply_binary_left(scalar_value, a_node, op)?;
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
    let a_storage = a.storage();
    let c_storage = c.storage_mut();

    let inputs: Vec<*const ()> = vec![
        &scalar as *const T as *const (),
        a_storage as *const _ as *const (),
    ];
    let outputs: Vec<*mut ()> = vec![c_storage as *mut _ as *mut ()];

    // TODO: Real execution would compute C = scalar op A
    function.execute(&inputs, &outputs)?;

    Ok(())
}

/// Apply binary operation with scalar on right to matrix: C = A op scalar
///
/// # Arguments
/// * `c` - Output matrix (will be modified)
/// * `mask` - Optional mask matrix
/// * `a` - Input matrix
/// * `scalar` - Scalar value on right
/// * `op` - Binary operation to apply
/// * `desc` - Optional descriptor
pub fn apply_binary_right_matrix<T: GraphBLASType>(
    c: &mut Matrix<T>,
    mask: Option<&Matrix<bool>>,
    a: &Matrix<T>,
    scalar: T,
    op: BinaryOpKind,
    desc: Option<&Descriptor>,
) -> Result<()> {
    // Validate dimensions
    if c.nrows() != a.nrows() || c.ncols() != a.ncols() {
        return Err(GraphBlasError::DimensionMismatch);
    }

    // Check mask dimensions
    if let Some(m) = mask {
        if m.nrows() != c.nrows() || m.ncols() != c.ncols() {
            return Err(GraphBlasError::DimensionMismatch);
        }
    }

    // Validate descriptor
    if let Some(d) = desc {
        d.validate_for_unary()?;
    }

    // Build IR graph
    let mut builder = GraphBuilder::new();
    let scalar_type =
        ScalarType::from_type_code(T::TYPE_CODE).ok_or(GraphBlasError::InvalidValue)?;

    let a_shape = Shape::matrix(a.nrows(), a.ncols());
    let a_node = builder.input_matrix("A", scalar_type, a_shape)?;

    // Convert scalar to ScalarValue
    // TODO: Support all scalar types, not just f64
    let scalar_value = match scalar_type {
        ScalarType::Float64 => {
            ScalarValue::Float64(unsafe { *(&scalar as *const T as *const f64) })
        }
        _ => return Err(GraphBlasError::InvalidValue), // Only f64 supported for now
    };

    let result = builder.apply_binary_right(a_node, scalar_value, op)?;
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
    let a_storage = a.storage();
    let c_storage = c.storage_mut();

    let inputs: Vec<*const ()> = vec![
        a_storage as *const _ as *const (),
        &scalar as *const T as *const (),
    ];
    let outputs: Vec<*mut ()> = vec![c_storage as *mut _ as *mut ()];

    // TODO: Real execution would compute C = A op scalar
    function.execute(&inputs, &outputs)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_matrix_dimensions() {
        let mut c = Matrix::<f64>::new(10, 20).unwrap();
        let a = Matrix::<f64>::new(10, 20).unwrap();

        let result = apply_matrix(&mut c, None, &a, UnaryOpKind::Abs, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_apply_matrix_dimension_mismatch() {
        let mut c = Matrix::<f64>::new(10, 20).unwrap();
        let a = Matrix::<f64>::new(15, 20).unwrap(); // Wrong dimensions

        let result = apply_matrix(&mut c, None, &a, UnaryOpKind::Abs, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_apply_vector_dimensions() {
        let mut w = Vector::<f64>::new(10).unwrap();
        let u = Vector::<f64>::new(10).unwrap();

        let result = apply_vector(&mut w, None, &u, UnaryOpKind::Sqrt, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_apply_binary_left_matrix() {
        let mut c = Matrix::<f64>::new(10, 20).unwrap();
        let a = Matrix::<f64>::new(10, 20).unwrap();

        let result = apply_binary_left_matrix(&mut c, None, 2.0, &a, BinaryOpKind::Mul, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_apply_binary_right_matrix() {
        let mut c = Matrix::<f64>::new(10, 20).unwrap();
        let a = Matrix::<f64>::new(10, 20).unwrap();

        let result = apply_binary_right_matrix(&mut c, None, &a, 5.0, BinaryOpKind::Add, None);
        assert!(result.is_ok());
    }
}
