// High-Level Element-wise Operations
//
// Provides high-level API for element-wise operations that use the JIT compilation
// pipeline: GraphBuilder → Optimization → Code Generation → Execution
//
// Operations:
// - ewadd: Element-wise addition (union semantics)
// - ewmult: Element-wise multiplication (intersection semantics)

use crate::compiler::backend::Backend;
use crate::compiler::cache::{CacheKey, KernelCache};
use crate::compiler::cranelift_backend::CraneliftBackend;
use crate::core::error::{GraphBlasError, Result};
use crate::core::matrix::Matrix;
use crate::core::vector::Vector;
use crate::ir::builder::GraphBuilder;
use crate::ir::node::BinaryOpKind;
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

/// Element-wise addition (union semantics): C = A + B
///
/// # Arguments
/// * `c` - Output matrix (will be modified)
/// * `mask` - Optional mask matrix
/// * `a` - First input matrix
/// * `b` - Second input matrix
/// * `op` - Binary operation to apply
/// * `desc` - Optional descriptor
pub fn ewadd_matrix<T: GraphBLASType>(
    c: &mut Matrix<T>,
    mask: Option<&Matrix<bool>>,
    a: &Matrix<T>,
    b: &Matrix<T>,
    op: BinaryOpKind,
    desc: Option<&Descriptor>,
) -> Result<()> {
    // Validate dimensions
    if a.nrows() != b.nrows() || a.ncols() != b.ncols() {
        return Err(GraphBlasError::DimensionMismatch);
    }

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
        d.validate_for_ewise()?;
    }

    // Build IR graph
    let mut builder = GraphBuilder::new();
    let scalar_type = ScalarType::from_type_code(T::TYPE_CODE)
        .ok_or(GraphBlasError::InvalidValue)?;

    let a_shape = Shape::matrix(a.nrows(), a.ncols());
    let b_shape = Shape::matrix(b.nrows(), b.ncols());

    let a_node = builder.input_matrix("A", scalar_type, a_shape)?;
    let b_node = builder.input_matrix("B", scalar_type, b_shape)?;

    // Create element-wise operation
    let result = builder.ewise_add(a_node, b_node, op)?;
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
    let b_storage = b.storage();
    let c_storage = c.storage_mut();

    let inputs: Vec<*const ()> = vec![
        a_storage as *const _ as *const (),
        b_storage as *const _ as *const (),
    ];
    let outputs: Vec<*mut ()> = vec![
        c_storage as *mut _ as *mut (),
    ];

    // TODO: Real execution would compute C = A + B element-wise
    function.execute(&inputs, &outputs)?;

    Ok(())
}

/// Element-wise multiplication (intersection semantics): C = A .* B
///
/// # Arguments
/// * `c` - Output matrix (will be modified)
/// * `mask` - Optional mask matrix
/// * `a` - First input matrix
/// * `b` - Second input matrix
/// * `op` - Binary operation to apply
/// * `desc` - Optional descriptor
pub fn ewmult_matrix<T: GraphBLASType>(
    c: &mut Matrix<T>,
    mask: Option<&Matrix<bool>>,
    a: &Matrix<T>,
    b: &Matrix<T>,
    op: BinaryOpKind,
    desc: Option<&Descriptor>,
) -> Result<()> {
    // Validate dimensions
    if a.nrows() != b.nrows() || a.ncols() != b.ncols() {
        return Err(GraphBlasError::DimensionMismatch);
    }

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
        d.validate_for_ewise()?;
    }

    // Build IR graph
    let mut builder = GraphBuilder::new();
    let scalar_type = ScalarType::from_type_code(T::TYPE_CODE)
        .ok_or(GraphBlasError::InvalidValue)?;

    let a_shape = Shape::matrix(a.nrows(), a.ncols());
    let b_shape = Shape::matrix(b.nrows(), b.ncols());

    let a_node = builder.input_matrix("A", scalar_type, a_shape)?;
    let b_node = builder.input_matrix("B", scalar_type, b_shape)?;

    // Create element-wise multiplication
    let result = builder.ewise_mult(a_node, b_node, op)?;
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
    let b_storage = b.storage();
    let c_storage = c.storage_mut();

    let inputs: Vec<*const ()> = vec![
        a_storage as *const _ as *const (),
        b_storage as *const _ as *const (),
    ];
    let outputs: Vec<*mut ()> = vec![
        c_storage as *mut _ as *mut (),
    ];

    // TODO: Real execution would compute C = A .* B element-wise
    function.execute(&inputs, &outputs)?;

    Ok(())
}

/// Element-wise addition for vectors: w = u + v
///
/// # Arguments
/// * `w` - Output vector (will be modified)
/// * `mask` - Optional mask vector
/// * `u` - First input vector
/// * `v` - Second input vector
/// * `op` - Binary operation to apply
/// * `desc` - Optional descriptor
pub fn ewadd_vector<T: GraphBLASType>(
    w: &mut Vector<T>,
    mask: Option<&Vector<bool>>,
    u: &Vector<T>,
    v: &Vector<T>,
    op: BinaryOpKind,
    desc: Option<&Descriptor>,
) -> Result<()> {
    // Validate dimensions
    if u.size() != v.size() {
        return Err(GraphBlasError::DimensionMismatch);
    }

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
        d.validate_for_ewise()?;
    }

    // Build IR graph
    let mut builder = GraphBuilder::new();
    let scalar_type = ScalarType::from_type_code(T::TYPE_CODE)
        .ok_or(GraphBlasError::InvalidValue)?;

    let u_shape = Shape::vector(u.size());
    let v_shape = Shape::vector(v.size());

    let u_node = builder.input_vector("u", scalar_type, u_shape)?;
    let v_node = builder.input_vector("v", scalar_type, v_shape)?;

    let result = builder.ewise_add(u_node, v_node, op)?;
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
        v.indices().as_ptr() as *const (),
        v.values().as_ptr() as *const (),
    ];
    let outputs: Vec<*mut ()> = vec![
        w as *mut Vector<T> as *mut (),
    ];

    // TODO: Real execution would compute w = u + v element-wise
    function.execute(&inputs, &outputs)?;

    Ok(())
}

/// Element-wise multiplication for vectors: w = u .* v
///
/// # Arguments
/// * `w` - Output vector (will be modified)
/// * `mask` - Optional mask vector
/// * `u` - First input vector
/// * `v` - Second input vector
/// * `op` - Binary operation to apply
/// * `desc` - Optional descriptor
pub fn ewmult_vector<T: GraphBLASType>(
    w: &mut Vector<T>,
    mask: Option<&Vector<bool>>,
    u: &Vector<T>,
    v: &Vector<T>,
    op: BinaryOpKind,
    desc: Option<&Descriptor>,
) -> Result<()> {
    // Validate dimensions
    if u.size() != v.size() {
        return Err(GraphBlasError::DimensionMismatch);
    }

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
        d.validate_for_ewise()?;
    }

    // Build IR graph
    let mut builder = GraphBuilder::new();
    let scalar_type = ScalarType::from_type_code(T::TYPE_CODE)
        .ok_or(GraphBlasError::InvalidValue)?;

    let u_shape = Shape::vector(u.size());
    let v_shape = Shape::vector(v.size());

    let u_node = builder.input_vector("u", scalar_type, u_shape)?;
    let v_node = builder.input_vector("v", scalar_type, v_shape)?;

    let result = builder.ewise_mult(u_node, v_node, op)?;
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
        v.indices().as_ptr() as *const (),
        v.values().as_ptr() as *const (),
    ];
    let outputs: Vec<*mut ()> = vec![
        w as *mut Vector<T> as *mut (),
    ];

    // TODO: Real execution would compute w = u .* v element-wise
    function.execute(&inputs, &outputs)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ewadd_matrix_dimensions() {
        let mut c = Matrix::<f64>::new(10, 20).unwrap();
        let a = Matrix::<f64>::new(10, 20).unwrap();
        let b = Matrix::<f64>::new(10, 20).unwrap();

        let result = ewadd_matrix(&mut c, None, &a, &b, BinaryOpKind::Add, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_ewadd_matrix_dimension_mismatch() {
        let mut c = Matrix::<f64>::new(10, 20).unwrap();
        let a = Matrix::<f64>::new(10, 20).unwrap();
        let b = Matrix::<f64>::new(15, 20).unwrap(); // Wrong dimensions

        let result = ewadd_matrix(&mut c, None, &a, &b, BinaryOpKind::Add, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_ewmult_matrix_dimensions() {
        let mut c = Matrix::<f64>::new(10, 20).unwrap();
        let a = Matrix::<f64>::new(10, 20).unwrap();
        let b = Matrix::<f64>::new(10, 20).unwrap();

        let result = ewmult_matrix(&mut c, None, &a, &b, BinaryOpKind::Mul, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_ewadd_vector_dimensions() {
        let mut w = Vector::<f64>::new(10).unwrap();
        let u = Vector::<f64>::new(10).unwrap();
        let v = Vector::<f64>::new(10).unwrap();

        let result = ewadd_vector(&mut w, None, &u, &v, BinaryOpKind::Add, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_ewadd_vector_dimension_mismatch() {
        let mut w = Vector::<f64>::new(10).unwrap();
        let u = Vector::<f64>::new(10).unwrap();
        let v = Vector::<f64>::new(15).unwrap(); // Wrong dimension

        let result = ewadd_vector(&mut w, None, &u, &v, BinaryOpKind::Add, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_ewmult_vector_dimensions() {
        let mut w = Vector::<f64>::new(10).unwrap();
        let u = Vector::<f64>::new(10).unwrap();
        let v = Vector::<f64>::new(10).unwrap();

        let result = ewmult_vector(&mut w, None, &u, &v, BinaryOpKind::Mul, None);
        assert!(result.is_ok());
    }
}
