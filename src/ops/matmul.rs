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
use crate::core::container::SparseContainer;
use crate::core::error::{GraphBlasError, Result};
use crate::core::matrix::Matrix;
use crate::core::semiring::Semiring;
use crate::core::vector::Vector;
use crate::ir::builder::GraphBuilder;
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

/// Unified internal matmul compilation and execution
///
/// Following SuiteSparse design: vectors are n×1 matrices internally,
/// so all operations (mxm, mxv, vxm) use the same compilation path.
/// Shape metadata distinguishes the operation types.
///
/// This eliminates ~95% code duplication between mxm/mxv/vxm.
fn compile_and_execute_matmul<T: GraphBLASType>(
    output: &mut SparseContainer<T>,
    left: &SparseContainer<T>,
    right: &SparseContainer<T>,
    semiring: &Semiring<T>,
    desc: Option<&Descriptor>,
) -> Result<()> {
    // Build IR graph
    log::info!("Building IR graph for matrix multiplication");
    let mut builder = GraphBuilder::new();

    // Get scalar type
    let scalar_type =
        ScalarType::from_type_code(T::TYPE_CODE).ok_or(GraphBlasError::InvalidValue)?;
    log::debug!("Scalar type: {:?}", scalar_type);

    // Add inputs - use appropriate shape types
    // Vectors are n×1 internally, but we mark them as vectors in the IR
    let left_node = if left.is_vector() {
        let left_shape = Shape::vector(left.nrows());
        log::debug!("Left is vector, shape: {:?}", left_shape);
        builder.input_vector("u", scalar_type, left_shape)?
    } else {
        let left_shape = Shape::matrix(left.nrows(), left.ncols());
        log::debug!("Left is matrix, shape: {:?}", left_shape);
        builder.input_matrix("A", scalar_type, left_shape)?
    };

    let right_node = if right.is_vector() {
        let right_shape = Shape::vector(right.nrows());
        log::debug!("Right is vector, shape: {:?}", right_shape);
        builder.input_vector("v", scalar_type, right_shape)?
    } else {
        let right_shape = Shape::matrix(right.nrows(), right.ncols());
        log::debug!("Right is matrix, shape: {:?}", right_shape);
        builder.input_matrix("B", scalar_type, right_shape)?
    };

    log::trace!(
        "Created input nodes: left={}, right={}",
        left_node,
        right_node
    );

    // Apply transpose if requested
    let (left_node, right_node) = if let Some(descriptor) = desc {
        let left_final = if descriptor.transpose_first {
            builder.transpose(left_node)?
        } else {
            left_node
        };

        let right_final = if descriptor.transpose_second {
            builder.transpose(right_node)?
        } else {
            right_node
        };

        (left_final, right_final)
    } else {
        (left_node, right_node)
    };

    // Convert semiring from core layer to IR layer
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

    // Create appropriate operation based on shapes
    // Use the correct IR operation for better optimization and execution
    let result = if left.is_vector() && !right.is_vector() {
        // Vector × Matrix (after transpose, it becomes 1×m × m×n = 1×n)
        log::debug!(
            "Creating vecmat operation with semiring: add={:?}, mul={:?}",
            semiring_op.add_op.binary_op,
            semiring_op.mul_op
        );
        builder.vecmat(left_node, right_node, semiring_op)?
    } else if !left.is_vector() && right.is_vector() {
        // Matrix × Vector (m×n × n×1 = m×1)
        log::debug!(
            "Creating matvec operation with semiring: add={:?}, mul={:?}",
            semiring_op.add_op.binary_op,
            semiring_op.mul_op
        );
        builder.matvec(left_node, right_node, semiring_op)?
    } else {
        // Matrix × Matrix (m×n × n×p = m×p)
        log::debug!(
            "Creating matmul operation with semiring: add={:?}, mul={:?}",
            semiring_op.add_op.binary_op,
            semiring_op.mul_op
        );
        builder.matmul(left_node, right_node, semiring_op)?
    };
    log::trace!("Operation result node: {}", result);

    // Mark as output
    builder.output(result)?;

    // Build graph
    log::debug!("Building IR graph");
    let mut graph = builder.build();
    log::trace!("Graph has {} nodes", graph.nodes().len());

    // Try to get cached kernel
    let cache_key = CacheKey::from_graph(&graph);
    let function = if let Some(cached_fn) = KERNEL_CACHE.get(&cache_key) {
        log::info!("Using cached kernel");
        cached_fn
    } else {
        log::info!("Compiling new kernel");

        // Apply optimization passes
        log::info!("Running optimization passes");
        let mut pass_manager = PassManager::new();
        pass_manager.add_pass(Box::new(FusionPass::new()));
        pass_manager.add_pass(Box::new(CSEPass::new()));
        pass_manager.add_pass(Box::new(FormatSelectionPass::new()));

        pass_manager.run_all(&mut graph)?;

        // Compile the graph
        log::info!("JIT compiling with Cranelift backend");
        let backend = CraneliftBackend::new()?;
        let compiled = backend.compile(&graph)?;

        // Cache the result (estimate 10KB per kernel)
        KERNEL_CACHE.insert(cache_key, compiled.clone(), 10240);

        Arc::new(compiled)
    };

    // Execute the compiled kernel
    // Handle vectors specially for backwards compatibility with execution layer
    let inputs: Vec<*const ()> = if left.is_vector() && right.is_vector() {
        // Both vectors (rare case, but handle it)
        let left_data = left.vector_data().expect("Vector should have vector_data");
        let right_data = right.vector_data().expect("Vector should have vector_data");
        vec![
            left_data.0.as_ptr() as *const (),
            left_data.1.as_ptr() as *const (),
            right_data.0.as_ptr() as *const (),
            right_data.1.as_ptr() as *const (),
        ]
    } else if left.is_vector() {
        // Left is vector (vxm case - but after transpose in descriptor)
        let vec_data = left.vector_data().expect("Vector should have vector_data");
        vec![
            vec_data.0.as_ptr() as *const (),
            vec_data.1.as_ptr() as *const (),
            right.storage() as *const _ as *const (),
        ]
    } else if right.is_vector() {
        // Right is vector (mxv case)
        let vec_data = right.vector_data().expect("Vector should have vector_data");
        vec![
            left.storage() as *const _ as *const (),
            vec_data.0.as_ptr() as *const (),
            vec_data.1.as_ptr() as *const (),
        ]
    } else {
        // Both matrices (mxm case)
        vec![
            left.storage() as *const _ as *const (),
            right.storage() as *const _ as *const (),
        ]
    };

    let output_storage = output.storage_mut();
    let outputs: Vec<*mut ()> = vec![output_storage as *mut _ as *mut ()];
    log::trace!(
        "Prepared {} input(s) and {} output(s)",
        inputs.len(),
        outputs.len()
    );

    // Call the compiled kernel
    function.execute(&inputs, &outputs)?;

    Ok(())
}

/// Matrix-matrix multiplication: C = A * B
///
/// # Arguments
/// * `output` - Output matrix (will be modified)
/// * `mask` - Optional mask matrix
/// * `left_matrix` - First input matrix
/// * `right_matrix` - Second input matrix
/// * `semiring` - Semiring for multiplication
/// * `desc` - Optional descriptor
pub fn mxm<T: GraphBLASType>(
    output: &mut Matrix<T>,
    mask: Option<&Matrix<bool>>,
    left_matrix: &Matrix<T>,
    right_matrix: &Matrix<T>,
    semiring: &Semiring<T>,
    desc: Option<&Descriptor>,
) -> Result<()> {
    // Get effective dimensions (considering transpose)
    let (left_rows, left_cols) = if let Some(descriptor) = desc {
        if descriptor.transpose_first {
            (left_matrix.ncols(), left_matrix.nrows())
        } else {
            (left_matrix.nrows(), left_matrix.ncols())
        }
    } else {
        (left_matrix.nrows(), left_matrix.ncols())
    };

    let (right_rows, right_cols) = if let Some(descriptor) = desc {
        if descriptor.transpose_second {
            (right_matrix.ncols(), right_matrix.nrows())
        } else {
            (right_matrix.nrows(), right_matrix.ncols())
        }
    } else {
        (right_matrix.nrows(), right_matrix.ncols())
    };

    // Validate dimensions
    if left_cols != right_rows {
        return Err(GraphBlasError::DimensionMismatch);
    }

    if output.nrows() != left_rows || output.ncols() != right_cols {
        return Err(GraphBlasError::DimensionMismatch);
    }

    // Check if mask dimensions match if present
    if let Some(mask_matrix) = mask {
        if mask_matrix.nrows() != output.nrows() || mask_matrix.ncols() != output.ncols() {
            return Err(GraphBlasError::DimensionMismatch);
        }
    }

    // Delegate to unified implementation
    compile_and_execute_matmul(
        output.inner_mut(),
        left_matrix.inner(),
        right_matrix.inner(),
        semiring,
        desc,
    )
}

/// Matrix-vector multiplication: w = A * u
///
/// # Arguments
/// * `output` - Output vector (will be modified)
/// * `mask` - Optional mask vector
/// * `matrix` - Input matrix
/// * `input_vector` - Input vector
/// * `semiring` - Semiring for multiplication
/// * `desc` - Optional descriptor
pub fn mxv<T: GraphBLASType>(
    output: &mut Vector<T>,
    mask: Option<&Vector<bool>>,
    matrix: &Matrix<T>,
    input_vector: &Vector<T>,
    semiring: &Semiring<T>,
    desc: Option<&Descriptor>,
) -> Result<()> {
    // Validate dimensions
    if matrix.ncols() != input_vector.size() {
        return Err(GraphBlasError::DimensionMismatch);
    }

    if output.size() != matrix.nrows() {
        return Err(GraphBlasError::DimensionMismatch);
    }

    // Check mask dimensions
    if let Some(mask_vector) = mask {
        if mask_vector.size() != output.size() {
            return Err(GraphBlasError::DimensionMismatch);
        }
    }

    // Delegate to unified implementation
    // Vector is internally n×1, so this becomes matrix × n×1 = m×1
    compile_and_execute_matmul(
        output.inner_mut(),
        matrix.inner(),
        input_vector.inner(),
        semiring,
        desc,
    )
}

/// Vector-matrix multiplication: output = input_vector * matrix
///
/// # Arguments
/// * `output` - Output vector (will be modified)
/// * `mask` - Optional mask vector
/// * `input_vector` - Input vector
/// * `matrix` - Input matrix
/// * `semiring` - Semiring for multiplication
/// * `desc` - Optional descriptor
pub fn vxm<T: GraphBLASType>(
    output: &mut Vector<T>,
    mask: Option<&Vector<bool>>,
    input_vector: &Vector<T>,
    matrix: &Matrix<T>,
    semiring: &Semiring<T>,
    desc: Option<&Descriptor>,
) -> Result<()> {
    // Validate dimensions
    if input_vector.size() != matrix.nrows() {
        return Err(GraphBlasError::DimensionMismatch);
    }

    if output.size() != matrix.ncols() {
        return Err(GraphBlasError::DimensionMismatch);
    }

    // Check mask dimensions
    if let Some(mask_vector) = mask {
        if mask_vector.size() != output.size() {
            return Err(GraphBlasError::DimensionMismatch);
        }
    }

    // Delegate to unified implementation
    // Vector is internally m×1, but builder.vecmat() treats it as a row vector (1×m) semantically
    // No need to force transpose - the IR layer handles this correctly
    compile_and_execute_matmul(
        output.inner_mut(),
        input_vector.inner(),
        matrix.inner(),
        semiring,
        desc,
    )
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
        if let Err(e) = &result {
            eprintln!("vxm error: {:?}", e);
        }
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
