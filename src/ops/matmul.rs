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
    _semiring: &Semiring<T>,
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

    // Build IR graph
    let mut builder = GraphBuilder::new();

    // Get scalar type
    let scalar_type = ScalarType::from_type_code(T::TYPE_CODE)
        .ok_or(GraphBlasError::InvalidValue)?;

    // Add inputs
    let left_shape = Shape::matrix(left_matrix.nrows(), left_matrix.ncols());
    let right_shape = Shape::matrix(right_matrix.nrows(), right_matrix.ncols());

    let left_node = builder.input_matrix("A", scalar_type, left_shape)?;
    let right_node = builder.input_matrix("B", scalar_type, right_shape)?;

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

    // Convert semiring to SemiringOp
    let semiring_op = SemiringOp {
        add_op: MonoidOp {
            binary_op: BinaryOpKind::Add, // TODO: Get from semiring
            identity: ScalarValue::from_type(scalar_type, 0.0),
        },
        mul_op: BinaryOpKind::Mul, // TODO: Get from semiring
    };

    // Create matmul operation
    let result = builder.matmul(left_node, right_node, semiring_op)?;

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
    let left_storage = left_matrix.storage();
    let right_storage = right_matrix.storage();
    let output_storage = output.storage_mut();

    // Prepare input/output pointers for kernel execution
    // In a full implementation, these would point to the actual sparse arrays
    let inputs: Vec<*const ()> = vec![
        left_storage as *const _ as *const (),
        right_storage as *const _ as *const (),
    ];
    let outputs: Vec<*mut ()> = vec![
        output_storage as *mut _ as *mut (),
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

    // Build IR graph
    let mut builder = GraphBuilder::new();
    let scalar_type = ScalarType::from_type_code(T::TYPE_CODE)
        .ok_or(GraphBlasError::InvalidValue)?;

    let matrix_shape = Shape::matrix(matrix.nrows(), matrix.ncols());
    let vector_shape = Shape::vector(input_vector.size());

    let matrix_node = builder.input_matrix("A", scalar_type, matrix_shape)?;
    let vector_node = builder.input_vector("u", scalar_type, vector_shape)?;

    // Apply transpose if requested
    let matrix_node = if let Some(descriptor) = desc {
        if descriptor.transpose_first {
            builder.transpose(matrix_node)?
        } else {
            matrix_node
        }
    } else {
        matrix_node
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

    let result = builder.matvec(matrix_node, vector_node, semiring_op)?;
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
    let matrix_storage = matrix.storage();

    // Prepare input/output pointers for kernel execution
    let inputs: Vec<*const ()> = vec![
        matrix_storage as *const _ as *const (),
        input_vector.indices().as_ptr() as *const (),
        input_vector.values().as_ptr() as *const (),
    ];
    let outputs: Vec<*mut ()> = vec![
        output as *mut Vector<T> as *mut (),  // Pass the whole vector for now
    ];

    // Call the compiled kernel
    // TODO: Real execution would compute output = matrix * input_vector using compiled native code
    function.execute(&inputs, &outputs)?;

    Ok(())
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

    // Build IR graph
    let mut builder = GraphBuilder::new();
    let scalar_type = ScalarType::from_type_code(T::TYPE_CODE)
        .ok_or(GraphBlasError::InvalidValue)?;

    let vector_shape = Shape::vector(input_vector.size());
    let matrix_shape = Shape::matrix(matrix.nrows(), matrix.ncols());

    let vector_node = builder.input_vector("u", scalar_type, vector_shape)?;
    let matrix_node = builder.input_matrix("A", scalar_type, matrix_shape)?;

    // Apply transpose if requested
    let matrix_node = if let Some(descriptor) = desc {
        if descriptor.transpose_second {
            builder.transpose(matrix_node)?
        } else {
            matrix_node
        }
    } else {
        matrix_node
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

    let result = builder.vecmat(vector_node, matrix_node, semiring_op)?;
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
    let matrix_storage = matrix.storage();

    // Prepare input/output pointers for kernel execution
    let inputs: Vec<*const ()> = vec![
        input_vector.indices().as_ptr() as *const (),
        input_vector.values().as_ptr() as *const (),
        matrix_storage as *const _ as *const (),
    ];
    let outputs: Vec<*mut ()> = vec![
        output as *mut Vector<T> as *mut (),  // Pass the whole vector for now
    ];

    // Call the compiled kernel
    // TODO: Real execution would compute output = input_vector * matrix using compiled native code
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
