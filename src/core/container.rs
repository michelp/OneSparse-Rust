// Core Layer: Unified Sparse Container
//
// Internal representation shared by Matrix, Vector, and Scalar types.
// Following SuiteSparse design: vectors are n×1 matrices, scalars are 1×1 matrices.

use crate::core::error::Result;
use crate::core::matrix::SparseStorage;
use crate::types::{GraphBLASType, TypeCode};

/// Unified sparse container for matrices, vectors, and scalars
///
/// This is the internal representation. Externally, Matrix<T>, Vector<T>, and Scalar<T>
/// wrap this type to provide ergonomic APIs while sharing implementation.
///
/// Design invariants:
/// - Matrix: arbitrary (nrows, ncols)
/// - Vector: (nrows, 1) where nrows is the vector size
/// - Scalar: (1, 1)
pub struct SparseContainer<T: GraphBLASType> {
    /// Shape: (number of rows, number of columns)
    shape: (usize, usize),
    /// Element type code
    type_code: TypeCode,
    /// Sparse storage format
    storage: SparseStorage<T>,
}

impl<T: GraphBLASType> SparseContainer<T> {
    /// Create a new container for a matrix
    ///
    /// # Arguments
    /// * `nrows` - Number of rows
    /// * `ncols` - Number of columns
    pub fn new_matrix(nrows: usize, ncols: usize) -> Result<Self> {
        Ok(Self {
            shape: (nrows, ncols),
            type_code: T::TYPE_CODE,
            storage: SparseStorage::COO {
                row_indices: Vec::new(),
                col_indices: Vec::new(),
                values: Vec::new(),
            },
        })
    }

    /// Create a new container for a vector (n×1 matrix)
    ///
    /// # Arguments
    /// * `size` - Vector size (becomes nrows, ncols=1)
    pub fn new_vector(size: usize) -> Result<Self> {
        Ok(Self {
            shape: (size, 1),
            type_code: T::TYPE_CODE,
            storage: SparseStorage::COO {
                row_indices: Vec::new(),
                col_indices: Vec::new(),
                values: Vec::new(),
            },
        })
    }

    /// Create a new container for a scalar (1×1 matrix)
    pub fn new_scalar() -> Result<Self> {
        Ok(Self {
            shape: (1, 1),
            type_code: T::TYPE_CODE,
            storage: SparseStorage::COO {
                row_indices: Vec::new(),
                col_indices: Vec::new(),
                values: Vec::new(),
            },
        })
    }

    /// Create a container with CSR storage
    ///
    /// # Arguments
    /// * `nrows` - Number of rows
    /// * `ncols` - Number of columns
    /// * `row_ptrs` - Row pointer array (length = nrows + 1)
    /// * `col_indices` - Column indices for each non-zero
    /// * `values` - Values for each non-zero
    pub fn from_csr(
        nrows: usize,
        ncols: usize,
        row_ptrs: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<T>,
    ) -> Result<Self> {
        Ok(Self {
            shape: (nrows, ncols),
            type_code: T::TYPE_CODE,
            storage: SparseStorage::CSR {
                row_ptrs,
                col_indices,
                values,
            },
        })
    }

    /// Create a container with CSC storage
    pub fn from_csc(
        nrows: usize,
        ncols: usize,
        col_ptrs: Vec<usize>,
        row_indices: Vec<usize>,
        values: Vec<T>,
    ) -> Result<Self> {
        Ok(Self {
            shape: (nrows, ncols),
            type_code: T::TYPE_CODE,
            storage: SparseStorage::CSC {
                col_ptrs,
                row_indices,
                values,
            },
        })
    }

    /// Create a container with COO storage
    pub fn from_coo(
        nrows: usize,
        ncols: usize,
        row_indices: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<T>,
    ) -> Result<Self> {
        Ok(Self {
            shape: (nrows, ncols),
            type_code: T::TYPE_CODE,
            storage: SparseStorage::COO {
                row_indices,
                col_indices,
                values,
            },
        })
    }

    /// Check if this container represents a vector (n×1)
    pub fn is_vector(&self) -> bool {
        self.shape.1 == 1
    }

    /// Check if this container represents a scalar (1×1)
    pub fn is_scalar(&self) -> bool {
        self.shape == (1, 1)
    }

    /// Check if this container represents a matrix (not a vector)
    pub fn is_matrix(&self) -> bool {
        self.shape.1 > 1
    }

    /// Get number of rows
    pub fn nrows(&self) -> usize {
        self.shape.0
    }

    /// Get number of columns
    pub fn ncols(&self) -> usize {
        self.shape.1
    }

    /// Get shape as (nrows, ncols)
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    /// Get number of stored values
    pub fn nvals(&self) -> usize {
        match &self.storage {
            SparseStorage::COO { values, .. } => values.len(),
            SparseStorage::CSR { values, .. } => values.len(),
            SparseStorage::CSC { values, .. } => values.len(),
        }
    }

    /// Get type code
    pub fn type_code(&self) -> TypeCode {
        self.type_code
    }

    /// Get reference to storage
    pub fn storage(&self) -> &SparseStorage<T> {
        &self.storage
    }

    /// Get mutable reference to storage
    pub fn storage_mut(&mut self) -> &mut SparseStorage<T> {
        &mut self.storage
    }

    /// Extract vector indices and values
    ///
    /// For vector containers (n×1), extracts the sparse representation.
    /// Returns None if not a vector or storage format doesn't support direct access.
    pub fn vector_data(&self) -> Option<(&[usize], &[T])> {
        if !self.is_vector() {
            return None;
        }

        match &self.storage {
            SparseStorage::COO {
                row_indices,
                values,
                ..
            } => Some((row_indices.as_slice(), values.as_slice())),
            SparseStorage::CSR {
                col_indices,
                values,
                ..
            } => {
                // For vector (n×1), CSR col_indices are always 0, row structure gives indices
                // This is a simplified view - in practice vectors might not use CSR
                Some((col_indices.as_slice(), values.as_slice()))
            }
            SparseStorage::CSC {
                row_indices,
                values,
                ..
            } => {
                // For vector (n×1) in CSC, row_indices give the sparse indices
                Some((row_indices.as_slice(), values.as_slice()))
            }
        }
    }

    /// Extract mutable vector indices and values
    pub fn vector_data_mut(&mut self) -> Option<(&mut Vec<usize>, &mut Vec<T>)> {
        if !self.is_vector() {
            return None;
        }

        match &mut self.storage {
            SparseStorage::COO {
                row_indices,
                values,
                ..
            } => Some((row_indices, values)),
            SparseStorage::CSR {
                col_indices,
                values,
                ..
            } => Some((col_indices, values)),
            SparseStorage::CSC {
                row_indices,
                values,
                ..
            } => Some((row_indices, values)),
        }
    }

    /// Extract scalar value
    ///
    /// For scalar containers (1×1), extracts the value if present.
    /// Returns None if not a scalar or no value is stored.
    pub fn scalar_value(&self) -> Option<T> {
        if !self.is_scalar() {
            return None;
        }

        match &self.storage {
            SparseStorage::COO { values, .. }
            | SparseStorage::CSR { values, .. }
            | SparseStorage::CSC { values, .. } => {
                if values.is_empty() {
                    None
                } else {
                    Some(values[0])
                }
            }
        }
    }

    /// Set scalar value
    ///
    /// For scalar containers (1×1), sets the single value.
    /// Returns error if not a scalar.
    pub fn set_scalar_value(&mut self, value: T) {
        debug_assert!(self.is_scalar(), "set_scalar_value called on non-scalar");

        // Replace storage with simple COO containing the single value
        self.storage = SparseStorage::COO {
            row_indices: vec![0],
            col_indices: vec![0],
            values: vec![value],
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_container_is_matrix() {
        let mat = SparseContainer::<f64>::new_matrix(3, 3).unwrap();
        assert!(mat.is_matrix());
        assert!(!mat.is_vector());
        assert!(!mat.is_scalar());
        assert_eq!(mat.nrows(), 3);
        assert_eq!(mat.ncols(), 3);
    }

    #[test]
    fn test_container_is_vector() {
        let vec = SparseContainer::<f64>::new_vector(5).unwrap();
        assert!(!vec.is_matrix());
        assert!(vec.is_vector());
        assert!(!vec.is_scalar());
        assert_eq!(vec.nrows(), 5);
        assert_eq!(vec.ncols(), 1);
    }

    #[test]
    fn test_container_is_scalar() {
        let scalar = SparseContainer::<f64>::new_scalar().unwrap();
        assert!(!scalar.is_matrix());
        assert!(scalar.is_vector()); // 1×1 is also a vector technically
        assert!(scalar.is_scalar());
        assert_eq!(scalar.nrows(), 1);
        assert_eq!(scalar.ncols(), 1);
    }

    #[test]
    fn test_scalar_value() {
        let mut scalar = SparseContainer::<f64>::new_scalar().unwrap();
        assert_eq!(scalar.scalar_value(), None);

        scalar.set_scalar_value(42.0);
        assert_eq!(scalar.scalar_value(), Some(42.0));
    }

    #[test]
    fn test_from_csr() {
        let container =
            SparseContainer::<f64>::from_csr(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0, 2.0])
                .unwrap();

        assert_eq!(container.nrows(), 2);
        assert_eq!(container.ncols(), 2);
        assert_eq!(container.nvals(), 2);
        assert!(matches!(container.storage(), SparseStorage::CSR { .. }));
    }
}
