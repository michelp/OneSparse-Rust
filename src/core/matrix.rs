// Core Layer: Matrix implementation
//
// Idiomatic Rust implementation of sparse matrices

use crate::core::container::SparseContainer;
use crate::core::error::Result;
use crate::types::{GraphBLASType, TypeCode};

/// Sparse matrix storage (placeholder for actual implementation)
pub enum SparseStorage<T> {
    /// Compressed Sparse Row format
    CSR {
        row_ptrs: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<T>,
    },
    /// Compressed Sparse Column format
    CSC {
        col_ptrs: Vec<usize>,
        row_indices: Vec<usize>,
        values: Vec<T>,
    },
    /// Coordinate format
    COO {
        row_indices: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<T>,
    },
}

/// Sparse matrix wrapper around unified SparseContainer
///
/// Following SuiteSparse design: Matrix is a facade over the unified internal representation.
/// This maintains the ergonomic Matrix API while sharing implementation with Vector and Scalar.
pub struct Matrix<T: GraphBLASType>(SparseContainer<T>);

impl<T: GraphBLASType> Matrix<T> {
    /// Create a new empty matrix
    pub fn new(nrows: usize, ncols: usize) -> Result<Self> {
        Ok(Matrix(SparseContainer::new_matrix(nrows, ncols)?))
    }

    /// Get number of rows
    pub fn nrows(&self) -> usize {
        self.0.nrows()
    }

    /// Get number of columns
    pub fn ncols(&self) -> usize {
        self.0.ncols()
    }

    /// Get shape as (nrows, ncols)
    pub fn shape(&self) -> (usize, usize) {
        self.0.shape()
    }

    /// Get number of stored values
    pub fn nvals(&self) -> usize {
        self.0.nvals()
    }

    /// Get type code
    pub fn type_code(&self) -> TypeCode {
        self.0.type_code()
    }

    /// Get reference to storage
    pub fn storage(&self) -> &SparseStorage<T> {
        self.0.storage()
    }

    /// Get mutable reference to storage
    pub fn storage_mut(&mut self) -> &mut SparseStorage<T> {
        self.0.storage_mut()
    }

    /// Create a matrix with CSR storage
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
        Ok(Matrix(SparseContainer::from_csr(
            nrows,
            ncols,
            row_ptrs,
            col_indices,
            values,
        )?))
    }

    /// Create a matrix with CSC storage
    pub fn from_csc(
        nrows: usize,
        ncols: usize,
        col_ptrs: Vec<usize>,
        row_indices: Vec<usize>,
        values: Vec<T>,
    ) -> Result<Self> {
        Ok(Matrix(SparseContainer::from_csc(
            nrows,
            ncols,
            col_ptrs,
            row_indices,
            values,
        )?))
    }

    /// Create a matrix with COO storage
    pub fn from_coo(
        nrows: usize,
        ncols: usize,
        row_indices: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<T>,
    ) -> Result<Self> {
        Ok(Matrix(SparseContainer::from_coo(
            nrows,
            ncols,
            row_indices,
            col_indices,
            values,
        )?))
    }

    /// Get reference to the inner container (for internal use)
    pub(crate) fn inner(&self) -> &SparseContainer<T> {
        &self.0
    }

    /// Get mutable reference to the inner container (for internal use)
    pub(crate) fn inner_mut(&mut self) -> &mut SparseContainer<T> {
        &mut self.0
    }
}
