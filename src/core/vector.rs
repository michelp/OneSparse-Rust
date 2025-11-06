// Core Layer: Vector implementation

use crate::core::container::SparseContainer;
use crate::core::error::Result;
use crate::types::{GraphBLASType, TypeCode};

/// Sparse vector wrapper around unified SparseContainer
///
/// Following SuiteSparse design: Vector is internally represented as an n×1 matrix.
/// This maintains the ergonomic Vector API while sharing implementation with Matrix and Scalar.
pub struct Vector<T: GraphBLASType>(SparseContainer<T>);

impl<T: GraphBLASType> Vector<T> {
    /// Create a new empty vector
    pub fn new(size: usize) -> Result<Self> {
        Ok(Vector(SparseContainer::new_vector(size)?))
    }

    /// Get size (number of elements)
    pub fn size(&self) -> usize {
        self.0.nrows()
    }

    /// Get number of stored values
    pub fn nvals(&self) -> usize {
        self.0.nvals()
    }

    /// Get type code
    pub fn type_code(&self) -> TypeCode {
        self.0.type_code()
    }

    /// Get indices slice
    ///
    /// Returns the indices of stored values in the vector.
    /// Internally, vectors are n×1 matrices, so this extracts row indices.
    pub fn indices(&self) -> &[usize] {
        if let Some((indices, _)) = self.0.vector_data() {
            indices
        } else {
            &[] // Should never happen for a vector
        }
    }

    /// Get values slice
    pub fn values(&self) -> &[T] {
        if let Some((_, values)) = self.0.vector_data() {
            values
        } else {
            &[] // Should never happen for a vector
        }
    }

    /// Get mutable indices and values
    ///
    /// Returns mutable references to both indices and values vectors.
    pub fn indices_and_values_mut(&mut self) -> Option<(&mut Vec<usize>, &mut Vec<T>)> {
        self.0.vector_data_mut()
    }

    /// Get mutable indices
    pub fn indices_mut(&mut self) -> &mut Vec<usize> {
        self.0
            .vector_data_mut()
            .map(|(indices, _)| indices)
            .expect("Vector should always support vector_data_mut")
    }

    /// Get mutable values
    pub fn values_mut(&mut self) -> &mut Vec<T> {
        self.0
            .vector_data_mut()
            .map(|(_, values)| values)
            .expect("Vector should always support vector_data_mut")
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
