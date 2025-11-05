// Core Layer: Vector implementation

use crate::core::error::Result;
use crate::types::{GraphBLASType, TypeCode};

/// Sparse vector
pub struct Vector<T: GraphBLASType> {
    /// Size of vector
    size: usize,
    /// Element type code
    type_code: TypeCode,
    /// Indices of stored values
    indices: Vec<usize>,
    /// Stored values
    values: Vec<T>,
}

impl<T: GraphBLASType> Vector<T> {
    /// Create a new empty vector
    pub fn new(size: usize) -> Result<Self> {
        Ok(Self {
            size,
            type_code: T::TYPE_CODE,
            indices: Vec::new(),
            values: Vec::new(),
        })
    }

    /// Get size
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get number of stored values
    pub fn nvals(&self) -> usize {
        self.values.len()
    }

    /// Get type code
    pub fn type_code(&self) -> TypeCode {
        self.type_code
    }

    /// Get indices slice
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// Get values slice
    pub fn values(&self) -> &[T] {
        &self.values
    }

    /// Get mutable indices
    pub fn indices_mut(&mut self) -> &mut Vec<usize> {
        &mut self.indices
    }

    /// Get mutable values
    pub fn values_mut(&mut self) -> &mut Vec<T> {
        &mut self.values
    }
}
