// Core Layer: Scalar implementation

use crate::core::container::SparseContainer;
use crate::core::error::Result;
use crate::types::{GraphBLASType, TypeCode};

/// Scalar wrapper around unified SparseContainer
///
/// Following SuiteSparse design: Scalar is internally represented as a 1×1 matrix.
/// This maintains the ergonomic Scalar API while sharing implementation with Matrix and Vector.
pub struct Scalar<T: GraphBLASType>(SparseContainer<T>);

impl<T: GraphBLASType> Scalar<T> {
    /// Create a new empty scalar
    pub fn new() -> Result<Self> {
        Ok(Scalar(SparseContainer::new_scalar()?))
    }

    /// Create a scalar with a value
    pub fn from_value(value: T) -> Result<Self> {
        let mut container = SparseContainer::new_scalar()?;
        container.set_scalar_value(value);
        Ok(Scalar(container))
    }

    /// Set the scalar value
    pub fn set(&mut self, value: T) {
        self.0.set_scalar_value(value);
    }

    /// Get the scalar value
    pub fn get(&self) -> Option<T> {
        self.0.scalar_value()
    }

    /// Check if scalar has a value
    pub fn is_empty(&self) -> bool {
        self.0.scalar_value().is_none()
    }

    /// Get type code
    pub fn type_code(&self) -> TypeCode {
        self.0.type_code()
    }

    /// Get number of stored values (0 or 1)
    pub fn nvals(&self) -> usize {
        self.0.nvals()
    }
}
