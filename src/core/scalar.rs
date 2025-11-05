// Core Layer: Scalar implementation

use crate::core::error::Result;
use crate::types::{GraphBLASType, TypeCode};

/// Scalar (1x1 matrix)
pub struct Scalar<T: GraphBLASType> {
    /// Element type code
    type_code: TypeCode,
    /// Optional value (None if not set)
    value: Option<T>,
}

impl<T: GraphBLASType> Scalar<T> {
    /// Create a new empty scalar
    pub fn new() -> Result<Self> {
        Ok(Self {
            type_code: T::TYPE_CODE,
            value: None,
        })
    }

    /// Set the scalar value
    pub fn set(&mut self, value: T) {
        self.value = Some(value);
    }

    /// Get the scalar value
    pub fn get(&self) -> Option<T> {
        self.value
    }

    /// Check if scalar has a value
    pub fn is_empty(&self) -> bool {
        self.value.is_none()
    }

    /// Get type code
    pub fn type_code(&self) -> TypeCode {
        self.type_code
    }
}
