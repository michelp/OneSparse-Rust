// Core Layer: UnaryOp implementation

use crate::core::error::Result;
use crate::types::GraphBLASType;

/// Unary operator: z = f(x)
pub struct UnaryOp<T: GraphBLASType, U: GraphBLASType> {
    /// Function pointer: x -> z
    op: fn(T) -> U,
    /// Name of operation
    name: String,
}

impl<T: GraphBLASType, U: GraphBLASType> UnaryOp<T, U> {
    /// Create a new unary operator
    pub fn new(op: fn(T) -> U, name: String) -> Result<Self> {
        Ok(Self { op, name })
    }

    /// Apply the operation
    pub fn apply(&self, x: T) -> U {
        (self.op)(x)
    }

    /// Get operator name
    pub fn name(&self) -> &str {
        &self.name
    }
}
