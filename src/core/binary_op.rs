// Core Layer: BinaryOp implementation

use crate::core::error::Result;
use crate::types::GraphBLASType;

/// Binary operator: z = f(x, y)
pub struct BinaryOp<T: GraphBLASType, U: GraphBLASType, V: GraphBLASType> {
    /// Function pointer: (x, y) -> z
    op: fn(T, U) -> V,
    /// Name of operation
    name: String,
}

impl<T: GraphBLASType, U: GraphBLASType, V: GraphBLASType> BinaryOp<T, U, V> {
    /// Create a new binary operator
    pub fn new(op: fn(T, U) -> V, name: String) -> Result<Self> {
        Ok(Self { op, name })
    }

    /// Apply the operation
    pub fn apply(&self, x: T, y: U) -> V {
        (self.op)(x, y)
    }

    /// Get operator name
    pub fn name(&self) -> &str {
        &self.name
    }
}
