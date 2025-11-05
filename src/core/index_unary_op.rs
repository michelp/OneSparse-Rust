// Core Layer: IndexUnaryOp implementation

use crate::core::error::Result;
use crate::types::GraphBLASType;

/// Index-aware unary operator: z = f(x, i, j, thunk)
///
/// This operator is aware of the position (i, j) of the element
/// and can use an additional thunk parameter.
pub struct IndexUnaryOp<T: GraphBLASType, U: GraphBLASType, Thunk: Copy> {
    /// Function pointer: (x, i, j, thunk) -> z
    op: fn(T, u64, u64, Thunk) -> U,
    /// Name of operation
    name: String,
}

impl<T: GraphBLASType, U: GraphBLASType, Thunk: Copy> IndexUnaryOp<T, U, Thunk> {
    /// Create a new index-aware unary operator
    pub fn new(op: fn(T, u64, u64, Thunk) -> U, name: String) -> Result<Self> {
        Ok(Self { op, name })
    }

    /// Apply the operation
    pub fn apply(&self, x: T, i: u64, j: u64, thunk: Thunk) -> U {
        (self.op)(x, i, j, thunk)
    }

    /// Get operator name
    pub fn name(&self) -> &str {
        &self.name
    }
}
