// Core Layer: IndexBinaryOp implementation

use crate::core::error::Result;
use crate::types::GraphBLASType;

/// Index-aware binary operator: z = f(x, y, i, j, thunk)
///
/// This operator is aware of the position (i, j) of the elements
/// and can use an additional thunk parameter.
pub struct IndexBinaryOp<T: GraphBLASType, U: GraphBLASType, V: GraphBLASType, Thunk: Copy> {
    /// Function pointer: (x, y, i, j, thunk) -> z
    op: fn(T, U, u64, u64, Thunk) -> V,
    /// Name of operation
    name: String,
}

impl<T: GraphBLASType, U: GraphBLASType, V: GraphBLASType, Thunk: Copy>
    IndexBinaryOp<T, U, V, Thunk>
{
    /// Create a new index-aware binary operator
    pub fn new(op: fn(T, U, u64, u64, Thunk) -> V, name: String) -> Result<Self> {
        Ok(Self { op, name })
    }

    /// Apply the operation
    pub fn apply(&self, x: T, y: U, i: u64, j: u64, thunk: Thunk) -> V {
        (self.op)(x, y, i, j, thunk)
    }

    /// Get operator name
    pub fn name(&self) -> &str {
        &self.name
    }
}
