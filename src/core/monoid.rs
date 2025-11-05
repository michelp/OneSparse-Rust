// Core Layer: Monoid implementation

use crate::core::error::Result;
use crate::types::GraphBLASType;

/// Monoid: Associative binary operation with identity element
///
/// A monoid consists of:
/// - Binary operation that is associative: (a ⊕ b) ⊕ c = a ⊕ (b ⊕ c)
/// - Identity element: a ⊕ identity = identity ⊕ a = a
pub struct Monoid<T: GraphBLASType> {
    /// Binary operation function pointer
    op: fn(T, T) -> T,
    /// Identity element
    identity: T,
    /// Name of monoid
    name: String,
}

impl<T: GraphBLASType> Monoid<T> {
    /// Create a new monoid
    pub fn new(op: fn(T, T) -> T, identity: T, name: String) -> Result<Self> {
        Ok(Self { op, identity, name })
    }

    /// Apply the monoid operation
    pub fn apply(&self, a: T, b: T) -> T {
        (self.op)(a, b)
    }

    /// Get identity element
    pub fn identity(&self) -> T {
        self.identity
    }

    /// Get monoid name
    pub fn name(&self) -> &str {
        &self.name
    }
}
