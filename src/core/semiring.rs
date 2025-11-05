// Core Layer: Semiring implementation

use crate::core::error::Result;
use crate::types::GraphBLASType;

/// Semiring: Defines addition monoid and multiplication operator
///
/// A semiring consists of:
/// - Addition monoid (associative operation + identity element)
/// - Multiplication binary operation
/// - Distributive property: a * (b + c) = (a * b) + (a * c)
pub struct Semiring<T: GraphBLASType> {
    /// Addition operator function pointer
    add_op: fn(T, T) -> T,
    /// Multiplication operator function pointer
    mul_op: fn(T, T) -> T,
    /// Identity element for addition
    zero: T,
    /// Name of semiring
    name: String,
}

impl<T: GraphBLASType> Semiring<T> {
    /// Create a new semiring
    pub fn new(
        add_op: fn(T, T) -> T,
        mul_op: fn(T, T) -> T,
        zero: T,
        name: String,
    ) -> Result<Self> {
        Ok(Self {
            add_op,
            mul_op,
            zero,
            name,
        })
    }

    /// Apply addition operation
    pub fn add(&self, left: T, right: T) -> T {
        (self.add_op)(left, right)
    }

    /// Apply multiplication operation
    pub fn multiply(&self, left: T, right: T) -> T {
        (self.mul_op)(left, right)
    }

    /// Get zero element
    pub fn zero(&self) -> T {
        self.zero
    }

    /// Get semiring name
    pub fn name(&self) -> &str {
        &self.name
    }
}

// Common semirings for f64
impl Semiring<f64> {
    /// Plus-times semiring (standard arithmetic): (a + b, a * b, 0)
    pub fn plus_times() -> Result<Self> {
        Self::new(
            |a, b| a + b,
            |a, b| a * b,
            0.0,
            "plus_times".to_string(),
        )
    }

    /// Min-plus semiring (tropical): (min, +, ∞)
    pub fn min_plus() -> Result<Self> {
        Self::new(
            |a, b| a.min(b),
            |a, b| a + b,
            f64::INFINITY,
            "min_plus".to_string(),
        )
    }

    /// Max-times semiring: (max, *, -∞)
    pub fn max_times() -> Result<Self> {
        Self::new(
            |a, b| a.max(b),
            |a, b| a * b,
            f64::NEG_INFINITY,
            "max_times".to_string(),
        )
    }
}
