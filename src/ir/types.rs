// IR Type System
//
// This is separate from the runtime type system (types::TypeCode).
// IR types are used during compilation and optimization.
//
// Following SuiteSparse design: unified Tensor type with Shape metadata.
// Scalars are 0-d tensors, vectors are 1-d tensors, matrices are 2-d tensors.

use crate::ir::shape::Shape;
use crate::types::TypeCode;
use std::fmt;

/// IR value type - unified tensor representation
///
/// Following SuiteSparse design principle: all sparse objects are tensors
/// distinguished by their shape. This unifies the IR representation while
/// maintaining type safety through shape metadata.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IRType {
    /// Unified tensor type with element type and shape
    ///
    /// - Scalar: Tensor(ScalarType, Shape::Scalar)
    /// - Vector: Tensor(ScalarType, Shape::Vector(_))
    /// - Matrix: Tensor(ScalarType, Shape::Matrix(_, _))
    Tensor(ScalarType, Shape),
}

/// Scalar element types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalarType {
    Bool,
    Int8,
    Int16,
    Int32,
    Int64,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    Float32,
    Float64,
}

impl ScalarType {
    /// Convert from runtime TypeCode
    pub fn from_type_code(code: TypeCode) -> Option<Self> {
        match code {
            TypeCode::Bool => Some(ScalarType::Bool),
            TypeCode::Int8 => Some(ScalarType::Int8),
            TypeCode::Int16 => Some(ScalarType::Int16),
            TypeCode::Int32 => Some(ScalarType::Int32),
            TypeCode::Int64 => Some(ScalarType::Int64),
            TypeCode::Uint8 => Some(ScalarType::Uint8),
            TypeCode::Uint16 => Some(ScalarType::Uint16),
            TypeCode::Uint32 => Some(ScalarType::Uint32),
            TypeCode::Uint64 => Some(ScalarType::Uint64),
            TypeCode::Fp32 => Some(ScalarType::Float32),
            TypeCode::Fp64 => Some(ScalarType::Float64),
            TypeCode::UserDefined(_) => None, // Not supported in IR yet
        }
    }

    /// Get size in bytes
    pub fn size_bytes(&self) -> usize {
        match self {
            ScalarType::Bool => 1,
            ScalarType::Int8 => 1,
            ScalarType::Int16 => 2,
            ScalarType::Int32 => 4,
            ScalarType::Int64 => 8,
            ScalarType::Uint8 => 1,
            ScalarType::Uint16 => 2,
            ScalarType::Uint32 => 4,
            ScalarType::Uint64 => 8,
            ScalarType::Float32 => 4,
            ScalarType::Float64 => 8,
        }
    }

    /// Check if this is a floating-point type
    pub fn is_float(&self) -> bool {
        matches!(self, ScalarType::Float32 | ScalarType::Float64)
    }

    /// Check if this is an integer type
    pub fn is_integer(&self) -> bool {
        !self.is_float() && !matches!(self, ScalarType::Bool)
    }
}

impl fmt::Display for ScalarType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ScalarType::Bool => write!(f, "bool"),
            ScalarType::Int8 => write!(f, "i8"),
            ScalarType::Int16 => write!(f, "i16"),
            ScalarType::Int32 => write!(f, "i32"),
            ScalarType::Int64 => write!(f, "i64"),
            ScalarType::Uint8 => write!(f, "u8"),
            ScalarType::Uint16 => write!(f, "u16"),
            ScalarType::Uint32 => write!(f, "u32"),
            ScalarType::Uint64 => write!(f, "u64"),
            ScalarType::Float32 => write!(f, "f32"),
            ScalarType::Float64 => write!(f, "f64"),
        }
    }
}

impl IRType {
    /// Create a scalar IR type
    pub fn scalar(scalar_type: ScalarType) -> Self {
        IRType::Tensor(scalar_type, Shape::Scalar)
    }

    /// Create a vector IR type
    pub fn vector(scalar_type: ScalarType, size: usize) -> Self {
        IRType::Tensor(scalar_type, Shape::vector(size))
    }

    /// Create a matrix IR type
    pub fn matrix(scalar_type: ScalarType, nrows: usize, ncols: usize) -> Self {
        IRType::Tensor(scalar_type, Shape::matrix(nrows, ncols))
    }

    /// Create a symbolic vector IR type
    pub fn symbolic_vector(scalar_type: ScalarType, name: impl Into<String>) -> Self {
        IRType::Tensor(scalar_type, Shape::symbolic_vector(name))
    }

    /// Create a symbolic matrix IR type
    pub fn symbolic_matrix(
        scalar_type: ScalarType,
        nrows: impl Into<String>,
        ncols: impl Into<String>,
    ) -> Self {
        IRType::Tensor(scalar_type, Shape::symbolic_matrix(nrows, ncols))
    }

    /// Get the scalar element type
    pub fn scalar_type(&self) -> ScalarType {
        match self {
            IRType::Tensor(st, _) => *st,
        }
    }

    /// Get the shape
    pub fn shape(&self) -> &Shape {
        match self {
            IRType::Tensor(_, shape) => shape,
        }
    }

    /// Check if this is a scalar type
    pub fn is_scalar(&self) -> bool {
        matches!(self.shape(), Shape::Scalar)
    }

    /// Check if this is a vector type
    pub fn is_vector(&self) -> bool {
        matches!(self.shape(), Shape::Vector(_))
    }

    /// Check if this is a matrix type
    pub fn is_matrix(&self) -> bool {
        matches!(self.shape(), Shape::Matrix(_, _))
    }

    /// Get rank (number of dimensions)
    pub fn rank(&self) -> usize {
        self.shape().rank()
    }
}

impl fmt::Display for IRType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            IRType::Tensor(st, shape) => match shape {
                Shape::Scalar => write!(f, "scalar<{}>", st),
                Shape::Vector(_) => write!(f, "vector<{}>{}", st, shape),
                Shape::Matrix(_, _) => write!(f, "matrix<{}>{}", st, shape),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_type_sizes() {
        assert_eq!(ScalarType::Bool.size_bytes(), 1);
        assert_eq!(ScalarType::Int32.size_bytes(), 4);
        assert_eq!(ScalarType::Float64.size_bytes(), 8);
    }

    #[test]
    fn test_type_predicates() {
        assert!(ScalarType::Float32.is_float());
        assert!(!ScalarType::Int32.is_float());
        assert!(ScalarType::Int32.is_integer());
        assert!(!ScalarType::Bool.is_integer());
    }

    #[test]
    fn test_from_type_code() {
        assert_eq!(
            ScalarType::from_type_code(TypeCode::Int32),
            Some(ScalarType::Int32)
        );
        assert_eq!(
            ScalarType::from_type_code(TypeCode::Fp64),
            Some(ScalarType::Float64)
        );
    }
}
