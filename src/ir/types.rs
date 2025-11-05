// IR Type System
//
// This is separate from the runtime type system (types::TypeCode).
// IR types are used during compilation and optimization.

use crate::types::TypeCode;
use std::fmt;

/// IR value type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IRType {
    /// Scalar value
    Scalar(ScalarType),
    /// Vector (size can be symbolic)
    Vector(ScalarType),
    /// Matrix (dimensions can be symbolic)
    Matrix(ScalarType),
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

impl fmt::Display for IRType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            IRType::Scalar(st) => write!(f, "scalar<{}>", st),
            IRType::Vector(st) => write!(f, "vector<{}>", st),
            IRType::Matrix(st) => write!(f, "matrix<{}>", st),
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
