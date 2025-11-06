// Core Layer: Rust Error Types
// Idiomatic Rust error handling

use crate::ffi::error::*;
use std::fmt;

/// Rust-native GraphBLAS error type
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphBlasError {
    /// Object has not been initialized
    UninitializedObject,
    /// Input pointer is null
    NullPointer,
    /// Invalid parameter value
    InvalidValue,
    /// Invalid index
    InvalidIndex,
    /// Type mismatch between operands
    DomainMismatch,
    /// Dimension mismatch between operands
    DimensionMismatch,
    /// Output object must be empty
    OutputNotEmpty,
    /// Method not implemented
    NotImplemented,
    /// Memory allocation failed
    OutOfMemory,
    /// Insufficient space in output
    InsufficientSpace,
    /// Object is invalid or corrupted
    InvalidObject,
    /// Index exceeds valid bounds
    IndexOutOfBounds,
    /// Object is empty
    EmptyObject,
    /// Panic occurred
    Panic(String),
    /// Loop break (control flow, not an error)
    LoopBreak,
    /// Loop continue (control flow, not an error)
    LoopContinue,
}

impl GraphBlasError {
    /// Convert Rust error to C API error code
    pub fn to_grb_info(&self) -> GrB_Info {
        match self {
            Self::UninitializedObject => GrB_UNINITIALIZED_OBJECT,
            Self::NullPointer => GrB_NULL_POINTER,
            Self::InvalidValue => GrB_INVALID_VALUE,
            Self::InvalidIndex => GrB_INVALID_INDEX,
            Self::DomainMismatch => GrB_DOMAIN_MISMATCH,
            Self::DimensionMismatch => GrB_DIMENSION_MISMATCH,
            Self::OutputNotEmpty => GrB_OUTPUT_NOT_EMPTY,
            Self::NotImplemented => GrB_NOT_IMPLEMENTED,
            Self::OutOfMemory => GrB_OUT_OF_MEMORY,
            Self::InsufficientSpace => GrB_INSUFFICIENT_SPACE,
            Self::InvalidObject => GrB_INVALID_OBJECT,
            Self::IndexOutOfBounds => GrB_INDEX_OUT_OF_BOUNDS,
            Self::EmptyObject => GrB_EMPTY_OBJECT,
            Self::Panic(_) => GrB_PANIC,
            // Control flow signals don't map to C API codes
            Self::LoopBreak => GrB_PANIC,
            Self::LoopContinue => GrB_PANIC,
        }
    }

    /// Convert C API error code to Rust error
    pub fn from_grb_info(info: GrB_Info) -> Option<Self> {
        match info {
            GrB_SUCCESS | GrB_NO_VALUE => None,
            GrB_UNINITIALIZED_OBJECT => Some(Self::UninitializedObject),
            GrB_NULL_POINTER => Some(Self::NullPointer),
            GrB_INVALID_VALUE => Some(Self::InvalidValue),
            GrB_INVALID_INDEX => Some(Self::InvalidIndex),
            GrB_DOMAIN_MISMATCH => Some(Self::DomainMismatch),
            GrB_DIMENSION_MISMATCH => Some(Self::DimensionMismatch),
            GrB_OUTPUT_NOT_EMPTY => Some(Self::OutputNotEmpty),
            GrB_NOT_IMPLEMENTED => Some(Self::NotImplemented),
            GrB_OUT_OF_MEMORY => Some(Self::OutOfMemory),
            GrB_INSUFFICIENT_SPACE => Some(Self::InsufficientSpace),
            GrB_INVALID_OBJECT => Some(Self::InvalidObject),
            GrB_INDEX_OUT_OF_BOUNDS => Some(Self::IndexOutOfBounds),
            GrB_EMPTY_OBJECT => Some(Self::EmptyObject),
            GrB_PANIC => Some(Self::Panic("Unknown panic".to_string())),
            _ => Some(Self::InvalidValue),
        }
    }
}

impl fmt::Display for GraphBlasError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UninitializedObject => write!(f, "Object has not been initialized"),
            Self::NullPointer => write!(f, "Input pointer is null"),
            Self::InvalidValue => write!(f, "Invalid parameter value"),
            Self::InvalidIndex => write!(f, "Invalid index"),
            Self::DomainMismatch => write!(f, "Type mismatch between operands"),
            Self::DimensionMismatch => write!(f, "Dimension mismatch between operands"),
            Self::OutputNotEmpty => write!(f, "Output object must be empty"),
            Self::NotImplemented => write!(f, "Method not implemented"),
            Self::OutOfMemory => write!(f, "Memory allocation failed"),
            Self::InsufficientSpace => write!(f, "Insufficient space in output"),
            Self::InvalidObject => write!(f, "Object is invalid or corrupted"),
            Self::IndexOutOfBounds => write!(f, "Index exceeds valid bounds"),
            Self::EmptyObject => write!(f, "Object is empty"),
            Self::Panic(msg) => write!(f, "Panic: {}", msg),
            Self::LoopBreak => write!(f, "Break outside of loop"),
            Self::LoopContinue => write!(f, "Continue outside of loop"),
        }
    }
}

impl std::error::Error for GraphBlasError {}

/// Result type for GraphBLAS operations
pub type Result<T> = std::result::Result<T, GraphBlasError>;
