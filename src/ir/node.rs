// IR Node Types
//
// Defines all operations that can be represented in the IR

use crate::ir::shape::Shape;
use crate::ir::types::{IRType, ScalarType};
use std::fmt;

/// Unique node identifier in a computation graph
pub type NodeId = usize;

/// Sparse matrix storage format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StorageFormat {
    /// Format-agnostic (let optimizer decide)
    Any,
    /// Compressed Sparse Row
    CSR,
    /// Compressed Sparse Column
    CSC,
    /// Coordinate format
    COO,
}

/// IR Node representing an operation
#[derive(Debug, Clone)]
pub struct IRNode {
    /// Unique identifier
    pub id: NodeId,
    /// Operation kind
    pub op: Operation,
    /// Input node IDs
    pub inputs: Vec<NodeId>,
    /// Output type
    pub output_type: IRType,
    /// Output shape
    pub output_shape: Shape,
}

/// Operation types
#[derive(Debug, Clone, PartialEq)]
pub enum Operation {
    // ===== Input/Output Operations =====
    /// Input placeholder (for function parameters)
    Input { name: String, format: StorageFormat },

    /// Output (marks a result)
    Output,

    // ===== Matrix Multiplication =====
    /// Matrix-matrix multiplication (C = A × B)
    MatMul {
        semiring: SemiringOp,
        format: StorageFormat,
    },

    /// Matrix-vector multiplication (w = A × u)
    MatVec { semiring: SemiringOp },

    /// Vector-matrix multiplication (w = u × A)
    VecMat { semiring: SemiringOp },

    // ===== Element-wise Operations =====
    /// Element-wise addition (union of sparse structures)
    EWiseAdd { binary_op: BinaryOpKind },

    /// Element-wise multiplication (intersection of sparse structures)
    EWiseMult { binary_op: BinaryOpKind },

    // ===== Apply Operations =====
    /// Apply unary operator
    Apply { unary_op: UnaryOpKind },

    /// Apply binary operator with bound left operand
    ApplyBinaryLeft {
        binary_op: BinaryOpKind,
        scalar: ScalarValue,
    },

    /// Apply binary operator with bound right operand
    ApplyBinaryRight {
        binary_op: BinaryOpKind,
        scalar: ScalarValue,
    },

    // ===== Selection =====
    /// Select elements based on predicate
    Select { predicate: SelectOp },

    // ===== Structural Operations =====
    /// Transpose matrix
    Transpose,

    /// Format conversion
    ConvertFormat {
        from: StorageFormat,
        to: StorageFormat,
    },

    /// Extract submatrix/subvector
    Extract {
        // TODO: Add index ranges
    },

    /// Assign to submatrix/subvector
    Assign {
        // TODO: Add index ranges
    },

    // ===== Reduction Operations =====
    /// Reduce matrix to vector (along rows or columns)
    ReduceMatrix { monoid: MonoidOp, axis: Axis },

    /// Reduce vector to scalar
    ReduceVector { monoid: MonoidOp },
}

/// Axis for reduction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    Rows,
    Columns,
}

/// Semiring operations (add_op, mul_op, zero)
#[derive(Debug, Clone, PartialEq)]
pub struct SemiringOp {
    pub add_op: MonoidOp,
    pub mul_op: BinaryOpKind,
}

/// Monoid operations (binary_op, identity)
#[derive(Debug, Clone, PartialEq)]
pub struct MonoidOp {
    pub binary_op: BinaryOpKind,
    pub identity: ScalarValue,
}

/// Binary operation kinds
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOpKind {
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,

    // Logical
    And,
    Or,
    Xor,

    // Comparison
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,

    // Custom (index into registry)
    Custom(usize),
}

/// Unary operation kinds
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOpKind {
    // Arithmetic
    Neg,
    Abs,
    Sqrt,
    Exp,
    Log,

    // Logical
    Not,

    // Rounding
    Floor,
    Ceil,
    Round,

    // Custom (index into registry)
    Custom(usize),
}

/// Select predicate operations
#[derive(Debug, Clone, PartialEq)]
pub enum SelectOp {
    /// Select non-zero elements
    NonZero,

    /// Select elements greater than threshold
    GreaterThan(ScalarValue),

    /// Select elements less than threshold
    LessThan(ScalarValue),

    /// Select elements in range [min, max]
    InRange(ScalarValue, ScalarValue),

    /// Custom predicate (index into registry)
    Custom(usize),
}

/// Scalar value (constant)
#[derive(Debug, Clone, PartialEq)]
pub enum ScalarValue {
    Bool(bool),
    Int8(i8),
    Int16(i16),
    Int32(i32),
    Int64(i64),
    Uint8(u8),
    Uint16(u16),
    Uint32(u32),
    Uint64(u64),
    Float32(f32),
    Float64(f64),
}

impl ScalarValue {
    pub fn scalar_type(&self) -> ScalarType {
        match self {
            ScalarValue::Bool(_) => ScalarType::Bool,
            ScalarValue::Int8(_) => ScalarType::Int8,
            ScalarValue::Int16(_) => ScalarType::Int16,
            ScalarValue::Int32(_) => ScalarType::Int32,
            ScalarValue::Int64(_) => ScalarType::Int64,
            ScalarValue::Uint8(_) => ScalarType::Uint8,
            ScalarValue::Uint16(_) => ScalarType::Uint16,
            ScalarValue::Uint32(_) => ScalarType::Uint32,
            ScalarValue::Uint64(_) => ScalarType::Uint64,
            ScalarValue::Float32(_) => ScalarType::Float32,
            ScalarValue::Float64(_) => ScalarType::Float64,
        }
    }

    /// Create a zero value of the given scalar type
    pub fn from_type(scalar_type: ScalarType, _value: f64) -> Self {
        match scalar_type {
            ScalarType::Bool => ScalarValue::Bool(false),
            ScalarType::Int8 => ScalarValue::Int8(0),
            ScalarType::Int16 => ScalarValue::Int16(0),
            ScalarType::Int32 => ScalarValue::Int32(0),
            ScalarType::Int64 => ScalarValue::Int64(0),
            ScalarType::Uint8 => ScalarValue::Uint8(0),
            ScalarType::Uint16 => ScalarValue::Uint16(0),
            ScalarType::Uint32 => ScalarValue::Uint32(0),
            ScalarType::Uint64 => ScalarValue::Uint64(0),
            ScalarType::Float32 => ScalarValue::Float32(0.0),
            ScalarType::Float64 => ScalarValue::Float64(0.0),
        }
    }
}

impl fmt::Display for StorageFormat {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            StorageFormat::Any => write!(f, "any"),
            StorageFormat::CSR => write!(f, "csr"),
            StorageFormat::CSC => write!(f, "csc"),
            StorageFormat::COO => write!(f, "coo"),
        }
    }
}

impl fmt::Display for BinaryOpKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            BinaryOpKind::Add => write!(f, "add"),
            BinaryOpKind::Sub => write!(f, "sub"),
            BinaryOpKind::Mul => write!(f, "mul"),
            BinaryOpKind::Div => write!(f, "div"),
            BinaryOpKind::Min => write!(f, "min"),
            BinaryOpKind::Max => write!(f, "max"),
            BinaryOpKind::And => write!(f, "and"),
            BinaryOpKind::Or => write!(f, "or"),
            BinaryOpKind::Xor => write!(f, "xor"),
            BinaryOpKind::Eq => write!(f, "eq"),
            BinaryOpKind::Ne => write!(f, "ne"),
            BinaryOpKind::Lt => write!(f, "lt"),
            BinaryOpKind::Le => write!(f, "le"),
            BinaryOpKind::Gt => write!(f, "gt"),
            BinaryOpKind::Ge => write!(f, "ge"),
            BinaryOpKind::Custom(id) => write!(f, "custom_{}", id),
        }
    }
}

impl fmt::Display for UnaryOpKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            UnaryOpKind::Neg => write!(f, "neg"),
            UnaryOpKind::Abs => write!(f, "abs"),
            UnaryOpKind::Sqrt => write!(f, "sqrt"),
            UnaryOpKind::Exp => write!(f, "exp"),
            UnaryOpKind::Log => write!(f, "log"),
            UnaryOpKind::Not => write!(f, "not"),
            UnaryOpKind::Floor => write!(f, "floor"),
            UnaryOpKind::Ceil => write!(f, "ceil"),
            UnaryOpKind::Round => write!(f, "round"),
            UnaryOpKind::Custom(id) => write!(f, "custom_{}", id),
        }
    }
}
