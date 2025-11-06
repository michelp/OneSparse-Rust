// Abstract Syntax Tree for Lisp DSL
//
// Defines the AST nodes produced by parsing S-expressions

use crate::ir::ScalarType;

/// Top-level form in a Lisp program
#[derive(Debug, Clone, PartialEq)]
pub enum Form {
    /// Kernel definition: (defkernel name [params] body)
    DefKernel {
        name: String,
        params: Vec<Param>,
        body: Expr,
    },
    /// Expression to evaluate
    Expr(Expr),
}

/// Function parameter with optional type annotation
#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub name: String,
    pub type_annotation: Option<TypeAnnotation>,
}

impl Param {
    pub fn new(name: String) -> Self {
        Self {
            name,
            type_annotation: None,
        }
    }

    pub fn with_type(name: String, type_annotation: TypeAnnotation) -> Self {
        Self {
            name,
            type_annotation: Some(type_annotation),
        }
    }
}

/// Type annotation for parameters
#[derive(Debug, Clone, PartialEq)]
pub enum TypeAnnotation {
    Scalar(ScalarType),
    Vector(ScalarType),
    Matrix(ScalarType),
}

/// Expression in the Lisp DSL
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// Literal value: 42, 3.14, true
    Literal(Literal),

    /// Variable reference: x, my_matrix
    Variable(String),

    /// Function call: (op arg1 arg2 ...)
    FuncCall { func: FuncName, args: Vec<Expr> },

    /// Let binding: (let ((x expr)) body)
    Let {
        bindings: Vec<(String, Expr)>,
        body: Box<Expr>,
    },

    /// Conditional: (if condition then-expr else-expr)
    If {
        condition: Box<Expr>,
        then_branch: Box<Expr>,
        else_branch: Box<Expr>,
    },

    /// While loop: (while condition body...)
    While {
        condition: Box<Expr>,
        body: Box<Expr>,
    },

    /// For loop: (for var start end body...) or (for var start end step body...)
    For {
        var: String,
        start: Box<Expr>,
        end: Box<Expr>,
        step: Option<Box<Expr>>,
        body: Box<Expr>,
    },

    /// Cond (multi-way branching): (cond (test1 result1) (test2 result2) ... (else default))
    Cond {
        clauses: Vec<(Expr, Expr)>,
        else_clause: Option<Box<Expr>>,
    },

    /// Block/sequence of expressions: (begin expr1 expr2 ...)
    /// Returns the value of the last expression
    Block(Vec<Expr>),

    /// Break from loop: (break) or (break value)
    Break(Option<Box<Expr>>),

    /// Continue to next iteration: (continue)
    Continue,
}

/// Function names in the DSL
#[derive(Debug, Clone, PartialEq)]
pub enum FuncName {
    /// Semiring operations
    PlusTimes,
    MinPlus,
    MaxTimes,
    OrAnd,

    /// Matrix operations
    Transpose,
    MatMul,
    MxV,
    VxM,

    /// Element-wise operations
    EWiseAdd,
    EWiseMult,

    /// Vector operations
    VecAdd,
    VecMult,

    /// Apply operations (unary)
    Apply(UnaryOp),

    /// Apply operations (binary with scalar)
    ApplyLeft(BinaryOp),
    ApplyRight(BinaryOp),

    /// Reduction operations
    ReduceRow,
    ReduceCol,
    ReduceVector,

    /// Selection/filtering
    Select(Predicate),

    /// Format conversion
    ToCSR,
    ToCSC,
    ToDense,

    /// User-defined kernel
    UserKernel(String),
}

/// Unary operations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UnaryOp {
    Abs,
    Neg,
    Sqrt,
    Exp,
    Log,
}

/// Binary operations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,
}

/// Predicates for selection
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Predicate {
    GreaterThan(f64),
    LessThan(f64),
    Equal(f64),
    NotEqual(f64),
    GreaterOrEqual(f64),
    LessOrEqual(f64),
}

/// Literal values
#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Int32(i32),
    Int64(i64),
    Float32(f32),
    Float64(f64),
    Bool(bool),
}

impl Literal {
    /// Get the scalar type of this literal
    pub fn scalar_type(&self) -> ScalarType {
        match self {
            Literal::Int32(_) => ScalarType::Int32,
            Literal::Int64(_) => ScalarType::Int64,
            Literal::Float32(_) => ScalarType::Float32,
            Literal::Float64(_) => ScalarType::Float64,
            Literal::Bool(_) => ScalarType::Bool,
        }
    }

    /// Try to convert to f64 for numeric operations
    pub fn to_f64(&self) -> Option<f64> {
        match self {
            Literal::Int32(i) => Some(*i as f64),
            Literal::Int64(i) => Some(*i as f64),
            Literal::Float32(f) => Some(*f as f64),
            Literal::Float64(f) => Some(*f),
            Literal::Bool(_) => None,
        }
    }
}

impl std::fmt::Display for FuncName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FuncName::PlusTimes => write!(f, "plus-times"),
            FuncName::MinPlus => write!(f, "min-plus"),
            FuncName::MaxTimes => write!(f, "max-times"),
            FuncName::OrAnd => write!(f, "or-and"),
            FuncName::Transpose => write!(f, "transpose"),
            FuncName::MatMul => write!(f, "matmul"),
            FuncName::MxV => write!(f, "mxv"),
            FuncName::VxM => write!(f, "vxm"),
            FuncName::EWiseAdd => write!(f, "ewise-add"),
            FuncName::EWiseMult => write!(f, "ewise-mult"),
            FuncName::VecAdd => write!(f, "vec-add"),
            FuncName::VecMult => write!(f, "vec-mult"),
            FuncName::Apply(op) => write!(f, "apply-{:?}", op),
            FuncName::ApplyLeft(op) => write!(f, "apply-left-{:?}", op),
            FuncName::ApplyRight(op) => write!(f, "apply-right-{:?}", op),
            FuncName::ReduceRow => write!(f, "reduce-row"),
            FuncName::ReduceCol => write!(f, "reduce-col"),
            FuncName::ReduceVector => write!(f, "reduce-vector"),
            FuncName::Select(pred) => write!(f, "select-{:?}", pred),
            FuncName::ToCSR => write!(f, "to-csr"),
            FuncName::ToCSC => write!(f, "to-csc"),
            FuncName::ToDense => write!(f, "to-dense"),
            FuncName::UserKernel(name) => write!(f, "{}", name),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_literal_types() {
        assert_eq!(Literal::Int32(42).scalar_type(), ScalarType::Int32);
        assert_eq!(Literal::Float64(3.14).scalar_type(), ScalarType::Float64);
        assert_eq!(Literal::Bool(true).scalar_type(), ScalarType::Bool);
    }

    #[test]
    fn test_literal_conversion() {
        assert_eq!(Literal::Int32(42).to_f64(), Some(42.0));
        assert_eq!(Literal::Float64(3.14).to_f64(), Some(3.14));
        assert_eq!(Literal::Bool(true).to_f64(), None);
    }

    #[test]
    fn test_param_creation() {
        let p1 = Param::new("x".to_string());
        assert_eq!(p1.name, "x");
        assert_eq!(p1.type_annotation, None);

        let p2 = Param::with_type("y".to_string(), TypeAnnotation::Matrix(ScalarType::Float64));
        assert_eq!(p2.name, "y");
        assert!(p2.type_annotation.is_some());
    }

    #[test]
    fn test_expr_structure() {
        // Test simple literal
        let lit = Expr::Literal(Literal::Int32(42));
        assert!(matches!(lit, Expr::Literal(_)));

        // Test variable
        let var = Expr::Variable("x".to_string());
        assert!(matches!(var, Expr::Variable(_)));

        // Test function call
        let call = Expr::FuncCall {
            func: FuncName::PlusTimes,
            args: vec![
                Expr::Variable("a".to_string()),
                Expr::Variable("b".to_string()),
            ],
        };
        assert!(matches!(call, Expr::FuncCall { .. }));
    }
}
