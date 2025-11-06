// Type inference for Lisp DSL
//
// Implements bidirectional type checking with unification

use crate::core::error::{GraphBlasError, Result};
use crate::ir::ScalarType;
use crate::lisp::ast::*;
use std::collections::HashMap;

/// Type of an expression
#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    /// Scalar value
    Scalar(ScalarType),
    /// Vector with element type
    Vector(ScalarType),
    /// Matrix with element type
    Matrix(ScalarType),
    /// Unknown type (for inference)
    Unknown,
}

impl Type {
    /// Get the scalar type if this is a scalar/vector/matrix
    pub fn scalar_type(&self) -> Option<ScalarType> {
        match self {
            Type::Scalar(t) | Type::Vector(t) | Type::Matrix(t) => Some(*t),
            Type::Unknown => None,
        }
    }

    /// Check if this type is compatible with another for unification
    pub fn unify(&self, other: &Type) -> Result<Type> {
        match (self, other) {
            (Type::Unknown, t) | (t, Type::Unknown) => Ok(t.clone()),
            (Type::Scalar(t1), Type::Scalar(t2)) if t1 == t2 => Ok(Type::Scalar(*t1)),
            (Type::Vector(t1), Type::Vector(t2)) if t1 == t2 => Ok(Type::Vector(*t1)),
            (Type::Matrix(t1), Type::Matrix(t2)) if t1 == t2 => Ok(Type::Matrix(*t1)),
            _ => Err(GraphBlasError::DomainMismatch),
        }
    }

    /// Promote numeric types (i32 -> i64 -> f32 -> f64)
    pub fn promote_with(&self, other: &Type) -> Result<Type> {
        use ScalarType::*;

        match (self.scalar_type(), other.scalar_type()) {
            (Some(t1), Some(t2)) => {
                let promoted = match (t1, t2) {
                    // Same type
                    (t, s) if t == s => t,
                    // Int32 promotions
                    (Int32, Int64) | (Int64, Int32) => Int64,
                    (Int32, Float32) | (Float32, Int32) => Float32,
                    (Int32, Float64) | (Float64, Int32) => Float64,
                    // Int64 promotions
                    (Int64, Float32) | (Float32, Int64) => Float32,
                    (Int64, Float64) | (Float64, Int64) => Float64,
                    // Float32 promotions
                    (Float32, Float64) | (Float64, Float32) => Float64,
                    // Bool doesn't promote
                    _ => return Err(GraphBlasError::DomainMismatch),
                };

                // Preserve the structure (scalar/vector/matrix) from self
                match self {
                    Type::Scalar(_) => Ok(Type::Scalar(promoted)),
                    Type::Vector(_) => Ok(Type::Vector(promoted)),
                    Type::Matrix(_) => Ok(Type::Matrix(promoted)),
                    Type::Unknown => Ok(Type::Scalar(promoted)),
                }
            }
            _ => Err(GraphBlasError::DomainMismatch),
        }
    }
}

/// Type environment for tracking variable types
#[derive(Debug, Clone)]
pub struct TypeEnv {
    bindings: HashMap<String, Type>,
}

impl TypeEnv {
    pub fn new() -> Self {
        Self {
            bindings: HashMap::new(),
        }
    }

    pub fn bind(&mut self, name: String, ty: Type) {
        self.bindings.insert(name, ty);
    }

    pub fn lookup(&self, name: &str) -> Option<&Type> {
        self.bindings.get(name)
    }

    pub fn extend(&self, name: String, ty: Type) -> Self {
        let mut new_env = self.clone();
        new_env.bind(name, ty);
        new_env
    }
}

impl Default for TypeEnv {
    fn default() -> Self {
        Self::new()
    }
}

/// Type checker
pub struct TypeChecker {
    env: TypeEnv,
}

impl TypeChecker {
    pub fn new() -> Self {
        Self {
            env: TypeEnv::new(),
        }
    }

    pub fn with_env(env: TypeEnv) -> Self {
        Self { env }
    }

    /// Infer the type of an expression
    pub fn infer_expr(&mut self, expr: &Expr) -> Result<Type> {
        match expr {
            Expr::Literal(lit) => Ok(Type::Scalar(lit.scalar_type())),

            Expr::Variable(name) => self
                .env
                .lookup(name)
                .cloned()
                .ok_or(GraphBlasError::InvalidValue),

            Expr::FuncCall { func, args } => self.infer_func_call(func, args),

            Expr::Let { bindings, body } => {
                // Create new environment with bindings
                let mut new_env = self.env.clone();
                for (var, expr) in bindings {
                    let ty = self.infer_expr(expr)?;
                    new_env.bind(var.clone(), ty);
                }

                // Type check body in new environment
                let mut checker = TypeChecker::with_env(new_env);
                checker.infer_expr(body)
            }

            Expr::If {
                condition,
                then_branch,
                else_branch,
            } => {
                // Check condition is boolean-ish
                let _cond_ty = self.infer_expr(condition)?;
                // TODO: Verify condition is boolean or numeric

                // Type is the union of both branches (for now, just use then branch)
                let then_ty = self.infer_expr(then_branch)?;
                let _else_ty = self.infer_expr(else_branch)?;
                // TODO: Unify types
                Ok(then_ty)
            }

            Expr::While { condition, body } => {
                let _cond_ty = self.infer_expr(condition)?;
                let _body_ty = self.infer_expr(body)?;
                // While loops return nil
                Ok(Type::Scalar(ScalarType::Bool)) // Using bool as stand-in for nil
            }

            Expr::For {
                var: _,
                start,
                end,
                step,
                body,
            } => {
                let _start_ty = self.infer_expr(start)?;
                let _end_ty = self.infer_expr(end)?;
                if let Some(step_expr) = step {
                    let _step_ty = self.infer_expr(step_expr)?;
                }
                let _body_ty = self.infer_expr(body)?;
                // For loops return nil
                Ok(Type::Scalar(ScalarType::Bool))
            }

            Expr::Cond {
                clauses,
                else_clause,
            } => {
                // Type check all clauses
                let mut result_ty = None;
                for (test, result) in clauses {
                    let _test_ty = self.infer_expr(test)?;
                    let res_ty = self.infer_expr(result)?;
                    if result_ty.is_none() {
                        result_ty = Some(res_ty);
                    }
                    // TODO: Unify all clause result types
                }

                if let Some(else_expr) = else_clause {
                    let _else_ty = self.infer_expr(else_expr)?;
                }

                result_ty.ok_or(GraphBlasError::InvalidValue)
            }

            Expr::Block(exprs) => {
                let mut last_ty = Type::Scalar(ScalarType::Bool); // nil type
                for expr in exprs {
                    last_ty = self.infer_expr(expr)?;
                }
                Ok(last_ty)
            }

            Expr::Break(_) | Expr::Continue => {
                // Control flow doesn't have a value type
                Ok(Type::Scalar(ScalarType::Bool))
            }
        }
    }

    /// Infer the type of a function call
    fn infer_func_call(&mut self, func: &FuncName, args: &[Expr]) -> Result<Type> {
        match func {
            // Semiring operations: (op matrix matrix) -> matrix
            FuncName::PlusTimes
            | FuncName::MinPlus
            | FuncName::MaxTimes
            | FuncName::OrAnd
            | FuncName::MatMul => {
                if args.len() != 2 {
                    return Err(GraphBlasError::DimensionMismatch);
                }

                let t1 = self.infer_expr(&args[0])?;
                let t2 = self.infer_expr(&args[1])?;

                // Both must be matrices
                match (&t1, &t2) {
                    (Type::Matrix(s1), Type::Matrix(s2)) => {
                        let unified = Type::Matrix(*s1).promote_with(&Type::Matrix(*s2))?;
                        Ok(unified)
                    }
                    _ => Err(GraphBlasError::DomainMismatch),
                }
            }

            // Matrix-vector multiply: (mxv matrix vector) -> vector
            FuncName::MxV => {
                if args.len() != 2 {
                    return Err(GraphBlasError::DimensionMismatch);
                }

                let t1 = self.infer_expr(&args[0])?;
                let t2 = self.infer_expr(&args[1])?;

                match (&t1, &t2) {
                    (Type::Matrix(s1), Type::Vector(s2)) => {
                        if s1 != s2 {
                            return Err(GraphBlasError::DomainMismatch);
                        }
                        Ok(Type::Vector(*s1))
                    }
                    _ => Err(GraphBlasError::DomainMismatch),
                }
            }

            // Vector-matrix multiply: (vxm vector matrix) -> vector
            FuncName::VxM => {
                if args.len() != 2 {
                    return Err(GraphBlasError::DimensionMismatch);
                }

                let t1 = self.infer_expr(&args[0])?;
                let t2 = self.infer_expr(&args[1])?;

                match (&t1, &t2) {
                    (Type::Vector(s1), Type::Matrix(s2)) => {
                        if s1 != s2 {
                            return Err(GraphBlasError::DomainMismatch);
                        }
                        Ok(Type::Vector(*s1))
                    }
                    _ => Err(GraphBlasError::DomainMismatch),
                }
            }

            // Transpose: (transpose matrix) -> matrix
            FuncName::Transpose => {
                if args.len() != 1 {
                    return Err(GraphBlasError::DimensionMismatch);
                }

                let t = self.infer_expr(&args[0])?;
                match t {
                    Type::Matrix(_) => Ok(t),
                    _ => Err(GraphBlasError::DomainMismatch),
                }
            }

            // Element-wise operations: preserve structure
            FuncName::EWiseAdd | FuncName::EWiseMult => {
                if args.len() != 2 {
                    return Err(GraphBlasError::DimensionMismatch);
                }

                let t1 = self.infer_expr(&args[0])?;
                let t2 = self.infer_expr(&args[1])?;

                t1.unify(&t2)
            }

            // Vector operations
            FuncName::VecAdd | FuncName::VecMult => {
                if args.len() != 2 {
                    return Err(GraphBlasError::DimensionMismatch);
                }

                let t1 = self.infer_expr(&args[0])?;
                let t2 = self.infer_expr(&args[1])?;

                match (&t1, &t2) {
                    (Type::Vector(s1), Type::Vector(s2)) => {
                        if s1 != s2 {
                            return Err(GraphBlasError::DomainMismatch);
                        }
                        Ok(Type::Vector(*s1))
                    }
                    _ => Err(GraphBlasError::DomainMismatch),
                }
            }

            // Unary apply: preserves structure
            FuncName::Apply(_) => {
                if args.len() != 1 {
                    return Err(GraphBlasError::DimensionMismatch);
                }

                self.infer_expr(&args[0])
            }

            // Binary apply with scalar
            FuncName::ApplyLeft(_) | FuncName::ApplyRight(_) => {
                if args.len() != 2 {
                    return Err(GraphBlasError::DimensionMismatch);
                }

                // First arg is the matrix/vector, second is scalar
                let t1 = self.infer_expr(&args[0])?;
                let t2 = self.infer_expr(&args[1])?;

                // Second arg must be scalar
                if !matches!(t2, Type::Scalar(_)) {
                    return Err(GraphBlasError::DomainMismatch);
                }

                // Return type of first arg
                Ok(t1)
            }

            // Reduction operations
            FuncName::ReduceRow => {
                if args.len() != 1 {
                    return Err(GraphBlasError::DimensionMismatch);
                }

                let t = self.infer_expr(&args[0])?;
                match t {
                    Type::Matrix(s) => Ok(Type::Vector(s)),
                    _ => Err(GraphBlasError::DomainMismatch),
                }
            }

            FuncName::ReduceCol => {
                if args.len() != 1 {
                    return Err(GraphBlasError::DimensionMismatch);
                }

                let t = self.infer_expr(&args[0])?;
                match t {
                    Type::Matrix(s) => Ok(Type::Vector(s)),
                    _ => Err(GraphBlasError::DomainMismatch),
                }
            }

            FuncName::ReduceVector => {
                if args.len() != 1 {
                    return Err(GraphBlasError::DimensionMismatch);
                }

                let t = self.infer_expr(&args[0])?;
                match t {
                    Type::Vector(s) => Ok(Type::Scalar(s)),
                    _ => Err(GraphBlasError::DomainMismatch),
                }
            }

            // Selection preserves type
            FuncName::Select(_) => {
                if args.len() != 1 {
                    return Err(GraphBlasError::DimensionMismatch);
                }

                self.infer_expr(&args[0])
            }

            // Format conversion preserves type and element type
            FuncName::ToCSR | FuncName::ToCSC | FuncName::ToDense => {
                if args.len() != 1 {
                    return Err(GraphBlasError::DimensionMismatch);
                }

                self.infer_expr(&args[0])
            }

            // User-defined kernels: need to look up in kernel registry
            FuncName::UserKernel(_) => {
                // For now, return unknown type
                // TODO: Look up kernel signature in registry
                Ok(Type::Unknown)
            }
        }
    }

    /// Check if an expression has a specific type
    pub fn check_expr(&mut self, expr: &Expr, expected: &Type) -> Result<()> {
        let inferred = self.infer_expr(expr)?;
        inferred.unify(expected)?;
        Ok(())
    }
}

impl Default for TypeChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_unification() {
        let t1 = Type::Matrix(ScalarType::Float64);
        let t2 = Type::Matrix(ScalarType::Float64);
        assert_eq!(t1.unify(&t2).unwrap(), Type::Matrix(ScalarType::Float64));

        let t3 = Type::Unknown;
        assert_eq!(t1.unify(&t3).unwrap(), Type::Matrix(ScalarType::Float64));
    }

    #[test]
    fn test_type_promotion() {
        let t1 = Type::Scalar(ScalarType::Int32);
        let t2 = Type::Scalar(ScalarType::Float64);
        assert_eq!(
            t1.promote_with(&t2).unwrap(),
            Type::Scalar(ScalarType::Float64)
        );

        let t3 = Type::Matrix(ScalarType::Int32);
        let t4 = Type::Matrix(ScalarType::Int64);
        assert_eq!(
            t3.promote_with(&t4).unwrap(),
            Type::Matrix(ScalarType::Int64)
        );
    }

    #[test]
    fn test_infer_literal() {
        let mut checker = TypeChecker::new();
        let expr = Expr::Literal(Literal::Float64(3.14));
        let ty = checker.infer_expr(&expr).unwrap();
        assert_eq!(ty, Type::Scalar(ScalarType::Float64));
    }

    #[test]
    fn test_infer_variable() {
        let mut env = TypeEnv::new();
        env.bind("x".to_string(), Type::Matrix(ScalarType::Float64));

        let mut checker = TypeChecker::with_env(env);
        let expr = Expr::Variable("x".to_string());
        let ty = checker.infer_expr(&expr).unwrap();
        assert_eq!(ty, Type::Matrix(ScalarType::Float64));
    }

    #[test]
    fn test_infer_transpose() {
        let mut env = TypeEnv::new();
        env.bind("a".to_string(), Type::Matrix(ScalarType::Float64));

        let mut checker = TypeChecker::with_env(env);
        let expr = Expr::FuncCall {
            func: FuncName::Transpose,
            args: vec![Expr::Variable("a".to_string())],
        };

        let ty = checker.infer_expr(&expr).unwrap();
        assert_eq!(ty, Type::Matrix(ScalarType::Float64));
    }

    #[test]
    fn test_infer_mxv() {
        let mut env = TypeEnv::new();
        env.bind("a".to_string(), Type::Matrix(ScalarType::Float64));
        env.bind("u".to_string(), Type::Vector(ScalarType::Float64));

        let mut checker = TypeChecker::with_env(env);
        let expr = Expr::FuncCall {
            func: FuncName::MxV,
            args: vec![
                Expr::Variable("a".to_string()),
                Expr::Variable("u".to_string()),
            ],
        };

        let ty = checker.infer_expr(&expr).unwrap();
        assert_eq!(ty, Type::Vector(ScalarType::Float64));
    }

    #[test]
    fn test_type_mismatch() {
        let mut env = TypeEnv::new();
        env.bind("a".to_string(), Type::Matrix(ScalarType::Float64));
        env.bind("b".to_string(), Type::Vector(ScalarType::Float64));

        let mut checker = TypeChecker::with_env(env);
        let expr = Expr::FuncCall {
            func: FuncName::MatMul,
            args: vec![
                Expr::Variable("a".to_string()),
                Expr::Variable("b".to_string()), // Wrong: should be matrix
            ],
        };

        assert!(checker.infer_expr(&expr).is_err());
    }
}
