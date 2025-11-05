// Lisp Evaluator/Interpreter
//
// Executes Lisp expressions directly (without JIT compilation)
// for REPL and testing purposes

use crate::core::error::{GraphBlasError, Result};
use crate::core::matrix::Matrix;
use crate::core::semiring::Semiring;
use crate::core::vector::Vector;
use crate::lisp::ast::*;
use crate::lisp::parser::parse_program;
use crate::ops::{mxv, vxm};
use std::collections::HashMap;
use std::fmt;

/// Runtime value in the Lisp evaluator
pub enum Value {
    /// Scalar number
    Number(f64),
    /// Boolean value
    Bool(bool),
    /// Vector
    Vector(Vector<f64>),
    /// Matrix
    Matrix(Matrix<f64>),
    /// Nil/Unit value
    Nil,
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Number(num) => write!(f, "{}", num),
            Value::Bool(boolean) => write!(f, "{}", boolean),
            Value::Vector(vector) => {
                write!(f, "#[")?;
                for (idx, &val) in vector.values().iter().enumerate() {
                    if idx > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "{}", val)?;
                }
                write!(f, "]")
            }
            Value::Matrix(matrix) => {
                write!(f, "#<Matrix {}x{}>", matrix.nrows(), matrix.ncols())
            }
            Value::Nil => write!(f, "nil"),
        }
    }
}

/// Evaluation environment
pub struct Environment {
    /// Variable bindings
    bindings: HashMap<String, Value>,
    /// Kernel definitions
    kernels: HashMap<String, (Vec<Param>, Expr)>,
}

impl Environment {
    pub fn new() -> Self {
        Self {
            bindings: HashMap::new(),
            kernels: HashMap::new(),
        }
    }

    pub fn bind(&mut self, name: String, value: Value) {
        self.bindings.insert(name, value);
    }

    pub fn lookup(&self, name: &str) -> Result<&Value> {
        self.bindings.get(name).ok_or(GraphBlasError::InvalidValue)
    }

    pub fn define_kernel(&mut self, name: String, params: Vec<Param>, body: Expr) {
        self.kernels.insert(name, (params, body));
    }

    pub fn get_kernel(&self, name: &str) -> Option<&(Vec<Param>, Expr)> {
        self.kernels.get(name)
    }
}

impl Default for Environment {
    fn default() -> Self {
        Self::new()
    }
}

/// Lisp evaluator
pub struct Evaluator {
    env: Environment,
}

impl Evaluator {
    pub fn new() -> Self {
        Self {
            env: Environment::new(),
        }
    }

    /// Evaluate a Lisp program (multiple forms)
    pub fn eval_program(&mut self, source: &str) -> Result<Vec<Value>> {
        let forms = parse_program(source)?;
        let mut results = Vec::new();

        for form in forms {
            let result = self.eval_form(&form)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Evaluate a single form
    pub fn eval_form(&mut self, form: &Form) -> Result<Value> {
        match form {
            Form::DefKernel { name, params, body } => {
                self.env.define_kernel(name.clone(), params.clone(), body.clone());
                Ok(Value::Nil)
            }
            Form::Expr(expr) => self.eval_expr(expr),
        }
    }

    /// Evaluate an expression
    pub fn eval_expr(&mut self, expr: &Expr) -> Result<Value> {
        match expr {
            Expr::Literal(lit) => Ok(self.eval_literal(lit)),

            Expr::Variable(_name) => {
                // Variables containing matrices/vectors not supported yet
                // due to lack of Clone on Matrix/Vector types
                Err(GraphBlasError::NotImplemented)
            }

            Expr::FuncCall { func, args } => self.eval_func_call(func, args),

            Expr::Let { .. } => {
                // Let bindings not supported yet due to lack of Clone
                Err(GraphBlasError::NotImplemented)
            }
        }
    }

    fn eval_literal(&self, lit: &Literal) -> Value {
        match lit {
            Literal::Int32(i) => Value::Number(*i as f64),
            Literal::Int64(i) => Value::Number(*i as f64),
            Literal::Float32(f) => Value::Number(*f as f64),
            Literal::Float64(f) => Value::Number(*f),
            Literal::Bool(b) => Value::Bool(*b),
        }
    }

    fn eval_func_call(&mut self, func: &FuncName, args: &[Expr]) -> Result<Value> {
        match func {
            // Semiring matrix operations
            FuncName::PlusTimes => self.eval_semiring_op(args, "plus_times"),
            FuncName::MinPlus => self.eval_semiring_op(args, "min_plus"),
            FuncName::MaxTimes => self.eval_semiring_op(args, "max_times"),
            FuncName::OrAnd => self.eval_semiring_op(args, "or_and"),

            // Matrix-vector operations
            FuncName::MxV => self.eval_mxv(args),
            FuncName::VxM => self.eval_vxm(args),

            // Transpose
            FuncName::Transpose => {
                // Transpose not yet implemented in Matrix type
                Err(GraphBlasError::NotImplemented)
            }

            // Special forms for creating matrices/vectors
            FuncName::UserKernel(name) if name == "vector" => self.eval_create_vector(args),
            FuncName::UserKernel(name) if name == "matrix" => self.eval_create_matrix(args),

            // Other operations not yet implemented
            _ => Err(GraphBlasError::NotImplemented),
        }
    }

    fn eval_semiring_op(&mut self, args: &[Expr], semiring_name: &str) -> Result<Value> {
        if args.len() != 2 {
            return Err(GraphBlasError::DimensionMismatch);
        }

        let lhs = self.eval_expr(&args[0])?;
        let rhs = self.eval_expr(&args[1])?;

        match (lhs, rhs) {
            (Value::Matrix(left_matrix), Value::Matrix(right_matrix)) => {
                let semiring = match semiring_name {
                    "plus_times" => Semiring::plus_times()?,
                    "min_plus" => Semiring::min_plus()?,
                    "max_times" => Semiring::max_times()?,
                    "or_and" => return Err(GraphBlasError::NotImplemented), // or_and not yet implemented
                    _ => return Err(GraphBlasError::InvalidValue),
                };

                // Matrix multiply: output = left_matrix * right_matrix
                let mut output = Matrix::new(left_matrix.nrows(), right_matrix.ncols())?;
                crate::ops::mxm(&mut output, None, &left_matrix, &right_matrix, &semiring, None)?;
                Ok(Value::Matrix(output))
            }
            _ => Err(GraphBlasError::DomainMismatch),
        }
    }

    fn eval_mxv(&mut self, args: &[Expr]) -> Result<Value> {
        if args.len() != 2 {
            return Err(GraphBlasError::DimensionMismatch);
        }

        let matrix = self.eval_expr(&args[0])?;
        let vector = self.eval_expr(&args[1])?;

        match (matrix, vector) {
            (Value::Matrix(a), Value::Vector(u)) => {
                let mut w = Vector::new(a.nrows())?;
                // Pre-allocate output vector with dense storage
                // The kernel expects a pre-allocated buffer to write into
                w.values_mut().resize(a.nrows(), 0.0);
                w.indices_mut().extend(0..a.nrows());

                let semiring = Semiring::plus_times()?;
                mxv(&mut w, None, &a, &u, &semiring, None)?;
                Ok(Value::Vector(w))
            }
            _ => Err(GraphBlasError::DomainMismatch),
        }
    }

    fn eval_vxm(&mut self, args: &[Expr]) -> Result<Value> {
        if args.len() != 2 {
            return Err(GraphBlasError::DimensionMismatch);
        }

        let vector = self.eval_expr(&args[0])?;
        let matrix = self.eval_expr(&args[1])?;

        match (vector, matrix) {
            (Value::Vector(u), Value::Matrix(a)) => {
                let mut w = Vector::new(a.ncols())?;
                // Pre-allocate output vector with dense storage
                // The kernel expects a pre-allocated buffer to write into
                w.values_mut().resize(a.ncols(), 0.0);
                w.indices_mut().extend(0..a.ncols());

                let semiring = Semiring::plus_times()?;
                vxm(&mut w, None, &u, &a, &semiring, None)?;
                Ok(Value::Vector(w))
            }
            _ => Err(GraphBlasError::DomainMismatch),
        }
    }

    /// Create a vector from literal values: (vector 1 2 3)
    fn eval_create_vector(&mut self, args: &[Expr]) -> Result<Value> {
        let mut values = Vec::new();

        for arg in args {
            let val = self.eval_expr(arg)?;
            match val {
                Value::Number(num) => values.push(num),
                _ => return Err(GraphBlasError::DomainMismatch),
            }
        }

        let mut vector = Vector::new(values.len())?;
        vector.values_mut().extend_from_slice(&values);
        vector.indices_mut().extend(0..values.len());

        Ok(Value::Vector(vector))
    }

    /// Create a matrix from rows: (matrix (vector 1 2) (vector 3 4))
    fn eval_create_matrix(&mut self, args: &[Expr]) -> Result<Value> {
        if args.is_empty() {
            return Err(GraphBlasError::DimensionMismatch);
        }

        let mut rows = Vec::new();
        let mut ncols = 0;

        for arg in args {
            let val = self.eval_expr(arg)?;
            match val {
                Value::Vector(vector) => {
                    if ncols == 0 {
                        ncols = vector.values().len();
                    } else if vector.values().len() != ncols {
                        return Err(GraphBlasError::DimensionMismatch);
                    }
                    rows.push(vector.values().to_vec());
                }
                _ => return Err(GraphBlasError::DomainMismatch),
            }
        }

        let nrows = rows.len();

        // Convert to CSR format
        let mut row_ptrs = vec![0];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        for row in rows {
            for (col_idx, &val) in row.iter().enumerate() {
                if val != 0.0 {
                    col_indices.push(col_idx);
                    values.push(val);
                }
            }
            row_ptrs.push(col_indices.len());
        }

        let matrix = Matrix::from_csr(nrows, ncols, row_ptrs, col_indices, values)?;
        Ok(Value::Matrix(matrix))
    }

    /// Get a reference to the environment
    pub fn env(&self) -> &Environment {
        &self.env
    }

    /// Get a mutable reference to the environment
    pub fn env_mut(&mut self) -> &mut Environment {
        &mut self.env
    }
}

impl Default for Evaluator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eval_literal() {
        let mut eval = Evaluator::new();
        let results = eval.eval_program("42").unwrap();
        assert_eq!(results.len(), 1);
        assert!(matches!(results[0], Value::Number(n) if (n - 42.0).abs() < 1e-10));
    }

    #[test]
    fn test_eval_vector_creation() {
        let mut eval = Evaluator::new();
        let results = eval.eval_program("(vector 1 2 3)").unwrap();
        assert_eq!(results.len(), 1);

        if let Value::Vector(v) = &results[0] {
            assert_eq!(v.values().len(), 3);
            assert_eq!(v.values()[0], 1.0);
            assert_eq!(v.values()[1], 2.0);
            assert_eq!(v.values()[2], 3.0);
        } else {
            panic!("Expected vector");
        }
    }

    #[test]
    fn test_eval_matrix_creation() {
        let mut eval = Evaluator::new();
        let results = eval
            .eval_program("(matrix (vector 1 2) (vector 3 4))")
            .unwrap();
        assert_eq!(results.len(), 1);

        if let Value::Matrix(m) = &results[0] {
            assert_eq!(m.nrows(), 2);
            assert_eq!(m.ncols(), 2);
        } else {
            panic!("Expected matrix");
        }
    }

    // #[test]
    // fn test_eval_let_binding() {
    //     // Let bindings not yet supported
    // }
}
