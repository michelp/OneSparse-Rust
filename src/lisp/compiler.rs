// AST to IR compiler
//
// Translates Lisp AST nodes to IR graph nodes

use crate::core::error::{GraphBlasError, Result};
use crate::ir::{
    BinaryOpKind, GraphBuilder, NodeId, ScalarType, ScalarValue, UnaryOpKind,
};
use crate::lisp::ast::*;
use crate::lisp::types::{Type, TypeEnv};
use std::collections::HashMap;

/// Compiler context for AST to IR translation
pub struct Compiler {
    builder: GraphBuilder,
    bindings: HashMap<String, NodeId>,
    type_env: TypeEnv,
}

impl Compiler {
    pub fn new() -> Self {
        Self {
            builder: GraphBuilder::new(),
            bindings: HashMap::new(),
            type_env: TypeEnv::new(),
        }
    }

    /// Compile an expression to IR, returning the output node ID
    pub fn compile_expr(&mut self, expr: &Expr) -> Result<NodeId> {
        log::debug!("Compiling expression to IR");
        log::trace!("Expression: {:?}", expr);

        let result = match expr {
            Expr::Literal(lit) => {
                log::trace!("Compiling literal");
                self.compile_literal(lit)
            }

            Expr::Variable(name) => {
                log::trace!("Resolving variable: {}", name);
                self
                    .bindings
                    .get(name)
                    .copied()
                    .ok_or(GraphBlasError::InvalidValue)
            }

            Expr::FuncCall { func, args } => {
                log::debug!("Compiling function call: {:?}", func);
                self.compile_func_call(func, args)
            }

            Expr::Let { bindings, body } => {
                log::debug!("Compiling let bindings");
                // Compile bindings and add to environment
                let mut new_bindings = Vec::new();
                for (var, expr) in bindings {
                    let node_id = self.compile_expr(expr)?;
                    new_bindings.push((var.clone(), node_id));
                }

                // Add bindings to context
                for (var, node_id) in new_bindings {
                    self.bindings.insert(var, node_id);
                }

                // Compile body
                self.compile_expr(body)
            }
        };

        if let Ok(node_id) = result {
            log::trace!("Compiled to node ID: {}", node_id);
        }

        result
    }

    /// Compile a literal value
    fn compile_literal(&mut self, lit: &Literal) -> Result<NodeId> {
        // For now, literals need to be inputs
        // In a full implementation, we'd support constant folding
        let scalar_type = lit.scalar_type();

        // Create an input node for the literal
        // TODO: Support actual constant values in IR
        self.builder
            .input_scalar("const", scalar_type)
    }

    /// Compile a function call
    fn compile_func_call(&mut self, func: &FuncName, args: &[Expr]) -> Result<NodeId> {
        match func {
            // Semiring matrix operations
            FuncName::PlusTimes => self.compile_semiring_op(args, "plus_times"),
            FuncName::MinPlus => self.compile_semiring_op(args, "min_plus"),
            FuncName::MaxTimes => self.compile_semiring_op(args, "max_times"),
            FuncName::OrAnd => self.compile_semiring_op(args, "or_and"),

            // Matrix multiply (explicit)
            FuncName::MatMul => self.compile_semiring_op(args, "plus_times"),

            // Matrix-vector operations
            FuncName::MxV => self.compile_mxv(args),
            FuncName::VxM => self.compile_vxm(args),

            // Transpose
            FuncName::Transpose => {
                if args.len() != 1 {
                    return Err(GraphBlasError::DimensionMismatch);
                }
                let arg = self.compile_expr(&args[0])?;
                self.builder.transpose(arg)
            }

            // Element-wise operations
            FuncName::EWiseAdd => {
                if args.len() != 2 {
                    return Err(GraphBlasError::DimensionMismatch);
                }
                let lhs = self.compile_expr(&args[0])?;
                let rhs = self.compile_expr(&args[1])?;
                self.builder.ewise_add(lhs, rhs, BinaryOpKind::Add)
            }

            FuncName::EWiseMult => {
                if args.len() != 2 {
                    return Err(GraphBlasError::DimensionMismatch);
                }
                let lhs = self.compile_expr(&args[0])?;
                let rhs = self.compile_expr(&args[1])?;
                self.builder.ewise_mult(lhs, rhs, BinaryOpKind::Mul)
            }

            // Vector operations
            FuncName::VecAdd => {
                if args.len() != 2 {
                    return Err(GraphBlasError::DimensionMismatch);
                }
                let lhs = self.compile_expr(&args[0])?;
                let rhs = self.compile_expr(&args[1])?;
                self.builder.ewise_add(lhs, rhs, BinaryOpKind::Add)
            }

            FuncName::VecMult => {
                if args.len() != 2 {
                    return Err(GraphBlasError::DimensionMismatch);
                }
                let lhs = self.compile_expr(&args[0])?;
                let rhs = self.compile_expr(&args[1])?;
                self.builder.ewise_mult(lhs, rhs, BinaryOpKind::Mul)
            }

            // Apply unary operations
            FuncName::Apply(op) => {
                if args.len() != 1 {
                    return Err(GraphBlasError::DimensionMismatch);
                }
                let arg = self.compile_expr(&args[0])?;
                let unary_op = self.convert_unary_op(*op);
                self.builder.apply(arg, unary_op)
            }

            // Apply binary operations with scalar
            FuncName::ApplyLeft(op) => {
                if args.len() != 2 {
                    return Err(GraphBlasError::DimensionMismatch);
                }
                let arg = self.compile_expr(&args[0])?;
                let scalar_expr = &args[1];

                // Extract scalar value
                let scalar = self.extract_scalar(scalar_expr)?;
                let binary_op = self.convert_binary_op(*op);

                self.builder.apply_binary_left(scalar, arg, binary_op)
            }

            FuncName::ApplyRight(op) => {
                if args.len() != 2 {
                    return Err(GraphBlasError::DimensionMismatch);
                }
                let arg = self.compile_expr(&args[0])?;
                let scalar_expr = &args[1];

                let scalar = self.extract_scalar(scalar_expr)?;
                let binary_op = self.convert_binary_op(*op);

                self.builder.apply_binary_right(arg, scalar, binary_op)
            }

            // Reduction operations
            FuncName::ReduceRow => {
                // TODO: Implement reduce_matrix in GraphBuilder
                let _arg = self.compile_expr(&args[0])?;
                Err(GraphBlasError::NotImplemented)
            }

            FuncName::ReduceCol => {
                // TODO: Implement reduce_matrix in GraphBuilder
                let _arg = self.compile_expr(&args[0])?;
                Err(GraphBlasError::NotImplemented)
            }

            FuncName::ReduceVector => {
                // TODO: Implement reduce_vector in GraphBuilder
                let _arg = self.compile_expr(&args[0])?;
                Err(GraphBlasError::NotImplemented)
            }

            // Selection/filtering
            FuncName::Select(pred) => {
                if args.len() != 1 {
                    return Err(GraphBlasError::DimensionMismatch);
                }
                let arg = self.compile_expr(&args[0])?;
                let predicate = self.convert_predicate(*pred);
                self.builder.select(arg, predicate)
            }

            // Format conversion
            FuncName::ToCSR => {
                // TODO: Implement convert_format in GraphBuilder
                let _arg = self.compile_expr(&args[0])?;
                Err(GraphBlasError::NotImplemented)
            }

            FuncName::ToCSC => {
                // TODO: Implement convert_format in GraphBuilder
                let _arg = self.compile_expr(&args[0])?;
                Err(GraphBlasError::NotImplemented)
            }

            FuncName::ToDense => {
                // TODO: Implement convert_format in GraphBuilder
                let _arg = self.compile_expr(&args[0])?;
                Err(GraphBlasError::NotImplemented)
            }

            // User-defined kernels
            FuncName::UserKernel(_name) => {
                // TODO: Look up kernel in registry and inline its IR
                Err(GraphBlasError::InvalidValue)
            }
        }
    }

    /// Compile a semiring operation (matrix multiply)
    fn compile_semiring_op(&mut self, args: &[Expr], semiring_name: &str) -> Result<NodeId> {
        if args.len() != 2 {
            return Err(GraphBlasError::DimensionMismatch);
        }

        let lhs = self.compile_expr(&args[0])?;
        let rhs = self.compile_expr(&args[1])?;

        // Infer scalar type from arguments
        let scalar_type = self.infer_scalar_type(lhs)?;

        // Get semiring
        let semiring = match semiring_name {
            "plus_times" => crate::ir::semirings::plus_times(scalar_type),
            "min_plus" => crate::ir::semirings::min_plus(scalar_type),
            "max_times" => crate::ir::semirings::max_times(scalar_type),
            "or_and" => crate::ir::semirings::or_and(),
            _ => return Err(GraphBlasError::InvalidValue),
        };

        self.builder.matmul(lhs, rhs, semiring)
    }

    /// Compile matrix-vector multiply
    fn compile_mxv(&mut self, args: &[Expr]) -> Result<NodeId> {
        if args.len() != 2 {
            return Err(GraphBlasError::DimensionMismatch);
        }

        let matrix = self.compile_expr(&args[0])?;
        let vector = self.compile_expr(&args[1])?;

        let scalar_type = self.infer_scalar_type(matrix)?;
        let semiring = crate::ir::semirings::plus_times(scalar_type);

        self.builder.matvec(matrix, vector, semiring)
    }

    /// Compile vector-matrix multiply
    fn compile_vxm(&mut self, args: &[Expr]) -> Result<NodeId> {
        if args.len() != 2 {
            return Err(GraphBlasError::DimensionMismatch);
        }

        let vector = self.compile_expr(&args[0])?;
        let matrix = self.compile_expr(&args[1])?;

        let scalar_type = self.infer_scalar_type(matrix)?;
        let semiring = crate::ir::semirings::plus_times(scalar_type);

        self.builder.vecmat(vector, matrix, semiring)
    }

    /// Helper: infer scalar type from a node
    fn infer_scalar_type(&self, node_id: NodeId) -> Result<ScalarType> {
        let node = self
            .builder
            .graph()
            .get_node(node_id)
            .ok_or(GraphBlasError::InvalidValue)?;

        // Extract scalar type from IRType
        match node.output_type {
            crate::ir::IRType::Scalar(st) |
            crate::ir::IRType::Vector(st) |
            crate::ir::IRType::Matrix(st) => Ok(st),
        }
    }

    /// Helper: extract a scalar value from an expression
    fn extract_scalar(&self, expr: &Expr) -> Result<ScalarValue> {
        match expr {
            Expr::Literal(Literal::Int32(i)) => Ok(ScalarValue::Int32(*i)),
            Expr::Literal(Literal::Int64(i)) => Ok(ScalarValue::Int64(*i)),
            Expr::Literal(Literal::Float32(f)) => Ok(ScalarValue::Float32(*f)),
            Expr::Literal(Literal::Float64(f)) => Ok(ScalarValue::Float64(*f)),
            Expr::Literal(Literal::Bool(b)) => Ok(ScalarValue::Bool(*b)),
            _ => Err(GraphBlasError::InvalidValue),
        }
    }

    /// Convert AST unary op to IR unary op
    fn convert_unary_op(&self, op: UnaryOp) -> UnaryOpKind {
        match op {
            UnaryOp::Abs => UnaryOpKind::Abs,
            UnaryOp::Neg => UnaryOpKind::Neg,
            UnaryOp::Sqrt => UnaryOpKind::Sqrt,
            UnaryOp::Exp => UnaryOpKind::Exp,
            UnaryOp::Log => UnaryOpKind::Log,
        }
    }

    /// Convert AST binary op to IR binary op
    fn convert_binary_op(&self, op: BinaryOp) -> BinaryOpKind {
        match op {
            BinaryOp::Add => BinaryOpKind::Add,
            BinaryOp::Sub => BinaryOpKind::Sub,
            BinaryOp::Mul => BinaryOpKind::Mul,
            BinaryOp::Div => BinaryOpKind::Div,
            BinaryOp::Min => BinaryOpKind::Min,
            BinaryOp::Max => BinaryOpKind::Max,
        }
    }

    /// Convert AST predicate to IR select operation
    fn convert_predicate(&self, pred: Predicate) -> crate::ir::SelectOp {
        match pred {
            Predicate::GreaterThan(v) => crate::ir::SelectOp::GreaterThan(ScalarValue::Float64(v)),
            Predicate::LessThan(v) => crate::ir::SelectOp::LessThan(ScalarValue::Float64(v)),
            // For Equal/NotEqual/etc, we need to adapt since SelectOp doesn't have all these
            // For now, map them to GreaterThan/LessThan
            Predicate::Equal(_v) => {
                // TODO: Add Equal variant to SelectOp or use range
                crate::ir::SelectOp::NonZero
            }
            Predicate::NotEqual(_v) => {
                crate::ir::SelectOp::NonZero
            }
            Predicate::GreaterOrEqual(v) => {
                // Use GreaterThan for now
                crate::ir::SelectOp::GreaterThan(ScalarValue::Float64(v))
            }
            Predicate::LessOrEqual(v) => {
                crate::ir::SelectOp::LessThan(ScalarValue::Float64(v))
            }
        }
    }

    /// Add a binding for an input variable
    pub fn bind_input(&mut self, name: String, node_id: NodeId, ty: Type) {
        self.bindings.insert(name.clone(), node_id);
        self.type_env.bind(name, ty);
    }

    /// Get the IR graph
    pub fn into_graph(self) -> crate::ir::IRGraph {
        self.builder.build()
    }

    /// Get a reference to the graph builder
    pub fn builder(&self) -> &GraphBuilder {
        &self.builder
    }

    /// Get a mutable reference to the graph builder
    pub fn builder_mut(&mut self) -> &mut GraphBuilder {
        &mut self.builder
    }
}

impl Default for Compiler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Shape;

    #[test]
    fn test_compile_literal() {
        let mut compiler = Compiler::new();
        let expr = Expr::Literal(Literal::Float64(3.14));
        let node_id = compiler.compile_expr(&expr).unwrap();

        let graph = compiler.into_graph();
        let node = graph.get_node(node_id).unwrap();
        assert!(matches!(node.output_type, crate::ir::IRType::Scalar(crate::ir::ScalarType::Float64)));
    }

    #[test]
    fn test_compile_transpose() {
        let mut compiler = Compiler::new();

        // Create input matrix
        let a_id = compiler
            .builder
            .input_matrix("A", ScalarType::Float64, Shape::matrix(10, 20))
            .unwrap();
        compiler.bind_input("a".to_string(), a_id, Type::Matrix(ScalarType::Float64));

        // Compile (transpose a)
        let expr = Expr::FuncCall {
            func: FuncName::Transpose,
            args: vec![Expr::Variable("a".to_string())],
        };

        let result = compiler.compile_expr(&expr).unwrap();
        let graph = compiler.into_graph();

        let node = graph.get_node(result).unwrap();
        assert!(matches!(
            node.op,
            crate::ir::Operation::Transpose
        ));
    }

    #[test]
    fn test_compile_matmul() {
        let mut compiler = Compiler::new();

        // Create input matrices
        let a_id = compiler
            .builder
            .input_matrix("A", ScalarType::Float64, Shape::matrix(10, 20))
            .unwrap();
        let b_id = compiler
            .builder
            .input_matrix("B", ScalarType::Float64, Shape::matrix(20, 30))
            .unwrap();

        compiler.bind_input("a".to_string(), a_id, Type::Matrix(ScalarType::Float64));
        compiler.bind_input("b".to_string(), b_id, Type::Matrix(ScalarType::Float64));

        // Compile (plus-times a b)
        let expr = Expr::FuncCall {
            func: FuncName::PlusTimes,
            args: vec![
                Expr::Variable("a".to_string()),
                Expr::Variable("b".to_string()),
            ],
        };

        let result = compiler.compile_expr(&expr).unwrap();
        let graph = compiler.into_graph();

        let node = graph.get_node(result).unwrap();
        assert!(matches!(
            node.op,
            crate::ir::Operation::MatMul { .. }
        ));
    }
}
