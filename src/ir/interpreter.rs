// Simple IR Interpreter for Testing
//
// Executes IR graphs to produce actual results for validation.
// This is NOT the JIT compiler - it's a reference implementation for testing.

use crate::core::error::{GraphBlasError, Result};
use crate::ir::graph::IRGraph;
use crate::ir::node::{BinaryOpKind, NodeId, Operation, ScalarValue, UnaryOpKind};
use crate::ir::types::ScalarType;
use std::collections::HashMap;

/// Value that can be stored in the interpreter
#[derive(Debug, Clone)]
pub enum InterpreterValue {
    Scalar(f64),
    Vector(Vec<f64>),
    Matrix(Vec<Vec<f64>>),
}

impl InterpreterValue {
    /// Get as scalar
    pub fn as_scalar(&self) -> Result<f64> {
        match self {
            InterpreterValue::Scalar(v) => Ok(*v),
            _ => Err(GraphBlasError::InvalidValue),
        }
    }

    /// Get as vector
    pub fn as_vector(&self) -> Result<&Vec<f64>> {
        match self {
            InterpreterValue::Vector(v) => Ok(v),
            _ => Err(GraphBlasError::InvalidValue),
        }
    }

    /// Get as matrix
    pub fn as_matrix(&self) -> Result<&Vec<Vec<f64>>> {
        match self {
            InterpreterValue::Matrix(m) => Ok(m),
            _ => Err(GraphBlasError::InvalidValue),
        }
    }
}

/// Simple IR interpreter
pub struct Interpreter {
    /// Input values provided by user
    inputs: HashMap<String, InterpreterValue>,
    /// Computed values during execution
    values: HashMap<NodeId, InterpreterValue>,
}

impl Interpreter {
    /// Create a new interpreter
    pub fn new() -> Self {
        Self {
            inputs: HashMap::new(),
            values: HashMap::new(),
        }
    }

    /// Set an input value
    pub fn set_input(&mut self, name: impl Into<String>, value: InterpreterValue) {
        self.inputs.insert(name.into(), value);
    }

    /// Execute the graph and return output values
    pub fn execute(&mut self, graph: &IRGraph) -> Result<Vec<InterpreterValue>> {
        // Get topological order
        let topo_order = graph.topological_order()?;

        // Execute each node in order
        for &node_id in &topo_order {
            if let Some(node) = graph.get_node(node_id) {
                let result = match &node.op {
                    Operation::Input { name, .. } => {
                        self.inputs.get(name).cloned()
                            .ok_or(GraphBlasError::InvalidValue)?
                    }

                    Operation::Output => {
                        // Output nodes just pass through their input
                        if node.inputs.is_empty() {
                            return Err(GraphBlasError::InvalidValue);
                        }
                        self.values.get(&node.inputs[0]).cloned()
                            .ok_or(GraphBlasError::InvalidValue)?
                    }

                    Operation::MatMul { semiring, .. } => {
                        self.execute_matmul(node_id, &node.inputs, semiring)?
                    }

                    Operation::MatVec { semiring } => {
                        self.execute_matvec(node_id, &node.inputs, semiring)?
                    }

                    Operation::VecMat { semiring } => {
                        self.execute_vecmat(node_id, &node.inputs, semiring)?
                    }

                    Operation::EWiseAdd { binary_op } => {
                        self.execute_ewise_add(&node.inputs, binary_op)?
                    }

                    Operation::EWiseMult { binary_op } => {
                        self.execute_ewise_mult(&node.inputs, binary_op)?
                    }

                    Operation::Apply { unary_op } => {
                        self.execute_apply(&node.inputs, unary_op)?
                    }

                    Operation::ApplyBinaryLeft { binary_op, scalar } => {
                        self.execute_apply_binary_left(&node.inputs, scalar, binary_op)?
                    }

                    Operation::ApplyBinaryRight { binary_op, scalar } => {
                        self.execute_apply_binary_right(&node.inputs, scalar, binary_op)?
                    }

                    Operation::Transpose => {
                        self.execute_transpose(&node.inputs)?
                    }

                    _ => {
                        // Other operations not yet implemented in interpreter
                        InterpreterValue::Scalar(0.0)
                    }
                };

                self.values.insert(node_id, result);
            }
        }

        // Collect output values
        let mut outputs = Vec::new();
        for &output_id in graph.outputs() {
            if let Some(value) = self.values.get(&output_id) {
                outputs.push(value.clone());
            }
        }

        Ok(outputs)
    }

    fn execute_matmul(
        &self,
        _node_id: NodeId,
        inputs: &[NodeId],
        _semiring: &crate::ir::node::SemiringOp,
    ) -> Result<InterpreterValue> {
        if inputs.len() != 2 {
            return Err(GraphBlasError::InvalidValue);
        }

        let a = self.values.get(&inputs[0])
            .ok_or(GraphBlasError::InvalidValue)?
            .as_matrix()?;
        let b = self.values.get(&inputs[1])
            .ok_or(GraphBlasError::InvalidValue)?
            .as_matrix()?;

        let m = a.len();
        let n = b[0].len();
        let k = a[0].len();

        if k != b.len() {
            return Err(GraphBlasError::DimensionMismatch);
        }

        let mut result = vec![vec![0.0; n]; m];

        // Simple dense matmul (C = A * B)
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for kk in 0..k {
                    sum += a[i][kk] * b[kk][j];
                }
                result[i][j] = sum;
            }
        }

        Ok(InterpreterValue::Matrix(result))
    }

    fn execute_matvec(
        &self,
        _node_id: NodeId,
        inputs: &[NodeId],
        _semiring: &crate::ir::node::SemiringOp,
    ) -> Result<InterpreterValue> {
        if inputs.len() != 2 {
            return Err(GraphBlasError::InvalidValue);
        }

        let a = self.values.get(&inputs[0])
            .ok_or(GraphBlasError::InvalidValue)?
            .as_matrix()?;
        let u = self.values.get(&inputs[1])
            .ok_or(GraphBlasError::InvalidValue)?
            .as_vector()?;

        let m = a.len();
        let n = a[0].len();

        if n != u.len() {
            return Err(GraphBlasError::DimensionMismatch);
        }

        let mut result = vec![0.0; m];

        // w = A * u
        for i in 0..m {
            let mut sum = 0.0;
            for j in 0..n {
                sum += a[i][j] * u[j];
            }
            result[i] = sum;
        }

        Ok(InterpreterValue::Vector(result))
    }

    fn execute_vecmat(
        &self,
        _node_id: NodeId,
        inputs: &[NodeId],
        _semiring: &crate::ir::node::SemiringOp,
    ) -> Result<InterpreterValue> {
        if inputs.len() != 2 {
            return Err(GraphBlasError::InvalidValue);
        }

        let u = self.values.get(&inputs[0])
            .ok_or(GraphBlasError::InvalidValue)?
            .as_vector()?;
        let a = self.values.get(&inputs[1])
            .ok_or(GraphBlasError::InvalidValue)?
            .as_matrix()?;

        let m = a.len();
        let n = a[0].len();

        if u.len() != m {
            return Err(GraphBlasError::DimensionMismatch);
        }

        let mut result = vec![0.0; n];

        // w = u * A
        for j in 0..n {
            let mut sum = 0.0;
            for i in 0..m {
                sum += u[i] * a[i][j];
            }
            result[j] = sum;
        }

        Ok(InterpreterValue::Vector(result))
    }

    fn execute_ewise_add(&self, inputs: &[NodeId], op: &BinaryOpKind) -> Result<InterpreterValue> {
        if inputs.len() != 2 {
            return Err(GraphBlasError::InvalidValue);
        }

        let a = self.values.get(&inputs[0])
            .ok_or(GraphBlasError::InvalidValue)?;
        let b = self.values.get(&inputs[1])
            .ok_or(GraphBlasError::InvalidValue)?;

        match (a, b) {
            (InterpreterValue::Matrix(a_mat), InterpreterValue::Matrix(b_mat)) => {
                let m = a_mat.len();
                let n = a_mat[0].len();
                let mut result = vec![vec![0.0; n]; m];

                for i in 0..m {
                    for j in 0..n {
                        result[i][j] = self.apply_binary_op(a_mat[i][j], b_mat[i][j], op);
                    }
                }

                Ok(InterpreterValue::Matrix(result))
            }
            (InterpreterValue::Vector(a_vec), InterpreterValue::Vector(b_vec)) => {
                let result: Vec<f64> = a_vec.iter()
                    .zip(b_vec.iter())
                    .map(|(&a_val, &b_val)| self.apply_binary_op(a_val, b_val, op))
                    .collect();

                Ok(InterpreterValue::Vector(result))
            }
            _ => Err(GraphBlasError::InvalidValue),
        }
    }

    fn execute_ewise_mult(&self, inputs: &[NodeId], op: &BinaryOpKind) -> Result<InterpreterValue> {
        // For dense interpretation, ewise mult is same as ewise add
        self.execute_ewise_add(inputs, op)
    }

    fn execute_apply(&self, inputs: &[NodeId], op: &UnaryOpKind) -> Result<InterpreterValue> {
        if inputs.is_empty() {
            return Err(GraphBlasError::InvalidValue);
        }

        let input = self.values.get(&inputs[0])
            .ok_or(GraphBlasError::InvalidValue)?;

        match input {
            InterpreterValue::Matrix(mat) => {
                let m = mat.len();
                let n = mat[0].len();
                let mut result = vec![vec![0.0; n]; m];

                for i in 0..m {
                    for j in 0..n {
                        result[i][j] = self.apply_unary_op(mat[i][j], op);
                    }
                }

                Ok(InterpreterValue::Matrix(result))
            }
            InterpreterValue::Vector(vec) => {
                let result: Vec<f64> = vec.iter()
                    .map(|&val| self.apply_unary_op(val, op))
                    .collect();

                Ok(InterpreterValue::Vector(result))
            }
            InterpreterValue::Scalar(val) => {
                Ok(InterpreterValue::Scalar(self.apply_unary_op(*val, op)))
            }
        }
    }

    fn execute_apply_binary_left(
        &self,
        inputs: &[NodeId],
        scalar: &ScalarValue,
        op: &BinaryOpKind,
    ) -> Result<InterpreterValue> {
        if inputs.is_empty() {
            return Err(GraphBlasError::InvalidValue);
        }

        let scalar_val = match scalar {
            ScalarValue::Float64(v) => *v,
            _ => 0.0, // TODO: support other types
        };

        let input = self.values.get(&inputs[0])
            .ok_or(GraphBlasError::InvalidValue)?;

        match input {
            InterpreterValue::Matrix(mat) => {
                let m = mat.len();
                let n = mat[0].len();
                let mut result = vec![vec![0.0; n]; m];

                for i in 0..m {
                    for j in 0..n {
                        result[i][j] = self.apply_binary_op(scalar_val, mat[i][j], op);
                    }
                }

                Ok(InterpreterValue::Matrix(result))
            }
            InterpreterValue::Vector(vec) => {
                let result: Vec<f64> = vec.iter()
                    .map(|&val| self.apply_binary_op(scalar_val, val, op))
                    .collect();

                Ok(InterpreterValue::Vector(result))
            }
            _ => Err(GraphBlasError::InvalidValue),
        }
    }

    fn execute_apply_binary_right(
        &self,
        inputs: &[NodeId],
        scalar: &ScalarValue,
        op: &BinaryOpKind,
    ) -> Result<InterpreterValue> {
        if inputs.is_empty() {
            return Err(GraphBlasError::InvalidValue);
        }

        let scalar_val = match scalar {
            ScalarValue::Float64(v) => *v,
            _ => 0.0, // TODO: support other types
        };

        let input = self.values.get(&inputs[0])
            .ok_or(GraphBlasError::InvalidValue)?;

        match input {
            InterpreterValue::Matrix(mat) => {
                let m = mat.len();
                let n = mat[0].len();
                let mut result = vec![vec![0.0; n]; m];

                for i in 0..m {
                    for j in 0..n {
                        result[i][j] = self.apply_binary_op(mat[i][j], scalar_val, op);
                    }
                }

                Ok(InterpreterValue::Matrix(result))
            }
            InterpreterValue::Vector(vec) => {
                let result: Vec<f64> = vec.iter()
                    .map(|&val| self.apply_binary_op(val, scalar_val, op))
                    .collect();

                Ok(InterpreterValue::Vector(result))
            }
            _ => Err(GraphBlasError::InvalidValue),
        }
    }

    fn execute_transpose(&self, inputs: &[NodeId]) -> Result<InterpreterValue> {
        if inputs.is_empty() {
            return Err(GraphBlasError::InvalidValue);
        }

        let input = self.values.get(&inputs[0])
            .ok_or(GraphBlasError::InvalidValue)?;

        match input {
            InterpreterValue::Matrix(mat) => {
                let m = mat.len();
                let n = mat[0].len();
                let mut result = vec![vec![0.0; m]; n];

                for i in 0..m {
                    for j in 0..n {
                        result[j][i] = mat[i][j];
                    }
                }

                Ok(InterpreterValue::Matrix(result))
            }
            _ => Err(GraphBlasError::InvalidValue),
        }
    }

    fn apply_unary_op(&self, val: f64, op: &UnaryOpKind) -> f64 {
        match op {
            UnaryOpKind::Neg => -val,
            UnaryOpKind::Abs => val.abs(),
            UnaryOpKind::Sqrt => val.sqrt(),
            UnaryOpKind::Exp => val.exp(),
            UnaryOpKind::Log => val.ln(),
            UnaryOpKind::Not => if val == 0.0 { 1.0 } else { 0.0 },
            _ => val,
        }
    }

    fn apply_binary_op(&self, a: f64, b: f64, op: &BinaryOpKind) -> f64 {
        match op {
            BinaryOpKind::Add => a + b,
            BinaryOpKind::Sub => a - b,
            BinaryOpKind::Mul => a * b,
            BinaryOpKind::Div => a / b,
            BinaryOpKind::Min => a.min(b),
            BinaryOpKind::Max => a.max(b),
            BinaryOpKind::And => if a != 0.0 && b != 0.0 { 1.0 } else { 0.0 },
            BinaryOpKind::Or => if a != 0.0 || b != 0.0 { 1.0 } else { 0.0 },
            BinaryOpKind::Eq => if (a - b).abs() < 1e-10 { 1.0 } else { 0.0 },
            BinaryOpKind::Ne => if (a - b).abs() >= 1e-10 { 1.0 } else { 0.0 },
            BinaryOpKind::Lt => if a < b { 1.0 } else { 0.0 },
            BinaryOpKind::Le => if a <= b { 1.0 } else { 0.0 },
            BinaryOpKind::Gt => if a > b { 1.0 } else { 0.0 },
            BinaryOpKind::Ge => if a >= b { 1.0 } else { 0.0 },
            _ => 0.0,
        }
    }
}

impl Default for Interpreter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::builder::GraphBuilder;
    use crate::ir::node::{BinaryOpKind, MonoidOp, ScalarValue, SemiringOp, UnaryOpKind};
    use crate::ir::types::ScalarType;
    use crate::ir::Shape;

    #[test]
    fn test_interpreter_matmul() {
        let mut builder = GraphBuilder::new();

        let a = builder
            .input_matrix("A", ScalarType::Float64, Shape::matrix(2, 3))
            .unwrap();
        let b = builder
            .input_matrix("B", ScalarType::Float64, Shape::matrix(3, 2))
            .unwrap();

        let semiring = SemiringOp {
            add_op: MonoidOp {
                binary_op: BinaryOpKind::Add,
                identity: ScalarValue::Float64(0.0),
            },
            mul_op: BinaryOpKind::Mul,
        };

        let c = builder.matmul(a, b, semiring).unwrap();
        builder.output(c).unwrap();

        let graph = builder.build();

        let mut interp = Interpreter::new();
        interp.set_input("A", InterpreterValue::Matrix(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ]));
        interp.set_input("B", InterpreterValue::Matrix(vec![
            vec![7.0, 8.0],
            vec![9.0, 10.0],
            vec![11.0, 12.0],
        ]));

        let outputs = interp.execute(&graph).unwrap();
        assert_eq!(outputs.len(), 1);

        let result = outputs[0].as_matrix().unwrap();
        // [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
        // [[58, 64], [139, 154]]
        assert_eq!(result[0][0], 58.0);
        assert_eq!(result[0][1], 64.0);
        assert_eq!(result[1][0], 139.0);
        assert_eq!(result[1][1], 154.0);
    }

    #[test]
    fn test_interpreter_ewise_add() {
        let mut builder = GraphBuilder::new();

        let a = builder
            .input_matrix("A", ScalarType::Float64, Shape::matrix(2, 2))
            .unwrap();
        let b = builder
            .input_matrix("B", ScalarType::Float64, Shape::matrix(2, 2))
            .unwrap();

        let c = builder.ewise_add(a, b, BinaryOpKind::Add).unwrap();
        builder.output(c).unwrap();

        let graph = builder.build();

        let mut interp = Interpreter::new();
        interp.set_input("A", InterpreterValue::Matrix(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ]));
        interp.set_input("B", InterpreterValue::Matrix(vec![
            vec![5.0, 6.0],
            vec![7.0, 8.0],
        ]));

        let outputs = interp.execute(&graph).unwrap();
        let result = outputs[0].as_matrix().unwrap();

        assert_eq!(result[0][0], 6.0);
        assert_eq!(result[0][1], 8.0);
        assert_eq!(result[1][0], 10.0);
        assert_eq!(result[1][1], 12.0);
    }

    #[test]
    fn test_interpreter_apply() {
        let mut builder = GraphBuilder::new();

        let a = builder
            .input_vector("a", ScalarType::Float64, Shape::vector(3))
            .unwrap();

        let b = builder.apply(a, UnaryOpKind::Abs).unwrap();
        builder.output(b).unwrap();

        let graph = builder.build();

        let mut interp = Interpreter::new();
        interp.set_input("a", InterpreterValue::Vector(vec![-1.0, 2.0, -3.0]));

        let outputs = interp.execute(&graph).unwrap();
        let result = outputs[0].as_vector().unwrap();

        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], 2.0);
        assert_eq!(result[2], 3.0);
    }
}
