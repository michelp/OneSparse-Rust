// IR Builder: Public API for composing computation graphs
//
// Provides a fluent interface for building IR graphs

use crate::core::error::{GraphBlasError, Result};
use crate::ir::graph::IRGraph;
use crate::ir::node::{
    BinaryOpKind, MonoidOp, NodeId, Operation, ScalarValue, SelectOp, SemiringOp,
    StorageFormat, UnaryOpKind,
};
use crate::ir::shape::Shape;
use crate::ir::types::{IRType, ScalarType};
use std::collections::HashMap;

/// Builder for constructing IR graphs
pub struct GraphBuilder {
    graph: IRGraph,
    /// Map from user-provided names to node IDs
    named_nodes: HashMap<String, NodeId>,
}

impl GraphBuilder {
    /// Create a new graph builder
    pub fn new() -> Self {
        Self {
            graph: IRGraph::new(),
            named_nodes: HashMap::new(),
        }
    }

    /// Add an input matrix placeholder
    pub fn input_matrix(
        &mut self,
        name: impl Into<String>,
        elem_type: ScalarType,
        shape: Shape,
    ) -> Result<NodeId> {
        let name = name.into();

        if !matches!(shape, Shape::Matrix(_, _)) {
            return Err(GraphBlasError::InvalidValue);
        }

        let node_id = self.graph.add_node(
            Operation::Input {
                name: name.clone(),
                format: StorageFormat::Any,
            },
            vec![],
            IRType::Matrix(elem_type),
            shape,
        )?;

        self.graph.add_input(node_id)?;
        self.named_nodes.insert(name, node_id);
        Ok(node_id)
    }

    /// Add an input vector placeholder
    pub fn input_vector(
        &mut self,
        name: impl Into<String>,
        elem_type: ScalarType,
        shape: Shape,
    ) -> Result<NodeId> {
        let name = name.into();

        if !matches!(shape, Shape::Vector(_)) {
            return Err(GraphBlasError::InvalidValue);
        }

        let node_id = self.graph.add_node(
            Operation::Input {
                name: name.clone(),
                format: StorageFormat::Any,
            },
            vec![],
            IRType::Vector(elem_type),
            shape,
        )?;

        self.graph.add_input(node_id)?;
        self.named_nodes.insert(name, node_id);
        Ok(node_id)
    }

    /// Add an input scalar placeholder
    pub fn input_scalar(
        &mut self,
        name: impl Into<String>,
        elem_type: ScalarType,
    ) -> Result<NodeId> {
        let name = name.into();

        let node_id = self.graph.add_node(
            Operation::Input {
                name: name.clone(),
                format: StorageFormat::Any,
            },
            vec![],
            IRType::Scalar(elem_type),
            Shape::Scalar,
        )?;

        self.graph.add_input(node_id)?;
        self.named_nodes.insert(name, node_id);
        Ok(node_id)
    }

    /// Matrix-matrix multiplication
    pub fn matmul(
        &mut self,
        a: NodeId,
        b: NodeId,
        semiring: SemiringOp,
    ) -> Result<NodeId> {
        let a_node = self.graph.get_node(a)
            .ok_or(GraphBlasError::InvalidValue)?;
        let b_node = self.graph.get_node(b)
            .ok_or(GraphBlasError::InvalidValue)?;

        // Infer output shape
        let output_shape = Shape::matmul(&a_node.output_shape, &b_node.output_shape)
            .ok_or(GraphBlasError::DimensionMismatch)?;

        // Get element type (should match semiring)
        let elem_type = match a_node.output_type {
            IRType::Matrix(t) => t,
            _ => return Err(GraphBlasError::InvalidValue),
        };

        self.graph.add_node(
            Operation::MatMul {
                semiring,
                format: StorageFormat::Any,
            },
            vec![a, b],
            IRType::Matrix(elem_type),
            output_shape,
        )
    }

    /// Matrix-vector multiplication
    pub fn matvec(
        &mut self,
        a: NodeId,
        u: NodeId,
        semiring: SemiringOp,
    ) -> Result<NodeId> {
        let a_node = self.graph.get_node(a)
            .ok_or(GraphBlasError::InvalidValue)?;
        let u_node = self.graph.get_node(u)
            .ok_or(GraphBlasError::InvalidValue)?;

        let output_shape = Shape::matmul(&a_node.output_shape, &u_node.output_shape)
            .ok_or(GraphBlasError::DimensionMismatch)?;

        let elem_type = match a_node.output_type {
            IRType::Matrix(t) => t,
            _ => return Err(GraphBlasError::InvalidValue),
        };

        self.graph.add_node(
            Operation::MatVec { semiring },
            vec![a, u],
            IRType::Vector(elem_type),
            output_shape,
        )
    }

    /// Vector-matrix multiplication
    pub fn vecmat(
        &mut self,
        u: NodeId,
        a: NodeId,
        semiring: SemiringOp,
    ) -> Result<NodeId> {
        let u_node = self.graph.get_node(u)
            .ok_or(GraphBlasError::InvalidValue)?;
        let a_node = self.graph.get_node(a)
            .ok_or(GraphBlasError::InvalidValue)?;

        let output_shape = Shape::matmul(&u_node.output_shape, &a_node.output_shape)
            .ok_or(GraphBlasError::DimensionMismatch)?;

        let elem_type = match a_node.output_type {
            IRType::Matrix(t) => t,
            _ => return Err(GraphBlasError::InvalidValue),
        };

        self.graph.add_node(
            Operation::VecMat { semiring },
            vec![u, a],
            IRType::Vector(elem_type),
            output_shape,
        )
    }

    /// Element-wise addition (union)
    pub fn ewise_add(
        &mut self,
        a: NodeId,
        b: NodeId,
        binary_op: BinaryOpKind,
    ) -> Result<NodeId> {
        let a_node = self.graph.get_node(a)
            .ok_or(GraphBlasError::InvalidValue)?;
        let b_node = self.graph.get_node(b)
            .ok_or(GraphBlasError::InvalidValue)?;

        let output_shape = Shape::ewise(&a_node.output_shape, &b_node.output_shape)
            .ok_or(GraphBlasError::DimensionMismatch)?;

        let output_type = a_node.output_type;

        self.graph.add_node(
            Operation::EWiseAdd { binary_op },
            vec![a, b],
            output_type,
            output_shape,
        )
    }

    /// Element-wise multiplication (intersection)
    pub fn ewise_mult(
        &mut self,
        a: NodeId,
        b: NodeId,
        binary_op: BinaryOpKind,
    ) -> Result<NodeId> {
        let a_node = self.graph.get_node(a)
            .ok_or(GraphBlasError::InvalidValue)?;
        let b_node = self.graph.get_node(b)
            .ok_or(GraphBlasError::InvalidValue)?;

        let output_shape = Shape::ewise(&a_node.output_shape, &b_node.output_shape)
            .ok_or(GraphBlasError::DimensionMismatch)?;

        let output_type = a_node.output_type;

        self.graph.add_node(
            Operation::EWiseMult { binary_op },
            vec![a, b],
            output_type,
            output_shape,
        )
    }

    /// Apply unary operator
    pub fn apply(
        &mut self,
        a: NodeId,
        unary_op: UnaryOpKind,
    ) -> Result<NodeId> {
        let a_node = self.graph.get_node(a)
            .ok_or(GraphBlasError::InvalidValue)?;

        self.graph.add_node(
            Operation::Apply { unary_op },
            vec![a],
            a_node.output_type,
            a_node.output_shape.clone(),
        )
    }

    /// Apply binary operator with bound left scalar
    pub fn apply_binary_left(
        &mut self,
        scalar: ScalarValue,
        a: NodeId,
        binary_op: BinaryOpKind,
    ) -> Result<NodeId> {
        let a_node = self.graph.get_node(a)
            .ok_or(GraphBlasError::InvalidValue)?;

        self.graph.add_node(
            Operation::ApplyBinaryLeft { binary_op, scalar },
            vec![a],
            a_node.output_type,
            a_node.output_shape.clone(),
        )
    }

    /// Apply binary operator with bound right scalar
    pub fn apply_binary_right(
        &mut self,
        a: NodeId,
        scalar: ScalarValue,
        binary_op: BinaryOpKind,
    ) -> Result<NodeId> {
        let a_node = self.graph.get_node(a)
            .ok_or(GraphBlasError::InvalidValue)?;

        self.graph.add_node(
            Operation::ApplyBinaryRight { binary_op, scalar },
            vec![a],
            a_node.output_type,
            a_node.output_shape.clone(),
        )
    }

    /// Select elements based on predicate
    pub fn select(
        &mut self,
        a: NodeId,
        predicate: SelectOp,
    ) -> Result<NodeId> {
        let a_node = self.graph.get_node(a)
            .ok_or(GraphBlasError::InvalidValue)?;

        self.graph.add_node(
            Operation::Select { predicate },
            vec![a],
            a_node.output_type,
            a_node.output_shape.clone(),
        )
    }

    /// Transpose matrix
    pub fn transpose(&mut self, a: NodeId) -> Result<NodeId> {
        let a_node = self.graph.get_node(a)
            .ok_or(GraphBlasError::InvalidValue)?;

        let output_shape = a_node.output_shape.transpose()
            .ok_or(GraphBlasError::InvalidValue)?;

        self.graph.add_node(
            Operation::Transpose,
            vec![a],
            a_node.output_type,
            output_shape,
        )
    }

    /// Mark a node as an output
    pub fn output(&mut self, node: NodeId) -> Result<NodeId> {
        let node_data = self.graph.get_node(node)
            .ok_or(GraphBlasError::InvalidValue)?;

        let output_id = self.graph.add_node(
            Operation::Output,
            vec![node],
            node_data.output_type,
            node_data.output_shape.clone(),
        )?;

        self.graph.add_output(output_id)?;
        Ok(output_id)
    }

    /// Get the underlying IR graph
    pub fn graph(&self) -> &IRGraph {
        &self.graph
    }

    /// Consume the builder and return the graph
    pub fn build(self) -> IRGraph {
        self.graph
    }

    /// Get a node by name
    pub fn get_by_name(&self, name: &str) -> Option<NodeId> {
        self.named_nodes.get(name).copied()
    }
}

impl Default for GraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper for creating common semirings
pub mod semirings {
    use super::*;

    /// Plus-Times semiring (standard linear algebra)
    pub fn plus_times(elem_type: ScalarType) -> SemiringOp {
        let identity = match elem_type {
            ScalarType::Float32 => ScalarValue::Float32(0.0),
            ScalarType::Float64 => ScalarValue::Float64(0.0),
            ScalarType::Int32 => ScalarValue::Int32(0),
            ScalarType::Int64 => ScalarValue::Int64(0),
            _ => ScalarValue::Int32(0),
        };

        SemiringOp {
            add_op: MonoidOp {
                binary_op: BinaryOpKind::Add,
                identity,
            },
            mul_op: BinaryOpKind::Mul,
        }
    }

    /// Min-Plus semiring (shortest path)
    pub fn min_plus(elem_type: ScalarType) -> SemiringOp {
        let identity = match elem_type {
            ScalarType::Float32 => ScalarValue::Float32(f32::INFINITY),
            ScalarType::Float64 => ScalarValue::Float64(f64::INFINITY),
            ScalarType::Int32 => ScalarValue::Int32(i32::MAX),
            ScalarType::Int64 => ScalarValue::Int64(i64::MAX),
            _ => ScalarValue::Int32(i32::MAX),
        };

        SemiringOp {
            add_op: MonoidOp {
                binary_op: BinaryOpKind::Min,
                identity,
            },
            mul_op: BinaryOpKind::Add,
        }
    }

    /// Max-Times semiring (maximum weighted path)
    pub fn max_times(elem_type: ScalarType) -> SemiringOp {
        let identity = match elem_type {
            ScalarType::Float32 => ScalarValue::Float32(0.0),
            ScalarType::Float64 => ScalarValue::Float64(0.0),
            ScalarType::Int32 => ScalarValue::Int32(0),
            ScalarType::Int64 => ScalarValue::Int64(0),
            _ => ScalarValue::Int32(0),
        };

        SemiringOp {
            add_op: MonoidOp {
                binary_op: BinaryOpKind::Max,
                identity,
            },
            mul_op: BinaryOpKind::Mul,
        }
    }

    /// Or-And semiring (Boolean logic)
    pub fn or_and() -> SemiringOp {
        SemiringOp {
            add_op: MonoidOp {
                binary_op: BinaryOpKind::Or,
                identity: ScalarValue::Bool(false),
            },
            mul_op: BinaryOpKind::And,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_create() {
        let builder = GraphBuilder::new();
        assert_eq!(builder.graph().nodes().len(), 0);
    }

    #[test]
    fn test_input_matrix() {
        let mut builder = GraphBuilder::new();
        let a = builder
            .input_matrix("A", ScalarType::Float64, Shape::matrix(10, 20))
            .unwrap();

        assert_eq!(a, 0);
        assert_eq!(builder.graph().inputs().len(), 1);
        assert_eq!(builder.get_by_name("A"), Some(0));
    }

    #[test]
    fn test_matmul() {
        let mut builder = GraphBuilder::new();

        let a = builder
            .input_matrix("A", ScalarType::Float64, Shape::matrix(10, 20))
            .unwrap();
        let b = builder
            .input_matrix("B", ScalarType::Float64, Shape::matrix(20, 30))
            .unwrap();

        let semiring = semirings::plus_times(ScalarType::Float64);
        let c = builder.matmul(a, b, semiring).unwrap();

        let c_node = builder.graph().get_node(c).unwrap();
        assert_eq!(c_node.output_shape, Shape::matrix(10, 30));
    }

    #[test]
    fn test_symbolic_matmul() {
        let mut builder = GraphBuilder::new();

        let a = builder
            .input_matrix("A", ScalarType::Float64, Shape::symbolic_matrix("m", "n"))
            .unwrap();
        let b = builder
            .input_matrix("B", ScalarType::Float64, Shape::symbolic_matrix("n", "k"))
            .unwrap();

        let semiring = semirings::plus_times(ScalarType::Float64);
        let c = builder.matmul(a, b, semiring).unwrap();

        let c_node = builder.graph().get_node(c).unwrap();
        assert_eq!(c_node.output_shape, Shape::symbolic_matrix("m", "k"));
    }

    #[test]
    fn test_ewise_operations() {
        let mut builder = GraphBuilder::new();

        let a = builder
            .input_matrix("A", ScalarType::Float64, Shape::matrix(10, 20))
            .unwrap();
        let b = builder
            .input_matrix("B", ScalarType::Float64, Shape::matrix(10, 20))
            .unwrap();

        let c = builder.ewise_add(a, b, BinaryOpKind::Add).unwrap();
        let d = builder.ewise_mult(c, b, BinaryOpKind::Mul).unwrap();

        assert_eq!(builder.graph().nodes().len(), 4);
        let d_node = builder.graph().get_node(d).unwrap();
        assert_eq!(d_node.output_shape, Shape::matrix(10, 20));
    }

    #[test]
    fn test_apply_operations() {
        let mut builder = GraphBuilder::new();

        let a = builder
            .input_matrix("A", ScalarType::Float64, Shape::matrix(10, 20))
            .unwrap();

        let b = builder.apply(a, UnaryOpKind::Abs).unwrap();
        let _c = builder.apply_binary_right(
            b,
            ScalarValue::Float64(2.0),
            BinaryOpKind::Mul,
        ).unwrap();

        assert_eq!(builder.graph().nodes().len(), 3);
    }

    #[test]
    fn test_transpose() {
        let mut builder = GraphBuilder::new();

        let a = builder
            .input_matrix("A", ScalarType::Float64, Shape::matrix(10, 20))
            .unwrap();

        let b = builder.transpose(a).unwrap();

        let b_node = builder.graph().get_node(b).unwrap();
        assert_eq!(b_node.output_shape, Shape::matrix(20, 10));
    }

    #[test]
    fn test_dimension_mismatch() {
        let mut builder = GraphBuilder::new();

        let a = builder
            .input_matrix("A", ScalarType::Float64, Shape::matrix(10, 20))
            .unwrap();
        let b = builder
            .input_matrix("B", ScalarType::Float64, Shape::matrix(15, 30))
            .unwrap();

        let semiring = semirings::plus_times(ScalarType::Float64);
        let result = builder.matmul(a, b, semiring);

        assert!(result.is_err());
    }
}
