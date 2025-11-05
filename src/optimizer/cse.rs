// Common Subexpression Elimination Pass
//
// Identifies and eliminates redundant computations by finding nodes
// with identical operations and inputs.
//
// Two nodes are equivalent if:
// - They have the same operation type
// - They have the same inputs (in the same order)
// - They have the same operation parameters (semiring, binary_op, etc.)

use crate::core::error::Result;
use crate::ir::{IRGraph, IRNode, NodeId, Operation};
use crate::optimizer::pass::OptimizationPass;
use std::collections::HashMap;

/// Common subexpression elimination pass
pub struct CSEPass {
    /// Whether to be aggressive (eliminate more) or conservative
    _aggressive: bool,
}

impl CSEPass {
    /// Create a new CSE pass
    pub fn new() -> Self {
        Self { _aggressive: false }
    }

    /// Create an aggressive CSE pass
    pub fn aggressive() -> Self {
        Self { _aggressive: true }
    }

    /// Check if two operations are semantically equivalent
    fn operations_equal(&self, op1: &Operation, op2: &Operation) -> bool {
        use Operation::*;

        match (op1, op2) {
            // Input/Output nodes are never equivalent (have side effects)
            (Input { .. }, _) | (_, Input { .. }) => false,
            (Output, _) | (_, Output) => false,

            // Transpose
            (Transpose, Transpose) => true,

            // MatMul: compare semirings and format
            (MatMul { semiring: s1, format: f1 }, MatMul { semiring: s2, format: f2 }) => {
                s1.add_op == s2.add_op && s1.mul_op == s2.mul_op && f1 == f2
            }

            // MatVec: compare semirings
            (MatVec { semiring: s1 }, MatVec { semiring: s2 }) => {
                s1.add_op == s2.add_op && s1.mul_op == s2.mul_op
            }

            // VecMat: compare semirings
            (VecMat { semiring: s1 }, VecMat { semiring: s2 }) => {
                s1.add_op == s2.add_op && s1.mul_op == s2.mul_op
            }

            // EWiseAdd: compare binary operations
            (EWiseAdd { binary_op: op1 }, EWiseAdd { binary_op: op2 }) => op1 == op2,

            // EWiseMult: compare binary operations
            (EWiseMult { binary_op: op1 }, EWiseMult { binary_op: op2 }) => op1 == op2,

            // Apply: compare unary operations
            (Apply { unary_op: op1 }, Apply { unary_op: op2 }) => op1 == op2,

            // ApplyBinaryLeft: compare binary op and scalar
            (
                ApplyBinaryLeft {
                    binary_op: op1,
                    scalar: s1,
                },
                ApplyBinaryLeft {
                    binary_op: op2,
                    scalar: s2,
                },
            ) => op1 == op2 && s1 == s2,

            // ApplyBinaryRight: compare binary op and scalar
            (
                ApplyBinaryRight {
                    binary_op: op1,
                    scalar: s1,
                },
                ApplyBinaryRight {
                    binary_op: op2,
                    scalar: s2,
                },
            ) => op1 == op2 && s1 == s2,

            // Select: compare predicates
            (Select { predicate: op1 }, Select { predicate: op2 }) => op1 == op2,

            // Extract: no parameters (TODO)
            (Extract {}, Extract {}) => true,

            // Assign: no parameters (TODO)
            (Assign {}, Assign {}) => true,

            // ReduceMatrix: compare monoids and axis
            (
                ReduceMatrix {
                    monoid: m1,
                    axis: a1,
                },
                ReduceMatrix {
                    monoid: m2,
                    axis: a2,
                },
            ) => m1 == m2 && a1 == a2,

            // ReduceVector: compare monoids
            (ReduceVector { monoid: m1 }, ReduceVector { monoid: m2 }) => m1 == m2,

            // ConvertFormat: compare target formats
            (ConvertFormat { to: f1, .. }, ConvertFormat { to: f2, .. }) => f1 == f2,

            // Different operation types are never equal
            _ => false,
        }
    }

    /// Check if two nodes are equivalent
    fn nodes_equivalent(&self, node1: &IRNode, node2: &IRNode) -> bool {
        // Must have same inputs in same order
        if node1.inputs != node2.inputs {
            return false;
        }

        // Must have same output type
        if node1.output_type != node2.output_type {
            return false;
        }

        // Check if operations are equivalent
        self.operations_equal(&node1.op, &node2.op)
    }

    /// Find groups of equivalent nodes
    fn find_equivalence_classes(&self, graph: &IRGraph) -> HashMap<NodeId, NodeId> {
        let mut canonical_map = HashMap::new();
        let topo_order = graph.topological_order().unwrap_or_default();

        // For each node, find if there's an earlier equivalent node
        for (i, &node_id) in topo_order.iter().enumerate() {
            if let Some(node) = graph.get_node(node_id) {
                // Skip nodes that shouldn't be eliminated
                if matches!(node.op, Operation::Input { .. } | Operation::Output) {
                    continue;
                }

                // Check all earlier nodes for equivalence
                let mut found_canonical = None;
                for &earlier_id in topo_order.iter().take(i) {
                    if let Some(earlier_node) = graph.get_node(earlier_id) {
                        // Check if inputs have been remapped
                        let remapped_inputs: Vec<NodeId> = node
                            .inputs
                            .iter()
                            .map(|&input| canonical_map.get(&input).copied().unwrap_or(input))
                            .collect();

                        // Create temporary node with remapped inputs for comparison
                        let mut temp_node = node.clone();
                        temp_node.inputs = remapped_inputs.clone();

                        if self.nodes_equivalent(earlier_node, &temp_node) {
                            found_canonical = Some(earlier_id);
                            break;
                        }
                    }
                }

                // Map this node to its canonical representative
                if let Some(canonical_id) = found_canonical {
                    canonical_map.insert(node_id, canonical_id);
                }
            }
        }

        canonical_map
    }

    /// Count how many nodes would be eliminated
    #[allow(dead_code)]
    fn count_redundant(&self, graph: &IRGraph) -> usize {
        let canonical_map = self.find_equivalence_classes(graph);
        canonical_map.len()
    }
}

impl OptimizationPass for CSEPass {
    fn run(&mut self, graph: &mut IRGraph) -> Result<bool> {
        let canonical_map = self.find_equivalence_classes(graph);

        if canonical_map.is_empty() {
            return Ok(false);
        }

        // Rewire all uses of redundant nodes to canonical nodes
        let node_ids: Vec<NodeId> = graph.nodes().keys().copied().collect();

        for node_id in node_ids {
            let node = match graph.get_node(node_id) {
                Some(n) => n.clone(),
                None => continue,
            };

            // Remap inputs to use canonical nodes
            let new_inputs: Vec<NodeId> = node
                .inputs
                .iter()
                .map(|&input_id| canonical_map.get(&input_id).copied().unwrap_or(input_id))
                .collect();

            // Update if inputs changed
            if new_inputs != node.inputs {
                graph.update_node_inputs(node_id, new_inputs)?;
            }
        }

        Ok(true)
    }

    fn name(&self) -> &str {
        "cse"
    }
}

impl Default for CSEPass {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BinaryOpKind, GraphBuilder, ScalarType, Shape, UnaryOpKind, semirings};

    #[test]
    fn test_cse_pass_creation() {
        let pass = CSEPass::new();
        assert_eq!(pass.name(), "cse");
        assert!(!pass._aggressive);
    }

    #[test]
    fn test_find_duplicate_transpose() {
        let mut builder = GraphBuilder::new();

        let a = builder
            .input_matrix("A", ScalarType::Float64, Shape::matrix(10, 20))
            .unwrap();

        // Create two identical transpose operations
        builder.transpose(a).unwrap();
        builder.transpose(a).unwrap();

        let graph = builder.build();

        let pass = CSEPass::new();
        let canonical_map = pass.find_equivalence_classes(&graph);

        // Should find one redundant transpose
        assert_eq!(canonical_map.len(), 1);
    }

    #[test]
    fn test_find_duplicate_matmul() {
        let mut builder = GraphBuilder::new();

        let a = builder
            .input_matrix("A", ScalarType::Float64, Shape::matrix(10, 20))
            .unwrap();
        let b = builder
            .input_matrix("B", ScalarType::Float64, Shape::matrix(20, 30))
            .unwrap();

        let semiring = semirings::plus_times(ScalarType::Float64);

        // Create two identical matmuls
        builder.matmul(a, b, semiring.clone()).unwrap();
        builder.matmul(a, b, semiring).unwrap();

        let graph = builder.build();

        let pass = CSEPass::new();
        let canonical_map = pass.find_equivalence_classes(&graph);

        // Should find one redundant matmul
        assert_eq!(canonical_map.len(), 1);
    }

    #[test]
    fn test_find_duplicate_apply() {
        let mut builder = GraphBuilder::new();

        let a = builder
            .input_matrix("A", ScalarType::Float64, Shape::matrix(10, 20))
            .unwrap();

        // Create two identical apply operations
        builder.apply(a, UnaryOpKind::Abs).unwrap();
        builder.apply(a, UnaryOpKind::Abs).unwrap();

        let graph = builder.build();

        let pass = CSEPass::new();
        let canonical_map = pass.find_equivalence_classes(&graph);

        // Should find one redundant apply
        assert_eq!(canonical_map.len(), 1);
    }

    #[test]
    fn test_no_cse_different_operations() {
        let mut builder = GraphBuilder::new();

        let a = builder
            .input_matrix("A", ScalarType::Float64, Shape::matrix(10, 20))
            .unwrap();

        // Create different operations on same input
        builder.apply(a, UnaryOpKind::Abs).unwrap();
        builder.apply(a, UnaryOpKind::Sqrt).unwrap();

        let graph = builder.build();

        let pass = CSEPass::new();
        let canonical_map = pass.find_equivalence_classes(&graph);

        // Should not find any redundancy (different operations)
        assert_eq!(canonical_map.len(), 0);
    }

    #[test]
    fn test_cse_chain_elimination() {
        let mut builder = GraphBuilder::new();

        let a = builder
            .input_matrix("A", ScalarType::Float64, Shape::matrix(10, 20))
            .unwrap();

        // Create a chain: A → transpose → abs
        let t1 = builder.transpose(a).unwrap();
        builder.apply(t1, UnaryOpKind::Abs).unwrap();

        // Create same chain again
        let t2 = builder.transpose(a).unwrap();
        builder.apply(t2, UnaryOpKind::Abs).unwrap();

        let graph = builder.build();

        let pass = CSEPass::new();
        let canonical_map = pass.find_equivalence_classes(&graph);

        // Should find two redundancies (duplicate transpose and duplicate apply)
        assert_eq!(canonical_map.len(), 2);
    }

    #[test]
    fn test_cse_ewise_operations() {
        let mut builder = GraphBuilder::new();

        let a = builder
            .input_matrix("A", ScalarType::Float64, Shape::matrix(10, 20))
            .unwrap();
        let b = builder
            .input_matrix("B", ScalarType::Float64, Shape::matrix(10, 20))
            .unwrap();

        // Create two identical ewise operations
        builder.ewise_add(a, b, BinaryOpKind::Add).unwrap();
        builder.ewise_add(a, b, BinaryOpKind::Add).unwrap();

        let graph = builder.build();

        let pass = CSEPass::new();
        let canonical_map = pass.find_equivalence_classes(&graph);

        // Should find one redundant ewise
        assert_eq!(canonical_map.len(), 1);
    }

    #[test]
    fn test_cse_pass_run() {
        let mut builder = GraphBuilder::new();

        let a = builder
            .input_matrix("A", ScalarType::Float64, Shape::matrix(10, 20))
            .unwrap();

        // Create duplicate operations
        builder.transpose(a).unwrap();
        builder.transpose(a).unwrap();

        let mut graph = builder.build();

        let mut pass = CSEPass::new();
        let changed = pass.run(&mut graph).unwrap();

        // Should detect CSE opportunities
        assert!(changed);
    }

    #[test]
    fn test_cse_rewires_graph() {
        let mut builder = GraphBuilder::new();

        let a = builder
            .input_matrix("A", ScalarType::Float64, Shape::matrix(10, 20))
            .unwrap();

        // Create duplicate transpose operations
        let t1 = builder.transpose(a).unwrap();
        let t2 = builder.transpose(a).unwrap();

        // Use both transposes in apply operations
        let abs1 = builder.apply(t1, UnaryOpKind::Abs).unwrap();
        let abs2 = builder.apply(t2, UnaryOpKind::Abs).unwrap();

        let mut graph = builder.build();

        // Before CSE: abs1 uses t1, abs2 uses t2
        let abs1_node_before = graph.get_node(abs1).unwrap();
        let abs2_node_before = graph.get_node(abs2).unwrap();
        assert_eq!(abs1_node_before.inputs[0], t1);
        assert_eq!(abs2_node_before.inputs[0], t2);

        // Run CSE
        let mut pass = CSEPass::new();
        let changed = pass.run(&mut graph).unwrap();
        assert!(changed);

        // After CSE: both abs nodes should use t1 (the canonical transpose)
        let abs1_node_after = graph.get_node(abs1).unwrap();
        let abs2_node_after = graph.get_node(abs2).unwrap();
        assert_eq!(abs1_node_after.inputs[0], t1);
        assert_eq!(abs2_node_after.inputs[0], t1); // Now also uses t1
    }
}
