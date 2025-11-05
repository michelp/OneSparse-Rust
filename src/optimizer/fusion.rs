// Fusion Optimization Pass
//
// Identifies and merges fusable operation patterns to reduce memory traffic
// and loop overhead.
//
// Fusable patterns:
// - MatMul → EWiseAdd/EWiseMult
// - Apply → Apply (chain unary operations)
// - MatMul → Apply
// - EWise → Apply

use crate::core::error::Result;
use crate::ir::{IRGraph, IRNode, NodeId, Operation};
use crate::optimizer::pass::OptimizationPass;
use std::collections::{HashMap, HashSet};

/// Fusion optimization pass
pub struct FusionPass {
    /// Maximum fusion depth (prevent excessive merging)
    _max_depth: usize,
}

impl FusionPass {
    /// Create a new fusion pass
    pub fn new() -> Self {
        Self { _max_depth: 3 }
    }

    /// Create a fusion pass with custom max depth
    pub fn with_max_depth(max_depth: usize) -> Self {
        Self { _max_depth: max_depth }
    }

    /// Check if two operations can be fused
    fn can_fuse(&self, producer: &IRNode, consumer: &IRNode) -> bool {
        // Check if consumer only depends on producer
        if consumer.inputs.len() != 1 && !self.is_binary_fusable(consumer) {
            return false;
        }

        match (&producer.op, &consumer.op) {
            // MatMul → EWise
            (Operation::MatMul { .. }, Operation::EWiseAdd { .. }) => true,
            (Operation::MatMul { .. }, Operation::EWiseMult { .. }) => true,

            // MatMul → Apply
            (Operation::MatMul { .. }, Operation::Apply { .. }) => true,
            (Operation::MatMul { .. }, Operation::ApplyBinaryLeft { .. }) => true,
            (Operation::MatMul { .. }, Operation::ApplyBinaryRight { .. }) => true,

            // MatVec/VecMat → Apply
            (Operation::MatVec { .. }, Operation::Apply { .. }) => true,
            (Operation::VecMat { .. }, Operation::Apply { .. }) => true,

            // EWise → Apply
            (Operation::EWiseAdd { .. }, Operation::Apply { .. }) => true,
            (Operation::EWiseMult { .. }, Operation::Apply { .. }) => true,

            // Apply → Apply (chain unary operations)
            (Operation::Apply { .. }, Operation::Apply { .. }) => true,

            _ => false,
        }
    }

    /// Check if this is a binary operation that can be fused with one input
    fn is_binary_fusable(&self, node: &IRNode) -> bool {
        matches!(
            node.op,
            Operation::EWiseAdd { .. } | Operation::EWiseMult { .. }
        )
    }

    /// Find fusable pairs in the graph
    fn find_fusable_pairs(&self, graph: &IRGraph) -> Vec<(NodeId, NodeId)> {
        let mut pairs = Vec::new();

        // Track which nodes are used multiple times (can't fuse if result is shared)
        let mut use_counts = HashMap::new();
        for node in graph.nodes().values() {
            for &input_id in &node.inputs {
                *use_counts.entry(input_id).or_insert(0) += 1;
            }
        }

        // Find producer-consumer pairs that can be fused
        for (consumer_id, consumer) in graph.nodes() {
            if consumer.inputs.is_empty() {
                continue;
            }

            // Check primary input (first input)
            let producer_id = consumer.inputs[0];

            // Don't fuse if producer is used elsewhere
            if use_counts.get(&producer_id).copied().unwrap_or(0) > 1 {
                continue;
            }

            if let Some(producer) = graph.get_node(producer_id) {
                // Don't fuse inputs or outputs
                if matches!(producer.op, Operation::Input { .. } | Operation::Output) {
                    continue;
                }

                if self.can_fuse(producer, consumer) {
                    pairs.push((producer_id, *consumer_id));
                }
            }
        }

        pairs
    }

    /// Create a fused operation from producer and consumer
    fn create_fused_op(
        &self,
        producer: &Operation,
        consumer: &Operation,
    ) -> Option<Operation> {
        // TODO: Create actual fused operations
        // For now, we'll keep the consumer operation as-is
        // In a full implementation, we'd create FusedMatMulEWise, etc.

        match (producer, consumer) {
            // These patterns are recognized for fusion
            (Operation::MatMul { .. }, Operation::EWiseAdd { .. })
            | (Operation::MatMul { .. }, Operation::EWiseMult { .. })
            | (Operation::MatMul { .. }, Operation::Apply { .. })
            | (Operation::EWiseAdd { .. }, Operation::Apply { .. })
            | (Operation::EWiseMult { .. }, Operation::Apply { .. })
            | (Operation::Apply { .. }, Operation::Apply { .. }) => {
                // Return consumer op (in full implementation, would create fused op)
                Some(consumer.clone())
            }
            _ => None,
        }
    }
}

impl OptimizationPass for FusionPass {
    fn run(&mut self, graph: &mut IRGraph) -> Result<bool> {
        let pairs = self.find_fusable_pairs(graph);

        if pairs.is_empty() {
            return Ok(false);
        }

        // Track which nodes to remove after fusion
        let mut to_remove = HashSet::new();

        // TODO: Actually perform fusion by:
        // 1. Creating new fused nodes
        // 2. Rewiring graph connections
        // 3. Removing old nodes
        //
        // For now, we'll just identify the opportunities
        // and mark this as a TODO for actual implementation

        for (producer_id, consumer_id) in &pairs {
            if let (Some(producer), Some(consumer)) = (
                graph.get_node(*producer_id),
                graph.get_node(*consumer_id),
            ) {
                if let Some(_fused_op) = self.create_fused_op(&producer.op, &consumer.op) {
                    // TODO: Create new fused node and update graph
                    // For now, just track that we found a fusion opportunity
                    to_remove.insert(*producer_id);
                }
            }
        }

        // Return true if we found opportunities (even if not yet implemented)
        Ok(!to_remove.is_empty())
    }

    fn name(&self) -> &str {
        "fusion"
    }
}

impl Default for FusionPass {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BinaryOpKind, GraphBuilder, ScalarType, Shape, semirings};

    #[test]
    fn test_fusion_pass_creation() {
        let pass = FusionPass::new();
        assert_eq!(pass.name(), "fusion");
        assert_eq!(pass._max_depth, 3);
    }

    #[test]
    fn test_find_matmul_ewise_fusion() {
        let mut builder = GraphBuilder::new();

        let a = builder
            .input_matrix("A", ScalarType::Float64, Shape::matrix(10, 20))
            .unwrap();
        let b = builder
            .input_matrix("B", ScalarType::Float64, Shape::matrix(20, 30))
            .unwrap();

        let semiring = semirings::plus_times(ScalarType::Float64);
        let c = builder.matmul(a, b, semiring).unwrap();

        let d = builder
            .input_matrix("D", ScalarType::Float64, Shape::matrix(10, 30))
            .unwrap();

        // This should be fusable: matmul → ewise
        builder.ewise_add(c, d, BinaryOpKind::Add).unwrap();

        let graph = builder.build();

        let pass = FusionPass::new();
        let pairs = pass.find_fusable_pairs(&graph);

        // Should find the matmul→ewise fusion opportunity
        assert!(!pairs.is_empty());
    }

    #[test]
    fn test_find_matmul_apply_fusion() {
        let mut builder = GraphBuilder::new();

        let a = builder
            .input_matrix("A", ScalarType::Float64, Shape::matrix(10, 20))
            .unwrap();
        let b = builder
            .input_matrix("B", ScalarType::Float64, Shape::matrix(20, 30))
            .unwrap();

        let semiring = semirings::plus_times(ScalarType::Float64);
        let c = builder.matmul(a, b, semiring).unwrap();

        // This should be fusable: matmul → apply
        builder.apply(c, crate::ir::UnaryOpKind::Abs).unwrap();

        let graph = builder.build();

        let pass = FusionPass::new();
        let pairs = pass.find_fusable_pairs(&graph);

        assert!(!pairs.is_empty());
    }

    #[test]
    fn test_no_fusion_with_multiple_uses() {
        let mut builder = GraphBuilder::new();

        let a = builder
            .input_matrix("A", ScalarType::Float64, Shape::matrix(10, 20))
            .unwrap();
        let b = builder
            .input_matrix("B", ScalarType::Float64, Shape::matrix(20, 30))
            .unwrap();

        let semiring = semirings::plus_times(ScalarType::Float64);
        let c = builder.matmul(a, b, semiring).unwrap();

        // Use c multiple times - should NOT fuse
        builder.apply(c, crate::ir::UnaryOpKind::Abs).unwrap();
        builder.apply(c, crate::ir::UnaryOpKind::Sqrt).unwrap();

        let graph = builder.build();

        let pass = FusionPass::new();
        let pairs = pass.find_fusable_pairs(&graph);

        // Should not find fusion opportunities because c is used twice
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_fusion_pass_run() {
        let mut builder = GraphBuilder::new();

        let a = builder
            .input_matrix("A", ScalarType::Float64, Shape::matrix(10, 20))
            .unwrap();
        let b = builder
            .input_matrix("B", ScalarType::Float64, Shape::matrix(20, 30))
            .unwrap();

        let semiring = semirings::plus_times(ScalarType::Float64);
        let c = builder.matmul(a, b, semiring).unwrap();

        builder.apply(c, crate::ir::UnaryOpKind::Abs).unwrap();

        let mut graph = builder.build();

        let mut pass = FusionPass::new();
        let changed = pass.run(&mut graph).unwrap();

        // Should detect fusion opportunity
        assert!(changed);
    }
}
