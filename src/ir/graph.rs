// IR Computation Graph
//
// Represents a dataflow graph of operations

use crate::core::error::{GraphBlasError, Result};
use crate::ir::node::{IRNode, NodeId, Operation};
use crate::ir::shape::Shape;
use crate::ir::types::IRType;
use std::collections::HashMap;

/// Computation graph
#[derive(Debug, Clone)]
pub struct IRGraph {
    /// Nodes indexed by ID
    nodes: HashMap<NodeId, IRNode>,
    /// Next available node ID
    next_id: NodeId,
    /// Input nodes
    inputs: Vec<NodeId>,
    /// Output nodes
    outputs: Vec<NodeId>,
}

impl IRGraph {
    /// Create a new empty graph
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            next_id: 0,
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    /// Add a node to the graph
    pub fn add_node(
        &mut self,
        op: Operation,
        inputs: Vec<NodeId>,
        output_type: IRType,
        output_shape: Shape,
    ) -> Result<NodeId> {
        // Validate inputs exist
        for &input_id in &inputs {
            if !self.nodes.contains_key(&input_id) {
                return Err(GraphBlasError::InvalidValue);
            }
        }

        let id = self.next_id;
        self.next_id += 1;

        let node = IRNode {
            id,
            op,
            inputs,
            output_type,
            output_shape,
        };

        self.nodes.insert(id, node);
        Ok(id)
    }

    /// Get a node by ID
    pub fn get_node(&self, id: NodeId) -> Option<&IRNode> {
        self.nodes.get(&id)
    }

    /// Get mutable reference to a node
    pub fn get_node_mut(&mut self, id: NodeId) -> Option<&mut IRNode> {
        self.nodes.get_mut(&id)
    }

    /// Get all nodes
    pub fn nodes(&self) -> &HashMap<NodeId, IRNode> {
        &self.nodes
    }

    /// Update the inputs of a node
    pub fn update_node_inputs(&mut self, id: NodeId, new_inputs: Vec<NodeId>) -> Result<()> {
        if let Some(node) = self.nodes.get_mut(&id) {
            node.inputs = new_inputs;
            Ok(())
        } else {
            Err(GraphBlasError::InvalidValue)
        }
    }

    /// Mark a node as an input
    pub fn add_input(&mut self, id: NodeId) -> Result<()> {
        if !self.nodes.contains_key(&id) {
            return Err(GraphBlasError::InvalidValue);
        }
        if !self.inputs.contains(&id) {
            self.inputs.push(id);
        }
        Ok(())
    }

    /// Mark a node as an output
    pub fn add_output(&mut self, id: NodeId) -> Result<()> {
        if !self.nodes.contains_key(&id) {
            return Err(GraphBlasError::InvalidValue);
        }
        if !self.outputs.contains(&id) {
            self.outputs.push(id);
        }
        Ok(())
    }

    /// Get input nodes
    pub fn inputs(&self) -> &[NodeId] {
        &self.inputs
    }

    /// Get output nodes
    pub fn outputs(&self) -> &[NodeId] {
        &self.outputs
    }

    /// Get topologically sorted node IDs
    pub fn topological_order(&self) -> Result<Vec<NodeId>> {
        let mut visited = HashMap::new();
        let mut order = Vec::new();

        // Visit all nodes in ID order for deterministic results
        let mut node_ids: Vec<_> = self.nodes.keys().copied().collect();
        node_ids.sort();

        for &node_id in &node_ids {
            self.dfs_topo(node_id, &mut visited, &mut order)?;
        }

        // DFS post-order gives us reverse topological order, so reverse it
        // Actually, we add nodes after their dependencies, so no reverse needed
        Ok(order)
    }

    fn dfs_topo(
        &self,
        node_id: NodeId,
        visited: &mut HashMap<NodeId, bool>,
        order: &mut Vec<NodeId>,
    ) -> Result<()> {
        if let Some(&in_progress) = visited.get(&node_id) {
            if in_progress {
                return Err(GraphBlasError::InvalidValue); // Cycle detected
            }
            return Ok(()); // Already visited
        }

        visited.insert(node_id, true);

        if let Some(node) = self.nodes.get(&node_id) {
            for &input_id in &node.inputs {
                self.dfs_topo(input_id, visited, order)?;
            }
        }

        visited.insert(node_id, false);
        order.push(node_id);
        Ok(())
    }
}

impl Default for IRGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::node::StorageFormat;
    use crate::ir::types::ScalarType;

    #[test]
    fn test_empty_graph() {
        let graph = IRGraph::new();
        assert_eq!(graph.nodes().len(), 0);
    }

    #[test]
    fn test_add_node() {
        let mut graph = IRGraph::new();

        let input1 = graph
            .add_node(
                Operation::Input {
                    name: "A".to_string(),
                    format: StorageFormat::Any,
                },
                vec![],
                IRType::matrix(ScalarType::Float64, 2, 2),
                Shape::symbolic_matrix("m", "n"),
            )
            .unwrap();

        assert_eq!(input1, 0);
        assert_eq!(graph.nodes().len(), 1);
    }

    #[test]
    fn test_topological_order() {
        let mut graph = IRGraph::new();

        let a = graph
            .add_node(
                Operation::Input {
                    name: "A".to_string(),
                    format: StorageFormat::Any,
                },
                vec![],
                IRType::matrix(ScalarType::Float64, 2, 2),
                Shape::matrix(10, 10),
            )
            .unwrap();

        let _b = graph
            .add_node(
                Operation::Transpose,
                vec![a],
                IRType::matrix(ScalarType::Float64, 2, 2),
                Shape::matrix(10, 10),
            )
            .unwrap();

        let order = graph.topological_order().unwrap();
        assert_eq!(order.len(), 2);
        assert_eq!(order[0], 0); // Input first
        assert_eq!(order[1], 1); // Transpose second
    }
}
