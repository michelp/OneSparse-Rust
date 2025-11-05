// Format Selection Pass
//
// Chooses optimal sparse matrix storage format for each operation
// and inserts format conversion nodes as needed.
//
// Format selection heuristics:
// - MatMul: CSR for row-major, CSC for column-major
// - MatVec: CSR (efficient row access)
// - VecMat: CSC (efficient column access)
// - EWise: Match formats if possible, CSR default
// - Transpose: Swap CSR ↔ CSC
// - Build/Construction: COO (easy insertion)

use crate::core::error::Result;
use crate::ir::{IRGraph, IRNode, NodeId, Operation, StorageFormat};
use crate::optimizer::pass::OptimizationPass;
use std::collections::HashMap;

/// Format selection optimization pass
pub struct FormatSelectionPass {
    /// Prefer CSR or CSC as default
    default_format: StorageFormat,
}

impl FormatSelectionPass {
    /// Create a new format selection pass
    pub fn new() -> Self {
        Self {
            default_format: StorageFormat::CSR,
        }
    }

    /// Create with custom default format
    pub fn with_default(format: StorageFormat) -> Self {
        Self {
            default_format: format,
        }
    }

    /// Choose optimal format for an operation
    fn choose_format(&self, node: &IRNode, input_formats: &[StorageFormat]) -> StorageFormat {
        match &node.op {
            // Matrix multiplication prefers CSR
            Operation::MatMul { .. } => StorageFormat::CSR,

            // Matrix-vector prefers CSR (row access)
            Operation::MatVec { .. } => StorageFormat::CSR,

            // Vector-matrix prefers CSC (column access)
            Operation::VecMat { .. } => StorageFormat::CSC,

            // Element-wise operations: match first input format if possible
            Operation::EWiseAdd { .. } | Operation::EWiseMult { .. } => {
                if !input_formats.is_empty() && input_formats[0] != StorageFormat::Any {
                    input_formats[0]
                } else {
                    self.default_format
                }
            }

            // Apply operations: preserve input format
            Operation::Apply { .. }
            | Operation::ApplyBinaryLeft { .. }
            | Operation::ApplyBinaryRight { .. } => {
                if !input_formats.is_empty() && input_formats[0] != StorageFormat::Any {
                    input_formats[0]
                } else {
                    self.default_format
                }
            }

            // Transpose swaps CSR ↔ CSC
            Operation::Transpose => {
                if !input_formats.is_empty() {
                    match input_formats[0] {
                        StorageFormat::CSR => StorageFormat::CSC,
                        StorageFormat::CSC => StorageFormat::CSR,
                        StorageFormat::COO => StorageFormat::COO,
                        StorageFormat::Any => self.default_format,
                    }
                } else {
                    self.default_format
                }
            }

            // Select preserves format
            Operation::Select { .. } => {
                if !input_formats.is_empty() && input_formats[0] != StorageFormat::Any {
                    input_formats[0]
                } else {
                    self.default_format
                }
            }

            // Reductions: CSR for matrix→vector
            Operation::ReduceMatrix { .. } => StorageFormat::CSR,
            Operation::ReduceVector { .. } => StorageFormat::Any,

            // Format conversion: use target format
            Operation::ConvertFormat { to, .. } => *to,

            // Input/Output/Extract/Assign: use Any (determined by context)
            Operation::Input { .. }
            | Operation::Output
            | Operation::Extract { .. }
            | Operation::Assign { .. } => StorageFormat::Any,
        }
    }

    /// Determine if a format conversion is needed
    fn needs_conversion(
        &self,
        required_format: StorageFormat,
        actual_format: StorageFormat,
    ) -> bool {
        // No conversion needed if either is Any
        if required_format == StorageFormat::Any || actual_format == StorageFormat::Any {
            return false;
        }

        // Conversion needed if formats don't match
        required_format != actual_format
    }

    /// Assign formats to all nodes in topological order
    fn assign_formats(&self, graph: &IRGraph) -> Result<HashMap<NodeId, StorageFormat>> {
        let mut formats = HashMap::new();
        let topo_order = graph.topological_order()?;

        for &node_id in &topo_order {
            if let Some(node) = graph.get_node(node_id) {
                // Get formats of input nodes
                let input_formats: Vec<StorageFormat> = node
                    .inputs
                    .iter()
                    .filter_map(|&input_id| formats.get(&input_id).copied())
                    .collect();

                // Choose format for this node
                let format = match &node.op {
                    Operation::Input { format, .. } => {
                        // If input format is Any, use default
                        if *format == StorageFormat::Any {
                            self.default_format
                        } else {
                            *format
                        }
                    }
                    Operation::ConvertFormat { to, .. } => *to,
                    _ => self.choose_format(node, &input_formats),
                };

                formats.insert(node_id, format);
            }
        }

        Ok(formats)
    }

    /// Count how many conversion nodes would be needed
    fn count_conversions(&self, graph: &IRGraph, formats: &HashMap<NodeId, StorageFormat>) -> usize {
        let mut count = 0;

        for node in graph.nodes().values() {
            let node_format = formats.get(&node.id).copied().unwrap_or(StorageFormat::Any);

            for &input_id in &node.inputs {
                let input_format = formats.get(&input_id).copied().unwrap_or(StorageFormat::Any);

                if self.needs_conversion(node_format, input_format) {
                    count += 1;
                }
            }
        }

        count
    }
}

impl OptimizationPass for FormatSelectionPass {
    fn run(&mut self, graph: &mut IRGraph) -> Result<bool> {
        // Assign optimal formats to all nodes
        let formats = self.assign_formats(graph)?;

        // Track conversions we've inserted: (from_node, from_format, to_format) -> conversion_node
        let mut conversions: HashMap<(NodeId, StorageFormat, StorageFormat), NodeId> = HashMap::new();

        let mut changed = false;

        // Get list of nodes to process (can't iterate and modify at same time)
        let node_ids: Vec<NodeId> = graph.nodes().keys().copied().collect();

        for node_id in node_ids {
            let node = match graph.get_node(node_id) {
                Some(n) => n.clone(), // Clone to avoid borrow issues
                None => continue,
            };

            let node_format = formats.get(&node_id).copied().unwrap_or(StorageFormat::Any);

            // Check each input
            let mut new_inputs = node.inputs.clone();
            for (idx, &input_id) in node.inputs.iter().enumerate() {
                let input_format = formats.get(&input_id).copied().unwrap_or(StorageFormat::Any);

                if self.needs_conversion(node_format, input_format) {
                    // Check if we already created this conversion
                    let key = (input_id, input_format, node_format);
                    let conversion_node = if let Some(&existing) = conversions.get(&key) {
                        existing
                    } else {
                        // Create new conversion node
                        let input_node = graph.get_node(input_id).unwrap();
                        let conv_id = graph.add_node(
                            Operation::ConvertFormat {
                                from: input_format,
                                to: node_format,
                            },
                            vec![input_id],
                            input_node.output_type,
                            input_node.output_shape.clone(),
                        )?;
                        conversions.insert(key, conv_id);
                        changed = true;
                        conv_id
                    };

                    // Update input to use conversion node
                    new_inputs[idx] = conversion_node;
                }
            }

            // If we modified inputs, update the node
            if new_inputs != node.inputs {
                graph.update_node_inputs(node_id, new_inputs)?;
            }
        }

        Ok(changed)
    }

    fn name(&self) -> &str {
        "format_selection"
    }
}

impl Default for FormatSelectionPass {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{GraphBuilder, ScalarType, Shape, semirings};

    #[test]
    fn test_format_selection_creation() {
        let pass = FormatSelectionPass::new();
        assert_eq!(pass.name(), "format_selection");
        assert_eq!(pass.default_format, StorageFormat::CSR);
    }

    #[test]
    fn test_matmul_format_selection() {
        let mut builder = GraphBuilder::new();

        let a = builder
            .input_matrix("A", ScalarType::Float64, Shape::matrix(10, 20))
            .unwrap();
        let b = builder
            .input_matrix("B", ScalarType::Float64, Shape::matrix(20, 30))
            .unwrap();

        let semiring = semirings::plus_times(ScalarType::Float64);
        let c = builder.matmul(a, b, semiring).unwrap();

        let graph = builder.build();

        let pass = FormatSelectionPass::new();
        let formats = pass.assign_formats(&graph).unwrap();

        // MatMul should prefer CSR
        if let Some(c_node) = graph.get_node(c) {
            let c_format = formats.get(&c_node.id).copied().unwrap();
            assert_eq!(c_format, StorageFormat::CSR);
        }
    }

    #[test]
    fn test_transpose_format_swap() {
        let mut builder = GraphBuilder::new();

        let a = builder
            .input_matrix("A", ScalarType::Float64, Shape::matrix(10, 20))
            .unwrap();

        builder.transpose(a).unwrap();

        let graph = builder.build();

        let pass = FormatSelectionPass::new();
        let formats = pass.assign_formats(&graph).unwrap();

        // Input would be CSR (default), transpose should be CSC
        let transpose_format = formats.values().find(|&&f| f == StorageFormat::CSC);
        assert!(transpose_format.is_some());
    }

    #[test]
    fn test_format_selection_pass_run() {
        let mut builder = GraphBuilder::new();

        let a = builder
            .input_matrix("A", ScalarType::Float64, Shape::matrix(10, 20))
            .unwrap();

        builder.transpose(a).unwrap();

        let mut graph = builder.build();

        let mut pass = FormatSelectionPass::new();
        let changed = pass.run(&mut graph).unwrap();

        // Should detect format changes needed
        // (would need conversion between CSR and CSC)
        assert!(changed);
    }

    #[test]
    fn test_ewise_format_preservation() {
        let mut builder = GraphBuilder::new();

        let a = builder
            .input_matrix("A", ScalarType::Float64, Shape::matrix(10, 20))
            .unwrap();
        let b = builder
            .input_matrix("B", ScalarType::Float64, Shape::matrix(10, 20))
            .unwrap();

        builder.ewise_add(a, b, crate::ir::BinaryOpKind::Add).unwrap();

        let graph = builder.build();

        let pass = FormatSelectionPass::new();
        let formats = pass.assign_formats(&graph).unwrap();

        // Element-wise should preserve input formats
        // All should be CSR (default)
        let all_csr = formats.values().all(|&f| f == StorageFormat::CSR || f == StorageFormat::Any);
        assert!(all_csr);
    }

    #[test]
    fn test_conversion_nodes_inserted() {
        let mut builder = GraphBuilder::new();

        // Create a graph where conversion is needed
        let a = builder
            .input_matrix("A", ScalarType::Float64, Shape::matrix(10, 20))
            .unwrap();

        // Transpose will want CSC format, input defaults to CSR
        builder.transpose(a).unwrap();

        let mut graph = builder.build();
        let initial_node_count = graph.nodes().len();

        // Run the pass
        let mut pass = FormatSelectionPass::new();
        let changed = pass.run(&mut graph).unwrap();

        // Should have made changes
        assert!(changed);

        // Should have added conversion node(s)
        let final_node_count = graph.nodes().len();
        assert!(final_node_count > initial_node_count,
                "Expected conversion nodes to be inserted, initial: {}, final: {}",
                initial_node_count, final_node_count);

        // Check that a ConvertFormat node was created
        let has_convert = graph.nodes().values().any(|node| {
            matches!(node.op, Operation::ConvertFormat { .. })
        });
        assert!(has_convert, "Expected at least one ConvertFormat node");
    }
}
