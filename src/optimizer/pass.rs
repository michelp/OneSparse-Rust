// Optimization Pass Infrastructure

use crate::core::error::Result;
use crate::ir::IRGraph;

/// Optimization pass trait
pub trait OptimizationPass {
    /// Run the pass on a graph
    /// Returns true if the graph was modified
    fn run(&mut self, graph: &mut IRGraph) -> Result<bool>;

    /// Get pass name
    fn name(&self) -> &str;
}

/// Pass manager for running multiple passes
pub struct PassManager {
    passes: Vec<Box<dyn OptimizationPass>>,
}

impl PassManager {
    pub fn new() -> Self {
        Self { passes: Vec::new() }
    }

    pub fn add_pass(&mut self, pass: Box<dyn OptimizationPass>) {
        self.passes.push(pass);
    }

    pub fn run_all(&mut self, graph: &mut IRGraph) -> Result<()> {
        for pass in &mut self.passes {
            pass.run(graph)?;
        }
        Ok(())
    }
}

impl Default for PassManager {
    fn default() -> Self {
        Self::new()
    }
}
