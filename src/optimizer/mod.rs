// Optimizer Module: IR optimization passes

pub mod pass;
pub mod cse;
pub mod fusion;
pub mod format_select;

// Re-exports
pub use pass::OptimizationPass;
pub use cse::CSEPass;
pub use fusion::FusionPass;
pub use format_select::FormatSelectionPass;
