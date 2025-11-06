// Optimizer Module: IR optimization passes

pub mod cse;
pub mod format_select;
pub mod fusion;
pub mod pass;

// Re-exports
pub use cse::CSEPass;
pub use format_select::FormatSelectionPass;
pub use fusion::FusionPass;
pub use pass::OptimizationPass;
