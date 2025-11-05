// IR Module: Intermediate Representation for JIT compilation

pub mod builder;
pub mod graph;
pub mod node;
pub mod shape;
pub mod types;
pub mod interpreter;

// Re-exports
pub use builder::{GraphBuilder, semirings};
pub use graph::IRGraph;
pub use node::{
    Axis, BinaryOpKind, IRNode, MonoidOp, NodeId, Operation, ScalarValue, SelectOp,
    SemiringOp, StorageFormat, UnaryOpKind,
};
pub use shape::{Dim, Shape};
pub use types::{IRType, ScalarType};
pub use interpreter::{Interpreter, InterpreterValue};
