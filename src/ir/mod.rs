// IR Module: Intermediate Representation for JIT compilation

pub mod builder;
pub mod graph;
pub mod interpreter;
pub mod node;
pub mod shape;
pub mod types;

// Re-exports
pub use builder::{semirings, GraphBuilder};
pub use graph::IRGraph;
pub use interpreter::{Interpreter, InterpreterValue};
pub use node::{
    Axis, BinaryOpKind, IRNode, MonoidOp, NodeId, Operation, ScalarValue, SelectOp, SemiringOp,
    StorageFormat, UnaryOpKind,
};
pub use shape::{Dim, Shape};
pub use types::{IRType, ScalarType};
