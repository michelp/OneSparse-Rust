// Cranelift Backend: JIT compilation using Cranelift
//
// Generates machine code from IR graphs using Cranelift

use crate::compiler::backend::{Backend, BackendFeature, CompiledFunction};
use crate::core::error::{GraphBlasError, Result};
use crate::ir::{IRGraph, IRNode, Operation, ScalarType};
use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use std::collections::HashMap;

/// Cranelift JIT backend
pub struct CraneliftBackend {
    // ISA is determined at compile time, no need to store it
}

impl CraneliftBackend {
    /// Create a new Cranelift backend
    pub fn new() -> Result<Self> {
        // Just verify we can create a JIT module
        let builder = JITBuilder::new(cranelift_module::default_libcall_names())
            .map_err(|_| GraphBlasError::InvalidValue)?;
        let _module = JITModule::new(builder);

        Ok(Self {})
    }

    /// Generate code for an IR graph
    fn generate_code(&self, graph: &IRGraph) -> Result<CompiledFunction> {
        log::trace!("Analyzing IR graph for code generation");

        // Check if this is a simple MatVec operation we can compile
        let topo_order = graph.topological_order()?;
        log::trace!("Graph has {} nodes in topological order", topo_order.len());

        // Look for MatVec operation
        let mut matvec_node = None;
        for node_id in &topo_order {
            if let Some(node) = graph.get_node(*node_id) {
                if matches!(node.op, Operation::MatVec { .. }) {
                    log::debug!("Found MatVec operation, generating specialized SpMV kernel");
                    matvec_node = Some(node);
                    break;
                }
            }
        }

        // If we found a MatVec, generate specialized kernel
        if let Some(node) = matvec_node {
            // Extract semiring from the operation
            if let Operation::MatVec { semiring } = &node.op {
                return self.generate_spmv_kernel(semiring);
            }
        }

        // Otherwise, return stub for now
        log::trace!("No specialized kernel available, returning stub");
        Ok(CompiledFunction::new())
    }

    /// Generate a CSR SpMV kernel: y = A * x
    /// Function signature: (row_ptrs, col_indices, values, x, y, nrows)
    fn generate_spmv_kernel(&self, semiring: &crate::ir::SemiringOp) -> Result<CompiledFunction> {
        log::info!(
            "Generating SpMV kernel with semiring: add={:?}, mul={:?}",
            semiring.add_op.binary_op,
            semiring.mul_op
        );

        // Create JIT module
        log::debug!("Creating Cranelift JIT module");
        let builder = JITBuilder::new(cranelift_module::default_libcall_names())
            .map_err(|_| GraphBlasError::InvalidValue)?;
        let mut module = JITModule::new(builder);
        log::trace!("JIT module created");

        // Create function context
        let mut ctx = module.make_context();
        let mut func_ctx = FunctionBuilderContext::new();

        // Define function signature: void spmv(ptr, ptr, ptr, ptr, ptr, i64)
        log::debug!("Defining function signature");
        let ptr_type = module.target_config().pointer_type();
        ctx.func.signature.params.push(AbiParam::new(ptr_type)); // row_ptrs
        ctx.func.signature.params.push(AbiParam::new(ptr_type)); // col_indices
        ctx.func.signature.params.push(AbiParam::new(ptr_type)); // values
        ctx.func.signature.params.push(AbiParam::new(ptr_type)); // x
        ctx.func.signature.params.push(AbiParam::new(ptr_type)); // y
        ctx.func.signature.params.push(AbiParam::new(types::I64)); // nrows
        log::trace!("Function signature: void spmv(ptr, ptr, ptr, ptr, ptr, i64)");

        // Create function builder
        log::debug!("Building Cranelift IR for SpMV loop structure");
        let mut builder_fn = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);

        // Create blocks
        let entry_block = builder_fn.create_block();
        let loop_header = builder_fn.create_block();
        let loop_body = builder_fn.create_block();
        let inner_loop_header = builder_fn.create_block();
        let inner_loop_body = builder_fn.create_block();
        let inner_loop_exit = builder_fn.create_block();
        let loop_exit = builder_fn.create_block();

        // Append parameters to entry block
        builder_fn.append_block_params_for_function_params(entry_block);
        builder_fn.switch_to_block(entry_block);

        // Get parameters
        let row_ptrs = builder_fn.block_params(entry_block)[0];
        let col_indices = builder_fn.block_params(entry_block)[1];
        let values = builder_fn.block_params(entry_block)[2];
        let x = builder_fn.block_params(entry_block)[3];
        let y = builder_fn.block_params(entry_block)[4];
        let nrows = builder_fn.block_params(entry_block)[5];

        // Initialize outer loop counter i = 0
        let zero = builder_fn.ins().iconst(types::I64, 0);
        builder_fn.ins().jump(loop_header, &[zero]);
        builder_fn.seal_block(entry_block);

        // Outer loop header: while i < nrows
        builder_fn.switch_to_block(loop_header);
        builder_fn.append_block_param(loop_header, types::I64); // i
        let i = builder_fn.block_params(loop_header)[0];

        let cmp = builder_fn.ins().icmp(IntCC::UnsignedLessThan, i, nrows);
        builder_fn.ins().brif(cmp, loop_body, &[], loop_exit, &[]);

        // Outer loop body: process row i
        builder_fn.switch_to_block(loop_body);

        // Load row_start = row_ptrs[i]
        let i_offset = builder_fn.ins().imul_imm(i, 8); // sizeof(usize) = 8
        let row_start_addr = builder_fn.ins().iadd(row_ptrs, i_offset);
        let row_start = builder_fn
            .ins()
            .load(types::I64, MemFlags::trusted(), row_start_addr, 0);

        // Load row_end = row_ptrs[i+1]
        let i_plus_1 = builder_fn.ins().iadd_imm(i, 1);
        let i1_offset = builder_fn.ins().imul_imm(i_plus_1, 8);
        let row_end_addr = builder_fn.ins().iadd(row_ptrs, i1_offset);
        let row_end = builder_fn
            .ins()
            .load(types::I64, MemFlags::trusted(), row_end_addr, 0);

        // Initialize sum with semiring identity
        let identity_val = match &semiring.add_op.identity {
            crate::ir::ScalarValue::Float64(v) => builder_fn.ins().f64const(*v),
            crate::ir::ScalarValue::Float32(v) => builder_fn.ins().f32const(*v),
            crate::ir::ScalarValue::Int64(v) => builder_fn.ins().iconst(types::I64, *v),
            crate::ir::ScalarValue::Int32(v) => builder_fn.ins().iconst(types::I32, *v as i64),
            _ => builder_fn.ins().f64const(0.0), // Fallback
        };
        builder_fn
            .ins()
            .jump(inner_loop_header, &[row_start, identity_val]);

        // Inner loop header: while j < row_end
        builder_fn.switch_to_block(inner_loop_header);
        builder_fn.append_block_param(inner_loop_header, types::I64); // j
        builder_fn.append_block_param(inner_loop_header, types::F64); // sum
        let j = builder_fn.block_params(inner_loop_header)[0];
        let sum = builder_fn.block_params(inner_loop_header)[1];

        let inner_cmp = builder_fn.ins().icmp(IntCC::UnsignedLessThan, j, row_end);
        builder_fn
            .ins()
            .brif(inner_cmp, inner_loop_body, &[], inner_loop_exit, &[sum]);

        // Inner loop body: sum += values[j] * x[col_indices[j]]
        builder_fn.switch_to_block(inner_loop_body);

        // Load col = col_indices[j]
        let j_offset = builder_fn.ins().imul_imm(j, 8);
        let col_addr = builder_fn.ins().iadd(col_indices, j_offset);
        let col = builder_fn
            .ins()
            .load(types::I64, MemFlags::trusted(), col_addr, 0);

        // Load val = values[j]
        let val_offset = builder_fn.ins().imul_imm(j, 8); // sizeof(f64) = 8
        let val_addr = builder_fn.ins().iadd(values, val_offset);
        let val = builder_fn
            .ins()
            .load(types::F64, MemFlags::trusted(), val_addr, 0);

        // Load x_col = x[col]
        let x_offset = builder_fn.ins().imul_imm(col, 8);
        let x_addr = builder_fn.ins().iadd(x, x_offset);
        let x_val = builder_fn
            .ins()
            .load(types::F64, MemFlags::trusted(), x_addr, 0);

        // Compute semiring multiply: val ⊗ x_val
        let prod = self.emit_binary_op(&mut builder_fn, semiring.mul_op, val, x_val, types::F64);

        // Compute semiring add: sum ⊕ prod
        let new_sum = self.emit_binary_op(
            &mut builder_fn,
            semiring.add_op.binary_op,
            sum,
            prod,
            types::F64,
        );

        // Increment j
        let j_next = builder_fn.ins().iadd_imm(j, 1);
        builder_fn.ins().jump(inner_loop_header, &[j_next, new_sum]);

        // Inner loop exit: store sum to y[i]
        builder_fn.switch_to_block(inner_loop_exit);
        builder_fn.append_block_param(inner_loop_exit, types::F64); // final sum
        let final_sum = builder_fn.block_params(inner_loop_exit)[0];

        let y_offset = builder_fn.ins().imul_imm(i, 8);
        let y_addr = builder_fn.ins().iadd(y, y_offset);
        builder_fn
            .ins()
            .store(MemFlags::trusted(), final_sum, y_addr, 0);

        // Increment i and continue outer loop
        let i_next = builder_fn.ins().iadd_imm(i, 1);
        builder_fn.ins().jump(loop_header, &[i_next]);

        // Outer loop exit: return
        builder_fn.switch_to_block(loop_exit);
        builder_fn.ins().return_(&[]);

        // Seal all blocks
        builder_fn.seal_block(loop_header);
        builder_fn.seal_block(loop_body);
        builder_fn.seal_block(inner_loop_header);
        builder_fn.seal_block(inner_loop_body);
        builder_fn.seal_block(inner_loop_exit);
        builder_fn.seal_block(loop_exit);

        // Finalize function
        log::debug!("Finalizing Cranelift IR function");
        builder_fn.finalize();
        log::trace!("Cranelift IR construction complete");

        // Define function in module
        log::debug!("Declaring and defining function in JIT module");
        let id = module
            .declare_function("spmv", Linkage::Export, &ctx.func.signature)
            .map_err(|_| GraphBlasError::InvalidValue)?;

        module
            .define_function(id, &mut ctx)
            .map_err(|_| GraphBlasError::InvalidValue)?;

        // Finalize and get function pointer
        log::info!("Finalizing JIT compilation and generating native code");
        module.finalize_definitions().unwrap();
        let code_ptr = module.get_finalized_function(id);
        log::info!("Native code generated at address: {:p}", code_ptr);

        Ok(CompiledFunction::with_kernel(code_ptr))
    }

    #[allow(dead_code)]
    fn generate_node(
        &self,
        _builder: &mut FunctionBuilder,
        node: &IRNode,
        _value_map: &mut HashMap<usize, Value>,
    ) -> Result<()> {
        // TODO: Implement actual code generation for each operation type
        match &node.op {
            Operation::Input { .. } => {
                // Input values come from function parameters
                // TODO: Map to actual parameters
                Ok(())
            }
            Operation::Output => {
                // Output is handled in return statement
                Ok(())
            }
            Operation::MatMul { .. } => {
                // TODO: Generate matrix multiplication loop
                // This is complex and depends on storage format
                Ok(())
            }
            Operation::EWiseAdd { .. } => {
                // TODO: Generate element-wise addition
                Ok(())
            }
            Operation::EWiseMult { .. } => {
                // TODO: Generate element-wise multiplication
                Ok(())
            }
            Operation::Apply { .. } => {
                // TODO: Generate apply operation
                Ok(())
            }
            Operation::ApplyBinaryLeft { .. } => {
                // TODO: Generate apply with bound left
                Ok(())
            }
            Operation::ApplyBinaryRight { .. } => {
                // TODO: Generate apply with bound right
                Ok(())
            }
            Operation::Select { .. } => {
                // TODO: Generate select operation
                Ok(())
            }
            Operation::Transpose => {
                // TODO: Generate transpose
                Ok(())
            }
            Operation::ConvertFormat { .. } => {
                // TODO: Generate format conversion
                Ok(())
            }
            Operation::MatVec { .. } => {
                // TODO: Generate matrix-vector multiply
                Ok(())
            }
            Operation::VecMat { .. } => {
                // TODO: Generate vector-matrix multiply
                Ok(())
            }
            Operation::Extract { .. } => {
                // TODO: Generate extract
                Ok(())
            }
            Operation::Assign { .. } => {
                // TODO: Generate assign
                Ok(())
            }
            Operation::ReduceMatrix { .. } => {
                // TODO: Generate matrix reduction
                Ok(())
            }
            Operation::ReduceVector { .. } => {
                // TODO: Generate vector reduction
                Ok(())
            }
        }
    }

    #[allow(dead_code)]
    fn scalar_type_to_cranelift_type(&self, scalar_type: ScalarType) -> Type {
        match scalar_type {
            ScalarType::Bool => types::I8,
            ScalarType::Int8 => types::I8,
            ScalarType::Int16 => types::I16,
            ScalarType::Int32 => types::I32,
            ScalarType::Int64 => types::I64,
            ScalarType::Uint8 => types::I8,
            ScalarType::Uint16 => types::I16,
            ScalarType::Uint32 => types::I32,
            ScalarType::Uint64 => types::I64,
            ScalarType::Float32 => types::F32,
            ScalarType::Float64 => types::F64,
        }
    }

    /// Emit Cranelift IR for a binary operation
    ///
    /// Maps high-level BinaryOpKind to appropriate Cranelift instruction
    /// based on the operand type.
    fn emit_binary_op(
        &self,
        builder: &mut FunctionBuilder,
        op: crate::ir::BinaryOpKind,
        lhs: Value,
        rhs: Value,
        val_type: Type,
    ) -> Value {
        use crate::ir::BinaryOpKind;

        match (op, val_type) {
            // Floating point operations
            (BinaryOpKind::Add, types::F64) => builder.ins().fadd(lhs, rhs),
            (BinaryOpKind::Sub, types::F64) => builder.ins().fsub(lhs, rhs),
            (BinaryOpKind::Mul, types::F64) => builder.ins().fmul(lhs, rhs),
            (BinaryOpKind::Div, types::F64) => builder.ins().fdiv(lhs, rhs),
            (BinaryOpKind::Min, types::F64) => builder.ins().fmin(lhs, rhs),
            (BinaryOpKind::Max, types::F64) => builder.ins().fmax(lhs, rhs),

            (BinaryOpKind::Add, types::F32) => builder.ins().fadd(lhs, rhs),
            (BinaryOpKind::Sub, types::F32) => builder.ins().fsub(lhs, rhs),
            (BinaryOpKind::Mul, types::F32) => builder.ins().fmul(lhs, rhs),
            (BinaryOpKind::Div, types::F32) => builder.ins().fdiv(lhs, rhs),
            (BinaryOpKind::Min, types::F32) => builder.ins().fmin(lhs, rhs),
            (BinaryOpKind::Max, types::F32) => builder.ins().fmax(lhs, rhs),

            // Signed integer operations
            (BinaryOpKind::Add, types::I64) => builder.ins().iadd(lhs, rhs),
            (BinaryOpKind::Sub, types::I64) => builder.ins().isub(lhs, rhs),
            (BinaryOpKind::Mul, types::I64) => builder.ins().imul(lhs, rhs),
            (BinaryOpKind::Div, types::I64) => builder.ins().sdiv(lhs, rhs),
            (BinaryOpKind::Min, types::I64) => {
                // Min for signed integers: select(a < b, a, b)
                let cmp = builder.ins().icmp(IntCC::SignedLessThan, lhs, rhs);
                builder.ins().select(cmp, lhs, rhs)
            }
            (BinaryOpKind::Max, types::I64) => {
                // Max for signed integers: select(a > b, a, b)
                let cmp = builder.ins().icmp(IntCC::SignedGreaterThan, lhs, rhs);
                builder.ins().select(cmp, lhs, rhs)
            }

            (BinaryOpKind::Add, types::I32) => builder.ins().iadd(lhs, rhs),
            (BinaryOpKind::Sub, types::I32) => builder.ins().isub(lhs, rhs),
            (BinaryOpKind::Mul, types::I32) => builder.ins().imul(lhs, rhs),
            (BinaryOpKind::Div, types::I32) => builder.ins().sdiv(lhs, rhs),
            (BinaryOpKind::Min, types::I32) => {
                let cmp = builder.ins().icmp(IntCC::SignedLessThan, lhs, rhs);
                builder.ins().select(cmp, lhs, rhs)
            }
            (BinaryOpKind::Max, types::I32) => {
                let cmp = builder.ins().icmp(IntCC::SignedGreaterThan, lhs, rhs);
                builder.ins().select(cmp, lhs, rhs)
            }

            // Boolean operations (for Any-Or semiring)
            (BinaryOpKind::Or, types::I8) => builder.ins().bor(lhs, rhs),
            (BinaryOpKind::And, types::I8) => builder.ins().band(lhs, rhs),

            // Unsupported combinations
            _ => panic!(
                "Unsupported binary operation {:?} for type {:?}",
                op, val_type
            ),
        }
    }
}

impl Backend for CraneliftBackend {
    fn compile(&self, graph: &IRGraph) -> Result<CompiledFunction> {
        log::debug!("Cranelift backend compiling IR graph");
        let result = self.generate_code(graph)?;
        log::debug!("Cranelift compilation complete");
        Ok(result)
    }

    fn supports_feature(&self, feature: BackendFeature) -> bool {
        match feature {
            BackendFeature::SIMD => true,            // Cranelift supports SIMD
            BackendFeature::MultiThreading => false, // Not yet implemented
            BackendFeature::GPU => false,            // Cranelift is CPU-only
        }
    }

    fn name(&self) -> &str {
        "cranelift"
    }
}

impl Default for CraneliftBackend {
    fn default() -> Self {
        Self::new().expect("Failed to create Cranelift backend")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{GraphBuilder, Shape};

    #[test]
    fn test_backend_creation() {
        let backend = CraneliftBackend::new();
        assert!(backend.is_ok());
    }

    #[test]
    fn test_backend_features() {
        let backend = CraneliftBackend::new().unwrap();
        assert_eq!(backend.name(), "cranelift");
        assert!(backend.supports_feature(BackendFeature::SIMD));
        assert!(!backend.supports_feature(BackendFeature::GPU));
    }

    #[test]
    fn test_compile_simple_graph() {
        let backend = CraneliftBackend::new().unwrap();

        let mut builder = GraphBuilder::new();
        let a = builder
            .input_matrix("A", ScalarType::Float64, Shape::matrix(10, 20))
            .unwrap();
        builder.transpose(a).unwrap();

        let graph = builder.build();
        let result = backend.compile(&graph);

        // Should compile without errors (even though it's a stub)
        assert!(result.is_ok());
    }
}
