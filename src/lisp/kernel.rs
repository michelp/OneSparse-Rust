// Kernel Registry
//
// Manages user-defined kernels with lazy compilation and caching

use crate::compiler::backend::{Backend, CompiledFunction};
use crate::compiler::cranelift_backend::CraneliftBackend;
use crate::core::error::{GraphBlasError, Result};
use crate::ir::{IRGraph, Shape};
use crate::lisp::ast::*;
use crate::lisp::compiler::Compiler;
use crate::lisp::types::Type;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Kernel definition
#[derive(Debug, Clone)]
pub struct KernelDef {
    /// Kernel name
    pub name: String,
    /// Parameter list
    pub params: Vec<Param>,
    /// Body expression
    pub body: Expr,
}

impl KernelDef {
    pub fn new(name: String, params: Vec<Param>, body: Expr) -> Self {
        Self { name, params, body }
    }
}

/// Compiled kernel with metadata
struct CompiledKernel {
    /// The kernel definition
    definition: KernelDef,
    /// Compiled function (None if not yet compiled)
    function: Option<Arc<CompiledFunction>>,
    /// IR graph for this kernel
    graph: Option<IRGraph>,
}

/// Kernel registry with lazy compilation
pub struct KernelRegistry {
    /// Backend for JIT compilation
    backend: Arc<CraneliftBackend>,
    /// Registered kernels
    kernels: Mutex<HashMap<String, CompiledKernel>>,
}

impl KernelRegistry {
    /// Create a new kernel registry
    pub fn new() -> Self {
        Self {
            backend: Arc::new(CraneliftBackend::new().expect("Failed to create Cranelift backend")),
            kernels: Mutex::new(HashMap::new()),
        }
    }

    /// Register a kernel definition
    pub fn register(&self, definition: KernelDef) -> Result<()> {
        let mut kernels = self.kernels.lock().unwrap();

        kernels.insert(
            definition.name.clone(),
            CompiledKernel {
                definition,
                function: None,
                graph: None,
            },
        );

        Ok(())
    }

    /// Register a kernel from a defkernel form
    pub fn register_form(&self, form: &Form) -> Result<()> {
        match form {
            Form::DefKernel { name, params, body } => {
                let definition = KernelDef::new(name.clone(), params.clone(), body.clone());
                self.register(definition)
            }
            _ => Err(GraphBlasError::InvalidValue),
        }
    }

    /// Get a kernel definition by name
    pub fn get_definition(&self, name: &str) -> Option<KernelDef> {
        let kernels = self.kernels.lock().unwrap();
        kernels.get(name).map(|k| k.definition.clone())
    }

    /// Compile a kernel (if not already compiled)
    pub fn compile(&self, name: &str, input_types: &[Type]) -> Result<Arc<CompiledFunction>> {
        let mut kernels = self.kernels.lock().unwrap();

        let kernel = kernels
            .get_mut(name)
            .ok_or(GraphBlasError::InvalidValue)?;

        // If already compiled, return cached version
        if let Some(ref func) = kernel.function {
            return Ok(func.clone());
        }

        // Compile the kernel
        let definition = kernel.definition.clone();
        let (graph, function) = self.compile_kernel(&definition, input_types)?;

        // Cache the compiled function
        kernel.function = Some(Arc::new(function));
        kernel.graph = Some(graph);

        Ok(kernel.function.as_ref().unwrap().clone())
    }

    /// Internal: compile a kernel definition to IR and JIT code
    fn compile_kernel(
        &self,
        definition: &KernelDef,
        input_types: &[Type],
    ) -> Result<(IRGraph, CompiledFunction)> {
        // Validate parameter count
        if definition.params.len() != input_types.len() {
            return Err(GraphBlasError::DimensionMismatch);
        }

        // Create compiler
        let mut compiler = Compiler::new();

        // Bind input parameters to IR nodes
        for (param, ty) in definition.params.iter().zip(input_types.iter()) {
            let node_id = match ty {
                Type::Scalar(scalar_type) => compiler.builder_mut().input_scalar(
                    &param.name,
                    *scalar_type,
                )?,
                Type::Vector(scalar_type) => compiler.builder_mut().input_vector(
                    &param.name,
                    *scalar_type,
                    Shape::symbolic_vector(&param.name),
                )?,
                Type::Matrix(scalar_type) => compiler.builder_mut().input_matrix(
                    &param.name,
                    *scalar_type,
                    Shape::symbolic_matrix(&param.name, &param.name),
                )?,
                Type::Unknown => return Err(GraphBlasError::InvalidValue),
            };

            compiler.bind_input(param.name.clone(), node_id, ty.clone());
        }

        // Compile the body
        let _output_node = compiler.compile_expr(&definition.body)?;

        // Build the IR graph
        let graph = compiler.into_graph();

        // Compile to native code
        let function = self.backend.compile(&graph)?;

        Ok((graph, function))
    }

    /// Execute a kernel with given inputs
    pub fn execute(
        &self,
        name: &str,
        inputs: &[*const ()],
        outputs: &[*mut ()],
        input_types: &[Type],
    ) -> Result<()> {
        // Compile if needed
        let function = self.compile(name, input_types)?;

        // Execute
        function.execute(inputs, outputs)
    }

    /// List all registered kernels
    pub fn list_kernels(&self) -> Vec<String> {
        let kernels = self.kernels.lock().unwrap();
        kernels.keys().cloned().collect()
    }

    /// Check if a kernel is registered
    pub fn has_kernel(&self, name: &str) -> bool {
        let kernels = self.kernels.lock().unwrap();
        kernels.contains_key(name)
    }

    /// Clear all kernels
    pub fn clear(&self) {
        let mut kernels = self.kernels.lock().unwrap();
        kernels.clear();
    }
}

impl Default for KernelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

lazy_static::lazy_static! {
    static ref GLOBAL_REGISTRY: KernelRegistry = KernelRegistry::new();
}

/// Get the global kernel registry
pub fn global_registry() -> &'static KernelRegistry {
    &GLOBAL_REGISTRY
}

/// Register a kernel in the global registry
pub fn register_kernel(definition: KernelDef) -> Result<()> {
    global_registry().register(definition)
}

/// Register a kernel from a form in the global registry
pub fn register_kernel_form(form: &Form) -> Result<()> {
    global_registry().register_form(form)
}

/// Execute a kernel from the global registry
pub fn execute_kernel(
    name: &str,
    inputs: &[*const ()],
    outputs: &[*mut ()],
    input_types: &[Type],
) -> Result<()> {
    global_registry().execute(name, inputs, outputs, input_types)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::ScalarType;

    #[test]
    fn test_register_kernel() {
        let registry = KernelRegistry::new();

        let def = KernelDef::new(
            "test".to_string(),
            vec![Param::new("x".to_string())],
            Expr::Variable("x".to_string()),
        );

        registry.register(def).unwrap();
        assert!(registry.has_kernel("test"));
    }

    #[test]
    fn test_list_kernels() {
        let registry = KernelRegistry::new();

        let def1 = KernelDef::new(
            "kernel1".to_string(),
            vec![Param::new("x".to_string())],
            Expr::Variable("x".to_string()),
        );

        let def2 = KernelDef::new(
            "kernel2".to_string(),
            vec![Param::new("y".to_string())],
            Expr::Variable("y".to_string()),
        );

        registry.register(def1).unwrap();
        registry.register(def2).unwrap();

        let kernels = registry.list_kernels();
        assert_eq!(kernels.len(), 2);
        assert!(kernels.contains(&"kernel1".to_string()));
        assert!(kernels.contains(&"kernel2".to_string()));
    }

    #[test]
    fn test_compile_simple_kernel() {
        let registry = KernelRegistry::new();

        // (defkernel identity [x] x)
        let def = KernelDef::new(
            "identity".to_string(),
            vec![Param::new("x".to_string())],
            Expr::Variable("x".to_string()),
        );

        registry.register(def).unwrap();

        // Compile with matrix input
        let input_types = vec![Type::Matrix(ScalarType::Float64)];
        let result = registry.compile("identity", &input_types);

        assert!(result.is_ok());
    }

    #[test]
    fn test_compile_transpose_kernel() {
        let registry = KernelRegistry::new();

        // (defkernel my-transpose [a] (transpose a))
        let def = KernelDef::new(
            "my-transpose".to_string(),
            vec![Param::new("a".to_string())],
            Expr::FuncCall {
                func: FuncName::Transpose,
                args: vec![Expr::Variable("a".to_string())],
            },
        );

        registry.register(def).unwrap();

        // Compile with matrix input
        let input_types = vec![Type::Matrix(ScalarType::Float64)];
        let result = registry.compile("my-transpose", &input_types);

        assert!(result.is_ok());
    }
}
