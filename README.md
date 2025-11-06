# RustSparse

A high-performance sparse matrix library for Rust, implementing the GraphBLAS standard with JIT compilation and a Lisp-based DSL.

[![Tests](https://github.com/michelp/rustsparse/workflows/Tests/badge.svg)](https://github.com/michelp/rustsparse/actions)

## Project Goals

RustSparse aims to provide a modern, safe, and performant implementation of sparse linear algebra operations following the [GraphBLAS](https://graphblas.org/) specification. The project combines several innovative approaches:

1. **GraphBLAS Standard Compliance**: Implement the GraphBLAS API for graph algorithms expressed as sparse linear algebra
2. **JIT Compilation**: Use runtime code generation via Cranelift for optimal performance tailored to specific problem structures
3. **Lisp DSL**: Provide an expressive, interactive language for rapid prototyping and exploration
4. **SuiteSparse Architecture**: Follow proven design patterns from SuiteSparse for unified internal representation
5. **Memory Safety**: Leverage Rust's ownership system for safe sparse matrix operations without garbage collection
6. **Zero-Cost Abstractions**: Maintain performance while providing ergonomic high-level APIs

## Architecture Overview

RustSparse is organized into distinct layers, each with specific responsibilities:

```
┌─────────────────────────────────────────────────────────────┐
│                     User APIs                                │
│  ┌─────────────────┐        ┌──────────────────────┐       │
│  │   Lisp REPL     │        │   Rust Core API      │       │
│  │  (Interactive)  │        │  (Library Interface) │       │
│  └────────┬────────┘        └──────────┬───────────┘       │
└───────────┼────────────────────────────┼───────────────────┘
            │                            │
            ▼                            ▼
┌─────────────────────────────────────────────────────────────┐
│                 Core Layer (Rust)                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Unified Storage: SparseContainer<T>                  │  │
│  │   - Vectors as n×1 matrices                          │  │
│  │   - Scalars as 1×1 matrices                          │  │
│  │   - COO, CSR, CSC formats                            │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Type-Safe Wrappers: Matrix<T>, Vector<T>, Scalar<T> │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              Operations Layer                                │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Unified matmul: mxm, mxv, vxm → single path         │  │
│  │  Element-wise: ewadd, ewmult                         │  │
│  │  Semirings: plus-times, min-plus, max-times          │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│            IR (Intermediate Representation)                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Unified Type: IRType::Tensor(ScalarType, Shape)     │  │
│  │  Graph Builder API                                    │  │
│  │  Shape inference and validation                       │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  Optimizer                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  CSE (Common Subexpression Elimination)              │  │
│  │  Fusion (Operation fusion)                           │  │
│  │  Format Selection (CSR/CSC/COO)                      │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              JIT Compiler (Cranelift)                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Native code generation                              │  │
│  │  Kernel caching (100MB cache)                        │  │
│  │  Runtime optimization                                │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Working Components

### 1. Core Types (`src/core/`)

**Unified Storage Architecture** (following SuiteSparse):
- `SparseContainer<T>`: Internal unified representation
  - Vectors stored as n×1 matrices
  - Scalars stored as 1×1 matrices
  - Supports COO, CSR, CSC sparse formats

- **Type-Safe Wrappers**:
  - `Matrix<T>`: Arbitrary (m, n) matrices
  - `Vector<T>`: Ergonomic wrapper for n×1 matrices
  - `Scalar<T>`: Single value as 1×1 matrix

**Algebraic Structures**:
- `Semiring<T>`: Plus-times, min-plus, max-times
- `Monoid<T>`: Associative operations with identity
- `BinaryOp`, `UnaryOp`: Basic operations

### 2. Operations Layer (`src/ops/`)

**Unified Matrix Multiplication**:
- Single internal implementation for all variants
- `mxm(C, A, B)`: Matrix × Matrix
- `mxv(w, A, u)`: Matrix × Vector
- `vxm(w, u, A)`: Vector × Matrix
- Automatic shape-based dispatch

**Element-wise Operations**:
- `ewadd`: Union-based addition
- `ewmult`: Intersection-based multiplication
- Support for custom binary operators

**Descriptors**:
- Transpose operands
- Replace vs. accumulate semantics
- Mask operations

### 3. IR Layer (`src/ir/`)

**Unified Type System**:
- `IRType::Tensor(ScalarType, Shape)`: Single type for all tensors
- Shape-based reasoning (scalars: 0-d, vectors: 1-d, matrices: 2-d)
- Symbolic and concrete dimensions

**Graph Builder**:
- Fluent API for constructing computation graphs
- Automatic shape inference
- Type-safe operations

**Shape System**:
- `Shape::Scalar`, `Shape::Vector(Dim)`, `Shape::Matrix(Dim, Dim)`
- Symbolic dimensions for generic algorithms
- Compile-time shape validation

### 4. Optimizer (`src/optimizer/`)

**Optimization Passes**:
- **CSE Pass**: Eliminate redundant computations
- **Fusion Pass**: Combine operations (e.g., matmul + apply)
- **Format Selection**: Choose optimal sparse format (CSR/CSC/COO)

**Pass Manager**:
- Composable optimization pipeline
- Configurable pass ordering
- IR graph transformations

### 5. JIT Compiler (`src/compiler/`)

**Cranelift Backend**:
- Native code generation for sparse kernels
- Runtime optimization based on sparsity patterns
- Type-specialized implementations

**Kernel Cache**:
- LRU cache with configurable size (default: 100MB)
- Hash-based kernel lookup
- Automatic cache invalidation

### 6. Lisp DSL (`src/lisp/`)

**Interactive REPL**:
- S-expression based syntax
- Tab completion
- Multi-line input support
- Immediate feedback

**Language Features**:
- **Literals**: Integers, floats, booleans
- **Control Structures**: `if`, `cond`, `while`, `for`, `begin`, `break`, `continue`
- **Vector Operations**: `(vector 1 2 3 4 5)`
- **Matrix Operations**: `(matrix (vector 1 2) (vector 3 4))`
- **Matrix Multiplication**: `mxv`, `vxm`
- **Semirings**: `plus-times`, `min-plus`, `max-times`
- **Utilities**: `assert`, `set-log-level`, `get-log-level`

**Example Lisp Program**:
```lisp
;; Define a sparse matrix (adjacency matrix for graph)
(matrix
  (vector 0 1 5)
  (vector 9 0 2)
  (vector 6 4 0))

;; Shortest path computation using min-plus semiring
(min-plus graph graph)

;; Standard matrix multiplication
(plus-times
  (matrix (vector 1 2) (vector 3 4))
  (matrix (vector 5 6) (vector 7 8)))

;; Control flow
(for i 0 5
  (assert (if (< i 3) true false)))
```

## Getting Started

### Prerequisites

- Rust 1.70+ (latest stable recommended)
- Cargo

### Building

```bash
# Clone the repository
git clone https://github.com/michelp/rustsparse.git
cd rustsparse

# Build the project
cargo build --release

# Run tests
cargo test

# Run Lisp REPL
cargo run --bin lisp
```

### Running Tests

```bash
# All Rust tests (unit + integration)
cargo test

# Specific test suite
cargo test --test jit_spmv

# Lisp test suites
cat tests/lisp_comprehensive.lisp | cargo run --bin lisp
cat tests/lisp_advanced.lisp | cargo run --bin lisp
cat tests/lisp_examples.lisp | cargo run --bin lisp
```

### Usage Examples

#### Rust API

```rust
use rustsparse::core::{Matrix, Vector, Semiring};
use rustsparse::ops::mxv;

// Create a sparse matrix in CSR format
let a = Matrix::<f64>::from_csr(
    3, 3,
    vec![0, 2, 3, 5],           // row_ptrs
    vec![0, 2, 1, 0, 2],        // col_indices
    vec![1.0, 2.0, 3.0, 4.0, 5.0], // values
)?;

// Create a sparse vector
let mut u = Vector::<f64>::new(3)?;
u.indices_mut().extend_from_slice(&[0, 1, 2]);
u.values_mut().extend_from_slice(&[1.0, 2.0, 3.0]);

// Prepare output vector
let mut w = Vector::<f64>::new(3)?;

// Perform matrix-vector multiplication
let semiring = Semiring::plus_times()?;
mxv(&mut w, None, &a, &u, &semiring, None)?;

println!("Result: {:?}", w.values());
```

#### Lisp REPL

```lisp
>> (vector 1 2 3)
#[1 2 3]

>> (matrix (vector 1 0) (vector 0 1))
#<Matrix 2x2>

>> (mxv (matrix (vector 2 3) (vector 4 5)) (vector 10 20))
#[80 140]

>> (plus-times (matrix (vector 1 2) (vector 3 4))
               (matrix (vector 5 6) (vector 7 8)))
#<Matrix 2x2>
  [19 22]
  [43 50]
```

## Current Status

### ✅ Implemented

- **Core Types**: Full sparse matrix/vector/scalar infrastructure with unified storage
- **Sparse Formats**: COO, CSR, CSC support
- **Operations**: Matrix multiplication (all variants), element-wise operations
- **Semirings**: Plus-times, min-plus, max-times
- **IR Layer**: Graph-based intermediate representation with shape inference
- **Optimizer**: CSE, fusion, format selection passes
- **JIT Compilation**: Cranelift backend with kernel caching
- **Lisp DSL**: Full REPL with control structures and matrix operations
- **Type System**: Unified tensor types following SuiteSparse design
- **Testing**: 184 tests (152 unit + 32 integration)
- **CI/CD**: GitHub Actions workflow for automated testing

### 🚧 In Progress

- Advanced semiring operations
- Descriptor full implementation (masks, accumulate)
- Additional sparse formats (bitmap, full)

### 📋 Planned

- **GraphBLAS C API**: Full C FFI compatibility layer
- **GPU Backend**: CUDA/ROCm support via compile targets
- **Distributed Computing**: MPI-based distributed sparse operations
- **Advanced Algorithms**:
  - BFS, DFS, shortest paths
  - PageRank, triangle counting
  - Community detection
- **Python Bindings**: PyO3-based Python interface
- **Performance Benchmarks**: Comprehensive comparison with SuiteSparse
- **Documentation**: API docs, tutorials, algorithm guides

## Design Philosophy

### SuiteSparse Inspiration

RustSparse follows key design principles from [SuiteSparse](https://people.engr.tamu.edu/davis/suitesparse.html):

1. **Unified Internal Representation**: Vectors are n×1 matrices, scalars are 1×1 matrices
2. **Multiple Sparse Formats**: COO for construction, CSR/CSC for computation
3. **Semiring Abstraction**: Generalized matrix operations over algebraic structures
4. **Shape-Based Reasoning**: Types distinguished by metadata, not separate implementations

### Rust Advantages

- **Memory Safety**: No segfaults, no data races
- **Zero-Cost Abstractions**: High-level API with C-like performance
- **RAII**: Automatic resource management without GC overhead
- **Type System**: Compile-time guarantees for dimension compatibility
- **Fearless Concurrency**: Safe parallel operations (future work)

## Project Structure

```
rustsparse/
├── src/
│   ├── core/              # Core types (Matrix, Vector, Scalar, Semiring)
│   │   ├── container.rs   # Unified SparseContainer<T>
│   │   ├── matrix.rs      # Matrix<T> wrapper
│   │   ├── vector.rs      # Vector<T> wrapper
│   │   └── scalar.rs      # Scalar<T> wrapper
│   ├── ops/               # High-level operations
│   │   ├── matmul.rs      # Unified mxm/mxv/vxm implementation
│   │   ├── ewise.rs       # Element-wise operations
│   │   └── descriptor.rs  # Operation descriptors
│   ├── ir/                # Intermediate representation
│   │   ├── types.rs       # IRType::Tensor unified type
│   │   ├── shape.rs       # Shape system
│   │   ├── builder.rs     # Graph builder API
│   │   └── graph.rs       # IR graph data structure
│   ├── optimizer/         # Optimization passes
│   │   ├── cse.rs         # Common subexpression elimination
│   │   ├── fusion.rs      # Operation fusion
│   │   └── format_select.rs # Sparse format selection
│   ├── compiler/          # JIT compilation
│   │   ├── cranelift_backend.rs # Cranelift code generation
│   │   └── cache.rs       # Kernel cache
│   ├── lisp/              # Lisp DSL
│   │   ├── repl.rs        # Interactive REPL
│   │   ├── parser.rs      # S-expression parser
│   │   ├── eval.rs        # Interpreter
│   │   └── compiler.rs    # AST → IR compiler
│   ├── ffi/               # C API (GraphBLAS compatibility)
│   └── bin/
│       └── lisp_repl.rs   # REPL binary
├── tests/                 # Integration tests
│   ├── lisp_comprehensive.lisp  # Full Lisp test suite
│   ├── lisp_advanced.lisp       # Advanced tests
│   ├── lisp_examples.lisp       # Example programs
│   ├── jit_spmv.rs        # JIT compilation tests
│   └── e2e_operations.rs  # End-to-end operation tests
└── .github/
    └── workflows/
        └── test.yml       # CI/CD pipeline
```

## Contributing

Contributions are welcome! Areas of particular interest:

- Performance optimization and benchmarking
- Additional GraphBLAS operations
- GPU backend development
- Algorithm implementations (graph algorithms)
- Documentation and tutorials
- Python/Julia bindings

## References

- [GraphBLAS API Specification](https://graphblas.org/)
- [SuiteSparse](https://people.engr.tamu.edu/davis/suitesparse.html)
- [Cranelift Code Generator](https://cranelift.dev/)
- [The Rust Programming Language](https://www.rust-lang.org/)

## License

[To be determined - please specify your preferred license]

## Acknowledgments

- Design inspired by [SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse) by Tim Davis
- JIT compilation powered by [Cranelift](https://github.com/bytecodealliance/wasmtime/tree/main/cranelift)
- Built with [Rust](https://www.rust-lang.org/)
