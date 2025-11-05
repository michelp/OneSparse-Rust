# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements sparse linear algebra in Rust for solving graph problems. It provides a Rust implementation of the GraphBLAS API with a **link-time compatible C API** that mirrors SuiteSparse:GraphBLAS v10.2.0+ (conforming to GraphBLAS C API Specification v2.1.0), making it a drop-in replacement.

**Compatibility Target:** SuiteSparse:GraphBLAS 10.2.0+ / GraphBLAS C API v2.1.0

## Build Commands

```bash
# Build the project
cargo build

# Build with release optimizations
cargo build --release

# Run all tests
cargo test

# Run a specific test
cargo test test_name

# Run tests with output
cargo test -- --nocapture

# Run only FFI compatibility tests
cargo test --test ffi_compat

# Check code without building
cargo check

# Format code
cargo fmt

# Run linter
cargo clippy

# Build C-compatible shared library
cargo build --release --lib

# Run link compatibility tests (when implemented)
cargo test --test link_compat
```

## Architecture

This project uses a **three-layer architecture** that separates concerns between C compatibility, FFI safety, and idiomatic Rust implementation.

### Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Layer 1: C API (extern "C" facade)                     │
│  - Link-compatible with SuiteSparse:GraphBLAS           │
│  - GraphBLAS.h header file (single include)             │
│  - Opaque pointer types only                            │
│  - Exact function signatures (GrB_* / GxB_*)            │
│  - GrB_Info error codes                                 │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  Layer 2: FFI Safety Layer                              │
│  - Panic boundaries (catch_unwind)                      │
│  - Pointer validation                                   │
│  - Memory ownership tracking                            │
│  - Error translation (Rust ↔ C)                         │
│  - Thread safety coordination                           │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  Layer 3: Idiomatic Rust Implementation                 │
│  - Safe, typed API                                      │
│  - Traits and generics                                  │
│  - NOT constrained by C idioms                          │
│  - Leverage Rust type system                            │
│  - Zero-cost abstractions                               │
└─────────────────────────────────────────────────────────┘
```

**Key principle:** The C API layer is a **thin facade** that calls into safe Rust code. The Rust implementation should be designed idiomatically without concern for C limitations.

---

## Layer 1: C API Compatibility Requirements

### Header File as Source of Truth

**Single public header:** `GraphBLAS.h`
- Contains all public API declarations
- Must be kept in sync with the Rust `extern "C"` layer
- Located in `include/GraphBLAS.h` (standard location)

### Opaque Types (11 Core Types)

All GraphBLAS objects are **opaque pointers** - never expose internal structure:

```rust
// Correct: Opaque pointer type
#[repr(C)]
pub struct GrB_Matrix_opaque {
    _private: [u8; 0],
}
pub type GrB_Matrix = *mut GrB_Matrix_opaque;
```

**The 11 core opaque types:**

**Data containers:**
- `GrB_Scalar` - 1×1 matrix (typed single value)
- `GrB_Vector` - n×1 sparse vector
- `GrB_Matrix` - m×n sparse matrix

**Algebraic operators:**
- `GrB_Type` - Data type specification (INT64, FP64, BOOL, user-defined)
- `GrB_UnaryOp` - Unary function: z = f(x)
- `GrB_BinaryOp` - Binary function: z = f(x, y)
- `GrB_IndexUnaryOp` - Position-aware unary: z = f(x, i, j, thunk)
- `GrB_Monoid` - Associative operator with identity: (⊕, 0)
- `GrB_Semiring` - (Monoid, Multiply): (⊕, ⊗, 0)

**Control objects:**
- `GrB_Descriptor` - Operation modifiers (transpose, mask, accumulate)
- `GrB_Global` - System-wide configuration

### Function Naming Convention

**Pattern:** `Prefix_ObjectType_operation_detail`

**Prefixes:**
- `GrB_` - Standard GraphBLAS C API (portable across implementations)
- `GxB_` - SuiteSparse-specific extensions (non-portable)
- `GB_` - Internal use ONLY (must not be exposed to users)

**Examples:**
```c
GrB_Matrix_new()              // Create matrix
GrB_Matrix_nrows()            // Query rows
GrB_mxm()                     // Matrix-matrix multiply
GxB_Matrix_serialize()        // SuiteSparse extension
```

### Error Handling

**Return type:** `GrB_Info` (C enum, represented as `i32` in Rust)

```rust
pub type GrB_Info = i32;

// Success codes (>= 0)
pub const GrB_SUCCESS: GrB_Info = 0;
pub const GrB_NO_VALUE: GrB_Info = 1;  // Entry not present (sparse)

// Error codes (< 0)
pub const GrB_UNINITIALIZED_OBJECT: GrB_Info = -1;
pub const GrB_NULL_POINTER: GrB_Info = -2;
pub const GrB_INVALID_VALUE: GrB_Info = -3;
pub const GrB_INVALID_INDEX: GrB_Info = -4;
pub const GrB_DOMAIN_MISMATCH: GrB_Info = -5;
pub const GrB_DIMENSION_MISMATCH: GrB_Info = -6;
pub const GrB_OUTPUT_NOT_EMPTY: GrB_Info = -7;
pub const GrB_OUT_OF_MEMORY: GrB_Info = -102;
pub const GrB_INDEX_OUT_OF_BOUNDS: GrB_Info = -105;
```

**Every C API function returns `GrB_Info`** - never panic at the C boundary.

### Function Signature Matching

**Critical:** Function signatures must match exactly for link compatibility.

```rust
#[no_mangle]
pub extern "C" fn GrB_Matrix_new(
    A: *mut GrB_Matrix,           // Output parameter (pointer to pointer)
    type_: GrB_Type,              // Type of elements
    nrows: GrB_Index,             // Number of rows
    ncols: GrB_Index              // Number of columns
) -> GrB_Info {
    // Implementation...
}
```

**Common types:**
```rust
pub type GrB_Index = u64;         // Index type (unsigned 64-bit)
pub type GrB_Info = i32;          // Return status
```

### Key Function Categories

The C API includes 15+ operation categories. Here are the main groups:

**1. Initialization**
```c
GrB_init(GrB_Mode mode)
GrB_finalize()
```

**2. Object lifecycle (new/free pattern)**
```c
GrB_Matrix_new(&A, type, nrows, ncols)
GrB_Vector_new(&v, type, n)
GrB_Matrix_free(&A)                    // Sets A to NULL
```

**3. Matrix multiplication (semiring-based)**
```c
GrB_mxm(C, M, accum, semiring, A, B, desc)     // Matrix × Matrix
GrB_mxv(w, m, accum, semiring, A, u, desc)     // Matrix × Vector
GrB_vxm(w, m, accum, semiring, u, A, desc)     // Vector × Matrix
```

**4. Element-wise operations**
```c
GrB_eWiseAdd_Matrix_BinaryOp(C, M, accum, op, A, B, desc)   // Union
GrB_eWiseMult_Matrix_BinaryOp(C, M, accum, op, A, B, desc)  // Intersection
```

**5. Extract (submatrix/subvector)**
```c
GrB_Matrix_extract(C, M, accum, A, rows, nrows, cols, ncols, desc)
GrB_Vector_extract(w, m, accum, u, indices, ni, desc)
```

**6. Assign (write to submatrix)**
```c
GrB_Matrix_assign(C, M, accum, A, rows, nrows, cols, ncols, desc)
GrB_Row_assign(C, m, accum, u, row, cols, ncols, desc)
GrB_Col_assign(C, m, accum, u, rows, nrows, col, desc)
```

**7. Apply (transform with operators)**
```c
GrB_Matrix_apply(C, M, accum, op, A, desc)
GrB_Matrix_apply_BinaryOp1st(C, M, accum, op, x, A, desc)  // Bind left
GrB_Matrix_apply_BinaryOp2nd(C, M, accum, op, A, y, desc)  // Bind right
```

**8. Select (filter with predicates)**
```c
GrB_Matrix_select(C, M, accum, op, A, thunk, desc)
```

**9. Reduce (matrix→vector, vector→scalar)**
```c
GrB_Matrix_reduce_Monoid(w, m, accum, monoid, A, desc)
GrB_Vector_reduce(s, accum, monoid, u, desc)
```

**10. Transpose**
```c
GrB_transpose(C, M, accum, A, desc)
```

**11. Element access**
```c
GrB_Matrix_setElement(A, x, i, j)
GrB_Matrix_extractElement(&x, A, i, j)  // Returns GrB_NO_VALUE if sparse
GrB_Matrix_removeElement(A, i, j)
```

**12. Build from tuples (COO format)**
```c
GrB_Matrix_build(A, rows, cols, vals, nvals, dup_op)
```

**13. Extract tuples (to COO format)**
```c
GrB_Matrix_extractTuples(rows, cols, vals, &nvals, A)
```

**14. Property queries**
```c
GrB_Matrix_nrows(&n, A)
GrB_Matrix_ncols(&n, A)
GrB_Matrix_nvals(&n, A)     // Number of stored entries
GrB_Matrix_wait(A)          // Complete deferred operations
```

**15. Descriptor operations**
```c
GrB_Descriptor_set(desc, GrB_MASK, GrB_COMP)    // Complement mask
GrB_Descriptor_set(desc, GrB_INP0, GrB_TRAN)    // Transpose first input
```

### Memory Management

**Explicit lifecycle:**
- `GrB_*_new()` allocates
- `GrB_*_free()` deallocates and sets pointer to NULL
- No automatic garbage collection
- Manual reference counting required

**Critical rules:**
- Never mix Rust allocator with C allocator
- All memory passed to C must be C-allocated
- Use `std::alloc::System` or custom allocator for FFI memory

---

## Layer 2: FFI Safety Patterns

### Never Dereference Opaque Pointers

```rust
// CORRECT: Treat as opaque handle
fn matrix_to_internal(grb_matrix: GrB_Matrix) -> Option<Arc<MatrixImpl>> {
    unsafe {
        let handle = grb_matrix as usize;
        HANDLE_MAP.lock().get(&handle).cloned()
    }
}

// WRONG: Never do this
fn bad_access(grb_matrix: GrB_Matrix) -> usize {
    unsafe { (*grb_matrix).nrows }  // BREAKS LINK COMPATIBILITY
}
```

### Panic Boundaries

**Every `extern "C"` function must catch panics:**

```rust
#[no_mangle]
pub extern "C" fn GrB_Matrix_new(
    A: *mut GrB_Matrix,
    type_: GrB_Type,
    nrows: GrB_Index,
    ncols: GrB_Index
) -> GrB_Info {
    let result = std::panic::catch_unwind(|| {
        // Implementation that might panic
        matrix_new_impl(A, type_, nrows, ncols)
    });

    match result {
        Ok(info) => info,
        Err(_) => GrB_PANIC  // Special error code for panics
    }
}
```

### Pointer Validation

```rust
fn validate_output_ptr<T>(ptr: *mut T) -> Result<(), GrB_Info> {
    if ptr.is_null() {
        return Err(GrB_NULL_POINTER);
    }
    Ok(())
}

fn validate_input_ptr<T>(ptr: *const T) -> Result<(), GrB_Info> {
    if ptr.is_null() {
        return Err(GrB_NULL_POINTER);
    }
    Ok(())
}
```

### Memory Ownership Tracking

Use a handle-based system to manage Rust objects accessed through C:

```rust
use std::sync::{Arc, Mutex};
use std::collections::HashMap;

lazy_static! {
    static ref MATRIX_HANDLES: Mutex<HashMap<usize, Arc<MatrixImpl>>> =
        Mutex::new(HashMap::new());
}

// Allocate: Create Rust object, return opaque C pointer
fn create_handle(matrix: MatrixImpl) -> GrB_Matrix {
    let arc = Arc::new(matrix);
    let handle = Arc::as_ptr(&arc) as usize;
    MATRIX_HANDLES.lock().unwrap().insert(handle, arc);
    handle as GrB_Matrix
}

// Free: Remove from handle map
fn destroy_handle(handle: GrB_Matrix) -> Result<(), GrB_Info> {
    let handle_id = handle as usize;
    MATRIX_HANDLES.lock().unwrap().remove(&handle_id)
        .ok_or(GrB_UNINITIALIZED_OBJECT)?;
    Ok(())
}
```

### Runtime Version Checking

```rust
pub const GRB_VERSION: u32 = 2;
pub const GRB_SUBVERSION: u32 = 1;

#[no_mangle]
pub extern "C" fn GrB_getVersion(
    version: *mut u32,
    subversion: *mut u32
) -> GrB_Info {
    if version.is_null() || subversion.is_null() {
        return GrB_NULL_POINTER;
    }
    unsafe {
        *version = GRB_VERSION;
        *subversion = GRB_SUBVERSION;
    }
    GrB_SUCCESS
}
```

### Thread Safety

**Important:** SuiteSparse:GraphBLAS uses OpenMP for parallelism. If replicating behavior:
- Use `rayon` or similar for Rust-side parallelism
- Be aware that C code may call into library from OpenMP threads
- All public functions must be thread-safe
- Use `Arc<Mutex<T>>` or `Arc<RwLock<T>>` for shared mutable state

---

## Layer 3: Idiomatic Rust Implementation

### Design Principles

**The Rust implementation is NOT constrained by C limitations.** Design it idiomatically:

### Safe, Typed API

```rust
pub struct Matrix<T> {
    nrows: usize,
    ncols: usize,
    storage: SparseStorage<T>,
}

impl<T> Matrix<T> {
    pub fn new(nrows: usize, ncols: usize) -> Self { ... }
    pub fn nrows(&self) -> usize { self.nrows }
    pub fn ncols(&self) -> usize { self.ncols }
    pub fn nvals(&self) -> usize { self.storage.len() }
}
```

### Use Traits for Algebraic Structures

```rust
pub trait Semiring {
    type Elem;
    fn add(&self, a: Self::Elem, b: Self::Elem) -> Self::Elem;
    fn multiply(&self, a: Self::Elem, b: Self::Elem) -> Self::Elem;
    fn zero(&self) -> Self::Elem;
}

pub struct PlusTimesSemiring;
impl Semiring for PlusTimesSemiring {
    type Elem = f64;
    fn add(&self, a: f64, b: f64) -> f64 { a + b }
    fn multiply(&self, a: f64, b: f64) -> f64 { a * b }
    fn zero(&self) -> f64 { 0.0 }
}
```

### Leverage Generics

```rust
pub fn mxm<T, S>(
    c: &mut Matrix<T>,
    a: &Matrix<T>,
    b: &Matrix<T>,
    semiring: &S
) -> Result<(), Error>
where
    T: Copy + PartialEq,
    S: Semiring<Elem = T>
{
    // Matrix multiplication using arbitrary semiring
}
```

### Sparse Storage Formats

Implement multiple formats internally:

```rust
pub enum SparseStorage<T> {
    CSR(CsrMatrix<T>),
    CSC(CscMatrix<T>),
    COO(CooMatrix<T>),
}

pub struct CsrMatrix<T> {
    row_ptrs: Vec<usize>,
    col_indices: Vec<usize>,
    values: Vec<T>,
}
```

### Error Handling

Use Rust's `Result` type internally:

```rust
#[derive(Debug)]
pub enum GraphBlasError {
    UninitializedObject,
    NullPointer,
    InvalidValue,
    InvalidIndex,
    DomainMismatch,
    DimensionMismatch,
    OutputNotEmpty,
    OutOfMemory,
    IndexOutOfBounds,
}

impl GraphBlasError {
    pub fn to_grb_info(&self) -> GrB_Info {
        match self {
            Self::UninitializedObject => GrB_UNINITIALIZED_OBJECT,
            Self::NullPointer => GrB_NULL_POINTER,
            // ... map all variants
        }
    }
}
```

---

## Synchronization Workflow

**Keeping C header and Rust FFI in sync:**

### 1. Header as Source of Truth

When adding new API functions:

1. **Update `include/GraphBLAS.h`** with the C declaration
2. **Add corresponding Rust `extern "C"` function** in `src/ffi/` module
3. **Verify signature match** (types, parameter order, return type)
4. **Run link compatibility tests**

### 2. Automated Checking

**Use `bindgen` for verification (not generation):**

```bash
# Generate bindings from header to verify consistency
cargo install bindgen
bindgen include/GraphBLAS.h -o tests/verify_bindings.rs

# Compare with hand-written FFI layer
diff tests/verify_bindings.rs src/ffi/raw.rs
```

### 3. Testing Strategy

**FFI compatibility tests:**

```rust
#[test]
fn test_grb_matrix_new_signature() {
    // Verify function exists and has correct signature
    let _: unsafe extern "C" fn(
        *mut GrB_Matrix,
        GrB_Type,
        GrB_Index,
        GrB_Index
    ) -> GrB_Info = GrB_Matrix_new;
}
```

**Link compatibility tests:**

```bash
# Create a C test program that links against our library
gcc -o test_link test_link.c -L./target/release -lrustsparse -lgraphblas
./test_link
```

### 4. Version Validation

**On every build, verify:**
- API version constants match target (v2.1.0)
- All required functions are exported
- Function signatures match exactly

```rust
#[cfg(test)]
mod version_tests {
    #[test]
    fn verify_api_version() {
        assert_eq!(GRB_VERSION, 2);
        assert_eq!(GRB_SUBVERSION, 1);
    }
}
```

---

## Core Implementation Components

### Sparse Matrix Storage

Implement multiple formats for different operation performance:

- **CSR (Compressed Sparse Row)**: Fast row access, good for row-major operations
- **CSC (Compressed Sparse Column)**: Fast column access, good for column-major operations
- **COO (Coordinate)**: Fast construction, easy random insertion
- **Automatic format selection**: Switch formats based on operation patterns

### Semiring Operations

GraphBLAS defines operations over semirings. Common examples:

- **Plus-Times**: Standard linear algebra `(+, ×, 0)`
- **Min-Plus**: Shortest path problems `(min, +, ∞)`
- **Max-Times**: Maximum weighted path `(max, ×, 0)`
- **Any-Or**: Boolean logic `(∨, ∧, false)`

### Deferred Operations

SuiteSparse:GraphBLAS uses deferred execution:
- Operations may not execute immediately
- `GrB_Matrix_wait()` forces completion
- Allows optimization through fusion

Consider implementing similar deferred execution in Rust layer.

---

## Testing Strategy

### 1. Unit Tests (Rust Layer)

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_matrix_creation() {
        let m = Matrix::<f64>::new(10, 10);
        assert_eq!(m.nrows(), 10);
        assert_eq!(m.ncols(), 10);
        assert_eq!(m.nvals(), 0);
    }
}
```

### 2. FFI Safety Tests

```rust
#[test]
fn test_null_pointer_handling() {
    let result = unsafe {
        GrB_Matrix_new(std::ptr::null_mut(), GrB_FP64, 10, 10)
    };
    assert_eq!(result, GrB_NULL_POINTER);
}
```

### 3. Link Compatibility Tests

Create C test programs that verify binary compatibility:

```c
// test_link.c
#include "GraphBLAS.h"

int main() {
    GrB_init(GrB_NONBLOCKING);

    GrB_Matrix A;
    GrB_Info info = GrB_Matrix_new(&A, GrB_FP64, 100, 100);
    assert(info == GrB_SUCCESS);

    GrB_Matrix_free(&A);
    GrB_finalize();
    return 0;
}
```

### 4. Correctness Tests Against SuiteSparse

Compare results with SuiteSparse:GraphBLAS on standard test cases:
- LAGraph test suite
- GraphBLAS C API test suite
- Known graph algorithm results

### 5. Performance Benchmarks

Track performance on standard operations:
- Sparse matrix multiplication
- BFS (breadth-first search)
- PageRank
- Triangle counting

---

## Key Implementation Notes

### Never Expose Internal Layouts

```rust
// CORRECT: Opaque type
#[repr(C)]
pub struct GrB_Matrix_opaque {
    _private: [u8; 0],
}
pub type GrB_Matrix = *mut GrB_Matrix_opaque;

// WRONG: Exposing internals breaks compatibility
#[repr(C)]
pub struct GrB_Matrix {
    nrows: usize,
    ncols: usize,
    data: *mut f64,
}
```

### Don't Mix Memory Allocators

```rust
// When allocating memory for C
use std::alloc::{alloc, dealloc, Layout};

let layout = Layout::array::<f64>(len).unwrap();
let ptr = unsafe { alloc(layout) as *mut f64 };
```

### Respect OpenMP Threading

If implementing parallel operations:
- Don't assume thread ownership
- Use thread-safe data structures
- Consider nested parallelism from C caller

---

## Development Workflow

### Adding a New Function

1. **Add to `include/GraphBLAS.h`**:
   ```c
   GrB_Info GrB_Matrix_new(GrB_Matrix *A, GrB_Type type,
                           GrB_Index nrows, GrB_Index ncols);
   ```

2. **Implement in `src/ffi/matrix.rs`**:
   ```rust
   #[no_mangle]
   pub extern "C" fn GrB_Matrix_new(...) -> GrB_Info {
       // With panic boundary, pointer validation, error mapping
   }
   ```

3. **Call safe Rust implementation in `src/matrix.rs`**:
   ```rust
   impl<T> Matrix<T> {
       pub fn new(nrows: usize, ncols: usize) -> Self { ... }
   }
   ```

4. **Add tests**:
   - Rust unit test in `src/matrix.rs`
   - FFI test in `tests/ffi_matrix.rs`
   - C link test in `tests/link/test_matrix.c`

5. **Verify**:
   ```bash
   cargo test
   cargo build --release
   gcc -o test tests/link/test_matrix.c -L./target/release -lrustsparse
   ./test
   ```

---

## Common Pitfalls to Avoid

1. **Don't assume layout compatibility** - All GraphBLAS types are opaque
2. **Don't panic at C boundaries** - Use `catch_unwind`
3. **Don't mix allocators** - C-allocated memory stays with C allocator
4. **Don't expose Rust types to C** - Only use C-compatible types at boundary
5. **Don't skip version checking** - Verify runtime compatibility
6. **Don't forget thread safety** - All FFI functions must be thread-safe
7. **Don't ignore error codes** - Every C function must return `GrB_Info`

---

## References

- [GraphBLAS C API v2.1.0 Specification](http://graphblas.org/)
- [SuiteSparse:GraphBLAS GitHub](https://github.com/DrTimothyAldenDavis/GraphBLAS)
- [SuiteSparse Documentation](https://github.com/DrTimothyAldenDavis/SuiteSparse)
- GraphBLAS C API Specification (December 22, 2023)
