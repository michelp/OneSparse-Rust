# RustSparse Lisp Test Suite

Comprehensive test files for the RustSparse Lisp DSL.

## Test Files

### `lisp_comprehensive.lisp` - Full Feature Coverage
Comprehensive test suite covering all supported Lisp features with assertions.

**Features Tested:**
- Literals (numbers, booleans)
- Control structures (if, cond, while, for, begin, break, continue)
- Vector creation and operations
- Matrix creation
- Matrix-vector multiplication (mxv, vxm)
- Semiring operations (plus-times, min-plus, max-times)
- Nested control flow
- Complex expressions

**Usage:**
```bash
cat tests/lisp_comprehensive.lisp | cargo run --bin lisp
```

**Expected Output:** Multiple "true" values from successful assertions, ending with `42`

---

### `lisp_advanced.lisp` - Advanced Tests
Advanced tests for matrix/vector operations, edge cases, and stress testing.

**Features Tested:**
- Large vectors and matrices
- Sparse matrix operations
- Rectangular matrices
- Identity and zero matrix operations
- Chained semiring operations
- Deeply nested control structures
- Stress tests

**Usage:**
```bash
cat tests/lisp_advanced.lisp | cargo run --bin lisp
```

**Expected Output:** Multiple "true" values from successful assertions, ending with `999`

---

### `lisp_examples.lisp` - Practical Examples
Practical examples demonstrating real-world usage of the Lisp DSL.

**Examples Included:**
- Basic vector/matrix creation
- Matrix-vector multiplication
- Standard matrix multiplication
- Shortest path computation (min-plus)
- Control flow patterns
- Iterative operations
- Complex nested expressions

**Usage:**
```bash
cat tests/lisp_examples.lisp | cargo run --bin lisp
```

**Expected Output:** Various matrix and vector values demonstrating each feature

---

## The `assert` Function

All test files use the `assert` function for validation:

```lisp
;; Basic assertion
(assert true)

;; Assert with message (number shown on failure)
(assert condition 123)

;; Assert with computed result
(assert (if (cond (true 42) (else 0)) true false))
```

**Behavior:**
- **Success:** Returns `true`
- **Failure:** Returns error with message "Assertion failed" or custom message
- **Truthy values:** Non-zero numbers, `true`, vectors, matrices
- **Falsy values:** `0`, `false`, `nil`

---

## Running All Tests

Run all test suites in sequence:

```bash
cat tests/lisp_comprehensive.lisp | cargo run --bin lisp
cat tests/lisp_advanced.lisp | cargo run --bin lisp
cat tests/lisp_examples.lisp | cargo run --bin lisp
```

Or combine them:

```bash
(cat tests/lisp_comprehensive.lisp tests/lisp_advanced.lisp) | cargo run --bin lisp
```

---

## Currently Supported Features

### Literals
- Integers: `42`, `-123`
- Floats: `3.14`, `-0.5`
- Booleans: `true`, `false`

### Control Structures
- **if**: `(if condition then-expr else-expr)`
- **cond**: `(cond (test1 result1) (test2 result2) (else default))`
- **while**: `(while condition body...)`
- **for**: `(for var start end body...)` or `(for var start end step body...)`
- **begin**: `(begin expr1 expr2 ...)`
- **break**: `(break)` or `(break value)`
- **continue**: `(continue)`

### Vector Operations
- **vector**: `(vector 1 2 3 4 5)`

### Matrix Operations
- **matrix**: `(matrix (vector 1 2) (vector 3 4))`
- **mxv**: Matrix-vector multiply
- **vxm**: Vector-matrix multiply

### Semiring Operations
- **plus-times**: Standard matrix multiplication
- **min-plus**: Shortest path semiring
- **max-times**: Max-times semiring

### Utilities
- **assert**: `(assert condition)` or `(assert condition message)`
- **set-log-level**: `(set-log-level :info|:debug|:trace)`
- **get-log-level**: `(get-log-level)`

---

## Test Coverage

| Feature | Comprehensive | Advanced | Examples |
|---------|--------------|----------|----------|
| Literals | ✓ | ✓ | ✓ |
| if | ✓ | ✓ | ✓ |
| cond | ✓ | ✓ | ✓ |
| while | ✓ | ✓ | ✓ |
| for | ✓ | ✓ | ✓ |
| begin | ✓ | ✓ | ✓ |
| break/continue | ✓ | - | ✓ |
| vector | ✓ | ✓ | ✓ |
| matrix | ✓ | ✓ | ✓ |
| mxv/vxm | ✓ | ✓ | ✓ |
| plus-times | ✓ | ✓ | ✓ |
| min-plus | ✓ | ✓ | ✓ |
| max-times | ✓ | ✓ | ✓ |
| assert | ✓ | ✓ | ✓ |
| Nested expressions | ✓ | ✓ | ✓ |
| Edge cases | ✓ | ✓ | - |
| Stress tests | - | ✓ | - |

---

## Notes

### Interpreted vs JIT Compilation

Currently, control structures work **only in interpreted mode** (REPL). Using control structures inside `defkernel` will return a "NotImplemented" error. This is intentional for Phase 1.

### Known Limitations

- Variable bindings (`let`) not fully supported due to lack of Clone on Matrix/Vector types
- Variables cannot reference matrices/vectors
- Break with value not yet implemented
- Transpose operation not yet implemented

---

## Interactive Testing

You can also test interactively in the REPL:

```bash
cargo run --bin lisp
```

Then type expressions:

```lisp
>> (assert true)
true
>> (if true 42 0)
42
>> (vector 1 2 3)
#[1 2 3]
>> (plus-times (matrix (vector 1 0) (vector 0 1)) (matrix (vector 5 6) (vector 7 8)))
#<Matrix 2x2>
```

Use tab completion to explore available functions!
