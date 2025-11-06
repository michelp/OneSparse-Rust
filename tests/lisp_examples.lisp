;; RustSparse Lisp Examples
;; Practical examples demonstrating the Lisp DSL features
;; Usage: cat tests/lisp_examples.lisp | cargo run --bin lisp

;; ============================================================================
;; EXAMPLE 1: Basic Arithmetic with Vectors
;; ============================================================================

;; Create a simple vector
(vector 1 2 3 4 5)

;; Create a zero vector
(vector 0 0 0 0)

;; Create a vector with negative values
(vector -1 -2 -3)

;; ============================================================================
;; EXAMPLE 2: Matrix Creation
;; ============================================================================

;; Create a 2x2 identity matrix
(matrix
  (vector 1 0)
  (vector 0 1))

;; Create a 3x3 matrix
(matrix
  (vector 1 2 3)
  (vector 4 5 6)
  (vector 7 8 9))

;; Create a sparse matrix (diagonal)
(matrix
  (vector 5 0 0)
  (vector 0 3 0)
  (vector 0 0 7))

;; ============================================================================
;; EXAMPLE 3: Matrix-Vector Multiplication
;; ============================================================================

;; Multiply identity matrix by vector (returns the vector)
(mxv
  (matrix
    (vector 1 0 0)
    (vector 0 1 0)
    (vector 0 0 1))
  (vector 5 7 9))

;; Matrix-vector multiplication example
(mxv
  (matrix
    (vector 2 3)
    (vector 4 5))
  (vector 10 20))

;; ============================================================================
;; EXAMPLE 4: Vector-Matrix Multiplication
;; ============================================================================

;; Vector times identity matrix
(vxm
  (vector 5 7 9)
  (matrix
    (vector 1 0 0)
    (vector 0 1 0)
    (vector 0 0 1)))

;; ============================================================================
;; EXAMPLE 5: Standard Matrix Multiplication (Plus-Times Semiring)
;; ============================================================================

;; Multiply two 2x2 matrices
(plus-times
  (matrix
    (vector 1 2)
    (vector 3 4))
  (matrix
    (vector 5 6)
    (vector 7 8)))

;; Multiply by identity matrix
(plus-times
  (matrix
    (vector 1 2)
    (vector 3 4))
  (matrix
    (vector 1 0)
    (vector 0 1)))

;; ============================================================================
;; EXAMPLE 6: Shortest Path (Min-Plus Semiring)
;; ============================================================================

;; Compute shortest paths in a graph represented as adjacency matrix
;; 0 means direct connection, large numbers mean no edge
(min-plus
  (matrix
    (vector 0 1 5)
    (vector 9 0 2)
    (vector 6 4 0))
  (matrix
    (vector 0 1 5)
    (vector 9 0 2)
    (vector 6 4 0)))

;; ============================================================================
;; EXAMPLE 7: Control Flow - Conditional Operations
;; ============================================================================

;; Choose between two vectors based on condition
(if true
    (vector 1 2 3)
    (vector 9 9 9))

;; Multi-way selection with cond
(cond
  (false (vector 0 0 0))
  (false (vector 1 1 1))
  (true (vector 42 42 42))
  (else (vector 99 99 99)))

;; ============================================================================
;; EXAMPLE 8: Iterative Operations
;; ============================================================================

;; Create 5 vectors in a loop
(for i 0 5
  (vector 1 2 3))

;; Conditional loop (terminates immediately)
(while false
  (vector 1 2 3))

;; Loop with break
(for i 0 100
  (begin
    (vector 1 2 3)
    (break)))

;; ============================================================================
;; EXAMPLE 9: Sequential Operations
;; ============================================================================

;; Execute multiple operations in sequence
(begin
  (vector 1 2 3)
  (matrix (vector 1 0) (vector 0 1))
  (plus-times
    (matrix (vector 1 2) (vector 3 4))
    (matrix (vector 1 0) (vector 0 1)))
  42)

;; ============================================================================
;; EXAMPLE 10: Testing with Assert
;; ============================================================================

;; Assert that a condition is true
(assert true)

;; Assert with computed result
(assert (if (cond (true 42) (else 0)) true false))

;; Multiple asserts
(begin
  (assert true)
  (assert 1)
  (assert 42)
  (assert true))

;; ============================================================================
;; EXAMPLE 11: Combining Control Flow and Matrix Operations
;; ============================================================================

;; Conditional matrix multiplication
(if true
    (plus-times
      (matrix (vector 1 2) (vector 3 4))
      (matrix (vector 5 6) (vector 7 8)))
    (matrix (vector 0 0) (vector 0 0)))

;; Loop with matrix creation
(for i 0 3
  (matrix
    (vector 1 0)
    (vector 0 1)))

;; ============================================================================
;; EXAMPLE 12: Complex Nested Expressions
;; ============================================================================

;; Nested conditionals with matrix operations
(if true
    (begin
      (vector 1 2 3)
      (if true
          (matrix (vector 1 2) (vector 3 4))
          (vector 0 0 0))
      (mxv
        (matrix (vector 1 0) (vector 0 1))
        (vector 5 7)))
    (vector 0 0))

;; ============================================================================
;; SUCCESS!
;; ============================================================================

;; If you've made it this far without errors, congratulations!
;; You've seen all the major features of the RustSparse Lisp DSL.

(begin
  (assert true)
  (vector 100 200 300))

;; Output above should show the final vector #[100 200 300]
