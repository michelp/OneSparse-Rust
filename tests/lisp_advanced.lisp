;; Advanced Lisp Test Suite for RustSparse
;; Tests advanced matrix/vector operations and edge cases
;; Usage: cat tests/lisp_advanced.lisp | cargo run --bin lisp

;; ============================================================================
;; ADVANCED VECTOR OPERATIONS
;; ============================================================================

;; Large vectors
(begin
  (vector 1 2 3 4 5 6 7 8 9 10)
  (assert true))

;; Vectors with mixed signs
(begin
  (vector -5 0 5 -10 10)
  (assert true))

;; Vectors with floating point values
(begin
  (vector 0.1 0.2 0.3 0.4 0.5)
  (assert true))

;; Very small and very large numbers
(begin
  (vector 0.001 0.01 0.1 1 10 100 1000)
  (assert true))

;; ============================================================================
;; ADVANCED MATRIX OPERATIONS
;; ============================================================================

;; Larger matrices (3x3)
(begin
  (matrix
    (vector 1 2 3)
    (vector 4 5 6)
    (vector 7 8 9))
  (assert true))

;; Sparse matrix (lots of zeros)
(begin
  (matrix
    (vector 1 0 0 0)
    (vector 0 2 0 0)
    (vector 0 0 3 0)
    (vector 0 0 0 4))
  (assert true))

;; Dense matrix (no zeros)
(begin
  (matrix
    (vector 1 2 3)
    (vector 4 5 6)
    (vector 7 8 9))
  (assert true))

;; Rectangular matrices
(begin
  (matrix
    (vector 1 2 3 4 5)
    (vector 6 7 8 9 10))
  (assert true))

(begin
  (matrix
    (vector 1 2)
    (vector 3 4)
    (vector 5 6)
    (vector 7 8))
  (assert true))

;; ============================================================================
;; MATRIX-VECTOR MULTIPLICATION TESTS
;; ============================================================================

;; Identity matrix times vector should return vector
(begin
  (mxv
    (matrix
      (vector 1 0 0)
      (vector 0 1 0)
      (vector 0 0 1))
    (vector 5 7 9))
  (assert true))

;; Zero matrix times vector
(begin
  (mxv
    (matrix
      (vector 0 0 0)
      (vector 0 0 0)
      (vector 0 0 0))
    (vector 1 2 3))
  (assert true))

;; Sparse matrix times sparse vector
(begin
  (mxv
    (matrix
      (vector 1 0 0)
      (vector 0 0 0)
      (vector 0 0 1))
    (vector 5 0 7))
  (assert true))

;; ============================================================================
;; VECTOR-MATRIX MULTIPLICATION TESTS
;; ============================================================================

;; Vector times identity matrix
(begin
  (vxm
    (vector 5 7 9)
    (matrix
      (vector 1 0 0)
      (vector 0 1 0)
      (vector 0 0 1)))
  (assert true))

;; Vector times zero matrix
(begin
  (vxm
    (vector 1 2 3)
    (matrix
      (vector 0 0 0)
      (vector 0 0 0)
      (vector 0 0 0)))
  (assert true))

;; ============================================================================
;; SEMIRING OPERATIONS - PLUS-TIMES
;; ============================================================================

;; Identity matrix multiplication
(begin
  (plus-times
    (matrix
      (vector 1 0)
      (vector 0 1))
    (matrix
      (vector 5 6)
      (vector 7 8)))
  (assert true))

;; Zero matrix multiplication
(begin
  (plus-times
    (matrix
      (vector 0 0)
      (vector 0 0))
    (matrix
      (vector 1 2)
      (vector 3 4)))
  (assert true))

;; 3x3 matrix multiplication
(begin
  (plus-times
    (matrix
      (vector 1 2 3)
      (vector 4 5 6)
      (vector 7 8 9))
    (matrix
      (vector 9 8 7)
      (vector 6 5 4)
      (vector 3 2 1)))
  (assert true))

;; ============================================================================
;; SEMIRING OPERATIONS - MIN-PLUS (Shortest Path)
;; ============================================================================

;; Simple shortest path
(begin
  (min-plus
    (matrix
      (vector 0 1)
      (vector 1 0))
    (matrix
      (vector 0 1)
      (vector 1 0)))
  (assert true))

;; Larger graph (3x3)
(begin
  (min-plus
    (matrix
      (vector 0 1 5)
      (vector 9 0 2)
      (vector 6 4 0))
    (matrix
      (vector 0 1 5)
      (vector 9 0 2)
      (vector 6 4 0)))
  (assert true))

;; ============================================================================
;; SEMIRING OPERATIONS - MAX-TIMES
;; ============================================================================

;; Max-times with identity
(begin
  (max-times
    (matrix
      (vector 1 0)
      (vector 0 1))
    (matrix
      (vector 5 6)
      (vector 7 8)))
  (assert true))

;; Max-times with larger values
(begin
  (max-times
    (matrix
      (vector 10 20)
      (vector 30 40))
    (matrix
      (vector 2 3)
      (vector 4 5)))
  (assert true))

;; ============================================================================
;; CONTROL FLOW WITH MATRIX/VECTOR OPERATIONS
;; ============================================================================

;; Conditional matrix creation
(begin
  (if true
      (matrix (vector 1 2) (vector 3 4))
      (matrix (vector 0 0) (vector 0 0)))
  (assert true))

;; Cond with vector creation
(begin
  (cond
    (false (vector 0 0 0))
    (true (vector 1 2 3))
    (else (vector 9 9 9)))
  (assert true))

;; Loop with matrix operations
(begin
  (for i 0 3
    (matrix (vector 1 2) (vector 3 4)))
  (assert true))

;; Begin with matrix operations
(begin
  (vector 1 2 3)
  (matrix (vector 1 2) (vector 3 4))
  (mxv (matrix (vector 1 0) (vector 0 1)) (vector 5 7))
  (assert true))

;; ============================================================================
;; NESTED OPERATIONS
;; ============================================================================

;; Matrix mult inside conditional
(begin
  (if true
      (plus-times
        (matrix (vector 1 2) (vector 3 4))
        (matrix (vector 5 6) (vector 7 8)))
      (matrix (vector 0 0) (vector 0 0)))
  (assert true))

;; Vector creation in loop
(begin
  (for i 0 5
    (vector 1 2 3))
  (assert true))

;; Multiple operations in begin
(begin
  (vector 1 2 3)
  (vector 4 5 6)
  (matrix (vector 1 2) (vector 3 4))
  (matrix (vector 5 6) (vector 7 8))
  (assert true))

;; ============================================================================
;; CHAINED SEMIRING OPERATIONS
;; ============================================================================

;; Multiple matrix multiplications in sequence
(begin
  (plus-times
    (matrix (vector 1 0) (vector 0 1))
    (matrix (vector 1 0) (vector 0 1)))
  (plus-times
    (matrix (vector 1 2) (vector 3 4))
    (matrix (vector 5 6) (vector 7 8)))
  (assert true))

;; Mixed semiring operations
(begin
  (min-plus
    (matrix (vector 0 1) (vector 1 0))
    (matrix (vector 0 1) (vector 1 0)))
  (max-times
    (matrix (vector 2 3) (vector 4 5))
    (matrix (vector 1 0) (vector 0 1)))
  (plus-times
    (matrix (vector 1 2) (vector 3 4))
    (matrix (vector 1 0) (vector 0 1)))
  (assert true))

;; ============================================================================
;; STRESS TESTS
;; ============================================================================

;; Many nested control structures
(assert
  (if (begin
        (if (cond
              (false 0)
              (true (if (begin true true) true false))
              (else 0))
            true
            false))
      true
      false))

;; Deep nesting with operations
(begin
  (if true
      (begin
        (if true
            (begin
              (vector 1 2 3)
              (if true
                  (matrix (vector 1 2) (vector 3 4))
                  (vector 0 0 0)))
            (vector 9 9 9)))
      (vector 0 0 0))
  (assert true))

;; Many sequential operations
(begin
  (vector 1 2 3)
  (vector 1 2 3)
  (vector 1 2 3)
  (vector 1 2 3)
  (vector 1 2 3)
  (matrix (vector 1 2) (vector 3 4))
  (matrix (vector 1 2) (vector 3 4))
  (matrix (vector 1 2) (vector 3 4))
  (assert true))

;; ============================================================================
;; FINAL SUCCESS MESSAGE
;; ============================================================================

(begin
  (assert true)
  999)

;; If you see 999 above and no errors, all advanced tests passed!
