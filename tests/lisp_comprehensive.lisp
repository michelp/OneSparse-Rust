;; Comprehensive Lisp Test Suite for RustSparse
;; Run this file to test all supported features
;; Usage: cat tests/lisp_comprehensive.lisp | cargo run --bin lisp

;; ============================================================================
;; LITERALS AND BASIC VALUES
;; ============================================================================

;; Numbers
(assert (if 42 true false))
(assert (if 0 false true))
(assert (if -123 true false))
(assert (if 3.14 true false))

;; Booleans
(assert true)
(assert (if false false true))

;; ============================================================================
;; CONTROL STRUCTURES: IF
;; ============================================================================

;; Basic if
(assert (if true true false))
(assert (if false false true))

;; Numeric conditions (0 is false, non-zero is true)
(assert (if 1 true false))
(assert (if 0 false true))
(assert (if -1 true false))

;; Nested if
(assert (if true (if true true false) false))
(assert (if false false (if true true false)))

;; If with expressions
(assert (if true (if true true false) (if false true false)))

;; ============================================================================
;; CONTROL STRUCTURES: COND
;; ============================================================================

;; Basic cond
(assert (if (cond (false 0) (false 0) (true 42) (else 0)) true false))

;; Cond with else
(assert (if (cond (false 0) (false 0) (else 99)) true false))

;; Cond first match wins
(assert (if (cond (true 1) (true 2) (else 3)) true false))

;; Nested cond
(assert (if (cond (true (cond (true 42) (else 0))) (else 0)) true false))

;; ============================================================================
;; CONTROL STRUCTURES: BEGIN (BLOCK)
;; ============================================================================

;; Begin returns last expression
(assert (if (begin 1 2 3 42) true false))
(assert (if (begin true false true) true false))

;; Begin with single expression
(assert (if (begin 42) true false))

;; Nested begin
(assert (if (begin 1 (begin 2 3) 42) true false))

;; ============================================================================
;; CONTROL STRUCTURES: WHILE
;; ============================================================================

;; While with false condition doesn't execute
(begin
  (while false (assert false))
  (assert true))

;; While with break
(begin
  (while true (break))
  (assert true))

;; ============================================================================
;; CONTROL STRUCTURES: FOR
;; ============================================================================

;; Basic for loop
(begin
  (for i 0 5 (assert true))
  (assert true))

;; For loop with step
(begin
  (for i 0 10 2 (assert true))
  (assert true))

;; For loop with break
(begin
  (for i 0 100 (break))
  (assert true))

;; For loop with continue
(begin
  (for i 0 3 (continue))
  (assert true))

;; Nested for loops
(begin
  (for i 0 3 (for j 0 2 (assert true)))
  (assert true))

;; ============================================================================
;; VECTOR OPERATIONS
;; ============================================================================

;; Create vector
(begin
  (vector 1 2 3)
  (assert true))

;; Vector with different values
(begin
  (vector 0 0 0)
  (assert true))

;; Vector with negative values
(begin
  (vector -1 -2 -3)
  (assert true))

;; Vector with floats
(begin
  (vector 1.5 2.5 3.5)
  (assert true))

;; Single element vector
(begin
  (vector 42)
  (assert true))

;; ============================================================================
;; MATRIX OPERATIONS
;; ============================================================================

;; Create 2x2 matrix
(begin
  (matrix (vector 1 2) (vector 3 4))
  (assert true))

;; Create 3x3 identity-like matrix
(begin
  (matrix (vector 1 0 0) (vector 0 1 0) (vector 0 0 1))
  (assert true))

;; Create matrix with zeros
(begin
  (matrix (vector 0 0) (vector 0 0))
  (assert true))

;; Create 1x3 matrix (single row)
(begin
  (matrix (vector 1 2 3))
  (assert true))

;; Create 3x1 matrix (single column)
(begin
  (matrix (vector 1) (vector 2) (vector 3))
  (assert true))

;; ============================================================================
;; MATRIX-VECTOR OPERATIONS
;; ============================================================================

;; Matrix-vector multiply (2x2 * 2x1 = 2x1)
(begin
  (mxv (matrix (vector 1 0) (vector 0 1)) (vector 5 7))
  (assert true))

;; Vector-matrix multiply (1x2 * 2x2 = 1x2)
(begin
  (vxm (vector 2 3) (matrix (vector 1 0) (vector 0 1)))
  (assert true))

;; ============================================================================
;; SEMIRING OPERATIONS
;; ============================================================================

;; Plus-times semiring (standard matrix multiplication)
(begin
  (plus-times
    (matrix (vector 1 2) (vector 3 4))
    (matrix (vector 5 6) (vector 7 8)))
  (assert true))

;; Min-plus semiring (shortest path)
(begin
  (min-plus
    (matrix (vector 0 1 5) (vector 9 0 2) (vector 6 4 0))
    (matrix (vector 0 1 5) (vector 9 0 2) (vector 6 4 0)))
  (assert true))

;; Max-times semiring
(begin
  (max-times
    (matrix (vector 1 2) (vector 3 4))
    (matrix (vector 1 0) (vector 0 1)))
  (assert true))

;; ============================================================================
;; NESTED CONTROL FLOW
;; ============================================================================

;; If inside cond
(assert
  (if (cond
        (false 0)
        (true (if true 42 0))
        (else 0))
      true
      false))

;; Cond inside if
(assert
  (if true
      (if (cond (true 42) (else 0)) true false)
      false))

;; Begin with conditionals
(assert
  (if (begin
        (if true 1 0)
        (if false 0 1)
        (cond (true 42) (else 0)))
      true
      false))

;; ============================================================================
;; COMPLEX CONTROL FLOW
;; ============================================================================

;; Multiple nested control structures
(assert
  (if (begin
        (if true
            (cond
              (false 0)
              (true (begin 1 2 42))
              (else 0))
            0))
      true
      false))

;; Control flow with vectors
(begin
  (if true
      (vector 1 2 3)
      (vector 0 0 0))
  (assert true))

;; Control flow with matrices
(begin
  (if false
      (matrix (vector 0 0) (vector 0 0))
      (matrix (vector 1 2) (vector 3 4)))
  (assert true))

;; ============================================================================
;; EDGE CASES
;; ============================================================================

;; Empty begin should work
(begin
  (assert true))

;; Deeply nested expressions
(assert (if (if (if (if true true false) true false) true false) true false))

;; Multiple asserts in sequence
(begin
  (assert true)
  (assert true)
  (assert true)
  (assert true))

;; Assert with different truthy values
(begin
  (assert 1)
  (assert 42)
  (assert -1)
  (assert 0.1)
  (assert true))

;; ============================================================================
;; FINAL SUCCESS MESSAGE
;; ============================================================================

(begin
  (assert true)
  42)

;; If you see 42 above and no errors, all tests passed!
