;; Semiring Operations Tests
;; Test various semiring operations on matrices

;; Test 1: Matrix multiply with plus-times
(plus-times
  (matrix (vector 1 2) (vector 3 4))
  (matrix (vector 5 6) (vector 7 8)))

;; Test 2: Matrix-vector multiply
(mxv
  (matrix (vector 1 0 2) (vector 0 3 0) (vector 4 0 5))
  (vector 1 2 3))

;; Test 3: Min-plus semiring (shortest path)
(min-plus
  (matrix (vector 0 1 5) (vector 9 0 2) (vector 6 4 0))
  (matrix (vector 0 1 5) (vector 9 0 2) (vector 6 4 0)))

;; Test 4: Max-times semiring
(max-times
  (matrix (vector 0.9 0.0 0.5) (vector 0.0 0.8 0.0) (vector 0.7 0.0 0.6))
  (matrix (vector 0.9 0.0 0.5) (vector 0.0 0.8 0.0) (vector 0.7 0.0 0.6)))
