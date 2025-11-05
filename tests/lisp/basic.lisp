;; Basic Lisp DSL Tests
;; Test vector and matrix creation

;; Test 1: Create a simple vector
(vector 1 2 3)

;; Test 2: Create another vector
(vector 4 5 6)

;; Test 3: Create a 2x2 matrix
(matrix (vector 1 2) (vector 3 4))

;; Test 4: Create a 3x3 identity-like matrix
(matrix (vector 1 0 0) (vector 0 1 0) (vector 0 0 1))

;; Test 5: Simple literal
42

;; Test 6: Another vector
(vector 7 8 9)
