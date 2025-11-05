// JIT SpMV Tests
//
// Tests that the JIT-compiled SpMV kernel produces correct results

use rustsparse::core::matrix::Matrix;
use rustsparse::core::semiring::Semiring;
use rustsparse::core::vector::Vector;
use rustsparse::ops::mxv;

#[test]
fn test_jit_spmv_simple() {
    // Create a simple CSR matrix:
    // [[1.0, 0.0, 2.0],
    //  [0.0, 3.0, 0.0],
    //  [4.0, 0.0, 5.0]]
    //
    // CSR format:
    // row_ptrs = [0, 2, 3, 5]
    // col_indices = [0, 2, 1, 0, 2]
    // values = [1.0, 2.0, 3.0, 4.0, 5.0]

    let row_ptrs = vec![0, 2, 3, 5];
    let col_indices = vec![0, 2, 1, 0, 2];
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let a = Matrix::from_csr(3, 3, row_ptrs, col_indices, values).unwrap();

    // Create input vector [1.0, 2.0, 3.0]
    let mut u = Vector::<f64>::new(3).unwrap();
    u.values_mut().extend_from_slice(&[1.0, 2.0, 3.0]);
    u.indices_mut().extend_from_slice(&[0, 1, 2]);

    // Create output vector
    let mut w = Vector::<f64>::new(3).unwrap();
    w.values_mut().resize(3, 0.0);
    w.indices_mut().extend_from_slice(&[0, 1, 2]);

    // Perform SpMV: w = A * u
    let semiring = Semiring::plus_times().unwrap();
    mxv(&mut w, None, &a, &u, &semiring, None).unwrap();

    // Expected result:
    // row 0: 1*1 + 2*3 = 1 + 6 = 7
    // row 1: 3*2 = 6
    // row 2: 4*1 + 5*3 = 4 + 15 = 19

    let result = w.values();
    assert_eq!(result.len(), 3);
    assert!((result[0] - 7.0).abs() < 1e-10, "Expected 7.0, got {}", result[0]);
    assert!((result[1] - 6.0).abs() < 1e-10, "Expected 6.0, got {}", result[1]);
    assert!((result[2] - 19.0).abs() < 1e-10, "Expected 19.0, got {}", result[2]);
}

#[test]
fn test_jit_spmv_identity() {
    // Test with identity matrix:
    // [[1, 0, 0],
    //  [0, 1, 0],
    //  [0, 0, 1]]

    let row_ptrs = vec![0, 1, 2, 3];
    let col_indices = vec![0, 1, 2];
    let values = vec![1.0, 1.0, 1.0];

    let a = Matrix::from_csr(3, 3, row_ptrs, col_indices, values).unwrap();

    let mut u = Vector::<f64>::new(3).unwrap();
    u.values_mut().extend_from_slice(&[2.0, 3.0, 5.0]);
    u.indices_mut().extend_from_slice(&[0, 1, 2]);

    let mut w = Vector::<f64>::new(3).unwrap();
    w.values_mut().resize(3, 0.0);
    w.indices_mut().extend_from_slice(&[0, 1, 2]);

    let semiring = Semiring::plus_times().unwrap();
    mxv(&mut w, None, &a, &u, &semiring, None).unwrap();

    // Identity matrix should return the same vector
    let result = w.values();
    assert_eq!(result[0], 2.0);
    assert_eq!(result[1], 3.0);
    assert_eq!(result[2], 5.0);
}

#[test]
fn test_jit_spmv_zero_row() {
    // Test with a row that has no non-zeros:
    // [[1, 2],
    //  [0, 0],
    //  [3, 4]]

    let row_ptrs = vec![0, 2, 2, 4];
    let col_indices = vec![0, 1, 0, 1];
    let values = vec![1.0, 2.0, 3.0, 4.0];

    let a = Matrix::from_csr(3, 2, row_ptrs, col_indices, values).unwrap();

    let mut u = Vector::<f64>::new(2).unwrap();
    u.values_mut().extend_from_slice(&[5.0, 6.0]);
    u.indices_mut().extend_from_slice(&[0, 1]);

    let mut w = Vector::<f64>::new(3).unwrap();
    w.values_mut().resize(3, 0.0);
    w.indices_mut().extend_from_slice(&[0, 1, 2]);

    let semiring = Semiring::plus_times().unwrap();
    mxv(&mut w, None, &a, &u, &semiring, None).unwrap();

    // Expected:
    // row 0: 1*5 + 2*6 = 17
    // row 1: 0 (no non-zeros)
    // row 2: 3*5 + 4*6 = 39

    let result = w.values();
    assert_eq!(result[0], 17.0);
    assert_eq!(result[1], 0.0);
    assert_eq!(result[2], 39.0);
}

#[test]
fn test_jit_spmv_large() {
    // Test with a larger matrix (10x10)
    // Diagonal matrix with values 1, 2, 3, ..., 10

    let row_ptrs: Vec<usize> = (0..11).collect();
    let col_indices: Vec<usize> = (0..10).collect();
    let values: Vec<f64> = (1..=10).map(|x| x as f64).collect();

    let a = Matrix::from_csr(10, 10, row_ptrs, col_indices, values).unwrap();

    let mut u = Vector::<f64>::new(10).unwrap();
    u.values_mut().extend((0..10).map(|x| x as f64));
    u.indices_mut().extend(0..10);

    let mut w = Vector::<f64>::new(10).unwrap();
    w.values_mut().resize(10, 0.0);
    w.indices_mut().extend(0..10);

    let semiring = Semiring::plus_times().unwrap();
    mxv(&mut w, None, &a, &u, &semiring, None).unwrap();

    // Expected: w[i] = (i+1) * i
    let result = w.values();
    for i in 0..10 {
        let expected = (i + 1) as f64 * i as f64;
        assert_eq!(result[i], expected, "Mismatch at index {}", i);
    }
}

#[test]
fn test_jit_spmv_min_plus_shortest_path() {
    // Test min-plus semiring (shortest path)
    // Matrix represents edge weights:
    // [[0, 1, 5],
    //  [9, 0, 2],
    //  [6, 4, 0]]

    let row_ptrs = vec![0, 3, 6, 9];
    let col_indices = vec![0, 1, 2, 0, 1, 2, 0, 1, 2];
    let values = vec![0.0, 1.0, 5.0, 9.0, 0.0, 2.0, 6.0, 4.0, 0.0];

    let a = Matrix::from_csr(3, 3, row_ptrs, col_indices, values).unwrap();

    // Input: distances from source [0, 3, inf]
    let mut u = Vector::<f64>::new(3).unwrap();
    u.values_mut().extend_from_slice(&[0.0, 3.0, f64::INFINITY]);
    u.indices_mut().extend_from_slice(&[0, 1, 2]);

    let mut w = Vector::<f64>::new(3).unwrap();
    w.values_mut().resize(3, f64::INFINITY);
    w.indices_mut().extend_from_slice(&[0, 1, 2]);

    let semiring = Semiring::min_plus().unwrap();
    mxv(&mut w, None, &a, &u, &semiring, None).unwrap();

    // Expected with min-plus:
    // row 0: min(0+0, 1+3, 5+inf) = min(0, 4, inf) = 0
    // row 1: min(9+0, 0+3, 2+inf) = min(9, 3, inf) = 3
    // row 2: min(6+0, 4+3, 0+inf) = min(6, 7, inf) = 6

    let result = w.values();
    assert_eq!(result[0], 0.0, "Row 0 shortest path");
    assert_eq!(result[1], 3.0, "Row 1 shortest path");
    assert_eq!(result[2], 6.0, "Row 2 shortest path");
}

#[test]
fn test_jit_spmv_max_times() {
    // Test max-times semiring (maximum weighted path)
    // Matrix represents probabilities/weights:
    // [[0.9, 0.0, 0.5],
    //  [0.0, 0.8, 0.0],
    //  [0.7, 0.0, 0.6]]

    let row_ptrs = vec![0, 2, 3, 5];
    let col_indices = vec![0, 2, 1, 0, 2];
    let values = vec![0.9, 0.5, 0.8, 0.7, 0.6];

    let a = Matrix::from_csr(3, 3, row_ptrs, col_indices, values).unwrap();

    // Input vector [1.0, 0.9, 0.8]
    let mut u = Vector::<f64>::new(3).unwrap();
    u.values_mut().extend_from_slice(&[1.0, 0.9, 0.8]);
    u.indices_mut().extend_from_slice(&[0, 1, 2]);

    let mut w = Vector::<f64>::new(3).unwrap();
    w.values_mut().resize(3, 0.0);
    w.indices_mut().extend_from_slice(&[0, 1, 2]);

    let semiring = Semiring::max_times().unwrap();
    mxv(&mut w, None, &a, &u, &semiring, None).unwrap();

    // Expected with max-times:
    // row 0: max(0.9*1.0, 0.5*0.8) = max(0.9, 0.4) = 0.9
    // row 1: max(0.8*0.9) = 0.72
    // row 2: max(0.7*1.0, 0.6*0.8) = max(0.7, 0.48) = 0.7

    let result = w.values();
    assert!((result[0] - 0.9).abs() < 1e-10, "Expected 0.9, got {}", result[0]);
    assert!((result[1] - 0.72).abs() < 1e-10, "Expected 0.72, got {}", result[1]);
    assert!((result[2] - 0.7).abs() < 1e-10, "Expected 0.7, got {}", result[2]);
}

#[test]
fn test_jit_spmv_different_semirings_same_structure() {
    // Critical test: Verify that different semirings produce different results
    // and are correctly cached separately

    let row_ptrs = vec![0, 2, 3];
    let col_indices = vec![0, 1, 1];
    let values = vec![2.0, 3.0, 4.0];

    let a = Matrix::from_csr(2, 2, row_ptrs, col_indices, values).unwrap();

    let mut u = Vector::<f64>::new(2).unwrap();
    u.values_mut().extend_from_slice(&[5.0, 6.0]);
    u.indices_mut().extend_from_slice(&[0, 1]);

    // Test plus-times
    let mut w1 = Vector::<f64>::new(2).unwrap();
    w1.values_mut().resize(2, 0.0);
    w1.indices_mut().extend_from_slice(&[0, 1]);

    let semiring_plus = Semiring::plus_times().unwrap();
    mxv(&mut w1, None, &a, &u, &semiring_plus, None).unwrap();

    // row 0: 2*5 + 3*6 = 10 + 18 = 28
    // row 1: 4*6 = 24
    assert_eq!(w1.values()[0], 28.0);
    assert_eq!(w1.values()[1], 24.0);

    // Test min-plus
    let mut w2 = Vector::<f64>::new(2).unwrap();
    w2.values_mut().resize(2, f64::INFINITY);
    w2.indices_mut().extend_from_slice(&[0, 1]);

    let semiring_min = Semiring::min_plus().unwrap();
    mxv(&mut w2, None, &a, &u, &semiring_min, None).unwrap();

    // row 0: min(2+5, 3+6) = min(7, 9) = 7
    // row 1: 4+6 = 10
    assert_eq!(w2.values()[0], 7.0);
    assert_eq!(w2.values()[1], 10.0);

    // Test max-times
    let mut w3 = Vector::<f64>::new(2).unwrap();
    w3.values_mut().resize(2, 0.0);
    w3.indices_mut().extend_from_slice(&[0, 1]);

    let semiring_max = Semiring::max_times().unwrap();
    mxv(&mut w3, None, &a, &u, &semiring_max, None).unwrap();

    // row 0: max(2*5, 3*6) = max(10, 18) = 18
    // row 1: 4*6 = 24
    assert_eq!(w3.values()[0], 18.0);
    assert_eq!(w3.values()[1], 24.0);

    // Verify all three produced different results
    assert_ne!(w1.values()[0], w2.values()[0]);
    assert_ne!(w1.values()[0], w3.values()[0]);
    assert_ne!(w2.values()[0], w3.values()[0]);
}
