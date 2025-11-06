// Integration tests for Lisp DSL
//
// Executes Lisp test files and verifies output

use rustsparse::lisp::eval::{Evaluator, Value};
use std::fs;

#[test]
fn test_basic_lisp() {
    let source = fs::read_to_string("tests/lisp/basic.lisp").expect("Failed to read basic.lisp");

    let mut eval = Evaluator::new();
    let results = eval
        .eval_program(&source)
        .expect("Failed to evaluate program");

    // Test 1: (vector 1 2 3) should create a 3-element vector
    if let Value::Vector(v) = &results[0] {
        assert_eq!(v.values().len(), 3);
        assert_eq!(v.values()[0], 1.0);
        assert_eq!(v.values()[1], 2.0);
        assert_eq!(v.values()[2], 3.0);
    } else {
        panic!("Expected vector in result 0");
    }

    // Test 2: (vector 4 5 6) should create another 3-element vector
    if let Value::Vector(v) = &results[1] {
        assert_eq!(v.values().len(), 3);
        assert_eq!(v.values()[0], 4.0);
        assert_eq!(v.values()[1], 5.0);
        assert_eq!(v.values()[2], 6.0);
    } else {
        panic!("Expected vector in result 1");
    }

    // Test 3: (matrix (vector 1 2) (vector 3 4)) should create a 2x2 matrix
    if let Value::Matrix(m) = &results[2] {
        assert_eq!(m.nrows(), 2);
        assert_eq!(m.ncols(), 2);
    } else {
        panic!("Expected matrix in result 2");
    }

    // Test 4: 3x3 matrix
    if let Value::Matrix(m) = &results[3] {
        assert_eq!(m.nrows(), 3);
        assert_eq!(m.ncols(), 3);
    } else {
        panic!("Expected matrix in result 3");
    }

    // Test 5: Literal 42
    if let Value::Number(n) = results[4] {
        assert!((n - 42.0).abs() < 1e-10);
    } else {
        panic!("Expected number 42 in result 4");
    }

    // Test 6: Another vector
    if let Value::Vector(v) = &results[5] {
        assert_eq!(v.values().len(), 3);
        assert_eq!(v.values()[0], 7.0);
        assert_eq!(v.values()[1], 8.0);
        assert_eq!(v.values()[2], 9.0);
    } else {
        panic!("Expected vector in result 5");
    }
}

#[test]
fn test_semiring_operations() {
    let source =
        fs::read_to_string("tests/lisp/semirings.lisp").expect("Failed to read semirings.lisp");

    let mut eval = Evaluator::new();
    let results = eval
        .eval_program(&source)
        .expect("Failed to evaluate program");

    // All results should be matrices
    for (i, result) in results.iter().enumerate() {
        match result {
            Value::Matrix(m) => {
                println!("Result {}: {}x{} matrix", i, m.nrows(), m.ncols());
            }
            Value::Vector(v) => {
                println!("Result {}: {}-element vector", i, v.values().len());
            }
            _ => {
                panic!("Result {} is not a matrix or vector", i);
            }
        }
    }

    // Test 1: Matrix multiply result should be 2x2
    if let Value::Matrix(m) = &results[0] {
        assert_eq!(m.nrows(), 2);
        assert_eq!(m.ncols(), 2);
    } else {
        panic!("Expected matrix in result 0");
    }

    // Test 2: Matrix-vector multiply should produce a 3-element vector
    if let Value::Vector(v) = &results[1] {
        assert_eq!(v.values().len(), 3);
        // Expected: [7, 6, 19] from the test in basic_spmv.rs
        assert!(
            (v.values()[0] - 7.0).abs() < 1e-10,
            "Expected 7.0, got {}",
            v.values()[0]
        );
        assert!(
            (v.values()[1] - 6.0).abs() < 1e-10,
            "Expected 6.0, got {}",
            v.values()[1]
        );
        assert!(
            (v.values()[2] - 19.0).abs() < 1e-10,
            "Expected 19.0, got {}",
            v.values()[2]
        );
    } else {
        panic!("Expected vector in result 1");
    }

    // Test 3: Min-plus semiring result
    if let Value::Matrix(m) = &results[2] {
        assert_eq!(m.nrows(), 3);
        assert_eq!(m.ncols(), 3);
    } else {
        panic!("Expected matrix in result 2");
    }

    // Test 4: Max-times semiring result
    if let Value::Matrix(m) = &results[3] {
        assert_eq!(m.nrows(), 3);
        assert_eq!(m.ncols(), 3);
    } else {
        panic!("Expected matrix in result 3");
    }
}

#[test]
fn test_eval_simple_expr() {
    let mut eval = Evaluator::new();

    // Test literal
    let results = eval.eval_program("42").unwrap();
    assert_eq!(results.len(), 1);
    if let Value::Number(n) = results[0] {
        assert_eq!(n, 42.0);
    } else {
        panic!("Expected number");
    }

    // Test vector creation
    let results = eval.eval_program("(vector 1 2 3)").unwrap();
    assert_eq!(results.len(), 1);
    if let Value::Vector(v) = &results[0] {
        assert_eq!(v.values().len(), 3);
    } else {
        panic!("Expected vector");
    }
}

#[test]
fn test_eval_matrix_vector_multiply() {
    let mut eval = Evaluator::new();

    // Simple 2x2 matrix times 2-element vector
    let source = r#"
        (mxv
          (matrix (vector 2 0) (vector 0 3))
          (vector 5 7))
    "#;

    let results = eval.eval_program(source).unwrap();
    assert_eq!(results.len(), 1);

    if let Value::Vector(v) = &results[0] {
        assert_eq!(v.values().len(), 2);
        // Expected: [2*5 + 0*7, 0*5 + 3*7] = [10, 21]
        assert_eq!(v.values()[0], 10.0);
        assert_eq!(v.values()[1], 21.0);
    } else {
        panic!("Expected vector result");
    }
}

#[test]
fn test_eval_matrix_multiply() {
    let mut eval = Evaluator::new();

    // 2x2 matrix multiplication
    let source = r#"
        (plus-times
          (matrix (vector 1 2) (vector 3 4))
          (matrix (vector 5 6) (vector 7 8)))
    "#;

    let results = eval.eval_program(source).unwrap();
    assert_eq!(results.len(), 1);

    if let Value::Matrix(m) = &results[0] {
        assert_eq!(m.nrows(), 2);
        assert_eq!(m.ncols(), 2);
        // Result should be:
        // [1*5 + 2*7,  1*6 + 2*8]   [19, 22]
        // [3*5 + 4*7,  3*6 + 4*8] = [43, 50]
    } else {
        panic!("Expected matrix result");
    }
}
