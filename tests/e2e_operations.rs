// End-to-End Integration Tests
//
// Tests the complete pipeline: API → IR Building → Optimization → Compilation → Execution
//
// These tests use the IR interpreter to validate that operations produce correct results
// through the entire JIT compilation pipeline.

use rustsparse::core::matrix::Matrix;
use rustsparse::core::semiring::Semiring;
use rustsparse::core::vector::Vector;
use rustsparse::ir::builder::GraphBuilder;
use rustsparse::ir::interpreter::{Interpreter, InterpreterValue};
use rustsparse::ir::node::{BinaryOpKind, UnaryOpKind};
use rustsparse::ir::types::ScalarType;
use rustsparse::ir::Shape;
use rustsparse::ops::*;

#[test]
fn test_e2e_matmul_builds_correct_graph() {
    // Test that mxm correctly builds an IR graph
    let mut c = Matrix::<f64>::new(2, 2).unwrap();
    let a = Matrix::<f64>::new(2, 3).unwrap();
    let b = Matrix::<f64>::new(3, 2).unwrap();

    let semiring = Semiring::plus_times().unwrap();

    // This should successfully build and execute the pipeline
    let result = mxm(&mut c, None, &a, &b, &semiring, None);
    assert!(result.is_ok());
}

#[test]
fn test_e2e_matmul_computation() {
    // Use interpreter to validate matmul produces correct results
    let mut builder = GraphBuilder::new();

    let a = builder
        .input_matrix("A", ScalarType::Float64, Shape::matrix(2, 3))
        .unwrap();
    let b = builder
        .input_matrix("B", ScalarType::Float64, Shape::matrix(3, 2))
        .unwrap();

    let semiring = rustsparse::ir::semirings::plus_times(ScalarType::Float64);
    let c = builder.matmul(a, b, semiring).unwrap();
    builder.output(c).unwrap();

    let graph = builder.build();

    let mut interp = Interpreter::new();
    interp.set_input(
        "A",
        InterpreterValue::Matrix(vec![vec![1.0, 0.0, 2.0], vec![0.0, 3.0, 0.0]]),
    );
    interp.set_input(
        "B",
        InterpreterValue::Matrix(vec![vec![4.0, 0.0], vec![0.0, 5.0], vec![6.0, 0.0]]),
    );

    let outputs = interp.execute(&graph).unwrap();
    let result = outputs[0].as_matrix().unwrap();

    // Expected: [[1*4+0*0+2*6, 1*0+0*5+2*0], [0*4+3*0+0*6, 0*0+3*5+0*0]]
    //         = [[16, 0], [0, 15]]
    assert_eq!(result[0][0], 16.0);
    assert_eq!(result[0][1], 0.0);
    assert_eq!(result[1][0], 0.0);
    assert_eq!(result[1][1], 15.0);
}

#[test]
fn test_e2e_matvec_computation() {
    let mut builder = GraphBuilder::new();

    let a = builder
        .input_matrix("A", ScalarType::Float64, Shape::matrix(3, 2))
        .unwrap();
    let u = builder
        .input_vector("u", ScalarType::Float64, Shape::vector(2))
        .unwrap();

    let semiring = rustsparse::ir::semirings::plus_times(ScalarType::Float64);
    let w = builder.matvec(a, u, semiring).unwrap();
    builder.output(w).unwrap();

    let graph = builder.build();

    let mut interp = Interpreter::new();
    interp.set_input(
        "A",
        InterpreterValue::Matrix(vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]),
    );
    interp.set_input("u", InterpreterValue::Vector(vec![2.0, 3.0]));

    let outputs = interp.execute(&graph).unwrap();
    let result = outputs[0].as_vector().unwrap();

    // Expected: [1*2+2*3, 3*2+4*3, 5*2+6*3] = [8, 18, 28]
    assert_eq!(result[0], 8.0);
    assert_eq!(result[1], 18.0);
    assert_eq!(result[2], 28.0);
}

#[test]
fn test_e2e_ewise_add_builds_correct_graph() {
    let mut c = Matrix::<f64>::new(2, 2).unwrap();
    let a = Matrix::<f64>::new(2, 2).unwrap();
    let b = Matrix::<f64>::new(2, 2).unwrap();

    let result = ewadd_matrix(&mut c, None, &a, &b, BinaryOpKind::Add, None);
    assert!(result.is_ok());
}

#[test]
fn test_e2e_ewise_add_computation() {
    let mut builder = GraphBuilder::new();

    let a = builder
        .input_matrix("A", ScalarType::Float64, Shape::matrix(2, 2))
        .unwrap();
    let b = builder
        .input_matrix("B", ScalarType::Float64, Shape::matrix(2, 2))
        .unwrap();

    let c = builder.ewise_add(a, b, BinaryOpKind::Add).unwrap();
    builder.output(c).unwrap();

    let graph = builder.build();

    let mut interp = Interpreter::new();
    interp.set_input(
        "A",
        InterpreterValue::Matrix(vec![vec![1.0, 2.0], vec![3.0, 4.0]]),
    );
    interp.set_input(
        "B",
        InterpreterValue::Matrix(vec![vec![10.0, 20.0], vec![30.0, 40.0]]),
    );

    let outputs = interp.execute(&graph).unwrap();
    let result = outputs[0].as_matrix().unwrap();

    assert_eq!(result[0][0], 11.0);
    assert_eq!(result[0][1], 22.0);
    assert_eq!(result[1][0], 33.0);
    assert_eq!(result[1][1], 44.0);
}

#[test]
fn test_e2e_ewise_mult_computation() {
    let mut builder = GraphBuilder::new();

    let a = builder
        .input_vector("a", ScalarType::Float64, Shape::vector(3))
        .unwrap();
    let b = builder
        .input_vector("b", ScalarType::Float64, Shape::vector(3))
        .unwrap();

    let c = builder.ewise_mult(a, b, BinaryOpKind::Mul).unwrap();
    builder.output(c).unwrap();

    let graph = builder.build();

    let mut interp = Interpreter::new();
    interp.set_input("a", InterpreterValue::Vector(vec![2.0, 3.0, 4.0]));
    interp.set_input("b", InterpreterValue::Vector(vec![5.0, 6.0, 7.0]));

    let outputs = interp.execute(&graph).unwrap();
    let result = outputs[0].as_vector().unwrap();

    assert_eq!(result[0], 10.0);
    assert_eq!(result[1], 18.0);
    assert_eq!(result[2], 28.0);
}

#[test]
fn test_e2e_apply_builds_correct_graph() {
    let mut c = Matrix::<f64>::new(2, 2).unwrap();
    let a = Matrix::<f64>::new(2, 2).unwrap();

    let result = apply_matrix(&mut c, None, &a, UnaryOpKind::Abs, None);
    assert!(result.is_ok());
}

#[test]
fn test_e2e_apply_computation() {
    let mut builder = GraphBuilder::new();

    let a = builder
        .input_matrix("A", ScalarType::Float64, Shape::matrix(2, 2))
        .unwrap();

    let b = builder.apply(a, UnaryOpKind::Abs).unwrap();
    builder.output(b).unwrap();

    let graph = builder.build();

    let mut interp = Interpreter::new();
    interp.set_input(
        "A",
        InterpreterValue::Matrix(vec![vec![-1.0, 2.0], vec![-3.0, -4.0]]),
    );

    let outputs = interp.execute(&graph).unwrap();
    let result = outputs[0].as_matrix().unwrap();

    assert_eq!(result[0][0], 1.0);
    assert_eq!(result[0][1], 2.0);
    assert_eq!(result[1][0], 3.0);
    assert_eq!(result[1][1], 4.0);
}

#[test]
fn test_e2e_apply_binary_left_builds_correct_graph() {
    let mut c = Matrix::<f64>::new(2, 2).unwrap();
    let a = Matrix::<f64>::new(2, 2).unwrap();

    let result = apply_binary_left_matrix(&mut c, None, 5.0, &a, BinaryOpKind::Mul, None);
    assert!(result.is_ok());
}

#[test]
fn test_e2e_apply_binary_right_builds_correct_graph() {
    let mut c = Matrix::<f64>::new(2, 2).unwrap();
    let a = Matrix::<f64>::new(2, 2).unwrap();

    let result = apply_binary_right_matrix(&mut c, None, &a, 10.0, BinaryOpKind::Add, None);
    assert!(result.is_ok());
}

#[test]
fn test_e2e_transpose_computation() {
    let mut builder = GraphBuilder::new();

    let a = builder
        .input_matrix("A", ScalarType::Float64, Shape::matrix(2, 3))
        .unwrap();

    let b = builder.transpose(a).unwrap();
    builder.output(b).unwrap();

    let graph = builder.build();

    let mut interp = Interpreter::new();
    interp.set_input(
        "A",
        InterpreterValue::Matrix(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]),
    );

    let outputs = interp.execute(&graph).unwrap();
    let result = outputs[0].as_matrix().unwrap();

    assert_eq!(result.len(), 3);
    assert_eq!(result[0].len(), 2);
    assert_eq!(result[0][0], 1.0);
    assert_eq!(result[0][1], 4.0);
    assert_eq!(result[1][0], 2.0);
    assert_eq!(result[1][1], 5.0);
    assert_eq!(result[2][0], 3.0);
    assert_eq!(result[2][1], 6.0);
}

#[test]
fn test_e2e_chained_operations() {
    // Test: C = (A * B) + D
    // This should trigger fusion optimization
    let mut builder = GraphBuilder::new();

    let a = builder
        .input_matrix("A", ScalarType::Float64, Shape::matrix(2, 2))
        .unwrap();
    let b = builder
        .input_matrix("B", ScalarType::Float64, Shape::matrix(2, 2))
        .unwrap();
    let d = builder
        .input_matrix("D", ScalarType::Float64, Shape::matrix(2, 2))
        .unwrap();

    let semiring = rustsparse::ir::semirings::plus_times(ScalarType::Float64);
    let ab = builder.matmul(a, b, semiring).unwrap();
    let c = builder.ewise_add(ab, d, BinaryOpKind::Add).unwrap();
    builder.output(c).unwrap();

    let graph = builder.build();

    let mut interp = Interpreter::new();
    interp.set_input(
        "A",
        InterpreterValue::Matrix(vec![vec![1.0, 2.0], vec![3.0, 4.0]]),
    );
    interp.set_input(
        "B",
        InterpreterValue::Matrix(vec![vec![5.0, 6.0], vec![7.0, 8.0]]),
    );
    interp.set_input(
        "D",
        InterpreterValue::Matrix(vec![vec![1.0, 1.0], vec![1.0, 1.0]]),
    );

    let outputs = interp.execute(&graph).unwrap();
    let result = outputs[0].as_matrix().unwrap();

    // A*B = [[19, 22], [43, 50]]
    // (A*B)+D = [[20, 23], [44, 51]]
    assert_eq!(result[0][0], 20.0);
    assert_eq!(result[0][1], 23.0);
    assert_eq!(result[1][0], 44.0);
    assert_eq!(result[1][1], 51.0);
}

#[test]
fn test_e2e_complex_expression() {
    // Test: result = abs(A^T * B) where ^T is transpose
    let mut builder = GraphBuilder::new();

    let a = builder
        .input_matrix("A", ScalarType::Float64, Shape::matrix(3, 2))
        .unwrap();
    let b = builder
        .input_matrix("B", ScalarType::Float64, Shape::matrix(3, 2))
        .unwrap();

    let at = builder.transpose(a).unwrap();
    let semiring = rustsparse::ir::semirings::plus_times(ScalarType::Float64);
    let atb = builder.matmul(at, b, semiring).unwrap();
    let result = builder.apply(atb, UnaryOpKind::Abs).unwrap();
    builder.output(result).unwrap();

    let graph = builder.build();

    let mut interp = Interpreter::new();
    interp.set_input(
        "A",
        InterpreterValue::Matrix(vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]),
    );
    interp.set_input(
        "B",
        InterpreterValue::Matrix(vec![vec![7.0, 8.0], vec![9.0, 10.0], vec![11.0, 12.0]]),
    );

    let outputs = interp.execute(&graph).unwrap();
    assert_eq!(outputs.len(), 1);

    let result = outputs[0].as_matrix().unwrap();
    // A^T is 2x3, B is 3x2, so result is 2x2
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].len(), 2);
}

#[test]
fn test_e2e_vector_operations() {
    let mut w = Vector::<f64>::new(3).unwrap();
    let u = Vector::<f64>::new(3).unwrap();
    let v = Vector::<f64>::new(3).unwrap();

    // Test vector element-wise operations work end-to-end
    let result = ewadd_vector(&mut w, None, &u, &v, BinaryOpKind::Add, None);
    assert!(result.is_ok());

    let result = ewmult_vector(&mut w, None, &u, &v, BinaryOpKind::Mul, None);
    assert!(result.is_ok());

    let result = apply_vector(&mut w, None, &u, UnaryOpKind::Sqrt, None);
    assert!(result.is_ok());
}
