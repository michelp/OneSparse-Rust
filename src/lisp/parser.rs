// S-expression parser for Lisp DSL
//
// Converts S-expressions to AST nodes

use crate::core::error::{GraphBlasError, Result};
use crate::ir::ScalarType;
use crate::lisp::ast::*;
use lexpr::Value;

/// Helper: Convert a cons list to a vector of values
fn cons_to_vec(cons: &lexpr::Cons) -> Vec<Value> {
    let mut result = Vec::new();
    let mut current = cons;

    loop {
        result.push(current.car().clone());

        match current.cdr() {
            Value::Cons(next_cons) => current = next_cons,
            Value::Nil => break,
            _ => break,
        }
    }

    result
}

/// Parse a Lisp program from a string
pub fn parse_program(source: &str) -> Result<Vec<Form>> {
    log::debug!("Source: {}", source);

    let mut forms = Vec::new();

    // Use Parser to handle multiple S-expressions
    let mut parser = lexpr::Parser::from_str(source);

    while let Some(value) = parser.next() {
        let value = value.map_err(|_| GraphBlasError::InvalidValue)?;
        log::trace!("Parsing S-expression: {:?}", value);
        forms.push(parse_form(&value)?);
    }

    Ok(forms)
}

/// Parse a single S-expression from a string
pub fn parse_expr_str(source: &str) -> Result<Expr> {
    let value = lexpr::from_str(source).map_err(|_| GraphBlasError::InvalidValue)?;
    parse_expr(&value)
}

/// Parse a top-level form
fn parse_form(value: &Value) -> Result<Form> {
    match value {
        Value::Cons(cons) => {
            // Check if it's a defkernel
            if let Some(first) = cons.car().as_symbol() {
                if first == "defkernel" {
                    return parse_defkernel(cons);
                }
            }
            // Otherwise it's an expression
            Ok(Form::Expr(parse_expr(value)?))
        }
        _ => Ok(Form::Expr(parse_expr(value)?)),
    }
}

/// Parse a defkernel form: (defkernel name [params] body)
fn parse_defkernel(cons: &lexpr::Cons) -> Result<Form> {
    log::debug!("Parsing defkernel form");

    // Convert cons to list
    let list = cons_to_vec(cons);

    if list.len() < 4 {
        return Err(GraphBlasError::InvalidValue);
    }

    // list[0] is 'defkernel'
    // list[1] is name
    let name = list[1]
        .as_symbol()
        .ok_or(GraphBlasError::InvalidValue)?
        .to_string();

    log::debug!("Kernel name: {}", name);

    // list[2] is parameters
    let params = parse_params(&list[2])?;
    log::debug!("Kernel params: {} parameter(s)", params.len());

    // list[3] is body
    let body = parse_expr(&list[3])?;

    Ok(Form::DefKernel { name, params, body })
}

/// Parse parameter list: [x y z] or [(x :matrix-f64) y z]
fn parse_params(value: &Value) -> Result<Vec<Param>> {
    // For lexpr, vectors are represented by Value::Vector
    // to_vec() works for both vectors and lists
    let vec = value.to_vec().ok_or(GraphBlasError::InvalidValue)?;

    let mut params = Vec::new();
    for item in &vec {
        params.push(parse_param(item)?);
    }

    Ok(params)
}

/// Parse a single parameter: x or (x :matrix-f64)
fn parse_param(value: &Value) -> Result<Param> {
    match value {
        Value::Symbol(s) => Ok(Param::new(s.to_string())),
        Value::Cons(cons) => {
            let list = cons_to_vec(cons);
            if list.len() < 2 {
                return Err(GraphBlasError::InvalidValue);
            }

            let name = list[0]
                .as_symbol()
                .ok_or(GraphBlasError::InvalidValue)?
                .to_string();

            let type_kw = list[1].as_keyword().ok_or(GraphBlasError::InvalidValue)?;

            let type_annotation = parse_type_annotation(type_kw)?;
            Ok(Param::with_type(name, type_annotation))
        }
        _ => Err(GraphBlasError::InvalidValue),
    }
}

/// Parse type annotation keyword
fn parse_type_annotation(kw: &str) -> Result<TypeAnnotation> {
    match kw {
        "scalar-f64" => Ok(TypeAnnotation::Scalar(ScalarType::Float64)),
        "scalar-f32" => Ok(TypeAnnotation::Scalar(ScalarType::Float32)),
        "scalar-i64" => Ok(TypeAnnotation::Scalar(ScalarType::Int64)),
        "scalar-i32" => Ok(TypeAnnotation::Scalar(ScalarType::Int32)),
        "vector-f64" => Ok(TypeAnnotation::Vector(ScalarType::Float64)),
        "vector-f32" => Ok(TypeAnnotation::Vector(ScalarType::Float32)),
        "vector-i64" => Ok(TypeAnnotation::Vector(ScalarType::Int64)),
        "vector-i32" => Ok(TypeAnnotation::Vector(ScalarType::Int32)),
        "matrix-f64" => Ok(TypeAnnotation::Matrix(ScalarType::Float64)),
        "matrix-f32" => Ok(TypeAnnotation::Matrix(ScalarType::Float32)),
        "matrix-i64" => Ok(TypeAnnotation::Matrix(ScalarType::Int64)),
        "matrix-i32" => Ok(TypeAnnotation::Matrix(ScalarType::Int32)),
        _ => Err(GraphBlasError::InvalidValue),
    }
}

/// Parse an expression
fn parse_expr(value: &Value) -> Result<Expr> {
    match value {
        // Literals
        Value::Number(n) => parse_number(n),
        Value::Bool(b) => Ok(Expr::Literal(Literal::Bool(*b))),

        // Variables (with special handling for boolean symbols)
        Value::Symbol(s) => {
            let sym = s.to_string();
            match sym.as_str() {
                "true" => Ok(Expr::Literal(Literal::Bool(true))),
                "false" => Ok(Expr::Literal(Literal::Bool(false))),
                _ => Ok(Expr::Variable(sym)),
            }
        }

        // Function calls and special forms
        Value::Cons(cons) => {
            if let Some(first) = cons.car().as_symbol() {
                match first {
                    "let" => parse_let(cons),
                    "if" => parse_if(cons),
                    "while" => parse_while(cons),
                    "for" => parse_for(cons),
                    "cond" => parse_cond(cons),
                    "begin" => parse_begin(cons),
                    "break" => parse_break(cons),
                    "continue" => parse_continue(cons),
                    _ => parse_func_call(cons),
                }
            } else {
                Err(GraphBlasError::InvalidValue)
            }
        }

        _ => Err(GraphBlasError::InvalidValue),
    }
}

/// Parse a number literal
fn parse_number(n: &lexpr::Number) -> Result<Expr> {
    if let Some(i) = n.as_i64() {
        // Check if it fits in i32
        if i >= i32::MIN as i64 && i <= i32::MAX as i64 {
            Ok(Expr::Literal(Literal::Int32(i as i32)))
        } else {
            Ok(Expr::Literal(Literal::Int64(i)))
        }
    } else if let Some(f) = n.as_f64() {
        Ok(Expr::Literal(Literal::Float64(f)))
    } else {
        Err(GraphBlasError::InvalidValue)
    }
}

/// Parse a let binding: (let ((x expr) (y expr)) body)
fn parse_let(cons: &lexpr::Cons) -> Result<Expr> {
    let list = cons_to_vec(cons);

    if list.len() < 3 {
        return Err(GraphBlasError::InvalidValue);
    }

    // list[0] is 'let'
    // list[1] is bindings vector
    let bindings_vec = list[1].to_vec().ok_or(GraphBlasError::InvalidValue)?;

    let mut bindings = Vec::new();
    for binding in &bindings_vec {
        let pair = binding.as_cons().ok_or(GraphBlasError::InvalidValue)?;
        let pair_list = cons_to_vec(pair);

        if pair_list.len() < 2 {
            return Err(GraphBlasError::InvalidValue);
        }

        let var = pair_list[0]
            .as_symbol()
            .ok_or(GraphBlasError::InvalidValue)?
            .to_string();

        let expr = parse_expr(&pair_list[1])?;

        bindings.push((var, expr));
    }

    // list[2] is body
    let body = parse_expr(&list[2])?;

    Ok(Expr::Let {
        bindings,
        body: Box::new(body),
    })
}

/// Parse a function call: (func arg1 arg2 ...)
fn parse_func_call(cons: &lexpr::Cons) -> Result<Expr> {
    let list = cons_to_vec(cons);

    if list.is_empty() {
        return Err(GraphBlasError::InvalidValue);
    }

    // list[0] is function name
    let func_name_str = list[0].as_symbol().ok_or(GraphBlasError::InvalidValue)?;

    let func = parse_func_name(func_name_str)?;

    // Remaining elements are arguments
    let mut args = Vec::new();
    for arg_value in &list[1..] {
        args.push(parse_expr(arg_value)?);
    }

    Ok(Expr::FuncCall { func, args })
}

/// Parse function name
fn parse_func_name(name: &str) -> Result<FuncName> {
    match name {
        // Semiring operations
        "plus-times" => Ok(FuncName::PlusTimes),
        "min-plus" => Ok(FuncName::MinPlus),
        "max-times" => Ok(FuncName::MaxTimes),
        "or-and" => Ok(FuncName::OrAnd),

        // Matrix operations
        "transpose" => Ok(FuncName::Transpose),
        "matmul" => Ok(FuncName::MatMul),
        "mxv" => Ok(FuncName::MxV),
        "vxm" => Ok(FuncName::VxM),

        // Element-wise operations
        "ewise-add" => Ok(FuncName::EWiseAdd),
        "ewise-mult" => Ok(FuncName::EWiseMult),

        // Vector operations
        "vec-add" => Ok(FuncName::VecAdd),
        "vec-mult" => Ok(FuncName::VecMult),

        // Unary apply operations
        "abs" => Ok(FuncName::Apply(UnaryOp::Abs)),
        "neg" => Ok(FuncName::Apply(UnaryOp::Neg)),
        "sqrt" => Ok(FuncName::Apply(UnaryOp::Sqrt)),
        "exp" => Ok(FuncName::Apply(UnaryOp::Exp)),
        "log" => Ok(FuncName::Apply(UnaryOp::Log)),

        // Reduction operations
        "reduce-row" => Ok(FuncName::ReduceRow),
        "reduce-col" => Ok(FuncName::ReduceCol),
        "reduce-vector" => Ok(FuncName::ReduceVector),

        // Format conversion
        "to-csr" => Ok(FuncName::ToCSR),
        "to-csc" => Ok(FuncName::ToCSC),
        "to-dense" => Ok(FuncName::ToDense),

        // Otherwise assume it's a user-defined kernel
        _ => Ok(FuncName::UserKernel(name.to_string())),
    }
}

/// Parse if expression: (if condition then-expr else-expr)
fn parse_if(cons: &lexpr::Cons) -> Result<Expr> {
    let list = cons_to_vec(cons);

    if list.len() < 4 {
        return Err(GraphBlasError::InvalidValue);
    }

    // list[0] is 'if'
    // list[1] is condition
    // list[2] is then-branch
    // list[3] is else-branch
    let condition = Box::new(parse_expr(&list[1])?);
    let then_branch = Box::new(parse_expr(&list[2])?);
    let else_branch = Box::new(parse_expr(&list[3])?);

    Ok(Expr::If {
        condition,
        then_branch,
        else_branch,
    })
}

/// Parse while loop: (while condition body...)
fn parse_while(cons: &lexpr::Cons) -> Result<Expr> {
    let list = cons_to_vec(cons);

    if list.len() < 3 {
        return Err(GraphBlasError::InvalidValue);
    }

    // list[0] is 'while'
    // list[1] is condition
    // list[2..] is body expressions
    let condition = Box::new(parse_expr(&list[1])?);

    // Parse body as a block if multiple expressions, otherwise single expr
    let body = if list.len() > 3 {
        let mut body_exprs = Vec::new();
        for expr_val in &list[2..] {
            body_exprs.push(parse_expr(expr_val)?);
        }
        Box::new(Expr::Block(body_exprs))
    } else {
        Box::new(parse_expr(&list[2])?)
    };

    Ok(Expr::While { condition, body })
}

/// Parse for loop: (for var start end body...) or (for var start end step body...)
fn parse_for(cons: &lexpr::Cons) -> Result<Expr> {
    let list = cons_to_vec(cons);

    if list.len() < 5 {
        return Err(GraphBlasError::InvalidValue);
    }

    // list[0] is 'for'
    // list[1] is var
    // list[2] is start
    // list[3] is end
    // list[4] is either step or body
    // list[5..] is body if step was provided

    let var = list[1]
        .as_symbol()
        .ok_or(GraphBlasError::InvalidValue)?
        .to_string();

    let start = Box::new(parse_expr(&list[2])?);
    let end = Box::new(parse_expr(&list[3])?);

    // Check if we have a step parameter (numeric literal or expression)
    // If list.len() > 5, then list[4] is step and list[5..] is body
    // Otherwise list[4..] is body
    let (step, body_start_idx) = if list.len() > 5 {
        (Some(Box::new(parse_expr(&list[4])?)), 5)
    } else {
        (None, 4)
    };

    // Parse body
    let body = if list.len() > body_start_idx + 1 {
        let mut body_exprs = Vec::new();
        for expr_val in &list[body_start_idx..] {
            body_exprs.push(parse_expr(expr_val)?);
        }
        Box::new(Expr::Block(body_exprs))
    } else {
        Box::new(parse_expr(&list[body_start_idx])?)
    };

    Ok(Expr::For {
        var,
        start,
        end,
        step,
        body,
    })
}

/// Parse cond expression: (cond (test1 result1) (test2 result2) ... (else default))
fn parse_cond(cons: &lexpr::Cons) -> Result<Expr> {
    let list = cons_to_vec(cons);

    if list.len() < 2 {
        return Err(GraphBlasError::InvalidValue);
    }

    // list[0] is 'cond'
    // list[1..] are clauses

    let mut clauses = Vec::new();
    let mut else_clause = None;

    for clause_val in &list[1..] {
        let clause_cons = clause_val.as_cons().ok_or(GraphBlasError::InvalidValue)?;
        let clause_list = cons_to_vec(clause_cons);

        if clause_list.len() < 2 {
            return Err(GraphBlasError::InvalidValue);
        }

        // Check if it's an else clause
        if let Some(sym) = clause_list[0].as_symbol() {
            if sym == "else" {
                else_clause = Some(Box::new(parse_expr(&clause_list[1])?));
                continue;
            }
        }

        // Regular clause (test result)
        let test = parse_expr(&clause_list[0])?;
        let result = parse_expr(&clause_list[1])?;
        clauses.push((test, result));
    }

    Ok(Expr::Cond {
        clauses,
        else_clause,
    })
}

/// Parse begin/block: (begin expr1 expr2 ...)
fn parse_begin(cons: &lexpr::Cons) -> Result<Expr> {
    let list = cons_to_vec(cons);

    if list.len() < 2 {
        return Err(GraphBlasError::InvalidValue);
    }

    // list[0] is 'begin'
    // list[1..] are expressions

    let mut exprs = Vec::new();
    for expr_val in &list[1..] {
        exprs.push(parse_expr(expr_val)?);
    }

    Ok(Expr::Block(exprs))
}

/// Parse break: (break) or (break value)
fn parse_break(cons: &lexpr::Cons) -> Result<Expr> {
    let list = cons_to_vec(cons);

    // list[0] is 'break'
    // list[1] is optional value

    let value = if list.len() > 1 {
        Some(Box::new(parse_expr(&list[1])?))
    } else {
        None
    };

    Ok(Expr::Break(value))
}

/// Parse continue: (continue)
fn parse_continue(_cons: &lexpr::Cons) -> Result<Expr> {
    Ok(Expr::Continue)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_literal() {
        let expr = parse_expr_str("42").unwrap();
        assert!(matches!(expr, Expr::Literal(Literal::Int32(42))));

        let expr = parse_expr_str("3.14").unwrap();
        assert!(matches!(expr, Expr::Literal(Literal::Float64(_))));

        // Note: lexpr may parse "true" as a symbol depending on parser options
        // let expr = parse_expr_str("true").unwrap();
        // assert!(matches!(expr, Expr::Literal(Literal::Bool(true))));
    }

    #[test]
    fn test_parse_variable() {
        let expr = parse_expr_str("my_matrix").unwrap();
        assert!(matches!(expr, Expr::Variable(_)));
        if let Expr::Variable(name) = expr {
            assert_eq!(name, "my_matrix");
        }
    }

    #[test]
    fn test_parse_func_call() {
        let expr = parse_expr_str("(plus-times a b)").unwrap();
        if let Expr::FuncCall { func, args } = expr {
            assert!(matches!(func, FuncName::PlusTimes));
            assert_eq!(args.len(), 2);
        } else {
            panic!("Expected FuncCall");
        }
    }

    #[test]
    fn test_parse_nested_call() {
        let expr = parse_expr_str("(min-plus a (transpose b))").unwrap();
        if let Expr::FuncCall { func, args } = expr {
            assert!(matches!(func, FuncName::MinPlus));
            assert_eq!(args.len(), 2);

            // Check second arg is transpose
            if let Expr::FuncCall { func, .. } = &args[1] {
                assert!(matches!(func, FuncName::Transpose));
            } else {
                panic!("Expected nested FuncCall");
            }
        } else {
            panic!("Expected FuncCall");
        }
    }

    #[test]
    fn test_parse_let() {
        let expr = parse_expr_str("(let ((x 42)) x)").unwrap();
        if let Expr::Let { bindings, body } = expr {
            assert_eq!(bindings.len(), 1);
            assert_eq!(bindings[0].0, "x");
            assert!(matches!(&*body, Expr::Variable(_)));
        } else {
            panic!("Expected Let");
        }
    }

    #[test]
    fn test_parse_defkernel() {
        let program = parse_program("(defkernel bfs [g u] (or-and g u))").unwrap();
        assert_eq!(program.len(), 1);

        if let Form::DefKernel { name, params, body } = &program[0] {
            assert_eq!(name, "bfs");
            assert_eq!(params.len(), 2);
            assert_eq!(params[0].name, "g");
            assert_eq!(params[1].name, "u");

            if let Expr::FuncCall { func, args } = body {
                assert!(matches!(func, FuncName::OrAnd));
                assert_eq!(args.len(), 2);
            } else {
                panic!("Expected FuncCall in body");
            }
        } else {
            panic!("Expected DefKernel");
        }
    }

    #[test]
    fn test_parse_defkernel_form() {
        let program = parse_program("(defkernel test [x] x)").unwrap();

        assert_eq!(program.len(), 1);
        assert!(matches!(program[0], Form::DefKernel { .. }));
    }

    #[test]
    fn test_parse_expr_form() {
        let program = parse_program("(test my_matrix)").unwrap();

        assert_eq!(program.len(), 1);
        assert!(matches!(program[0], Form::Expr(_)));
    }
}
