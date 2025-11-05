// Symbolic Shape Tracking
//
// Track matrix/vector dimensions symbolically to enable:
// - Shape inference
// - Fusion decisions
// - Error checking at compile time

use std::fmt;

/// Symbolic dimension (can be concrete or symbolic)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Dim {
    /// Concrete dimension size
    Concrete(usize),
    /// Symbolic dimension (e.g., "m", "n", "k")
    Symbolic(String),
}

impl Dim {
    pub fn is_concrete(&self) -> bool {
        matches!(self, Dim::Concrete(_))
    }

    pub fn is_symbolic(&self) -> bool {
        matches!(self, Dim::Symbolic(_))
    }

    /// Check if two dimensions are compatible (can be unified)
    pub fn compatible_with(&self, other: &Dim) -> bool {
        match (self, other) {
            (Dim::Concrete(a), Dim::Concrete(b)) => a == b,
            (Dim::Symbolic(a), Dim::Symbolic(b)) => a == b,
            _ => true, // Symbolic can unify with anything
        }
    }
}

impl fmt::Display for Dim {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Dim::Concrete(n) => write!(f, "{}", n),
            Dim::Symbolic(s) => write!(f, "{}", s),
        }
    }
}

/// Shape of a matrix or vector
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Shape {
    /// Scalar (no dimensions)
    Scalar,
    /// Vector with single dimension
    Vector(Dim),
    /// Matrix with two dimensions (rows, cols)
    Matrix(Dim, Dim),
}

impl Shape {
    /// Create a concrete scalar shape
    pub fn scalar() -> Self {
        Shape::Scalar
    }

    /// Create a concrete vector shape
    pub fn vector(size: usize) -> Self {
        Shape::Vector(Dim::Concrete(size))
    }

    /// Create a symbolic vector shape
    pub fn symbolic_vector(name: impl Into<String>) -> Self {
        Shape::Vector(Dim::Symbolic(name.into()))
    }

    /// Create a concrete matrix shape
    pub fn matrix(nrows: usize, ncols: usize) -> Self {
        Shape::Matrix(Dim::Concrete(nrows), Dim::Concrete(ncols))
    }

    /// Create a symbolic matrix shape
    pub fn symbolic_matrix(
        nrows: impl Into<String>,
        ncols: impl Into<String>,
    ) -> Self {
        Shape::Matrix(
            Dim::Symbolic(nrows.into()),
            Dim::Symbolic(ncols.into()),
        )
    }

    /// Create a matrix shape with mixed concrete/symbolic dimensions
    pub fn mixed_matrix(nrows: Dim, ncols: Dim) -> Self {
        Shape::Matrix(nrows, ncols)
    }

    /// Check if shape is fully concrete
    pub fn is_concrete(&self) -> bool {
        match self {
            Shape::Scalar => true,
            Shape::Vector(d) => d.is_concrete(),
            Shape::Matrix(r, c) => r.is_concrete() && c.is_concrete(),
        }
    }

    /// Get rank (number of dimensions)
    pub fn rank(&self) -> usize {
        match self {
            Shape::Scalar => 0,
            Shape::Vector(_) => 1,
            Shape::Matrix(_, _) => 2,
        }
    }

    /// Infer the result shape of matrix multiplication (A × B)
    pub fn matmul(a: &Shape, b: &Shape) -> Option<Shape> {
        match (a, b) {
            (Shape::Matrix(m, n1), Shape::Matrix(n2, k)) => {
                if n1.compatible_with(n2) {
                    Some(Shape::Matrix(m.clone(), k.clone()))
                } else {
                    None
                }
            }
            (Shape::Matrix(m, n1), Shape::Vector(n2)) => {
                if n1.compatible_with(n2) {
                    Some(Shape::Vector(m.clone()))
                } else {
                    None
                }
            }
            (Shape::Vector(n1), Shape::Matrix(n2, k)) => {
                if n1.compatible_with(n2) {
                    Some(Shape::Vector(k.clone()))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Infer the result shape of element-wise operations
    pub fn ewise(a: &Shape, b: &Shape) -> Option<Shape> {
        match (a, b) {
            (Shape::Scalar, s) | (s, Shape::Scalar) => Some(s.clone()),
            (Shape::Vector(n1), Shape::Vector(n2)) => {
                if n1.compatible_with(n2) {
                    Some(Shape::Vector(n1.clone()))
                } else {
                    None
                }
            }
            (Shape::Matrix(m1, n1), Shape::Matrix(m2, n2)) => {
                if m1.compatible_with(m2) && n1.compatible_with(n2) {
                    Some(Shape::Matrix(m1.clone(), n1.clone()))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Transpose shape
    pub fn transpose(&self) -> Option<Shape> {
        match self {
            Shape::Matrix(m, n) => Some(Shape::Matrix(n.clone(), m.clone())),
            _ => None,
        }
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Shape::Scalar => write!(f, "()"),
            Shape::Vector(d) => write!(f, "({})", d),
            Shape::Matrix(m, n) => write!(f, "({}, {})", m, n),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_concrete_shapes() {
        let s = Shape::scalar();
        assert!(s.is_concrete());
        assert_eq!(s.rank(), 0);

        let v = Shape::vector(100);
        assert!(v.is_concrete());
        assert_eq!(v.rank(), 1);

        let m = Shape::matrix(10, 20);
        assert!(m.is_concrete());
        assert_eq!(m.rank(), 2);
    }

    #[test]
    fn test_symbolic_shapes() {
        let v = Shape::symbolic_vector("n");
        assert!(!v.is_concrete());

        let m = Shape::symbolic_matrix("m", "n");
        assert!(!m.is_concrete());
    }

    #[test]
    fn test_matmul_inference() {
        let a = Shape::matrix(10, 20);
        let b = Shape::matrix(20, 30);
        let c = Shape::matmul(&a, &b).unwrap();
        assert_eq!(c, Shape::matrix(10, 30));

        // Incompatible dimensions
        let d = Shape::matrix(10, 15);
        assert!(Shape::matmul(&a, &d).is_none());

        // Symbolic
        let e = Shape::symbolic_matrix("m", "n");
        let f = Shape::symbolic_matrix("n", "k");
        let g = Shape::matmul(&e, &f).unwrap();
        assert_eq!(g, Shape::symbolic_matrix("m", "k"));
    }

    #[test]
    fn test_ewise_inference() {
        let a = Shape::matrix(10, 20);
        let b = Shape::matrix(10, 20);
        let c = Shape::ewise(&a, &b).unwrap();
        assert_eq!(c, a);

        // Incompatible
        let d = Shape::matrix(10, 30);
        assert!(Shape::ewise(&a, &d).is_none());

        // With scalar
        let s = Shape::scalar();
        let e = Shape::ewise(&s, &a).unwrap();
        assert_eq!(e, a);
    }

    #[test]
    fn test_transpose() {
        let a = Shape::matrix(10, 20);
        let b = a.transpose().unwrap();
        assert_eq!(b, Shape::matrix(20, 10));
    }
}
