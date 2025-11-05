/*
 * GraphBLAS C API Header
 *
 * Rust implementation of GraphBLAS C API v2.1.0
 * Link-compatible with SuiteSparse:GraphBLAS
 */

#ifndef GRAPHBLAS_H
#define GRAPHBLAS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Version Information
 */
#define GRB_VERSION 2
#define GRB_SUBVERSION 1

/*
 * Return Codes
 */
typedef int32_t GrB_Info;

/* Success codes (>= 0) */
#define GrB_SUCCESS 0
#define GrB_NO_VALUE 1

/* Error codes (< 0) */
#define GrB_UNINITIALIZED_OBJECT (-1)
#define GrB_NULL_POINTER (-2)
#define GrB_INVALID_VALUE (-3)
#define GrB_INVALID_INDEX (-4)
#define GrB_DOMAIN_MISMATCH (-5)
#define GrB_DIMENSION_MISMATCH (-6)
#define GrB_OUTPUT_NOT_EMPTY (-7)
#define GrB_NOT_IMPLEMENTED (-8)
#define GrB_PANIC (-101)
#define GrB_OUT_OF_MEMORY (-102)
#define GrB_INSUFFICIENT_SPACE (-103)
#define GrB_INVALID_OBJECT (-104)
#define GrB_INDEX_OUT_OF_BOUNDS (-105)
#define GrB_EMPTY_OBJECT (-106)

/*
 * Index Type
 */
typedef uint64_t GrB_Index;

/*
 * Opaque Object Types
 */

/* Type descriptor */
typedef struct GrB_Type_opaque *GrB_Type;

/* Matrix (m x n sparse matrix) */
typedef struct GrB_Matrix_opaque *GrB_Matrix;

/* Vector (n x 1 sparse vector) */
typedef struct GrB_Vector_opaque *GrB_Vector;

/* Scalar (1 x 1 matrix) */
typedef struct GrB_Scalar_opaque *GrB_Scalar;

/* Semiring */
typedef struct GrB_Semiring_opaque *GrB_Semiring;

/* Monoid */
typedef struct GrB_Monoid_opaque *GrB_Monoid;

/* Binary operator */
typedef struct GrB_BinaryOp_opaque *GrB_BinaryOp;

/* Unary operator */
typedef struct GrB_UnaryOp_opaque *GrB_UnaryOp;

/* Index binary operator */
typedef struct GrB_IndexBinaryOp_opaque *GrB_IndexBinaryOp;

/* Index unary operator */
typedef struct GrB_IndexUnaryOp_opaque *GrB_IndexUnaryOp;

/*
 * Predefined Types
 */

/* Boolean */
extern const GrB_Type GrB_BOOL;

/* Signed integers */
extern const GrB_Type GrB_INT8;
extern const GrB_Type GrB_INT16;
extern const GrB_Type GrB_INT32;
extern const GrB_Type GrB_INT64;

/* Unsigned integers */
extern const GrB_Type GrB_UINT8;
extern const GrB_Type GrB_UINT16;
extern const GrB_Type GrB_UINT32;
extern const GrB_Type GrB_UINT64;

/* Floating point */
extern const GrB_Type GrB_FP32;
extern const GrB_Type GrB_FP64;

/*
 * Type Functions
 */

/* Create a new user-defined type */
GrB_Info GrB_Type_new(
    GrB_Type *type,
    size_t sizeof_ctype
);

/* Free a type */
GrB_Info GrB_Type_free(
    GrB_Type *type
);

/*
 * Matrix Functions
 */

/* Create a new matrix */
GrB_Info GrB_Matrix_new(
    GrB_Matrix *A,
    GrB_Type type,
    GrB_Index nrows,
    GrB_Index ncols
);

/* Free a matrix */
GrB_Info GrB_Matrix_free(
    GrB_Matrix *A
);

/* Get number of rows */
GrB_Info GrB_Matrix_nrows(
    GrB_Index *nrows,
    GrB_Matrix A
);

/* Get number of columns */
GrB_Info GrB_Matrix_ncols(
    GrB_Index *ncols,
    GrB_Matrix A
);

/* Get number of stored values */
GrB_Info GrB_Matrix_nvals(
    GrB_Index *nvals,
    GrB_Matrix A
);

/*
 * Vector Functions
 */

/* Create a new vector */
GrB_Info GrB_Vector_new(
    GrB_Vector *v,
    GrB_Type type,
    GrB_Index size
);

/* Free a vector */
GrB_Info GrB_Vector_free(
    GrB_Vector *v
);

/* Get vector size */
GrB_Info GrB_Vector_size(
    GrB_Index *size,
    GrB_Vector v
);

/* Get number of stored values */
GrB_Info GrB_Vector_nvals(
    GrB_Index *nvals,
    GrB_Vector v
);

/*
 * Scalar Functions
 */

/* Create a new scalar */
GrB_Info GrB_Scalar_new(
    GrB_Scalar *s,
    GrB_Type type
);

/* Free a scalar */
GrB_Info GrB_Scalar_free(
    GrB_Scalar *s
);

/*
 * Semiring Functions
 */

/* Free a semiring */
GrB_Info GrB_Semiring_free(
    GrB_Semiring *semiring
);

/*
 * Monoid Functions
 */

/* Free a monoid */
GrB_Info GrB_Monoid_free(
    GrB_Monoid *monoid
);

/*
 * Binary Operator Functions
 */

/* Free a binary operator */
GrB_Info GrB_BinaryOp_free(
    GrB_BinaryOp *op
);

/*
 * Unary Operator Functions
 */

/* Free a unary operator */
GrB_Info GrB_UnaryOp_free(
    GrB_UnaryOp *op
);

/*
 * Index Binary Operator Functions
 */

/* Free an index binary operator */
GrB_Info GrB_IndexBinaryOp_free(
    GrB_IndexBinaryOp *op
);

/*
 * Index Unary Operator Functions
 */

/* Free an index unary operator */
GrB_Info GrB_IndexUnaryOp_free(
    GrB_IndexUnaryOp *op
);

#ifdef __cplusplus
}
#endif

#endif /* GRAPHBLAS_H */
